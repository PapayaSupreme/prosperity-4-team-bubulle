"""
Round 5 Product Behavior Classifier

Purpose
-------
Offline analysis script for Prosperity Round 5 unknown products.
It reads prices_round_5_day_*.csv and trades_round_5_day_*.csv files, computes
behavior attributes per product, classifies each product, and exports:
  - round5_product_behavior_classes.csv
  - round5_product_behavior_classes.json
  - round5_product_behavior_summary.txt

Classes
-------
ANCHOR_MEAN_REVERTING : bounded around stable anchor, frequent median crossing, negative autocorr
DRIFT                 : stable linear trend / line-like asset
RANDOM_WALK_PASSIVE   : no strong trend/anchor, only passive spread capture if traded
JUMPY_UNSTABLE         : large jumps/outliers; dangerous for naive market making
BAD_LIQUIDITY          : too wide spread / too few points / bad book quality
DISABLED              : no clear exploitable structure

Usage
-----
Put this script in the same folder as the round 5 CSVs and run:
    python round5_behavior_classifier.py

Or pass a data directory:
    python round5_behavior_classifier.py --data-dir path/to/data
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prosperity4.analysis.data import read_all_round5_prices, read_all_round5_trades


def _round5_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ROUND_5"


def load_round5_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ROUND_5 data using the same shared helpers as `round5_analysis.py`."""

    prices = read_all_round5_prices()
    trades = read_all_round5_trades()
    if prices.empty:
        raise FileNotFoundError(f"No ROUND_5 price data found in {_round5_data_dir()}")
    return prices, trades


@dataclass
class ProductBehavior:
    product: str
    behavior_class: str
    confidence: float
    strategy_hint: str

    # price / book basics
    n_points: int
    n_days: int
    first_mid: float | None
    last_mid: float | None
    mean_mid: float | None
    median_mid: float | None
    std_mid: float | None
    min_mid: float | None
    max_mid: float | None
    range_mid: float | None
    range_pct: float | None
    mean_spread: float | None
    median_spread: float | None
    spread_pct: float | None
    mean_top_depth: float | None
    missing_book_pct: float | None

    # dynamics
    total_change: float | None
    total_change_pct: float | None
    slope_per_tick: float | None
    slope_per_100: float | None
    r2_linear: float | None
    return_std: float | None
    diff_std: float | None
    diff_mean: float | None
    lag1_return_autocorr: float | None
    residual_lag1_autocorr: float | None
    median_crosses: int | None
    median_cross_rate: float | None
    mean_reversion_half_life_ticks: float | None
    jump_count: int | None
    jump_rate: float | None
    max_abs_jump: float | None

    # trade stats
    num_trades: int
    total_trade_volume: float
    avg_trade_qty: float | None

    notes: str


def build_market_frame(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    required = {"day", "timestamp", "product"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prices file is missing columns: {sorted(missing)}")

    numeric_cols = [
        "day", "timestamp", "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
        "bid_price_3", "bid_volume_3", "ask_price_1", "ask_volume_1", "ask_price_2",
        "ask_volume_2", "ask_price_3", "ask_volume_3", "mid_price", "profit_and_loss",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    df["global_time"] = df["day"] * 1_000_000 + df["timestamp"]

    has_book = df["bid_price_1"].notna() & df["ask_price_1"].notna()
    df["mid"] = np.where(has_book, (df["bid_price_1"] + df["ask_price_1"]) / 2.0, df.get("mid_price", np.nan))
    df["spread"] = np.where(has_book, df["ask_price_1"] - df["bid_price_1"], np.nan)
    df["depth_bid"] = df.get("bid_volume_1", pd.Series(0, index=df.index)).abs().fillna(0.0)
    df["depth_ask"] = df.get("ask_volume_1", pd.Series(0, index=df.index)).abs().fillna(0.0)
    df["top_depth"] = df["depth_bid"] + df["depth_ask"]

    grouped = df.groupby("product", group_keys=False)
    df["mid_diff"] = grouped["mid"].diff()
    df["mid_return"] = grouped["mid"].pct_change(fill_method=None)
    return df


def safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x) or not np.isfinite(float(x)):
            return None
        return float(x)
    except Exception:
        return None


def autocorr(x: np.ndarray, lag: int = 1) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag + 2:
        return None
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return safe_float(np.corrcoef(a, b)[0, 1])


def linear_fit_metrics(t: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, np.ndarray | None]:
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid].astype(float)
    y = y[valid].astype(float)
    if len(y) < 10 or np.std(t) < 1e-12:
        return None, None, None

    # normalize time for numerical stability; slope returned per original timestamp unit
    t0 = t[0]
    tn = t - t0
    try:
        slope, intercept = np.polyfit(tn, y, 1)
        pred = intercept + slope * tn
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return safe_float(slope), safe_float(r2), y - pred
    except Exception:
        return None, None, None


def estimate_half_life(residuals: np.ndarray) -> float | None:
    """AR(1)-style residual half-life. Lower = faster mean reversion."""
    rho = autocorr(residuals, lag=1)
    if rho is None or rho <= 0 or rho >= 0.999:
        return None
    try:
        return safe_float(-math.log(2.0) / math.log(rho))
    except Exception:
        return None


def count_median_crosses(y: np.ndarray, median: float) -> tuple[int, float]:
    centered = y - median
    centered = centered[np.isfinite(centered)]
    if len(centered) < 2:
        return 0, 0.0
    signs = np.sign(centered)
    # Fill zeros with previous non-zero sign to avoid fake crossings on exact median prints.
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    crosses = int(np.sum(signs[1:] * signs[:-1] < 0))
    return crosses, crosses / max(1, len(centered) - 1)


def classify_product(prod: pd.DataFrame, trades: pd.DataFrame) -> ProductBehavior:
    product = str(prod["product"].iloc[0])
    prod = prod.dropna(subset=["mid"]).copy()
    n = len(prod)

    if n == 0:
        return ProductBehavior(
            product=product,
            behavior_class="BAD_LIQUIDITY",
            confidence=0.0,
            strategy_hint="do_not_trade",
            n_points=0,
            n_days=0,
            first_mid=None,
            last_mid=None,
            mean_mid=None,
            median_mid=None,
            std_mid=None,
            min_mid=None,
            max_mid=None,
            range_mid=None,
            range_pct=None,
            mean_spread=None,
            median_spread=None,
            spread_pct=None,
            mean_top_depth=None,
            missing_book_pct=None,
            total_change=None,
            total_change_pct=None,
            slope_per_tick=None,
            slope_per_100=None,
            r2_linear=None,
            return_std=None,
            diff_std=None,
            diff_mean=None,
            lag1_return_autocorr=None,
            residual_lag1_autocorr=None,
            median_crosses=None,
            median_cross_rate=None,
            mean_reversion_half_life_ticks=None,
            jump_count=None,
            jump_rate=None,
            max_abs_jump=None,
            num_trades=0,
            total_trade_volume=0.0,
            avg_trade_qty=None,
            notes="no valid mid prices",
        )

    y = prod["mid"].to_numpy(dtype=float)
    t = prod["global_time"].to_numpy(dtype=float)
    diffs = np.diff(y)
    returns = pd.Series(y).pct_change().dropna().to_numpy(dtype=float)

    mean_mid = float(np.mean(y))
    median_mid = float(np.median(y))
    std_mid = float(np.std(y, ddof=1)) if n > 1 else 0.0
    min_mid = float(np.min(y))
    max_mid = float(np.max(y))
    range_mid = max_mid - min_mid
    range_pct = range_mid / median_mid if abs(median_mid) > 1e-12 else None

    spreads = prod["spread"].dropna().to_numpy(dtype=float)
    mean_spread = float(np.mean(spreads)) if len(spreads) else None
    median_spread = float(np.median(spreads)) if len(spreads) else None
    spread_pct = mean_spread / median_mid if mean_spread is not None and abs(median_mid) > 1e-12 else None
    mean_top_depth = float(prod["top_depth"].mean()) if "top_depth" in prod else None
    missing_book_pct = float(prod["spread"].isna().mean())

    slope, r2, residuals = linear_fit_metrics(t, y)
    slope_per_100 = slope * 100.0 if slope is not None else None
    total_change = float(y[-1] - y[0])
    total_change_pct = total_change / y[0] if abs(y[0]) > 1e-12 else None
    diff_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    diff_mean = float(np.mean(diffs)) if len(diffs) else 0.0
    return_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    ret_ac1 = autocorr(diffs, lag=1)
    resid_ac1 = autocorr(residuals, lag=1) if residuals is not None else None
    half_life = estimate_half_life(y - median_mid)
    crosses, cross_rate = count_median_crosses(y, median_mid)

    jump_threshold = max(3.0 * diff_std, (median_spread or 1.0) * 2.5, 1.0)
    jump_count = int(np.sum(np.abs(diffs - diff_mean) > jump_threshold)) if len(diffs) else 0
    jump_rate = jump_count / max(1, len(diffs))
    max_abs_jump = float(np.max(np.abs(diffs))) if len(diffs) else 0.0

    # Trade stats
    if not trades.empty and "symbol" in trades.columns:
        tr = trades[trades["symbol"] == product].copy()
    else:
        tr = pd.DataFrame()
    if not tr.empty:
        tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce")
        num_trades = int(len(tr))
        total_trade_volume = float(tr["quantity"].dropna().sum())
        avg_trade_qty = float(tr["quantity"].dropna().mean()) if tr["quantity"].notna().any() else None
    else:
        num_trades = 0
        total_trade_volume = 0.0
        avg_trade_qty = None

    n_days = int(prod["day"].nunique()) if "day" in prod else 0

    # Scores are intentionally conservative. We prefer DISABLED over false alpha.
    liquidity_bad = (
        n < 80
        or missing_book_pct > 0.10
        or (spread_pct is not None and spread_pct > 0.0040)
        or (mean_top_depth is not None and mean_top_depth < 4.0)
    )

    anchor_score = 0.0
    if range_pct is not None:
        anchor_score += max(0.0, 1.0 - range_pct / 0.030) * 2.0
    if abs(total_change_pct or 0.0) < 0.003:
        anchor_score += 1.0
    if cross_rate > 0.045:
        anchor_score += 1.5
    elif cross_rate > 0.025:
        anchor_score += 0.75
    if ret_ac1 is not None and ret_ac1 < -0.08:
        anchor_score += 1.0
    if half_life is not None and half_life < 20:
        anchor_score += 0.75
    if jump_rate > 0.035:
        anchor_score -= 1.5

    drift_score = 0.0
    if r2 is not None:
        drift_score += max(0.0, min(2.5, (r2 - 0.55) / 0.18))
    if abs(total_change_pct or 0.0) > 0.0035:
        drift_score += 1.0
    if resid_ac1 is not None and resid_ac1 < 0.85:
        drift_score += 0.5  # residuals do not simply wander away from the fitted line
    if cross_rate < 0.030:
        drift_score += 0.5
    if jump_rate > 0.035:
        drift_score -= 1.0

    random_walk_score = 0.0
    if r2 is not None and r2 < 0.35:
        random_walk_score += 1.0
    if abs(ret_ac1 or 0.0) < 0.12:
        random_walk_score += 1.0
    if range_pct is not None and 0.003 <= range_pct <= 0.080:
        random_walk_score += 0.75
    if jump_rate < 0.030:
        random_walk_score += 0.5

    notes_parts: list[str] = []
    if liquidity_bad:
        behavior = "BAD_LIQUIDITY"
        confidence = 0.80
        hint = "do_not_trade_or_size_1_only"
        notes_parts.append("bad liquidity / too few points / wide spread")
    elif jump_rate > 0.060 or (range_pct is not None and range_pct > 0.10):
        behavior = "JUMPY_UNSTABLE"
        confidence = min(0.95, 0.55 + jump_rate * 5.0)
        hint = "avoid_naive_mm; trade_only_with_specific_signal"
        notes_parts.append("large jumps/outliers")
    elif anchor_score >= 4.0 and anchor_score >= drift_score + 0.6:
        behavior = "ANCHOR_MEAN_REVERTING"
        confidence = min(0.98, anchor_score / 6.0)
        hint = f"AnchoredMarketMaker(anchor≈{median_mid:.2f})"
    elif drift_score >= 3.0 and drift_score >= anchor_score + 0.4:
        behavior = "DRIFT"
        confidence = min(0.98, drift_score / 5.0)
        hint = f"DriftMarketMaker(slope_per_100≈{slope_per_100 or 0.0:.3f})"
    elif random_walk_score >= 2.4:
        behavior = "RANDOM_WALK_PASSIVE"
        confidence = min(0.85, random_walk_score / 4.0)
        hint = "PassiveRandomWalkMM only when spread >= 2 ticks"
    else:
        behavior = "DISABLED"
        confidence = 0.55
        hint = "observe_only_until_stronger_pattern"
        notes_parts.append("no strong pattern")

    if ret_ac1 is not None:
        notes_parts.append(f"diff_ac1={ret_ac1:.3f}")
    if r2 is not None:
        notes_parts.append(f"linear_r2={r2:.3f}")
    notes_parts.append(f"anchor_score={anchor_score:.2f}")
    notes_parts.append(f"drift_score={drift_score:.2f}")
    notes_parts.append(f"rw_score={random_walk_score:.2f}")

    return ProductBehavior(
        product=product,
        behavior_class=behavior,
        confidence=round(float(confidence), 4),
        strategy_hint=hint,
        n_points=int(n),
        n_days=n_days,
        first_mid=safe_float(y[0]),
        last_mid=safe_float(y[-1]),
        mean_mid=safe_float(mean_mid),
        median_mid=safe_float(median_mid),
        std_mid=safe_float(std_mid),
        min_mid=safe_float(min_mid),
        max_mid=safe_float(max_mid),
        range_mid=safe_float(range_mid),
        range_pct=safe_float(range_pct),
        mean_spread=safe_float(mean_spread),
        median_spread=safe_float(median_spread),
        spread_pct=safe_float(spread_pct),
        mean_top_depth=safe_float(mean_top_depth),
        missing_book_pct=safe_float(missing_book_pct),
        total_change=safe_float(total_change),
        total_change_pct=safe_float(total_change_pct),
        slope_per_tick=safe_float(slope),
        slope_per_100=safe_float(slope_per_100),
        r2_linear=safe_float(r2),
        return_std=safe_float(return_std),
        diff_std=safe_float(diff_std),
        diff_mean=safe_float(diff_mean),
        lag1_return_autocorr=safe_float(ret_ac1),
        residual_lag1_autocorr=safe_float(resid_ac1),
        median_crosses=int(crosses),
        median_cross_rate=safe_float(cross_rate),
        mean_reversion_half_life_ticks=safe_float(half_life),
        jump_count=int(jump_count),
        jump_rate=safe_float(jump_rate),
        max_abs_jump=safe_float(max_abs_jump),
        num_trades=num_trades,
        total_trade_volume=safe_float(total_trade_volume) or 0.0,
        avg_trade_qty=safe_float(avg_trade_qty),
        notes="; ".join(notes_parts),
    )


def classify_all(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    market = build_market_frame(prices)
    rows = []
    for product, prod in market.groupby("product", sort=True):
        rows.append(asdict(classify_product(prod, trades)))
    out = pd.DataFrame(rows)
    order = {
        "ANCHOR_MEAN_REVERTING": 0,
        "DRIFT": 1,
        "RANDOM_WALK_PASSIVE": 2,
        "JUMPY_UNSTABLE": 3,
        "BAD_LIQUIDITY": 4,
        "DISABLED": 5,
    }
    out["class_rank"] = out["behavior_class"].map(order).fillna(99)
    out = out.sort_values(["class_rank", "confidence", "product"], ascending=[True, False, True])
    return out.drop(columns=["class_rank"])


def write_summary(df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("ROUND 5 PRODUCT BEHAVIOR CLASSIFICATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Class counts:")
    for cls, count in df["behavior_class"].value_counts().items():
        lines.append(f"  {cls:24s} {count:3d}")

    lines.append("")
    for cls in ["ANCHOR_MEAN_REVERTING", "DRIFT", "RANDOM_WALK_PASSIVE", "JUMPY_UNSTABLE", "BAD_LIQUIDITY", "DISABLED"]:
        sub = df[df["behavior_class"] == cls].copy()
        if sub.empty:
            continue
        lines.append("-" * 80)
        lines.append(cls)
        lines.append("-" * 80)
        cols = ["product", "confidence", "median_mid", "range_pct", "slope_per_100", "r2_linear", "lag1_return_autocorr", "mean_spread", "strategy_hint"]
        for _, row in sub[cols].head(25).iterrows():
            lines.append(
                f"{row['product']:32s} conf={row['confidence']:.2f} "
                f"anchor={row['median_mid']:.2f} range%={(row['range_pct'] or 0)*100:.3f} "
                f"slope100={(row['slope_per_100'] or 0):.3f} r2={(row['r2_linear'] or 0):.3f} "
                f"ac1={(row['lag1_return_autocorr'] or 0):.3f} spread={(row['mean_spread'] or 0):.2f} "
                f"=> {row['strategy_hint']}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = (args.out_dir or _round5_data_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prices, trades = load_round5_data()
    result = classify_all(prices, trades)

    csv_path = out_dir / "round5_product_behavior_classes.csv"
    json_path = out_dir / "round5_product_behavior_classes.json"
    summary_path = out_dir / "round5_product_behavior_summary.txt"

    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8")
    write_summary(result, summary_path)

    print(f"Loaded {len(prices)} price rows and {len(trades)} trade rows")
    print(f"Classified {len(result)} products")
    print("\nClass counts:")
    print(result["behavior_class"].value_counts().to_string())
    print(f"\nSaved CSV:     {csv_path}")
    print(f"Saved JSON:    {json_path}")
    print(f"Saved summary: {summary_path}")

    print("\nTop classified products:")
    display_cols = ["product", "behavior_class", "confidence", "median_mid", "range_pct", "slope_per_100", "r2_linear", "lag1_return_autocorr", "strategy_hint"]
    print(result[display_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
