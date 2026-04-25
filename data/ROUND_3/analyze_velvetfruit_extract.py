"""
Analyze VELVETFRUIT_EXTRACT price tendencies for Prosperity Round 3.

What it checks:
- anchor / mean-reversion strength around 5250 or estimated anchor
- AR(1) mean-reversion coefficient and half-life
- whether deviations above anchor revert differently than deviations below anchor
- order-book demand imbalance and whether it predicts next returns
- asymmetric behavior: upside vs downside volatility, return skew, jump frequency
- trade flow stats when trades files are available

Expected files in the same folder, or pass --data-dir:
    prices_round_3_day_0.csv, prices_round_3_day_1.csv, ...
    trades_round_3_day_0.csv, trades_round_3_day_1.csv, ...

Usage:
    python analyze_velvetfruit_extract.py --data-dir .
    python analyze_velvetfruit_extract.py --data-dir /path/to/files --anchor 5250 --plots
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting optional
    plt = None


PRODUCT = "VELVETFRUIT_EXTRACT"


@dataclass
class RegressionResult:
    alpha: float
    beta: float
    r2: float
    n: int


def _read_csv_semicolon(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def load_round3_files(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_paths = sorted(glob.glob(os.path.join(data_dir, "prices_round_3_day_*.csv")))
    trade_paths = sorted(glob.glob(os.path.join(data_dir, "trades_round_3_day_*.csv")))

    if not price_paths:
        raise FileNotFoundError(f"No prices_round_3_day_*.csv files found in {data_dir!r}")

    prices = pd.concat((_read_csv_semicolon(p) for p in price_paths), ignore_index=True)
    trades = pd.concat((_read_csv_semicolon(p) for p in trade_paths), ignore_index=True) if trade_paths else pd.DataFrame()

    return prices, trades


def add_book_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["product"] == PRODUCT].copy()
    if df.empty:
        raise ValueError(f"No {PRODUCT} rows found in prices file.")

    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)

    # Absolute volumes: Prosperity book volumes can be signed depending on parser/source.
    for col in [c for c in df.columns if "volume" in c]:
        df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    for col in [c for c in df.columns if "price" in c or c in ["mid_price"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["best_bid"] = df["bid_price_1"]
    df["best_ask"] = df["ask_price_1"]
    df["bid_vol_1"] = df["bid_volume_1"].fillna(0.0)
    df["ask_vol_1"] = df["ask_volume_1"].fillna(0.0)
    df["spread"] = df["best_ask"] - df["best_bid"]

    denom = df["bid_vol_1"] + df["ask_vol_1"]
    df["imbalance_l1"] = np.where(denom > 0, (df["bid_vol_1"] - df["ask_vol_1"]) / denom, np.nan)

    # Multi-level imbalance.
    bid_vol_cols = [c for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if c in df]
    ask_vol_cols = [c for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if c in df]
    df["bid_depth_3"] = df[bid_vol_cols].fillna(0).sum(axis=1)
    df["ask_depth_3"] = df[ask_vol_cols].fillna(0).sum(axis=1)
    denom3 = df["bid_depth_3"] + df["ask_depth_3"]
    df["imbalance_l3"] = np.where(denom3 > 0, (df["bid_depth_3"] - df["ask_depth_3"]) / denom3, np.nan)

    # Microprice: closer to ask when bid depth is large, closer to bid when ask depth is large.
    mp_denom = df["bid_vol_1"] + df["ask_vol_1"]
    df["microprice"] = np.where(
        mp_denom > 0,
        (df["best_ask"] * df["bid_vol_1"] + df["best_bid"] * df["ask_vol_1"]) / mp_denom,
        df["mid_price"],
    )

    # Time-series features per day to avoid day-boundary fake jumps.
    g = df.groupby("day", group_keys=False)
    df["ret_1"] = g["mid_price"].diff()
    df["ret_5"] = g["mid_price"].diff(5)
    df["fwd_ret_1"] = g["mid_price"].shift(-1) - df["mid_price"]
    df["fwd_ret_5"] = g["mid_price"].shift(-5) - df["mid_price"]
    return df


def ols_xy(x: Iterable[float], y: Iterable[float]) -> RegressionResult:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.var(x) == 0:
        return RegressionResult(np.nan, np.nan, np.nan, len(x))
    X = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = alpha + beta * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return RegressionResult(float(alpha), float(beta), float(r2), int(len(x)))


def half_life_from_phi(phi: float) -> float:
    # x_t = phi * x_{t-1}; half-life = ln(0.5)/ln(|phi|), only meaningful for 0 < phi < 1.
    if not np.isfinite(phi) or phi <= 0 or phi >= 1:
        return np.nan
    return float(math.log(0.5) / math.log(phi))


def mean_reversion_analysis(df: pd.DataFrame, anchor: float) -> dict[str, float]:
    x_prev = []
    x_next = []
    delta_next = []

    for _, d in df.groupby("day"):
        dev = d["mid_price"] - anchor
        x_prev.extend(dev.iloc[:-1].to_numpy())
        x_next.extend(dev.iloc[1:].to_numpy())
        delta_next.extend((d["mid_price"].shift(-1) - d["mid_price"]).iloc[:-1].to_numpy())

    ar = ols_xy(x_prev, x_next)              # dev_{t+1} = a + phi dev_t
    revert = ols_xy(x_prev, delta_next)      # price_change_{t+1} = a + b dev_t ; b should be negative

    x_prev_np = np.asarray(x_prev, dtype=float)
    dn = np.asarray(delta_next, dtype=float)
    mask = np.isfinite(x_prev_np) & np.isfinite(dn) & (np.abs(x_prev_np) > 0)
    hit_rate = np.mean(np.sign(dn[mask]) == -np.sign(x_prev_np[mask])) if np.any(mask) else np.nan

    abs_dev = np.abs(df["mid_price"] - anchor)
    return {
        "anchor_used": anchor,
        "mean_mid": float(df["mid_price"].mean()),
        "median_mid": float(df["mid_price"].median()),
        "std_mid": float(df["mid_price"].std()),
        "mean_abs_dev_from_anchor": float(abs_dev.mean()),
        "p95_abs_dev_from_anchor": float(abs_dev.quantile(0.95)),
        "ar1_phi_dev": ar.beta,
        "ar1_r2": ar.r2,
        "half_life_ticks": half_life_from_phi(ar.beta),
        "next_change_vs_dev_slope": revert.beta,
        "reversion_hit_rate_1tick": float(hit_rate),
    }


def asymmetry_analysis(df: pd.DataFrame, anchor: float) -> dict[str, float]:
    df = df.copy()
    df["dev"] = df["mid_price"] - anchor
    above = df[df["dev"] > 0]
    below = df[df["dev"] < 0]
    ret = df["ret_1"].dropna()

    return {
        "frac_time_above_anchor": float((df["dev"] > 0).mean()),
        "frac_time_below_anchor": float((df["dev"] < 0).mean()),
        "mean_dev_when_above": float(above["dev"].mean()) if len(above) else np.nan,
        "mean_abs_dev_when_below": float((-below["dev"]).mean()) if len(below) else np.nan,
        "upside_ret_std": float(ret[ret > 0].std()) if np.any(ret > 0) else np.nan,
        "downside_ret_std": float(ret[ret < 0].std()) if np.any(ret < 0) else np.nan,
        "return_skew": float(ret.skew()) if len(ret) else np.nan,
        "large_up_moves_gt_2std": float((ret > 2 * ret.std()).mean()) if len(ret) else np.nan,
        "large_down_moves_lt_minus_2std": float((ret < -2 * ret.std()).mean()) if len(ret) else np.nan,
    }


def imbalance_analysis(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for imb_col in ["imbalance_l1", "imbalance_l3", "microprice_minus_mid"]:
        if imb_col == "microprice_minus_mid":
            x = df["microprice"] - df["mid_price"]
        else:
            x = df[imb_col]
        for horizon in ["fwd_ret_1", "fwd_ret_5"]:
            y = df[horizon]
            reg = ols_xy(x, y)
            corr = pd.concat([x, y], axis=1).corr().iloc[0, 1]
            out[f"{imb_col}_to_{horizon}_beta"] = reg.beta
            out[f"{imb_col}_to_{horizon}_corr"] = float(corr) if np.isfinite(corr) else np.nan
            out[f"{imb_col}_to_{horizon}_r2"] = reg.r2
    return out


def trade_analysis(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "symbol" not in trades:
        return {}
    t = trades[trades["symbol"] == PRODUCT].copy()
    if t.empty:
        return {}
    t["price"] = pd.to_numeric(t["price"], errors="coerce")
    t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce")
    return {
        "trade_count": int(len(t)),
        "traded_volume": float(t["quantity"].sum()),
        "vwap": float((t["price"] * t["quantity"]).sum() / t["quantity"].sum()) if t["quantity"].sum() else np.nan,
        "mean_trade_size": float(t["quantity"].mean()),
        "max_trade_size": float(t["quantity"].max()),
    }


def print_section(title: str, d: dict[str, float]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in d.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k:36s}: {v}")
        elif isinstance(v, float):
            print(f"{k:36s}: {v: .6f}")
        else:
            print(f"{k:36s}: {v}")


def make_plots(df: pd.DataFrame, anchor: float, out_dir: str) -> None:
    if plt is None:
        print("matplotlib not installed; skipping plots.")
        return
    os.makedirs(out_dir, exist_ok=True)

    plot_x = df["timestamp"] + df["day"] * (df["timestamp"].max() + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_x, df["mid_price"], linewidth=1)
    plt.axhline(anchor, linestyle="--", linewidth=1)
    plt.title(f"{PRODUCT} mid price vs anchor")
    plt.xlabel("global timestamp")
    plt.ylabel("mid price")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "velvetfruit_price_anchor.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["mid_price"] - anchor, df["fwd_ret_1"], s=8, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.title("Mean reversion: next return vs deviation from anchor")
    plt.xlabel("mid - anchor")
    plt.ylabel("next 1-tick mid change")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mean_reversion_scatter.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["imbalance_l1"], df["fwd_ret_1"], s=8, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.title("Demand imbalance vs next return")
    plt.xlabel("L1 imbalance = (bid vol - ask vol)/(bid vol + ask vol)")
    plt.ylabel("next 1-tick mid change")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "imbalance_vs_next_return.png"), dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Folder containing prices_round_3_day_*.csv and trades_round_3_day_*.csv")
    parser.add_argument("--anchor", type=float, default=5250.0, help="Anchor to test for mean reversion")
    parser.add_argument("--estimate-anchor", action="store_true", help="Use median mid price as anchor instead of --anchor")
    parser.add_argument("--plots", action="store_true", help="Save diagnostic plots")
    parser.add_argument("--out-dir", default="analysis_output", help="Output folder for plots")
    args = parser.parse_args()

    prices, trades = load_round3_files(args.data_dir)
    df = add_book_features(prices)
    anchor = float(df["mid_price"].median()) if args.estimate_anchor else args.anchor

    print(f"Loaded {len(df)} {PRODUCT} book rows across days: {sorted(df['day'].unique().tolist())}")
    if not trades.empty:
        print(f"Loaded {len(trades)} total trades rows.")

    print_section("PRICE / MEAN REVERSION", mean_reversion_analysis(df, anchor))
    print_section("ASYMMETRY", asymmetry_analysis(df, anchor))
    print_section("ORDER-BOOK DEMAND IMBALANCE", imbalance_analysis(df))
    ta = trade_analysis(trades)
    if ta:
        print_section("TRADE FLOW", ta)

    # More readable interpretation hints.
    mr = mean_reversion_analysis(df, anchor)
    print("\n" + "=" * 80)
    print("INTERPRETATION HINTS")
    print("=" * 80)
    print("- Strong mean reversion: ar1_phi_dev clearly below 1, next_change_vs_dev_slope negative, hit_rate > 0.50.")
    print("- Half-life is in ticks: lower = faster pull back to anchor.")
    print("- Demand imbalance matters if imbalance beta/corr/R2 are meaningfully non-zero.")
    print("- Asymmetry exists if above-anchor and below-anchor deviations/reversions differ a lot, or return_skew is far from 0.")
    if np.isfinite(mr["next_change_vs_dev_slope"]):
        if mr["next_change_vs_dev_slope"] < 0:
            print("- Your data shows negative next_change_vs_dev_slope, i.e. deviations tend to be pulled back toward anchor.")
        else:
            print("- Warning: next_change_vs_dev_slope is not negative, so the chosen anchor may be weak/wrong or this sample is trending.")

    if args.plots:
        make_plots(df, anchor, os.path.join(args.data_dir, args.out_dir))
        print(f"\nSaved plots to: {os.path.join(args.data_dir, args.out_dir)}")


if __name__ == "__main__":
    main()
