from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

# Round 4 naming assumptions.
# VEV coupons are named like VEV_5400, VEV_5500, ... where the suffix is the strike.
# Their relevant underlying is VELVETFRUIT_EXTRACT.
UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
COUPON_PREFIX = "VEV_"


def load_data(price_csv_path: str | Path, trades_csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(price_csv_path, sep=";")
    trades = pd.read_csv(trades_csv_path, sep=";")
    prices.columns = prices.columns.str.strip()
    trades.columns = trades.columns.str.strip()
    return prices, trades


def load_round_days(data_dir: Path, days: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and stack price/trade files, adding day explicitly because timestamps reset by day."""
    price_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for day in days:
        prices_path = data_dir / f"prices_round_4_day_{day}.csv"
        trades_path = data_dir / f"trades_round_4_day_{day}.csv"
        prices_df, trades_df = load_data(prices_path, trades_path)
        prices_df["day"] = day
        trades_df["day"] = day
        price_frames.append(prices_df)
        trade_frames.append(trades_df)

    prices = pd.concat(price_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    return prices, trades


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def enrich_prices(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """
    Add own-product future mids/returns.
    Horizon is measured in product rows, not raw timestamp distance.
    """
    px = prices.copy()
    px = _numeric(px, ["day", "timestamp", "mid_price"])
    px["product"] = px["product"].astype(str).str.strip()
    px = px.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)

    for h in horizons:
        px[f"own_future_mid_t_plus_{h}"] = px.groupby(["day", "product"])["mid_price"].shift(-h)
        px[f"own_forward_return_t_plus_{h}"] = px[f"own_future_mid_t_plus_{h}"] - px["mid_price"]

    return px


def add_underlying_context(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Build a timestamp-aligned underlying table for VEV coupon trades."""
    und = prices[prices["product"] == UNDERLYING_SYMBOL].copy()
    keep = ["day", "timestamp", "mid_price"]
    for h in horizons:
        keep += [f"own_future_mid_t_plus_{h}", f"own_forward_return_t_plus_{h}"]
    und = und[keep].rename(columns={"mid_price": "underlying_mid"})
    for h in horizons:
        und = und.rename(
            columns={
                f"own_future_mid_t_plus_{h}": f"underlying_future_mid_t_plus_{h}",
                f"own_forward_return_t_plus_{h}": f"underlying_forward_return_t_plus_{h}",
            }
        )
    return und


def parse_coupon_strike(symbol: str) -> float | np.nan:
    m = re.fullmatch(r"VEV_(\d+)", str(symbol).strip())
    return float(m.group(1)) if m else np.nan


def merge_trades_with_market(prices: pd.DataFrame, trades: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    trades = trades.copy()
    trades.columns = trades.columns.str.strip()
    trades = _numeric(trades, ["day", "timestamp", "price", "quantity"])
    trades["symbol"] = trades["symbol"].astype(str).str.strip()

    price_cols = ["day", "timestamp", "product", "mid_price"]
    for h in horizons:
        price_cols += [f"own_future_mid_t_plus_{h}", f"own_forward_return_t_plus_{h}"]

    merged = trades.merge(
        prices[price_cols],
        left_on=["day", "timestamp", "symbol"],
        right_on=["day", "timestamp", "product"],
        how="left",
    ).drop(columns=["product"])

    underlying = add_underlying_context(prices, horizons)
    merged = merged.merge(underlying, on=["day", "timestamp"], how="left")

    merged["is_coupon"] = merged["symbol"].str.startswith(COUPON_PREFIX)
    merged["strike"] = merged["symbol"].map(parse_coupon_strike)
    return merged


def estimate_call_delta_proxy(underlying_mid: pd.Series, strike: pd.Series) -> pd.Series:
    """
    Rough call-like delta proxy for VEV coupons.
    This is not Black-Scholes. It only weights underlying direction more for near/in-the-money coupons.
    """
    x = (underlying_mid - strike) / 120.0
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def compute_scores(df: pd.DataFrame, horizons: tuple[int, ...], exec_weight: float = 0.15) -> pd.DataFrame:
    """
    Product trades:
        Score mostly by realized future PnL from the trade price to future mid.
        Execution edge vs current mid is only a small secondary term.

    VEV coupons:
        Score with two components:
        1) realized coupon future PnL, using coupon future mid;
        2) underlying-informed score, using VELVETFRUIT_EXTRACT future return and delta proxy.
    """
    out = df.copy()
    out["mid_price_missing"] = out["mid_price"].isna()

    out["buy_exec_edge"] = out["mid_price"] - out["price"]
    out["sell_exec_edge"] = out["price"] - out["mid_price"]
    out["buy_exec_score"] = out["buy_exec_edge"] * out["quantity"]
    out["sell_exec_score"] = out["sell_exec_edge"] * out["quantity"]

    out["delta_proxy"] = np.where(
        out["is_coupon"],
        estimate_call_delta_proxy(out["underlying_mid"], out["strike"]),
        np.nan,
    )

    for h in horizons:
        own_future = f"own_future_mid_t_plus_{h}"
        own_ret = f"own_forward_return_t_plus_{h}"
        und_ret = f"underlying_forward_return_t_plus_{h}"

        # Realized future PnL from the actual trade price, not just current mid.
        out[f"buy_realized_pnl_t_plus_{h}"] = (out[own_future] - out["price"]) * out["quantity"]
        out[f"sell_realized_pnl_t_plus_{h}"] = (out["price"] - out[own_future]) * out["quantity"]

        # Directional hit from own product future movement.
        out[f"buy_hit_t_plus_{h}"] = out[own_ret] > 0
        out[f"sell_hit_t_plus_{h}"] = out[own_ret] < 0

        # Normal products: future realized PnL dominates, current execution edge is just a tie-breaker.
        out[f"buy_product_score_t_plus_{h}"] = (
            out[f"buy_realized_pnl_t_plus_{h}"] + exec_weight * out["buy_exec_score"]
        )
        out[f"sell_product_score_t_plus_{h}"] = (
            out[f"sell_realized_pnl_t_plus_{h}"] + exec_weight * out["sell_exec_score"]
        )

        # Coupons: combine actual coupon PnL with an underlying directional component.
        # Buying calls is suspicious if underlying rises after the trade; selling is suspicious if underlying falls.
        out[f"buy_coupon_underlying_score_t_plus_{h}"] = (
            out[und_ret] * out["delta_proxy"] * out["quantity"]
        )
        out[f"sell_coupon_underlying_score_t_plus_{h}"] = (
            -out[und_ret] * out["delta_proxy"] * out["quantity"]
        )
        out[f"buy_coupon_score_t_plus_{h}"] = (
            out[f"buy_realized_pnl_t_plus_{h}"] + out[f"buy_coupon_underlying_score_t_plus_{h}"]
        )
        out[f"sell_coupon_score_t_plus_{h}"] = (
            out[f"sell_realized_pnl_t_plus_{h}"] + out[f"sell_coupon_underlying_score_t_plus_{h}"]
        )
        out[f"buy_coupon_underlying_hit_t_plus_{h}"] = out[und_ret] > 0
        out[f"sell_coupon_underlying_hit_t_plus_{h}"] = out[und_ret] < 0

    return out


def suspicious_trades(scored: pd.DataFrame, horizon: int, top_n: int = 20, coupons: bool = False) -> pd.DataFrame:
    """Return top suspicious individual buyer/seller trade actions."""
    rows = []
    subset = scored[scored["is_coupon"] == coupons].copy()
    subset = subset[~subset["mid_price_missing"]]

    if coupons:
        buy_score = f"buy_coupon_score_t_plus_{horizon}"
        sell_score = f"sell_coupon_score_t_plus_{horizon}"
        extra_cols = [
            "underlying_mid", "strike", "delta_proxy",
            f"underlying_forward_return_t_plus_{horizon}",
            f"buy_coupon_underlying_score_t_plus_{horizon}",
            f"sell_coupon_underlying_score_t_plus_{horizon}",
        ]
    else:
        buy_score = f"buy_product_score_t_plus_{horizon}"
        sell_score = f"sell_product_score_t_plus_{horizon}"
        extra_cols = [f"own_forward_return_t_plus_{horizon}"]

    base_cols = ["day", "timestamp", "symbol", "price", "quantity", "mid_price", "buyer", "seller"]

    b = subset[base_cols + extra_cols + [buy_score]].copy()
    b = b.rename(columns={"buyer": "participant", buy_score: "score"})
    b["side"] = "BUY"
    b = b[b["score"] > 0]

    s = subset[base_cols + extra_cols + [sell_score]].copy()
    s = s.rename(columns={"seller": "participant", sell_score: "score"})
    s["side"] = "SELL"
    s = s[s["score"] > 0]

    actions = pd.concat([b, s], ignore_index=True)
    actions["participant"] = actions["participant"].fillna("").astype(str).str.strip()
    actions = actions[actions["participant"] != ""]
    return actions.sort_values(["score", "quantity"], ascending=[False, False]).head(top_n)


def participant_summary(scored: pd.DataFrame, horizon: int, min_trades: int = 3, coupons: bool = False) -> pd.DataFrame:
    """Aggregate participant suspiciousness with hit rate and symbol breakdown."""
    subset = scored[scored["is_coupon"] == coupons].copy()

    if coupons:
        buy_score = f"buy_coupon_score_t_plus_{horizon}"
        sell_score = f"sell_coupon_score_t_plus_{horizon}"
        buy_hit = f"buy_coupon_underlying_hit_t_plus_{horizon}"
        sell_hit = f"sell_coupon_underlying_hit_t_plus_{horizon}"
    else:
        buy_score = f"buy_product_score_t_plus_{horizon}"
        sell_score = f"sell_product_score_t_plus_{horizon}"
        buy_hit = f"buy_hit_t_plus_{horizon}"
        sell_hit = f"sell_hit_t_plus_{horizon}"

    buys = subset[["buyer", "symbol", "quantity", buy_score, buy_hit]].rename(
        columns={"buyer": "participant", buy_score: "score", buy_hit: "hit"}
    )
    buys["side"] = "BUY"
    sells = subset[["seller", "symbol", "quantity", sell_score, sell_hit]].rename(
        columns={"seller": "participant", sell_score: "score", sell_hit: "hit"}
    )
    sells["side"] = "SELL"

    actions = pd.concat([buys, sells], ignore_index=True)
    actions["participant"] = actions["participant"].fillna("").astype(str).str.strip()
    actions = actions[(actions["participant"] != "") & actions["score"].notna()]
    actions["positive_score"] = actions["score"].clip(lower=0)

    if actions.empty:
        return pd.DataFrame()

    summary = actions.groupby("participant", as_index=False).agg(
        trades=("participant", "size"),
        total_qty=("quantity", "sum"),
        hit_rate=("hit", "mean"),
        mean_score=("score", "mean"),
        total_score=("score", "sum"),
        positive_score=("positive_score", "sum"),
        symbols=("symbol", lambda x: ", ".join(sorted(set(map(str, x)))[:8])),
    )

    # Final rank rewards total suspicious PnL, consistency, and not just one lucky print.
    summary["rank_score"] = summary["positive_score"] * (0.5 + summary["hit_rate"].fillna(0.0))
    summary = summary[summary["trades"] >= min_trades]
    return summary.sort_values(["rank_score", "positive_score", "hit_rate"], ascending=[False, False, False])


def run_analysis(data_dir: Path, days: list[int], horizons: tuple[int, ...] = (1, 3, 5, 10, 20), ranking_horizon: int = 10) -> None:
    prices, trades = load_round_days(data_dir, days)
    prices = enrich_prices(prices, horizons)
    merged = merge_trades_with_market(prices, trades, horizons)
    scored = compute_scores(merged, horizons)

    normal_top = suspicious_trades(scored, ranking_horizon, top_n=20, coupons=False)
    coupon_top = suspicious_trades(scored, ranking_horizon, top_n=20, coupons=True)
    normal_participants = participant_summary(scored, ranking_horizon, min_trades=3, coupons=False)
    coupon_participants = participant_summary(scored, ranking_horizon, min_trades=3, coupons=True)

    print(f"\nTOP PRODUCT TRADES, excluding VEV coupons, horizon t+{ranking_horizon}")
    print(normal_top.to_string(index=False) if not normal_top.empty else "No rows")

    print(f"\nTOP VEV COUPON TRADES, horizon t+{ranking_horizon}")
    print(coupon_top.to_string(index=False) if not coupon_top.empty else "No rows")

    print(f"\nPRODUCT PARTICIPANT SUMMARY, horizon t+{ranking_horizon}")
    print(normal_participants.to_string(index=False) if not normal_participants.empty else "No rows")

    print(f"\nVEV COUPON PARTICIPANT SUMMARY, horizon t+{ranking_horizon}")
    print(coupon_participants.to_string(index=False) if not coupon_participants.empty else "No rows")

    scored.to_csv("scored_all_trades.csv", index=False)
    normal_top.to_csv("top_product_trades.csv", index=False)
    coupon_top.to_csv("top_vev_coupon_trades.csv", index=False)
    normal_participants.to_csv("product_participant_summary.csv", index=False)
    coupon_participants.to_csv("vev_coupon_participant_summary.csv", index=False)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2] / "data" / "ROUND_4"
    run_analysis(base_dir, days=[1, 2, 3], ranking_horizon=10)


if __name__ == "__main__":
    main()
