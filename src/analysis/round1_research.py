from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from src.analysis.data import read_all_round1_prices, read_all_round1_trades


def build_market_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a clean top-of-book market frame with strategy-ready features."""
    df = prices.copy()
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    df["bid_price_1"] = pd.to_numeric(df["bid_price_1"], errors="coerce")
    df["ask_price_1"] = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["bid_volume_1"] = pd.to_numeric(df["bid_volume_1"], errors="coerce")
    df["ask_volume_1"] = pd.to_numeric(df["ask_volume_1"], errors="coerce")

    df["time_index"] = df["day"] * 1_000_000 + df["timestamp"]

    has_two_sided_top = df["bid_price_1"].notna() & df["ask_price_1"].notna()
    df["mid"] = np.where(
        has_two_sided_top,
        (df["bid_price_1"] + df["ask_price_1"]) / 2,
        pd.to_numeric(df["mid_price"], errors="coerce"),
    )
    df["spread"] = np.where(has_two_sided_top, df["ask_price_1"] - df["bid_price_1"], np.nan)

    df["depth_bid_1"] = df["bid_volume_1"].abs().fillna(0.0)
    df["depth_ask_1"] = df["ask_volume_1"].abs().fillna(0.0)
    top_depth = df["depth_bid_1"] + df["depth_ask_1"]

    # Imbalance is normalized in [-1, 1], where positive means bid-side pressure.
    df["imbalance"] = (df["depth_bid_1"] - df["depth_ask_1"]).where(top_depth > 0, 0.0) / top_depth.where(top_depth > 0, 1.0)

    # Microprice tilts toward the side with thinner liquidity and is often more predictive than mid.
    micro_valid = has_two_sided_top & (top_depth > 0)
    df["microprice"] = np.where(
        micro_valid,
        ((df["ask_price_1"] * df["depth_bid_1"]) + (df["bid_price_1"] * df["depth_ask_1"])) / top_depth,
        df["mid"],
    )
    df["micro_gap"] = df["microprice"] - df["mid"]

    grouped_mid = df.groupby("product", group_keys=False)["mid"]
    df["mid_diff"] = grouped_mid.diff().fillna(0.0)
    df["mid_return"] = grouped_mid.pct_change().fillna(0.0)

    # Forward returns are mandatory for real predictive testing (no look-ahead leakage).
    for horizon in (1, 2, 5, 10):
        fwd = grouped_mid.shift(-horizon) - df["mid"]
        df[f"fwd_mid_diff_{horizon}"] = fwd
        df[f"fwd_mid_bps_{horizon}"] = (fwd / df["mid"].where(df["mid"] != 0.0, np.nan)) * 10_000

    df["rolling_vol_100"] = (
        df.groupby("product")["mid_diff"].rolling(100).std().reset_index(level=0, drop=True)
    )

    # OFI proxy from L1 updates (useful when full order event feed is unavailable).
    prev_bid = df.groupby(["product", "day"], group_keys=False)["bid_price_1"].shift(1)
    prev_ask = df.groupby(["product", "day"], group_keys=False)["ask_price_1"].shift(1)
    prev_bid_v = df.groupby(["product", "day"], group_keys=False)["depth_bid_1"].shift(1).fillna(0.0)
    prev_ask_v = df.groupby(["product", "day"], group_keys=False)["depth_ask_1"].shift(1).fillna(0.0)

    bid_term = np.where(
        df["bid_price_1"] > prev_bid,
        df["depth_bid_1"],
        np.where(df["bid_price_1"] < prev_bid, -prev_bid_v, df["depth_bid_1"] - prev_bid_v),
    )
    ask_term = np.where(
        df["ask_price_1"] < prev_ask,
        df["depth_ask_1"],
        np.where(df["ask_price_1"] > prev_ask, -prev_ask_v, df["depth_ask_1"] - prev_ask_v),
    )
    ofi = bid_term - ask_term
    df["ofi_proxy"] = pd.Series(ofi, index=df.index).fillna(0.0)

    return df


def build_trade_frame(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Normalize trades and infer aggressor direction from contemporaneous top-of-book."""
    tr = trades.copy()
    tr = tr.rename(columns={"symbol": "product"})
    tr["time_index"] = tr["day"] * 1_000_000 + tr["timestamp"]
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0.0)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["notional"] = tr["price"] * tr["quantity"]

    touch = market[["day", "timestamp", "product", "mid", "bid_price_1", "ask_price_1"]]
    tr = tr.merge(touch, on=["day", "timestamp", "product"], how="left")

    def infer_side(row: pd.Series) -> float:
        mid = row["mid"]
        bid = row["bid_price_1"]
        ask = row["ask_price_1"]
        px = row["price"]

        if pd.isna(mid) or pd.isna(px):
            return 0.0
        if pd.notna(ask) and px >= ask:
            return 1.0
        if pd.notna(bid) and px <= bid:
            return -1.0
        if px > mid:
            return 1.0
        if px < mid:
            return -1.0
        return 0.0

    tr["aggressor_side"] = tr.apply(infer_side, axis=1)
    tr["signed_qty"] = tr["aggressor_side"] * tr["quantity"]
    tr["signed_notional"] = tr["aggressor_side"] * tr["notional"]
    return tr


def merge_trade_features(market: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate trade tape to market timestamps and merge into the market frame."""
    trade_agg = (
        trades.groupby(["product", "day", "timestamp", "time_index"], as_index=False)
        .agg(
            trade_count=("price", "count"),
            traded_qty=("quantity", "sum"),
            traded_notional=("notional", "sum"),
            signed_qty=("signed_qty", "sum"),
            signed_notional=("signed_notional", "sum"),
            trade_vwap=("price", "mean"),
        )
        .sort_values(["product", "day", "timestamp"])
    )

    enriched = market.merge(
        trade_agg,
        on=["product", "day", "timestamp", "time_index"],
        how="left",
    )
    for col in ("trade_count", "traded_qty", "traded_notional", "signed_qty", "signed_notional"):
        enriched[col] = enriched[col].fillna(0.0)

    # Rolling signed flow helps identify short pressure bursts vs reversion windows.
    enriched["rolling_signed_qty_50"] = (
        enriched.groupby("product")["signed_qty"].rolling(50).sum().reset_index(level=0, drop=True)
    )
    return enriched, trade_agg


def compute_signal_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    feature_cols = ["imbalance", "micro_gap", "ofi_proxy", "signed_qty", "rolling_signed_qty_50"]

    for product, g in df.groupby("product"):
        target = g["fwd_mid_bps_1"]
        for feature in feature_cols:
            x = g[feature]
            valid = x.notna() & target.notna()
            if valid.sum() < 20:
                continue
            x_valid = x[valid]
            y_valid = target[valid]
            deciles = pd.qcut(x_valid.rank(method="first"), 10, labels=False)
            decile_ret = y_valid.groupby(deciles).mean()
            long_short_spread = float(decile_ret.iloc[-1] - decile_ret.iloc[0])

            rows.append(
                {
                    "product": product,
                    "feature": feature,
                    "pearson": float(x_valid.corr(y_valid, method="pearson")),
                    "spearman": float(x_valid.corr(y_valid, method="spearman")),
                    "decile_10_minus_1_bps": long_short_spread,
                    "samples": float(valid.sum()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["product", "spearman"], ascending=[True, False]).reset_index(drop=True)


def compute_autocorr_table(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for product, g in df.groupby("product"):
        series = g["mid_diff"].fillna(0.0)
        for lag in range(1, max_lag + 1):
            rows.append(
                {
                    "product": product,
                    "lag": lag,
                    "autocorr_mid_diff": float(series.autocorr(lag=lag)),
                }
            )
    return pd.DataFrame(rows)


def compute_reversal_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for product, g in df.groupby("product"):
        curr = g["mid_diff"]
        prev = curr.shift(1)
        valid = prev.ne(0.0) & curr.ne(0.0)
        if valid.sum() < 30:
            continue

        prev_valid = prev[valid]
        curr_valid = curr[valid]
        reversal = (np.sign(curr_valid) == -np.sign(prev_valid)).astype(float)
        bucket = pd.qcut(prev_valid.abs(), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")

        grouped = (
            pd.DataFrame({"bucket": bucket, "reversal": reversal})
            .groupby("bucket", observed=False)
            .agg(reversal_prob=("reversal", "mean"), samples=("reversal", "count"))
            .reset_index()
        )
        grouped["product"] = product
        rows.extend(grouped.to_dict(orient="records"))

    return pd.DataFrame(rows)


def build_plots(df: pd.DataFrame, trade_agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = df.groupby("product", as_index=False, group_keys=False).head(3500).copy()

    # 1) Core top-of-book view.
    price_panel = sample.melt(
        id_vars=["product", "time_index"],
        value_vars=["bid_price_1", "mid", "ask_price_1"],
        var_name="series",
        value_name="price",
    )
    fig = px.line(price_panel, x="time_index", y="price", color="series", facet_row="product", title="Top-of-Book Prices")
    fig.write_html(str(out_dir / "01_top_of_book_prices.html"))

    # 2) Spread behavior through time and distribution.
    fig = px.line(sample, x="time_index", y="spread", color="product", title="Spread Through Time")
    fig.write_html(str(out_dir / "02_spread_time.html"))

    fig = px.box(df, x="product", y="spread", color="product", points=False, title="Spread Distribution")
    fig.write_html(str(out_dir / "03_spread_distribution.html"))

    # 3) Microstructure pressure signals.
    fig = px.line(sample, x="time_index", y="imbalance", color="product", title="L1 Imbalance Through Time")
    fig.write_html(str(out_dir / "04_imbalance_time.html"))

    fig = px.line(sample, x="time_index", y="micro_gap", color="product", title="Microprice - Mid Gap Through Time")
    fig.write_html(str(out_dir / "05_micro_gap_time.html"))

    # 4) Volatility and order-flow regime visuals.
    fig = px.line(sample, x="time_index", y="rolling_vol_100", color="product", title="Rolling Volatility (window=100)")
    fig.write_html(str(out_dir / "06_rolling_volatility.html"))

    fig = px.line(sample, x="time_index", y="ofi_proxy", color="product", title="OFI Proxy Through Time")
    fig.write_html(str(out_dir / "07_ofi_proxy.html"))

    # 5) Tape activity from trades dataset.
    fig = px.line(trade_agg, x="time_index", y="traded_qty", color="product", title="Executed Quantity Through Time")
    fig.write_html(str(out_dir / "08_traded_quantity.html"))

    fig = px.line(trade_agg, x="time_index", y="signed_qty", color="product", title="Signed Quantity Through Time")
    fig.write_html(str(out_dir / "09_signed_quantity.html"))

    fig = px.histogram(trade_agg, x="trade_count", color="product", barmode="overlay", nbins=40, title="Trade Count per Timestamp")
    fig.write_html(str(out_dir / "10_trade_count_histogram.html"))

    # 6) Predictive signal sanity checks (forward return targets).
    signal_sample = df[["product", "imbalance", "micro_gap", "fwd_mid_bps_1", "spread"]].dropna()
    signal_sample = signal_sample.sample(min(len(signal_sample), 8000), random_state=7)

    fig = px.scatter(
        signal_sample,
        x="imbalance",
        y="fwd_mid_bps_1",
        color="product",
        opacity=0.35,
        title="Imbalance vs Forward Mid Return (1 tick)",
    )
    fig.write_html(str(out_dir / "11_imbalance_vs_forward_return.html"))

    fig = px.scatter(
        signal_sample,
        x="micro_gap",
        y="fwd_mid_bps_1",
        color="product",
        opacity=0.35,
        title="Micro Gap vs Forward Mid Return (1 tick)",
    )
    fig.write_html(str(out_dir / "12_micro_gap_vs_forward_return.html"))

    # Heatmap: conditional expected forward return by spread and imbalance states.
    heat = df[["product", "spread", "imbalance", "fwd_mid_bps_1"]].dropna().copy()
    heat["spread_bucket"] = pd.qcut(heat["spread"], q=5, labels=False, duplicates="drop")
    heat["imb_bucket"] = pd.qcut(heat["imbalance"], q=5, labels=False, duplicates="drop")
    fig = px.density_heatmap(
        heat,
        x="spread_bucket",
        y="imb_bucket",
        z="fwd_mid_bps_1",
        facet_col="product",
        histfunc="avg",
        title="Expected Forward Return by Spread/Imbalance Regime",
    )
    fig.write_html(str(out_dir / "13_regime_heatmap.html"))


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "round1_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = read_all_round1_prices()
    trades = read_all_round1_trades()
    if prices.empty:
        raise RuntimeError("No Round 1 prices found under data/ROUND_1.")
    if trades.empty:
        raise RuntimeError("No Round 1 trades found under data/ROUND_1.")

    market = build_market_frame(prices)
    trade_frame = build_trade_frame(trades, market)
    market_enriched, trade_agg = merge_trade_features(market, trade_frame)

    # Save validation tables so this can be used as a reusable research artifact.
    coverage = (
        market_enriched.groupby(["product", "day"], as_index=False)
        .agg(
            price_rows=("timestamp", "count"),
            non_null_spread=("spread", lambda s: float(s.notna().mean())),
            active_trade_timestamps=("trade_count", lambda s: float((s > 0).sum())),
        )
        .sort_values(["product", "day"])
    )
    coverage.to_csv(out_dir / "coverage_checks.csv", index=False)

    liquidity = (
        market_enriched.groupby("product", as_index=False)
        .agg(
            mean_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            mean_bid_depth=("depth_bid_1", "mean"),
            mean_ask_depth=("depth_ask_1", "mean"),
            mean_abs_imbalance=("imbalance", lambda s: s.abs().mean()),
            mean_trade_count=("trade_count", "mean"),
            mean_signed_qty=("signed_qty", "mean"),
        )
        .sort_values("product")
    )
    liquidity.to_csv(out_dir / "liquidity_summary.csv", index=False)

    signal_scorecard = compute_signal_scorecard(market_enriched)
    signal_scorecard.to_csv(out_dir / "signal_scorecard.csv", index=False)

    autocorr = compute_autocorr_table(market_enriched, max_lag=12)
    autocorr.to_csv(out_dir / "autocorr_mid_diff.csv", index=False)

    reversal = compute_reversal_table(market_enriched)
    reversal.to_csv(out_dir / "reversal_prob_by_move_bucket.csv", index=False)

    build_plots(market_enriched, trade_agg, out_dir)

    print("Round 1 research pack created.")
    print(f"Output directory: {out_dir}")
    print("Key tables:")
    print(" - coverage_checks.csv")
    print(" - liquidity_summary.csv")
    print(" - signal_scorecard.csv")
    print(" - autocorr_mid_diff.csv")
    print(" - reversal_prob_by_move_bucket.csv")
    print("13 interactive HTML plots were saved in the same folder.")


if __name__ == "__main__":
    main()

