from pathlib import Path

import pandas as pd


def load_data(price_csv_path: str | Path, trades_csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the price and trades CSV files.

    Price CSV separator: ;
    Trades CSV separator: ;

    Returns:
        prices_df, trades_df
    """
    prices = pd.read_csv(price_csv_path, sep=";")
    trades = pd.read_csv(trades_csv_path, sep=";")

    # Clean column names just in case
    prices.columns = prices.columns.str.strip()
    trades.columns = trades.columns.str.strip()

    return prices, trades


def load_tutorial_days(data_dir: Path, days: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and stack multiple tutorial days while preserving day as merge key.
    """
    price_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for day in days:
        prices_path = data_dir / f"prices_round_1_day_{day}.csv"
        trades_path = data_dir / f"trades_round_1_day_{day}.csv"

        prices_df, trades_df = load_data(prices_path, trades_path)
        prices_df["day"] = day
        trades_df["day"] = day

        price_frames.append(prices_df)
        trade_frames.append(trades_df)

    prices = pd.concat(price_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True)
    return prices, trades


def enrich_prices_with_forward_returns(prices: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """
    Add future mid-price and forward return columns per product/day.
    Horizons are measured in price-row steps for each product/day stream.
    """
    px = prices.copy()
    px["timestamp"] = pd.to_numeric(px["timestamp"], errors="coerce")
    px["mid_price"] = pd.to_numeric(px["mid_price"], errors="coerce")

    px = px.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)
    group_cols = ["day", "product"]

    for horizon in horizons:
        future_mid_col = f"future_mid_t_plus_{horizon}"
        ret_col = f"forward_return_t_plus_{horizon}"
        px[future_mid_col] = px.groupby(group_cols)["mid_price"].shift(-horizon)
        px[ret_col] = px[future_mid_col] - px["mid_price"]

    return px


def merge_trades_with_prices(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Merge trades with market data using timestamp + symbol/product.

    Assumption:
    - trades.symbol corresponds to prices.product
    - timestamp is enough to align rows

    If your data also has a 'day' column in trades later, add it to the merge.
    """
    merge_cols = ["day", "timestamp", "symbol"]
    price_cols = ["day", "timestamp", "product", "mid_price"] + [
        col for col in prices.columns if col.startswith("forward_return_t_plus_")
    ]

    merged = trades.merge(
        prices[price_cols],
        left_on=merge_cols,
        right_on=["day", "timestamp", "product"],
        how="left",
    )

    # Drop duplicate merge helper column
    merged = merged.drop(columns=['product'])

    return merged


def compute_suspicion_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute suspiciousness metrics for both buyer and seller side.

    buy_edge:
        How much cheaper than mid the buyer got the trade.
        Positive is favorable for buyer.

    sell_edge:
        How much higher than mid the seller got the trade.
        Positive is favorable for seller.

    Scores:
        edge * quantity
    """
    df = df.copy()

    # Basic checks
    df["mid_price_missing"] = df["mid_price"].isna()

    # Favorability vs fair value proxy
    df["buy_edge"] = df["mid_price"] - df["price"]
    df["sell_edge"] = df["price"] - df["mid_price"]

    # Size-weighted suspiciousness
    df["buy_exec_score"] = df["buy_edge"] * df["quantity"]
    df["sell_exec_score"] = df["sell_edge"] * df["quantity"]

    # Add predictive component: buyer should benefit from positive future return,
    # seller should benefit from negative future return.
    forward_cols = [col for col in df.columns if col.startswith("forward_return_t_plus_")]
    for ret_col in forward_cols:
        suffix = ret_col.replace("forward_return_", "")
        buy_info_col = f"buy_info_score_{suffix}"
        sell_info_col = f"sell_info_score_{suffix}"
        buy_total_col = f"buy_total_score_{suffix}"
        sell_total_col = f"sell_total_score_{suffix}"

        df[buy_info_col] = df[ret_col] * df["quantity"]
        df[sell_info_col] = -df[ret_col] * df["quantity"]
        df[buy_total_col] = df["buy_exec_score"] + df[buy_info_col]
        df[sell_total_col] = df["sell_exec_score"] + df[sell_info_col]

    return df


def top_suspicious_buys(df: pd.DataFrame, top_n: int = 10, horizon: int = 3) -> pd.DataFrame:
    """
    Return top N suspicious buys.

    Interpretation:
    Buyer got a low price relative to mid, with large quantity.
    """
    total_col = f"buy_total_score_t_plus_{horizon}"
    info_col = f"buy_info_score_t_plus_{horizon}"
    ret_col = f"forward_return_t_plus_{horizon}"
    if total_col not in df.columns:
        raise ValueError(f"Horizon {horizon} not available in scored data")

    buys = df[df["mid_price_missing"] == False].copy()
    buys = buys[buys[ret_col].notna()]

    # Keep only trades favorable to buyer
    buys = buys[buys[total_col] > 0]

    buys = buys.sort_values(
        by=[total_col, info_col, "buy_exec_score", "quantity"],
        ascending=[False, False, False, False],
    )

    cols = [
        "day",
        "timestamp",
        "buyer",
        "seller",
        "symbol",
        "price",
        "quantity",
        "mid_price",
        "buy_edge",
        "buy_exec_score",
        ret_col,
        info_col,
        total_col,
    ]
    return buys[cols].head(top_n)


def top_suspicious_sells(df: pd.DataFrame, top_n: int = 10, horizon: int = 3) -> pd.DataFrame:
    """
    Return top N suspicious sells.

    Interpretation:
    Seller got a high price relative to mid, with large quantity.
    """
    total_col = f"sell_total_score_t_plus_{horizon}"
    info_col = f"sell_info_score_t_plus_{horizon}"
    ret_col = f"forward_return_t_plus_{horizon}"
    if total_col not in df.columns:
        raise ValueError(f"Horizon {horizon} not available in scored data")

    sells = df[df["mid_price_missing"] == False].copy()
    sells = sells[sells[ret_col].notna()]

    # Keep only trades favorable to seller
    sells = sells[sells[total_col] > 0]

    sells = sells.sort_values(
        by=[total_col, info_col, "sell_exec_score", "quantity"],
        ascending=[False, False, False, False],
    )

    cols = [
        "day",
        "timestamp",
        "buyer",
        "seller",
        "symbol",
        "price",
        "quantity",
        "mid_price",
        "sell_edge",
        "sell_exec_score",
        ret_col,
        info_col,
        total_col,
    ]
    return sells[cols].head(top_n)


def participant_summary(df: pd.DataFrame, horizon: int = 3, min_trades: int = 3) -> pd.DataFrame:
    """
    Aggregate participant-level suspicious scores for non-anonymous buyers/sellers.
    """
    buy_col = f"buy_total_score_t_plus_{horizon}"
    sell_col = f"sell_total_score_t_plus_{horizon}"

    buys = df[["buyer", "quantity", buy_col]].copy()
    buys = buys.rename(columns={"buyer": "participant", buy_col: "score"})

    sells = df[["seller", "quantity", sell_col]].copy()
    sells = sells.rename(columns={"seller": "participant", sell_col: "score"})

    side_scores = pd.concat([buys, sells], ignore_index=True)
    side_scores["participant"] = side_scores["participant"].fillna("").astype(str).str.strip()
    side_scores = side_scores[side_scores["participant"] != ""]
    side_scores = side_scores[side_scores["score"].notna()]

    if side_scores.empty:
        return pd.DataFrame(columns=["participant", "trades", "total_qty", "mean_score", "total_score"])

    summary = (
        side_scores.groupby("participant", as_index=False)
        .agg(
            trades=("participant", "size"),
            total_qty=("quantity", "sum"),
            mean_score=("score", "mean"),
            total_score=("score", "sum"),
        )
        .sort_values(["total_score", "mean_score"], ascending=[False, False])
    )

    return summary[summary["trades"] >= min_trades]


def main():
    base_dir = Path(__file__).resolve().parents[2] / "data" / "ROUND_4"
    days = [-2, -1]
    horizons = (1, 3, 5)
    ranking_horizon = 3

    prices, trades = load_tutorial_days(base_dir, days)
    prices_with_forward = enrich_prices_with_forward_returns(prices, horizons=horizons)
    merged = merge_trades_with_prices(prices_with_forward, trades)
    scored = compute_suspicion_scores(merged)

    suspicious_buys = top_suspicious_buys(scored, top_n=10, horizon=ranking_horizon)
    suspicious_sells = top_suspicious_sells(scored, top_n=10, horizon=ranking_horizon)
    participant_scores = participant_summary(scored, horizon=ranking_horizon, min_trades=3)

    print(f"\nTOP 10 SUSPICIOUS BUYS (horizon t+{ranking_horizon})")
    print(suspicious_buys.to_string(index=False))

    print(f"\nTOP 10 SUSPICIOUS SELLS (horizon t+{ranking_horizon})")
    print(suspicious_sells.to_string(index=False))

    print(f"\nPARTICIPANT SUMMARY (horizon t+{ranking_horizon}, min_trades=3)")
    if participant_scores.empty:
        print("No non-anonymous participant data available in this dataset.")
    else:
        print(participant_scores.to_string(index=False))

    # Optional: save results for notebook/chart review.
    suspicious_buys.to_csv("top_10_suspicious_buys.csv", index=False)
    suspicious_sells.to_csv("top_10_suspicious_sells.csv", index=False)
    participant_scores.to_csv("participant_suspicion_summary.csv", index=False)


if __name__ == '__main__':
    main()
