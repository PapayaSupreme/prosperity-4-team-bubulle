"""
Round 5 Comprehensive Product Analysis
Analyzes all ~50 tradable goods from ROUND_5 data with focus on:
  - Price statistics (mean, std, min, max, range)
  - Spread analysis (bid-ask spreads, depth)
  - Volume and liquidity
  - Price movement patterns
  - Trading activity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from prosperity4.analysis.data import read_all_round5_prices, read_all_round5_trades


def build_market_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a clean top-of-book market frame with strategy-ready features."""
    df = prices.copy()
    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    # Coerce to numeric
    for col in ["bid_price_1", "ask_price_1", "bid_volume_1", "ask_volume_1", "mid_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build time index
    df["time_index"] = df["day"] * 1_000_000 + df["timestamp"]

    # Compute mid and spread
    has_two_sided = (df["bid_price_1"].notna()) & (df["ask_price_1"].notna())
    df["mid"] = np.where(
        has_two_sided,
        (df["bid_price_1"] + df["ask_price_1"]) / 2,
        df["mid_price"],
    )
    df["spread"] = np.where(has_two_sided, df["ask_price_1"] - df["bid_price_1"], np.nan)

    # Depth
    df["depth_bid"] = df["bid_volume_1"].abs().fillna(0.0)
    df["depth_ask"] = df["ask_volume_1"].abs().fillna(0.0)
    total_depth = df["depth_bid"] + df["depth_ask"]

    # Imbalance (-1 to 1, positive = bid pressure)
    df["imbalance"] = np.where(
        total_depth > 0,
        (df["depth_bid"] - df["depth_ask"]) / total_depth,
        0.0,
    )

    # Price changes
    grouped_mid = df.groupby("product", group_keys=False)["mid"]
    df["mid_diff"] = grouped_mid.diff().fillna(0.0)
    df["mid_return"] = grouped_mid.pct_change().fillna(0.0)

    return df


def analyze_product_prices(prices_df: pd.DataFrame, product: str) -> dict:
    """Compute price statistics for a single product."""
    prod_prices = prices_df[prices_df["product"] == product].copy()

    if prod_prices.empty:
        return {"product": product, "data_points": 0}

    mid = prod_prices["mid"].dropna()
    spread = prod_prices["spread"].dropna()

    stats = {
        "product": product,
        "data_points": len(mid),
        "mid_min": float(mid.min()) if len(mid) > 0 else np.nan,
        "mid_max": float(mid.max()) if len(mid) > 0 else np.nan,
        "mid_mean": float(mid.mean()) if len(mid) > 0 else np.nan,
        "mid_std": float(mid.std()) if len(mid) > 0 else np.nan,
        "mid_range": float(mid.max() - mid.min()) if len(mid) > 0 else np.nan,
        "spread_mean": float(spread.mean()) if len(spread) > 0 else np.nan,
        "spread_max": float(spread.max()) if len(spread) > 0 else np.nan,
        "spread_min": float(spread.min()) if len(spread) > 0 else np.nan,
        "depth_avg_bid": float(prod_prices["depth_bid"].mean()),
        "depth_avg_ask": float(prod_prices["depth_ask"].mean()),
        "imbalance_mean": float(prod_prices["imbalance"].mean()),
        "returns_std": float(prod_prices["mid_return"].std()) if len(prod_prices) > 1 else np.nan,
    }

    return stats


def analyze_product_trades(trades_df: pd.DataFrame, product: str) -> dict:
    """Compute trade statistics for a single product."""
    prod_trades = trades_df[trades_df["symbol"] == product].copy()

    if prod_trades.empty:
        return {"product": product, "num_trades": 0}

    prod_trades["price"] = pd.to_numeric(prod_trades["price"], errors="coerce")
    prod_trades["quantity"] = pd.to_numeric(prod_trades["quantity"], errors="coerce")

    valid_trades = prod_trades[prod_trades["price"].notna() & prod_trades["quantity"].notna()]

    stats = {
        "product": product,
        "num_trades": len(prod_trades),
        "trade_price_min": float(valid_trades["price"].min()) if len(valid_trades) > 0 else np.nan,
        "trade_price_max": float(valid_trades["price"].max()) if len(valid_trades) > 0 else np.nan,
        "trade_price_mean": float(valid_trades["price"].mean()) if len(valid_trades) > 0 else np.nan,
        "avg_trade_quantity": float(valid_trades["quantity"].mean()) if len(valid_trades) > 0 else np.nan,
        "total_volume": float(valid_trades["quantity"].sum()) if len(valid_trades) > 0 else 0.0,
        "unique_buyers": len(valid_trades["buyer"].dropna().unique()),
        "unique_sellers": len(valid_trades["seller"].dropna().unique()),
    }

    return stats


def generate_summary_table(price_stats: list[dict], trade_stats: list[dict]) -> pd.DataFrame:
    """Merge price and trade stats into a single analysis table."""
    prices_df = pd.DataFrame(price_stats).set_index("product")
    trades_df = pd.DataFrame(trade_stats).set_index("product")

    combined = prices_df.join(trades_df, how="outer", rsuffix="_trades")

    # Sort by number of trades (descending)
    combined = combined.sort_values("num_trades", ascending=False, na_position="last")

    return combined


def print_product_summary(summary_df: pd.DataFrame) -> None:
    """Print ergonomic summary of all products."""
    print("\n" + "=" * 160)
    print("ROUND 5 COMPREHENSIVE PRODUCT ANALYSIS")
    print("=" * 160)

    print(f"\nTotal unique products: {len(summary_df)}")

    # Print by trading activity
    print("\n" + "-" * 160)
    print("TOP 20 MOST TRADED PRODUCTS")
    print("-" * 160)

    top_traded = summary_df.nlargest(20, "num_trades")[
        ["data_points", "mid_mean", "mid_std", "spread_mean", "num_trades", "total_volume", "returns_std"]
    ]

    print(
        "Rank | Product" + " " * 30 + "| Mid Price | Volatility | Avg Spread | Trades | Volume  | Ret. Std"
    )
    print("-" * 160)

    for rank, (product, row) in enumerate(top_traded.iterrows(), 1):
        mid = row.get("mid_mean", np.nan)
        std = row.get("mid_std", np.nan)
        spread = row.get("spread_mean", np.nan)
        trades = row.get("num_trades", 0)
        volume = row.get("total_volume", 0)
        ret_std = row.get("returns_std", np.nan)

        mid_str = f"{mid:10.2f}" if pd.notna(mid) else "       N/A"
        std_str = f"{std:8.4f}" if pd.notna(std) else "     N/A"
        spread_str = f"{spread:9.2f}" if pd.notna(spread) else "      N/A"
        trades_str = f"{int(trades):6d}"
        volume_str = f"{int(volume):7d}"
        ret_str = f"{ret_std:8.6f}" if pd.notna(ret_std) else "      N/A"

        print(
            f"{rank:3d}. | {product:32s} | {mid_str} | {std_str} | {spread_str} | {trades_str} | {volume_str} | {ret_str}"
        )

    # Print by liquidity (average depth)
    print("\n" + "-" * 160)
    print("TOP 15 PRODUCTS BY AVERAGE BID DEPTH (MOST LIQUID)")
    print("-" * 160)

    top_depth = summary_df.nlargest(15, "depth_avg_bid")[
        ["data_points", "mid_mean", "depth_avg_bid", "depth_avg_ask", "spread_mean", "num_trades"]
    ]

    print(
        "Rank | Product" + " " * 30 + "| Bid Depth | Ask Depth | Spread | Trades | Obs."
    )
    print("-" * 160)

    for rank, (product, row) in enumerate(top_depth.iterrows(), 1):
        bid_depth = row.get("depth_avg_bid", 0)
        ask_depth = row.get("depth_avg_ask", 0)
        spread = row.get("spread_mean", np.nan)
        trades = row.get("num_trades", 0)
        obs = row.get("data_points", 0)

        spread_str = f"{spread:7.2f}" if pd.notna(spread) else "    N/A"

        print(
            f"{rank:3d}. | {product:32s} | {bid_depth:9.1f} | {ask_depth:9.1f} | {spread_str} | {int(trades):6d} | {int(obs):5d}"
        )

    # Print by volatility
    print("\n" + "-" * 160)
    print("TOP 15 PRODUCTS BY VOLATILITY (PRICE STD DEV)")
    print("-" * 160)

    top_vol = summary_df.nlargest(15, "mid_std")[
        ["data_points", "mid_mean", "mid_std", "mid_range", "spread_mean", "returns_std"]
    ]

    print(
        "Rank | Product" + " " * 30 + "| Mean Price | Std Dev | Range   | Spread | Ret. Std"
    )
    print("-" * 160)

    for rank, (product, row) in enumerate(top_vol.iterrows(), 1):
        mean = row.get("mid_mean", np.nan)
        std = row.get("mid_std", np.nan)
        rng = row.get("mid_range", np.nan)
        spread = row.get("spread_mean", np.nan)
        ret_std = row.get("returns_std", np.nan)

        mean_str = f"{mean:10.2f}" if pd.notna(mean) else "       N/A"
        std_str = f"{std:7.4f}" if pd.notna(std) else "     N/A"
        range_str = f"{rng:7.2f}" if pd.notna(rng) else "     N/A"
        spread_str = f"{spread:6.2f}" if pd.notna(spread) else "    N/A"
        ret_str = f"{ret_std:8.6f}" if pd.notna(ret_std) else "      N/A"

        print(
            f"{rank:3d}. | {product:32s} | {mean_str} | {std_str} | {range_str} | {spread_str} | {ret_str}"
        )

    # Print by mid-price level (anchor points)
    print("\n" + "-" * 160)
    print("PRODUCTS BY ANCHOR PRICE LEVEL")
    print("-" * 160)

    summary_df_sorted = summary_df.sort_values("mid_mean", ascending=False)

    print("Rank | Product" + " " * 30 + "| Mid Price | Range   | Spread | Trades | Data Points")
    print("-" * 160)

    for rank, (product, row) in enumerate(summary_df_sorted.head(30).iterrows(), 1):
        mid = row.get("mid_mean", np.nan)
        rng = row.get("mid_range", np.nan)
        spread = row.get("spread_mean", np.nan)
        trades = row.get("num_trades", 0)
        obs = row.get("data_points", 0)

        mid_str = f"{mid:10.2f}" if pd.notna(mid) else "       N/A"
        range_str = f"{rng:7.2f}" if pd.notna(rng) else "     N/A"
        spread_str = f"{spread:6.2f}" if pd.notna(spread) else "    N/A"

        print(
            f"{rank:3d}. | {product:32s} | {mid_str} | {range_str} | {spread_str} | {int(trades):6d} | {int(obs):5d}"
        )

    # Overall summary statistics
    print("\n" + "-" * 160)
    print("OVERALL STATISTICS")
    print("-" * 160)

    avg_spread = summary_df["spread_mean"].mean()
    avg_std = summary_df["mid_std"].mean()
    total_trades = summary_df["num_trades"].sum()
    avg_depth_bid = summary_df["depth_avg_bid"].mean()
    avg_depth_ask = summary_df["depth_avg_ask"].mean()
    avg_imbalance = summary_df["imbalance_mean"].mean()

    print(f"Average Bid-Ask Spread (across all products): {avg_spread:8.2f}")
    print(f"Average Price Volatility (Std Dev): {avg_std:8.4f}")
    print(f"Total Trade Count: {int(total_trades):8d}")
    print(f"Average Bid Depth: {avg_depth_bid:8.1f}")
    print(f"Average Ask Depth: {avg_depth_ask:8.1f}")
    print(f"Average Order Imbalance: {avg_imbalance:8.4f}")

    print("\n" + "=" * 160)


def save_analysis_to_csv(summary_df: pd.DataFrame, output_path: Path = None) -> None:
    """Save full analysis to CSV for external review."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "round5_analysis_output.csv"

    summary_df.to_csv(output_path)
    print(f"\nFull analysis saved to: {output_path}")


def plot_product_midprices(prices_df: pd.DataFrame, products: list[str], output_dir: Path = None) -> None:
    """Generate individual Plotly plots for each product's mid-price over time."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "round5_outputs"

    output_dir.mkdir(exist_ok=True)

    print("\nGenerating mid-price plots for each product...")

    for product in products:
        prod_data = prices_df[prices_df["product"] == product].copy()

        if prod_data.empty or prod_data["mid"].isna().all():
            continue

        # Create time string for better x-axis labels
        prod_data["time_str"] = prod_data.apply(
            lambda row: f"Day {int(row['day'])} T{int(row['timestamp'])}", axis=1
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prod_data["time_index"],
                y=prod_data["mid"],
                mode="lines+markers",
                name=product,
                line=dict(color="steelblue", width=2),
                marker=dict(size=4),
                hovertemplate="<b>%{fullData.name}</b><br>Time: %{x}<br>Mid Price: %{y:.2f}<extra></extra>",
            )
        )

        # Add bid-ask bands as shaded area
        fig.add_trace(
            go.Scatter(
                x=prod_data["time_index"],
                y=prod_data["ask_price_1"],
                mode="lines",
                name="Ask Price",
                line=dict(color="rgba(255,0,0,0.3)", width=1),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=prod_data["time_index"],
                y=prod_data["bid_price_1"],
                mode="lines",
                name="Bid Price",
                line=dict(color="rgba(0,255,0,0.3)", width=1),
                fill="tonexty",
                fillcolor="rgba(0,128,255,0.1)",
                hoverinfo="skip",
            )
        )

        stats = analyze_product_prices(prices_df, product)

        fig.update_layout(
            title=f"<b>{product}</b> - Mid-Price Over Time<br>" +
                  f"<sub>Mean: {stats['mid_mean']:.2f} | Std: {stats['mid_std']:.4f} | Range: {stats['mid_range']:.2f}</sub>",
            xaxis_title="Time Index (day * 1M + timestamp)",
            yaxis_title="Price",
            template="plotly_white",
            hovermode="x unified",
            height=600,
            width=1200,
            showlegend=True,
        )

        fig.add_hline(
            y=stats["mid_mean"],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Mean: {stats['mid_mean']:.2f}",
            annotation_position="right",
        )

        output_file = output_dir / f"product_{product.replace(' ', '_')}_midprice.html"
        fig.write_html(str(output_file))
        print(f"  ✓ {product}")

    print(f"All plots saved to: {output_dir}")


def plot_all_products_overlay(prices_df: pd.DataFrame, products: list[str], output_dir: Path = None) -> None:
    """Create an overlay plot of all products' mid-prices (normalized)."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "round5_outputs"

    output_dir.mkdir(exist_ok=True)

    print("\nGenerating overlay plot of all products (normalized)...")

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for idx, product in enumerate(products):
        prod_data = prices_df[prices_df["product"] == product].copy()

        if prod_data.empty or prod_data["mid"].isna().all():
            continue

        # Normalize to start price
        mid = prod_data["mid"].dropna()
        if len(mid) == 0:
            continue

        start_price = mid.iloc[0]
        normalized = (mid - start_price) / start_price * 100  # Percent change

        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=prod_data[prod_data["mid"].notna()]["time_index"],
                y=normalized,
                mode="lines",
                name=product,
                line=dict(color=color, width=1.5),
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="<b>All Products Mid-Price Comparison</b><br><sub>Normalized to starting price (% change)</sub>",
        xaxis_title="Time Index (day * 1M + timestamp)",
        yaxis_title="Percent Change (%)",
        template="plotly_white",
        hovermode="x unified",
        height=700,
        width=1400,
        showlegend=True,
        legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
    )

    output_file = output_dir / "all_products_overlay_normalized.html"
    fig.write_html(str(output_file))
    print(f"  ✓ Saved to: {output_file}")


def plot_product_boxplot(prices_df: pd.DataFrame, products: list[str], output_dir: Path = None) -> None:
    """Create a boxplot comparing price distributions across all products."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "round5_outputs"

    output_dir.mkdir(exist_ok=True)

    print("\nGenerating price distribution boxplot...")

    # Prepare data for boxplot
    data_list = []
    for product in products:
        prod_data = prices_df[prices_df["product"] == product].copy()
        mid = prod_data["mid"].dropna()

        if len(mid) == 0:
            continue

        for price in mid:
            data_list.append({"Product": product, "Mid Price": price})

    df_box = pd.DataFrame(data_list)

    fig = px.box(
        df_box,
        x="Product",
        y="Mid Price",
        title="<b>Price Distribution by Product</b>",
        labels={"Mid Price": "Mid Price", "Product": "Product"},
        height=700,
        width=1600,
    )

    fig.update_xaxes(tickangle=-45)
    fig.update_layout(template="plotly_white", hovermode="closest")

    output_file = output_dir / "price_distribution_boxplot.html"
    fig.write_html(str(output_file))
    print(f"  ✓ Saved to: {output_file}")


def plot_top_products_detailed(prices_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path = None) -> None:
    """Create detailed subplots for the top 12 most-traded products."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "round5_outputs"

    output_dir.mkdir(exist_ok=True)

    print("\nGenerating detailed plots for top 12 products...")

    top_products = summary_df.nlargest(12, "num_trades").index.tolist()

    # Create 3x4 grid
    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=top_products,
        specs=[[{"secondary_y": False}] * 4] * 3,
    )

    for idx, product in enumerate(top_products):
        row = idx // 4 + 1
        col = idx % 4 + 1

        prod_data = prices_df[prices_df["product"] == product].copy()

        if prod_data.empty or prod_data["mid"].isna().all():
            continue

        fig.add_trace(
            go.Scatter(
                x=prod_data["time_index"],
                y=prod_data["mid"],
                mode="lines",
                name=product,
                line=dict(color="steelblue", width=1),
                hovertemplate=f"<b>{product}</b><br>Price: %{{y:.2f}}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(title_text="Price", row=row, col=col)

    fig.update_layout(
        title_text="<b>Top 12 Most-Traded Products - Mid-Price Evolution</b>",
        height=900,
        width=1600,
        template="plotly_white",
        showlegend=False,
    )

    output_file = output_dir / "top_12_products_detailed.html"
    fig.write_html(str(output_file))
    print(f"  ✓ Saved to: {output_file}")


def generate_all_plots(prices_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path = None) -> None:
    """Generate all visualization plots."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "round5_outputs"

    output_dir.mkdir(exist_ok=True)

    products = sorted(prices_df["product"].unique())

    plot_all_products_overlay(prices_df, products, output_dir)
    plot_product_boxplot(prices_df, products, output_dir)
    plot_top_products_detailed(prices_df, summary_df, output_dir)
    plot_product_midprices(prices_df, products[:20], output_dir)  # Plot first 20 individually

    print(f"\n✓ All visualizations completed and saved to: {output_dir}")


def main():
    """Run complete Round 5 analysis."""
    print("Loading Round 5 prices and trades...")
    prices = read_all_round5_prices()
    trades = read_all_round5_trades()

    if prices.empty or trades.empty:
        print("ERROR: Could not load data. Check that ROUND_5 files exist.")
        return

    print(f"Loaded {len(prices)} price observations and {len(trades)} trades.")

    # Build market frame
    print("\nBuilding market frame...")
    prices = build_market_frame(prices)

    # Get all unique products
    products = sorted(set(prices["product"].unique()) | set(trades["symbol"].unique()))
    print(f"\nDiscovered {len(products)} unique tradable products.")

    # Analyze each product
    print("\nAnalyzing products...")
    price_stats = [analyze_product_prices(prices, product) for product in products]
    trade_stats = [analyze_product_trades(trades, product) for product in products]

    # Combine into summary table
    summary = generate_summary_table(price_stats, trade_stats)

    # Print ergonomic summaries
    print_product_summary(summary)

    # Save to CSV
    save_analysis_to_csv(summary)

    # Generate and save all plots
    generate_all_plots(prices, summary)


if __name__ == "__main__":
    main()

