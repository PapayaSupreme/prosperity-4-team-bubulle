from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

_DAY_FROM_FILENAME = re.compile(r"day_(-?\d+)")


def _extract_day_from_path(path: Path) -> int:
    """Extract the simulation day from a trades' filename."""
    match = _DAY_FROM_FILENAME.search(path.name)
    if not match:
        raise ValueError(f"Could not parse day from filename: {path.name}")
    return int(match.group(1))


def _to_number(value: str | None) -> float | None:
    """Convert a CSV field to float, returning None for missing values."""
    if value is None or value == "":
        return None
    return float(value)


def _build_timeline(day: float, timestamp: float) -> int:
    """Create a sortable timeline across multiple days."""
    return int(day) * 100_000 + int(timestamp)


def _safe_imbalance(bid_volume: float | None, ask_volume: float | None) -> float | None:
    """
    Compute top-of-book imbalance.

    Formula:
        (bid_volume_1 - ask_volume_1) / (bid_volume_1 + ask_volume_1)

    Returns None if the required values are missing or if the denominator is zero.
    """
    if bid_volume is None or ask_volume is None:
        return None

    total = bid_volume + ask_volume
    if total == 0:
        return None

    return (bid_volume - ask_volume) / total


def plot_historical_chart(
    history_dir: str | Path = "tutorial/chart_history",
    products: Iterable[str] | None = None,
    include_trades: bool = False,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot historical market series from Prosperity chart history CSV files.

    The function reads:
    - `prices_*.csv` files for order book snapshots
    - optional `trades_*.csv` files for executed trades

    For each requested product, the figure contains three stacked charts:
    1. Best bid, best ask, and mid-price
    2. Spread (ask_price_1 - bid_price_1)
    3. Top-of-book imbalance

    Args:
        history_dir: Directory with `prices_*.csv` and optional `trades_*.csv` files.
        products: Product names to plot (e.g. ["EMERALDS", "TOMATOES"]).
            If None, plots all products found in prices files.
        include_trades: If True, overlays trade prices as scatter markers
            on the price chart.
        output_path: Optional path to save the rendered figure.
        show: If True, calls `plt.show()`.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    history_path = Path(history_dir)
    if not history_path.exists():
        raise FileNotFoundError(f"History directory not found: {history_path}")

    price_files = sorted(history_path.glob("prices_*.csv"))
    if not price_files:
        raise FileNotFoundError(f"No prices_*.csv files found in: {history_path}")

    # For each product, store the order book snapshot values we want to visualize.
    price_data: dict[str, list[dict[str, float]]] = defaultdict(list)

    for file_path in price_files:
        with file_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                product = row.get("product")
                day = _to_number(row.get("day"))
                timestamp = _to_number(row.get("timestamp"))

                bid_price_1 = _to_number(row.get("bid_price_1"))
                ask_price_1 = _to_number(row.get("ask_price_1"))
                mid_price = _to_number(row.get("mid_price"))
                bid_volume_1 = _to_number(row.get("bid_volume_1"))
                ask_volume_1 = _to_number(row.get("ask_volume_1"))

                if (
                    product is None
                    or day is None
                    or timestamp is None
                    or bid_price_1 is None
                    or ask_price_1 is None
                    or mid_price is None
                ):
                    continue

                timeline = _build_timeline(day, timestamp)
                spread = ask_price_1 - bid_price_1
                imbalance = _safe_imbalance(bid_volume_1, ask_volume_1)

                price_data[product].append(
                    {
                        "timeline": timeline,
                        "bid_price_1": bid_price_1,
                        "ask_price_1": ask_price_1,
                        "mid_price": mid_price,
                        "spread": spread,
                        "imbalance": imbalance if imbalance is not None else 0.0,
                    }
                )

    if not price_data:
        raise ValueError("No valid rows found in prices CSV files.")

    requested_products = list(products) if products is not None else sorted(price_data)
    missing_products = [product for product in requested_products if product not in price_data]
    if missing_products:
        raise ValueError(f"Products not found in prices data: {missing_products}")

    # Optional executed trade overlay.
    trade_data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if include_trades:
        for file_path in sorted(history_path.glob("trades_*.csv")):
            day = _extract_day_from_path(file_path)
            with file_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    symbol = row.get("symbol")
                    timestamp = _to_number(row.get("timestamp"))
                    price = _to_number(row.get("price"))
                    if symbol is None or timestamp is None or price is None:
                        continue

                    timeline = _build_timeline(day, timestamp)
                    trade_data[symbol].append((timeline, price))

    # Three charts per product:
    # - prices
    # - spread
    # - imbalance
    fig, axes = plt.subplots(
        nrows=3 * len(requested_products),
        ncols=1,
        figsize=(13, max(6, 7 * len(requested_products))),
        sharex=False,
    )

    # If only one product is requested, matplotlib may return a 1D array-like object.
    # Converting to a plain list keeps the indexing below simple and readable.
    if not isinstance(axes, (list, tuple)):
        axes = list(axes)

    for product_index, product in enumerate(requested_products):
        sorted_rows = sorted(price_data[product], key=lambda row: row["timeline"])

        x_values = [row["timeline"] for row in sorted_rows]
        bid_values = [row["bid_price_1"] for row in sorted_rows]
        ask_values = [row["ask_price_1"] for row in sorted_rows]
        mid_values = [row["mid_price"] for row in sorted_rows]
        spread_values = [row["spread"] for row in sorted_rows]
        imbalance_values = [row["imbalance"] for row in sorted_rows]

        price_axis = axes[3 * product_index]
        spread_axis = axes[3 * product_index + 1]
        imbalance_axis = axes[3 * product_index + 2]

        # 1) Price view: best bid, best ask, and mid-price.
        price_axis.plot(x_values, bid_values, label=f"{product} bid_1", linewidth=1.0)
        price_axis.plot(x_values, ask_values, label=f"{product} ask_1", linewidth=1.0)
        price_axis.plot(x_values, mid_values, label=f"{product} mid_price", linewidth=1.2)

        if include_trades and product in trade_data:
            sorted_trades = sorted(trade_data[product], key=lambda pair: pair[0])
            price_axis.scatter(
                [point[0] for point in sorted_trades],
                [point[1] for point in sorted_trades],
                s=10,
                alpha=0.5,
                label=f"{product} trades",
            )

        price_axis.set_title(f"{product} - Price view")
        price_axis.set_ylabel("Price")
        price_axis.grid(alpha=0.25)
        price_axis.legend(loc="best")

        # 2) Spread view: useful for market-making and liquidity inspection.
        spread_axis.plot(x_values, spread_values, label=f"{product} spread", linewidth=1.1)
        spread_axis.set_title(f"{product} - Spread")
        spread_axis.set_ylabel("Spread")
        spread_axis.grid(alpha=0.25)
        spread_axis.legend(loc="best")

        # 3) Imbalance view: simple directional pressure signal from top-of-book volume.
        imbalance_axis.plot(
            x_values,
            imbalance_values,
            label=f"{product} imbalance",
            linewidth=1.1,
        )
        imbalance_axis.set_title(f"{product} - Top-of-book imbalance")
        imbalance_axis.set_ylabel("Imbalance")
        imbalance_axis.set_xlabel("Timeline (day*100000 + timestamp)")
        imbalance_axis.grid(alpha=0.25)
        imbalance_axis.legend(loc="best")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    plot_historical_chart(include_trades=True)
