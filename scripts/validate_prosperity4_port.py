from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prosperity4.algorithms.datamodel import Listing, Observation, OrderDepth, Trade, TradingState
from prosperity4.algorithms.hybrid import Trader
from prosperity4.analysis.data import read_tutorial_prices, read_tutorial_trades


@dataclass
class ValidationResult:
    symbols: list[str]
    price_rows: int
    trade_rows: int
    produced_order_symbols: list[str]


def _to_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN check
        return None
    return int(cast(float | int, value))


def _build_state_for_timestamp(day: int = -1, timestamp: int = 0) -> TradingState:
    prices = read_tutorial_prices(day)
    trades = read_tutorial_trades(day)

    current = prices[prices["timestamp"] == timestamp]
    symbols = [str(symbol) for symbol in sorted(current["product"].astype(str).unique().tolist())]

    listings = {symbol: Listing(symbol, symbol, 1) for symbol in symbols}
    order_depths: dict[str, OrderDepth] = {}

    for _, row in current.iterrows():
        symbol = str(row["product"])
        depth = OrderDepth()

        for level in (1, 2, 3):
            bid_price = _to_int(row.get(f"bid_price_{level}"))
            bid_volume = _to_int(row.get(f"bid_volume_{level}"))
            ask_price = _to_int(row.get(f"ask_price_{level}"))
            ask_volume = _to_int(row.get(f"ask_volume_{level}"))

            if bid_price is not None and bid_volume is not None and bid_volume > 0:
                depth.buy_orders[bid_price] = bid_volume
            if ask_price is not None and ask_volume is not None and ask_volume > 0:
                depth.sell_orders[ask_price] = -ask_volume

        order_depths[symbol] = depth

    # Keep observation minimal for tutorial smoke checks.
    observations = Observation({}, {})

    market_trades: dict[str, list[Trade]] = {symbol: [] for symbol in symbols}
    current_trades = trades[trades["timestamp"] == timestamp]
    for _, row in current_trades.iterrows():
        symbol = str(row["symbol"])
        market_trades.setdefault(symbol, []).append(
            Trade(
                symbol=symbol,
                price=int(row["price"]),
                quantity=int(row["quantity"]),
                buyer=(row["buyer"] if isinstance(row["buyer"], str) else None),
                seller=(row["seller"] if isinstance(row["seller"], str) else None),
                timestamp=int(row["timestamp"]),
            )
        )

    state = TradingState(
        traderData="",
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades={symbol: [] for symbol in symbols},
        market_trades=market_trades,
        position={},
        observations=observations,
    )

    return state


def run_validation() -> ValidationResult:
    prices = read_tutorial_prices(-1)
    trades = read_tutorial_trades(-1)
    symbols = [str(symbol) for symbol in sorted(prices["product"].astype(str).unique().tolist())]

    trader = Trader()
    state = _build_state_for_timestamp(day=-1, timestamp=0)
    orders, conversions, trader_data = trader.run(state)

    if conversions != 0:
        raise RuntimeError(f"Expected no conversions in template strategy, got {conversions}")
    if not isinstance(trader_data, str):
        raise RuntimeError("Trader data is not a string")

    return ValidationResult(
        symbols=symbols,
        price_rows=len(prices),
        trade_rows=len(trades),
        produced_order_symbols=sorted(orders.keys()),
    )


if __name__ == "__main__":
    result = run_validation()
    print("Prosperity 4 port validation OK")
    print(f"Symbols in tutorial prices: {result.symbols}")
    print(f"Rows loaded - prices: {result.price_rows}, trades: {result.trade_rows}")
    print(f"Order symbols emitted at t=0: {result.produced_order_symbols}")
