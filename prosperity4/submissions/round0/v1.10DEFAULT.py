#emerald changed: no ar for tomato fair price
import json
from abc import abstractmethod
from typing import Any

try:
    # Competition website import path.
    from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
except ImportError:
    # Local development fallback.
    from prosperity4.algorithms.datamodel import (
        Listing,
        Observation,
        Order,
        OrderDepth,
        ProsperityEncoder,
        Symbol,
        Trade,
        TradingState,
    )


type JSON = dict[str, Any] | list[Any] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out

logger = Logger()


class Strategy:
    def __init__(self, symbol: Symbol, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol]

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders: list[Order] = []
        self.conversions = 0

        if all(
            symbol in state.order_depths
            and len(state.order_depths[symbol].buy_orders) > 0
            and len(state.order_depths[symbol].sell_orders) > 0
            for symbol in self.get_required_symbols()
        ):
            self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        if quantity > 0:
            self.orders.append(Order(self.symbol, price, -quantity))

    def get_mid_price(self, state: TradingState) -> float:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2


class StatefulStrategy(Strategy):
    @abstractmethod
    def save(self) -> JSON:
        raise NotImplementedError()

    @abstractmethod
    def load(self, data: JSON) -> None:
        raise NotImplementedError()


class EmeraldsMarketMaker(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.fair_value = 10_000

        # Hard thresholds for a product with fixed fair value.
        self.take_buy_price = 9_998   # buy anything cheaper than fair
        self.take_sell_price = 10_002 # sell anything richer than fair

        # Inventory bands.
        self.soft_pos = 40
        self.hard_pos = 70

    def act(self, state: TradingState) -> None:
        depth = state.order_depths[self.symbol]
        buy_orders = sorted(depth.buy_orders.items(), reverse=True)   # highest bid first
        sell_orders = sorted(depth.sell_orders.items())               # lowest ask first

        position = state.position.get(self.symbol, 0)

        buy_left = self.limit - position
        sell_left = self.limit + position

        # ============================================================
        # 1) TAKE OBVIOUS MISPRICINGS AROUND KNOWN FAIR = 10000
        # ============================================================

        # Buy all asks <= 9999, subject to position limit.
        for ask_price, ask_volume in sell_orders:
            if buy_left <= 0 or ask_price > self.take_buy_price:
                break
            size = min(buy_left, -ask_volume)
            self.buy(ask_price, size)
            buy_left -= size
            position += size

        # Sell all bids >= 10001, subject to position limit.
        for bid_price, bid_volume in buy_orders:
            if sell_left <= 0 or bid_price < self.take_sell_price:
                break
            size = min(sell_left, bid_volume)
            self.sell(bid_price, size)
            sell_left -= size
            position -= size

        # Recompute remaining capacity after aggressive fills.
        buy_left = self.limit - position
        sell_left = self.limit + position

        if buy_left <= 0 and sell_left <= 0:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        # ============================================================
        # 2) PASSIVE QUOTING: SIMPLE, DISCRETE, INVENTORY-AWARE
        # ============================================================

        # Default idea:
        # - improve best bid by 1 tick if still below fair
        # - improve best ask by 1 tick if still above fair
        bid_quote = min(best_bid + 1, self.fair_value - 1)
        ask_quote = max(best_ask - 1, self.fair_value + 1)

        # Safety: never cross.
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        # Inventory-dependent quote size and aggressiveness.
        if abs(position) < self.soft_pos:
            bid_size = min(buy_left, 32)
            ask_size = min(sell_left, 32)

        elif abs(position) < self.hard_pos:
            bid_size = min(buy_left, 20)
            ask_size = min(sell_left, 20)

            # Lean away from current inventory.
            if position > 0:
                # long -> less aggressive on bid, more aggressive on ask
                bid_quote = min(best_bid, self.fair_value - 1)
                ask_quote = max(best_bid + 1, self.fair_value + 1)
            elif position < 0:
                # short -> more aggressive on bid, less aggressive on ask
                bid_quote = min(best_ask - 1, self.fair_value - 1)
                ask_quote = max(best_ask, self.fair_value + 1)

        else:
            # Very stretched inventory: strongly prioritize flattening.
            bid_size = min(buy_left, 12)
            ask_size = min(sell_left, 12)

            if position > 0:
                # very long -> stop competing on bid, sell more aggressively
                bid_size = 0
                ask_quote = max(best_bid + 1, self.fair_value)
                ask_size = min(sell_left, 40)

            elif position < 0:
                # very short -> stop competing on ask, buy more aggressively
                ask_size = 0
                bid_quote = min(best_ask - 1, self.fair_value)
                bid_size = min(buy_left, 40)

        # ============================================================
        # 3) OPTIONAL INVENTORY FLATTENING AT FAIR WHEN BOOK TOUCHES IT
        # ============================================================

        # If fair is directly tradable and we are imbalanced, use it.
        if position > 0 and sell_left > 0:
            for bid_price, bid_volume in buy_orders:
                if bid_price == self.fair_value:
                    size = min(position, sell_left, bid_volume)
                    if size > 0:
                        self.sell(bid_price, size)
                        sell_left -= size
                        position -= size
                    break

        elif position < 0 < buy_left:
            for ask_price, ask_volume in sell_orders:
                if ask_price == self.fair_value:
                    size = min(-position, buy_left, -ask_volume)
                    if size > 0:
                        self.buy(ask_price, size)
                        buy_left -= size
                        position += size
                    break

        # Recompute after any flattening.
        buy_left = self.limit - position
        sell_left = self.limit + position

        # ============================================================
        # 4) POST PASSIVE ORDERS
        # ============================================================
        if buy_left > 0 and 'bid_size' in locals() and bid_size > 0:
            self.buy(bid_quote, min(buy_left, bid_size))

        if sell_left > 0 and 'ask_size' in locals() and ask_size > 0:
            self.sell(ask_quote, min(sell_left, ask_size))


class TomatoesAdaptiveMarketMaker(StatefulStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        # Simple AR(1) model: fair_value ≈ alpha + beta * prev_mid + drift
        self.fair_value: float | None = None
        self.mid_history: list[float] = []
        self.ar_beta = 0.95  # momentum coefficient: ~95% of prev price carries forward
        self.ar_alpha = 0.05  # mean-reversion coefficient: 5% pull toward long-term mean
        self.alpha = 0.08

        # Inventory bands (same as Emeralds).
        self.soft_pos = 40
        self.hard_pos = 70

    def _estimate_fair_value(self, current_mid: float) -> float:
        """current midprice"""
        return current_mid

    def act(self, state: TradingState) -> None:
        depth = state.order_depths[self.symbol]
        buy_orders = sorted(depth.buy_orders.items(), reverse=True)   # highest bid first
        sell_orders = sorted(depth.sell_orders.items())               # lowest ask first

        position = state.position.get(self.symbol, 0)
        buy_left = self.limit - position
        sell_left = self.limit + position

        # Calculate current mid price from best bid/ask
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        current_mid = (best_bid + best_ask) / 2

        # Update mid history and estimate fair value via AR(1)
        self.mid_history.append(current_mid)
        if len(self.mid_history) > 50:
            self.mid_history.pop(0)

        if self.fair_value is None:
            self.fair_value = current_mid
        else:
            self.fair_value = self._estimate_fair_value(current_mid)

        # ============================================================
        # 1) TAKE OBVIOUS MISPRICINGS AROUND ESTIMATED FAIR VALUE
        # ============================================================

        # Tighter thresholds than Emeralds since tomato fair value may drift.
        take_buy_price = self.fair_value - 1
        take_sell_price = self.fair_value + 1

        # Buy asks that are too cheap relative to fair value.
        for ask_price, ask_volume in sell_orders:
            if buy_left <= 0 or ask_price > take_buy_price:
                break
            size = min(buy_left, -ask_volume)
            self.buy(ask_price, size)
            buy_left -= size
            position += size

        # Sell bids that are too rich relative to fair value.
        for bid_price, bid_volume in buy_orders:
            if sell_left <= 0 or bid_price < take_sell_price:
                break
            size = min(sell_left, bid_volume)
            self.sell(bid_price, size)
            sell_left -= size
            position -= size

        # Recompute remaining capacity.
        buy_left = self.limit - position
        sell_left = self.limit + position

        if buy_left <= 0 and sell_left <= 0:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        # ============================================================
        # 2) PASSIVE QUOTING: SIMPLE, INVENTORY-AWARE (like Emeralds)
        # ============================================================

        # Default: improve best bid/ask by 1 tick, stay near fair value.
        bid_quote = min(best_bid + 1, int(self.fair_value) - 1)
        ask_quote = max(best_ask - 1, int(self.fair_value) + 1)

        # Safety: never cross.
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        # Inventory-dependent quote size and aggressiveness.
        if abs(position) < self.soft_pos:
            bid_size = min(buy_left, 32)
            ask_size = min(sell_left, 32)

        elif abs(position) < self.hard_pos:
            bid_size = min(buy_left, 20)
            ask_size = min(sell_left, 20)

            # Lean away from current inventory.
            if position > 0:
                # long -> less aggressive on bid, more aggressive on ask
                bid_quote = min(best_bid, int(self.fair_value) - 1)
                ask_quote = max(best_bid + 1, int(self.fair_value) + 1)
            elif position < 0:
                # short -> more aggressive on bid, less aggressive on ask
                bid_quote = min(best_ask - 1, int(self.fair_value) - 1)
                ask_quote = max(best_ask, int(self.fair_value) + 1)

        else:
            # Very stretched inventory: strongly prioritize flattening.
            bid_size = min(buy_left, 12)
            ask_size = min(sell_left, 12)

            if position > 0:
                # very long -> stop competing on bid, sell more aggressively
                bid_size = 0
                ask_quote = max(best_bid + 1, int(self.fair_value))
                ask_size = min(sell_left, 40)

            elif position < 0:
                # very short -> stop competing on ask, buy more aggressively
                ask_size = 0
                bid_quote = min(best_ask - 1, int(self.fair_value))
                bid_size = min(buy_left, 40)

        # ============================================================
        # 3) POST PASSIVE ORDERS
        # ============================================================
        if buy_left > 0 and 'bid_size' in locals() and bid_size > 0:
            self.buy(bid_quote, min(buy_left, bid_size))

        if sell_left > 0 and 'ask_size' in locals() and ask_size > 0:
            self.sell(ask_quote, min(sell_left, ask_size))

    def save(self) -> JSON:
        return {
            "fair_value": self.fair_value,
            "mid_history": self.mid_history,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        fair_value = data.get("fair_value")
        if isinstance(fair_value, (int, float)):
            self.fair_value = float(fair_value)

        mid_history = data.get("mid_history")
        if isinstance(mid_history, list):
            self.mid_history = [float(value) for value in mid_history][-50:]


class Trader:
    """
    Website entrypoint: run(state) -> (orders, conversions, trader_data)
    Local compatibility alias: trade(state)
    """

    def __init__(self) -> None:
        limits = {
            "EMERALDS": 80,
            "TOMATOES": 80,
        }

        self.strategies: dict[Symbol, Strategy] = {
            "EMERALDS": EmeraldsMarketMaker("EMERALDS", limits["EMERALDS"]),
            "TOMATOES": TomatoesAdaptiveMarketMaker("TOMATOES", limits["TOMATOES"]),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data: dict[str, JSON] = {}

        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        for symbol, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            strategy_orders, strategy_conversions = strategy.run(state)
            if strategy_orders:
                orders[symbol] = strategy_orders
            conversions += strategy_conversions

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    def trade(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        return self.run(state)
