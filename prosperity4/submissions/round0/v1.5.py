#emerald changed: take_edge, inventory_kew descreased, passive_clip and size increased
import json
from abc import abstractmethod
from math import ceil, floor, sqrt
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
        self.take_buy_price = 9_999   # buy anything cheaper than fair
        self.take_sell_price = 10_001 # sell anything richer than fair

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
        self.ema_mid: float | None = None
        self.prev_mid: float | None = None
        self.returns: list[float] = []
        self.alpha = 0.08
        self.inventory_skew = 0.06

    def _rolling_vol(self) -> float:
        if len(self.returns) < 8:
            return 1.0
        mean = sum(self.returns) / len(self.returns)
        var = sum((ret - mean) ** 2 for ret in self.returns) / len(self.returns)
        # Floor volatility so thresholds never collapse to zero in quiet windows.
        return max(1.0, sqrt(var))

    def act(self, state: TradingState) -> None:
        depth = state.order_depths[self.symbol]
        buy_orders = sorted(depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(depth.sell_orders.items())

        best_bid, best_bid_volume = buy_orders[0]
        best_ask, best_ask_volume = sell_orders[0]
        best_ask_volume = -best_ask_volume

        position = state.position.get(self.symbol, 0)
        buy_remaining = self.limit - position
        sell_remaining = self.limit + position

        mid = (best_bid + best_ask) / 2
        if self.ema_mid is None:
            self.ema_mid = mid
        else:
            # EMA is a slow anchor fair value that adapts to drift.
            self.ema_mid = (1 - self.alpha) * self.ema_mid + self.alpha * mid

        if self.prev_mid is not None:
            self.returns.append(mid - self.prev_mid)
            if len(self.returns) > 80:
                self.returns.pop(0)
        self.prev_mid = mid

        total_top_depth = best_bid_volume + best_ask_volume
        imbalance = 0.0
        if total_top_depth > 0:
            # Positive imbalance means bid side is stronger than ask side.
            imbalance = (best_bid_volume - best_ask_volume) / total_top_depth

        microprice = mid
        if total_top_depth > 0:
            # Microprice leans toward the side with more resting depth at top-of-book.
            microprice = (best_ask * best_bid_volume + best_bid * best_ask_volume) / total_top_depth

        rolling_vol = self._rolling_vol()
        # Blend slower EMA and faster microprice to get a robust but reactive fair value.
        fair_value = 0.6 * self.ema_mid + 0.4 * microprice
        # Directional tilt from order-book pressure and short-term momentum.
        signal = imbalance * 1.5 + 0.25 * (mid - self.ema_mid)
        # Inventory skew shifts quotes away from current exposure.
        reservation = fair_value + signal - self.inventory_skew * position

        # Higher volatility demands more edge before crossing the spread.
        take_threshold = max(1.0, 0.8 * rolling_vol)

        for ask_price, ask_volume_raw in sell_orders:
            if buy_remaining <= 0 or ask_price > reservation - take_threshold:
                break
            size = min(buy_remaining, -ask_volume_raw, 12)
            self.buy(ask_price, size)
            buy_remaining -= size

        for bid_price, bid_volume in buy_orders:
            if sell_remaining <= 0 or bid_price < reservation + take_threshold:
                break
            size = min(sell_remaining, bid_volume, 12)
            self.sell(bid_price, size)
            sell_remaining -= size

        spread = best_ask - best_bid
        # Widen passive quotes when realized volatility increases.
        half_width = max(1.0, min(4.0, spread / 2 + 0.3 * rolling_vol))

        bid_quote = floor(reservation - half_width)
        ask_quote = ceil(reservation + half_width)
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        # Post less size in one-sided books or when inventory is already extended.
        base_size = 18 if abs(imbalance) < 0.2 else 12
        quote_size = max(6, base_size - abs(position) // 8)

        if buy_remaining > 0:
            self.buy(bid_quote, min(buy_remaining, quote_size))
        if sell_remaining > 0:
            self.sell(ask_quote, min(sell_remaining, quote_size))

    def save(self) -> JSON:
        return {
            "ema_mid": self.ema_mid,
            "prev_mid": self.prev_mid,
            "returns": self.returns,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        ema_mid = data.get("ema_mid")
        if isinstance(ema_mid, (int, float)):
            self.ema_mid = float(ema_mid)

        prev_mid = data.get("prev_mid")
        if isinstance(prev_mid, (int, float)):
            self.prev_mid = float(prev_mid)

        returns = data.get("returns")
        if isinstance(returns, list):
            self.returns = [float(value) for value in returns][-80:]


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
