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
        self.anchor = 10_000 #static mid price value
        self.take_edge = 1 # difference threshold after which mid price is clearly underpriced
        self.inventory_skew = 0.05 # this * max_inventory = spread of emerald
        self.passive_clip = 34 # max between that - position // 2 and 8 will be the quote
        self.aggressive_clip = 22

    def act(self, state: TradingState) -> None:
        depth = state.order_depths[self.symbol]
        buy_orders = sorted(depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        buy_left = self.limit - position
        sell_left = self.limit + position

        # Aggressively take asks only when price is clearly cheap vs the fixed 10k anchor
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        spread = best_ask - best_bid

        recent_flow = 0
        for trade in state.market_trades.get(self.symbol, [])[-8:]:
            if trade.price > self.anchor:
                recent_flow += trade.quantity
            elif trade.price < self.anchor:
                recent_flow -= trade.quantity
        flow_bias = max(-1.0, min(1.0, recent_flow / 40))

        take_edge = self.take_edge + (1 if spread >= 4 else 0)
        for ask_price, ask_volume in sell_orders:
            if buy_left <= 0 or ask_price > self.anchor - take_edge:
                break
            size = min(buy_left, -ask_volume, self.aggressive_clip)
            self.buy(ask_price, size)
            buy_left -= size

        # Symmetric aggressive sells when bids are rich vs anchor
        for bid_price, bid_volume in buy_orders:
            if sell_left <= 0 or bid_price < self.anchor + take_edge:
                break
            size = min(sell_left, bid_volume, self.aggressive_clip)
            self.sell(bid_price, size)
            sell_left -= size

        if buy_left <= 0 and sell_left <= 0:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        spread = best_ask - best_bid

        # ensure reservation matches current inventory short / long
        reservation = self.anchor - self.inventory_skew * position
        # Keep a bounded quote width: tight in normal markets, wider when spread opens.
        half_width = max(1.0, min(3.0, spread / 2))

        bid_quote = floor(reservation - half_width)
        ask_quote = ceil(reservation + half_width)

        # Queue-jump one tick when safe, inspired by wall-based overbidding/underbidding.
        for bid_price, bid_volume in buy_orders:
            if bid_price < self.anchor and bid_volume >= 4:
                bid_quote = max(bid_quote, bid_price + 1)
                break
        for ask_price, ask_volume in sell_orders:
            if ask_price > self.anchor and -ask_volume >= 4:
                ask_quote = min(ask_quote, ask_price - 1)
                break

        # Never cross the inside unintentionally; remain passive when posting quotes.
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        # Reduce passive size as inventory grows to limit directional exposure.
        quote_size = max(8, self.passive_clip - abs(position) // 2)
        if abs(position) > 56:
            quote_size = max(6, quote_size - 8)

        can_post_bid = position < 72
        can_post_ask = position > -72

        if buy_left > 0 and can_post_bid:
            self.buy(bid_quote, min(buy_left, quote_size))
        if sell_left > 0 and can_post_ask:
            self.sell(ask_quote, min(sell_left, quote_size))


class TomatoesAdaptiveMarketMaker(StatefulStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.ema_mid: float | None = None
        self.fast_ema_mid: float | None = None
        self.prev_mid: float | None = None
        self.returns: list[float] = []
        self.alpha = 0.12
        self.fast_alpha = 0.26
        self.inventory_skew = 0.1
        self.adverse_fill_bias = 0.0
        self.last_own_trade_ts = -1

    def _rolling_vol(self) -> float:
        if len(self.returns) < 8:
            return 1.0
        mean = sum(self.returns) / len(self.returns)
        var = sum((ret - mean) ** 2 for ret in self.returns) / len(self.returns)
        # Floor volatility so thresholds never collapse to zero in quiet windows.
        return max(1.0, sqrt(var))

    def _update_fill_bias(self, state: TradingState) -> None:
        signed_volume = 0
        latest_ts = self.last_own_trade_ts

        for trade in state.own_trades.get(self.symbol, []):
            if trade.timestamp <= self.last_own_trade_ts:
                continue
            latest_ts = max(latest_ts, trade.timestamp)
            if trade.buyer == "SUBMISSION":
                signed_volume += trade.quantity
            elif trade.seller == "SUBMISSION":
                signed_volume -= trade.quantity

        self.last_own_trade_ts = latest_ts
        if signed_volume == 0:
            self.adverse_fill_bias *= 0.9
            return

        normalized = max(-1.0, min(1.0, signed_volume / self.limit))
        # Persistent one-sided fills are usually adverse; fade that side for a few steps.
        self.adverse_fill_bias = 0.75 * self.adverse_fill_bias + 0.25 * normalized

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

        if self.fast_ema_mid is None:
            self.fast_ema_mid = mid
        else:
            self.fast_ema_mid = (1 - self.fast_alpha) * self.fast_ema_mid + self.fast_alpha * mid

        if self.prev_mid is not None:
            self.returns.append(mid - self.prev_mid)
            if len(self.returns) > 80:
                self.returns.pop(0)
        self.prev_mid = mid
        self._update_fill_bias(state)

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
        slow_mid = self.ema_mid if self.ema_mid is not None else mid
        fast_mid = self.fast_ema_mid if self.fast_ema_mid is not None else mid
        trend = fast_mid - slow_mid
        high_vol = rolling_vol > 2.2

        # Blend slower/robust and faster/reactive components, then add trend tilt.
        fair_value = 0.45 * slow_mid + 0.55 * microprice + 0.5 * trend
        imbalance_weight = 2.0 if high_vol else 1.4
        signal = imbalance * imbalance_weight + 0.35 * trend
        # Fill bias and inventory skew dampen getting run over in one-way regimes.
        reservation = fair_value + signal - self.inventory_skew * position - 0.8 * self.adverse_fill_bias * rolling_vol

        # Higher volatility demands more edge before crossing the spread.
        take_threshold = max(1.0, (0.9 if high_vol else 0.6) * rolling_vol)
        aggressive_clip = 14 if high_vol else 10

        for ask_price, ask_volume_raw in sell_orders:
            if buy_remaining <= 0 or ask_price > reservation - take_threshold:
                break
            size = min(buy_remaining, -ask_volume_raw, aggressive_clip)
            self.buy(ask_price, size)
            buy_remaining -= size

        for bid_price, bid_volume in buy_orders:
            if sell_remaining <= 0 or bid_price < reservation + take_threshold:
                break
            size = min(sell_remaining, bid_volume, aggressive_clip)
            self.sell(bid_price, size)
            sell_remaining -= size

        spread = best_ask - best_bid
        # Widen passive quotes when realized volatility increases.
        half_width = max(1.0, min(5.0, spread / 2 + 0.35 * rolling_vol + (0.8 if high_vol else 0.0)))

        bid_side_penalty = 1.0 if self.adverse_fill_bias > 0.2 else 0.0
        ask_side_penalty = 1.0 if self.adverse_fill_bias < -0.2 else 0.0

        bid_quote = floor(reservation - half_width - bid_side_penalty)
        ask_quote = ceil(reservation + half_width + ask_side_penalty)
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        # Post less size in one-sided books or when inventory is already extended.
        base_size = 22 if abs(imbalance) < 0.2 else 14
        if high_vol:
            base_size -= 2
        quote_size = max(6, base_size - abs(position) // 6)

        can_post_bid = position < 70
        can_post_ask = position > -70

        if buy_remaining > 0 and can_post_bid:
            self.buy(bid_quote, min(buy_remaining, quote_size))
        if sell_remaining > 0 and can_post_ask:
            self.sell(ask_quote, min(sell_remaining, quote_size))

    def save(self) -> JSON:
        return {
            "ema_mid": self.ema_mid,
            "fast_ema_mid": self.fast_ema_mid,
            "prev_mid": self.prev_mid,
            "returns": self.returns,
            "adverse_fill_bias": self.adverse_fill_bias,
            "last_own_trade_ts": self.last_own_trade_ts,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        ema_mid = data.get("ema_mid")
        if isinstance(ema_mid, (int, float)):
            self.ema_mid = float(ema_mid)

        fast_ema_mid = data.get("fast_ema_mid")
        if isinstance(fast_ema_mid, (int, float)):
            self.fast_ema_mid = float(fast_ema_mid)

        prev_mid = data.get("prev_mid")
        if isinstance(prev_mid, (int, float)):
            self.prev_mid = float(prev_mid)

        returns = data.get("returns")
        if isinstance(returns, list):
            self.returns = [float(value) for value in returns][-80:]

        adverse_fill_bias = data.get("adverse_fill_bias")
        if isinstance(adverse_fill_bias, (int, float)):
            self.adverse_fill_bias = float(adverse_fill_bias)

        last_own_trade_ts = data.get("last_own_trade_ts")
        if isinstance(last_own_trade_ts, int):
            self.last_own_trade_ts = last_own_trade_ts


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
