#FROM v4.11
#PnL: TBD
import json
import math
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

OPTION_UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
OPTION_PREFIX = "VEV_"


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

    def _get_sorted_orders(self, state: TradingState) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        return buy_orders, sell_orders

    def _get_position_capacities(self, state: TradingState) -> tuple[int, int, int]:
        position = state.position.get(self.symbol, 0)
        buy_left = self.limit - position
        sell_left = self.limit + position
        return position, buy_left, sell_left


class StatefulStrategy(Strategy):
    @abstractmethod
    def save(self) -> JSON:
        raise NotImplementedError()

    @abstractmethod
    def load(self, data: JSON) -> None:
        raise NotImplementedError()


class AnchoredMarketMaker(StatefulStrategy):
    """Mean-reverting market maker around an anchor with light inventory skew."""

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        anchor: float,
        ema_alpha: float,
        residual_alpha: float,
        inventory_skew: float,
        take_edge: float,
    ) -> None:
        super().__init__(symbol, limit)
        self.anchor = anchor
        self.ema_alpha = ema_alpha
        self.residual_alpha = residual_alpha
        self.inventory_skew = inventory_skew
        self.take_edge = take_edge

        self.ema_mid: float | None = None
        self.fair_value: float | None = None

    def _estimate_fair_value(self, mid: float) -> float:
        if self.ema_mid is None:
            self.ema_mid = mid
        else:
            self.ema_mid = (1.0 - self.ema_alpha) * self.ema_mid + self.ema_alpha * mid

        adaptive_anchor = 0.75 * self.anchor + 0.25 * self.ema_mid
        residual = mid - adaptive_anchor
        return mid - self.residual_alpha * residual

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        best_bid, best_bid_volume = buy_orders[0]
        best_ask, best_ask_volume_raw = sell_orders[0]
        best_ask_volume = -best_ask_volume_raw

        position, buy_left, sell_left = self._get_position_capacities(state)

        mid = (best_bid + best_ask) / 2.0
        total_top_depth = best_bid_volume + best_ask_volume

        imbalance = 0.0
        if total_top_depth > 0:
            imbalance = (best_bid_volume - best_ask_volume) / total_top_depth

        fair = self._estimate_fair_value(mid)
        self.fair_value = fair

        reservation = fair - self.inventory_skew * position + 1.2 * imbalance

        for ask_price, ask_volume in sell_orders:
            if buy_left <= 0 or ask_price > reservation - self.take_edge:
                break
            size = min(buy_left, -ask_volume, 20)
            self.buy(ask_price, size)
            buy_left -= size
            position += size

        for bid_price, bid_volume in buy_orders:
            if sell_left <= 0 or bid_price < reservation + self.take_edge:
                break
            size = min(sell_left, bid_volume, 20)
            self.sell(bid_price, size)
            sell_left -= size
            position -= size

        buy_left = self.limit - position
        sell_left = self.limit + position

        if buy_left <= 0 and sell_left <= 0:
            return

        spread = best_ask - best_bid
        half_width = max(1.0, min(4.0, spread / 2.0 + abs(imbalance)))

        bid_quote = int(math.floor(reservation - half_width))
        ask_quote = int(math.ceil(reservation + half_width))

        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        quote_size = max(10, 26 - abs(position) // 10)

        if buy_left > 0:
            self.buy(bid_quote, min(buy_left, quote_size))
        if sell_left > 0:
            self.sell(ask_quote, min(sell_left, quote_size))

    def save(self) -> JSON:
        return {
            "ema_mid": self.ema_mid,
            "fair_value": self.fair_value,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        ema_mid = data.get("ema_mid")
        if isinstance(ema_mid, (int, float)):
            self.ema_mid = float(ema_mid)

        fair_value = data.get("fair_value")
        if isinstance(fair_value, (int, float)):
            self.fair_value = float(fair_value)

class InsiderOptionFollower:
    """
    Trades VEV options only from informed flow on the underlying.

    Logic:
    - If an informed trader buys VELVETFRUIT_EXTRACT, bullish underlying signal.
    - Buy call vouchers.
    - If an informed trader sells VELVETFRUIT_EXTRACT, bearish signal.
    - Sell call vouchers.
    """

    def __init__(self, option_limit: int) -> None:
        self.option_limit = option_limit
        self.max_order_size = 25
        self.max_abs_position = 120
        self.min_signal = 3.0

    def _option_symbols(self, state: TradingState) -> list[Symbol]:
        return sorted(
            [s for s in state.order_depths if s.startswith(OPTION_PREFIX)],
            key=lambda s: int(s.split("_")[-1])
        )

    def run_single_option(
        self,
        state: TradingState,
        symbol: Symbol,
        signal: float,
    ) -> dict[Symbol, list[Order]]:

        orders: dict[Symbol, list[Order]] = {}

        if abs(signal) < 1.5:
            return orders

        depth = state.order_depths.get(symbol)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        position = state.position.get(symbol, 0)

        if abs(position) >= self.max_abs_position:
            return orders

        buy_left = self.option_limit - position
        sell_left = self.option_limit + position

        size = min(
            self.max_order_size,
            int(8 + 5 * abs(signal)),
        )

        if signal > 0 and buy_left > 0:
            orders.setdefault(symbol, []).append(Order(symbol, best_ask, min(size, buy_left)))

        elif signal < 0 and sell_left > 0:
            orders.setdefault(symbol, []).append(Order(symbol, best_bid, -min(size, sell_left)))

        return orders

    def run(
        self,
        state: TradingState,
        underlying_signal: float,
    ) -> dict[Symbol, list[Order]]:

        orders: dict[Symbol, list[Order]] = {}

        if abs(underlying_signal) < self.min_signal:
            return orders

        option_symbols = self._option_symbols(state)
        if not option_symbols:
            return orders

        # Trade mostly ATM-ish options: middle strikes
        mid_index = len(option_symbols) // 2
        selected_options = option_symbols[max(0, mid_index - 1): mid_index + 2]

        for symbol in selected_options:
            depth = state.order_depths.get(symbol)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())

            position = state.position.get(symbol, 0)

            # Extra safety cap
            if abs(position) >= self.max_abs_position:
                continue

            buy_left = min(self.option_limit - position, self.max_abs_position - position)
            sell_left = min(self.option_limit + position, self.max_abs_position + position)

            size = min(
                self.max_order_size,
                int(6 + 4 * abs(underlying_signal)),
            )

            if underlying_signal > 0 and buy_left > 0:
                size = min(size, buy_left)

                # Aggressive, but only on strong signal
                orders.setdefault(symbol, []).append(Order(symbol, best_ask, size))

            elif underlying_signal < 0 and sell_left > 0:
                size = min(size, sell_left)

                # Short calls when informed flow is bearish
                orders.setdefault(symbol, []).append(Order(symbol, best_bid, -size))

        return orders

    def save(self) -> JSON:
        return {
            "ema_store": self.ema_store,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        ema_store = data.get("ema_store")
        if isinstance(ema_store, dict):
            loaded: dict[str, float] = {}
            for key, value in ema_store.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    loaded[key] = float(value)
            self.ema_store = loaded


class Trader:
    """
    Website entrypoint: run(state) -> (orders, conversions, trader_data)
    Local compatibility alias: trade(state)
    """

    def __init__(self) -> None:
        self.limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            "VELVETFRUIT_EXTRACT_VOUCHER": 300,
        }

        # Names come from the trades CSV buyer/seller columns (e.g. "Mark 01").
        self.informed_trader_names = ["Mark 14"]
        self.informed_base_order_size = 20
        self.informed_size_per_lot = 8
        self.informed_max_order_size = 60
        self.informed_flow_memory: dict[Symbol, float] = {}
        self.informed_flow_decay = 0.80
        self.informed_signal_threshold = 2.0

        self.strategies: dict[str, Any] = {
            "HYDROGEL_PACK": AnchoredMarketMaker(
                symbol="HYDROGEL_PACK",
                limit=self.limits["HYDROGEL_PACK"],
                anchor=9_994.0,
                ema_alpha=0.08,
                residual_alpha=0.30,
                inventory_skew=0.08,
                take_edge=1.0,
            ),
            "VELVETFRUIT_EXTRACT": AnchoredMarketMaker(
                symbol="VELVETFRUIT_EXTRACT",
                limit=self.limits["VELVETFRUIT_EXTRACT"],
                anchor=5_250.0,
                ema_alpha=0.08,
                residual_alpha=0.33,
                inventory_skew=0.07,
                take_edge=1.0,
            ),
        }

        self.option_insider = InsiderOptionFollower(
            option_limit=self.limits["VELVETFRUIT_EXTRACT_VOUCHER"])

    def _limit_for_symbol(self, symbol: Symbol) -> int:
        if symbol in self.limits:
            return self.limits[symbol]
        if symbol.startswith(OPTION_PREFIX):
            return self.limits["VELVETFRUIT_EXTRACT_VOUCHER"]
        return 0

    def _projected_position(self, state: TradingState, orders: dict[Symbol, list[Order]], symbol: Symbol) -> int:
        pending = sum(order.quantity for order in orders.get(symbol, []))
        return state.position.get(symbol, 0) + pending

    def _extract_informed_flow_signal(self, state: TradingState) -> dict[Symbol, float]:
        signal_by_symbol: dict[Symbol, float] = {}

        for symbol, trades in state.market_trades.items():

            # ONLY look at VEV options
            if not symbol.startswith(OPTION_PREFIX):
                continue

            if symbol not in state.order_depths:
                continue

            depth = state.order_depths[symbol]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2.0

            raw_score = 0.0
            total_qty = 0

            for trade in trades:
                if trade.buyer not in self.informed_trader_names and trade.seller not in self.informed_trader_names:
                    continue

                qty = int(trade.quantity)
                total_qty += qty

                if trade.buyer in self.informed_trader_names:
                    raw_score += qty

                if trade.seller in self.informed_trader_names:
                    raw_score -= qty

            if total_qty == 0:
                continue

            score = raw_score / total_qty

            prev = self.informed_flow_memory.get(symbol, 0.0)
            smoothed = 0.6 * prev + score
            self.informed_flow_memory[symbol] = smoothed

            if abs(smoothed) >= 1.5:
                signal_by_symbol[symbol] = smoothed

        return signal_by_symbol

    """def _apply_informed_overlay(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        signal_by_symbol: dict[Symbol, int],
    ) -> None:
        for symbol, score in sorted(signal_by_symbol.items(), key=lambda item: abs(item[1]), reverse=True):
            limit = self._limit_for_symbol(symbol)
            if limit <= 0:
                continue
            if symbol not in state.order_depths:
                continue

            depth = state.order_depths[symbol]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            projected_position = self._projected_position(state, orders, symbol)
            buy_left = limit - projected_position
            sell_left = limit + projected_position
            desired_size = min(
                self.informed_max_order_size,
                int(self.informed_base_order_size + self.informed_size_per_lot * abs(score)),
            )

            if score > 0 and buy_left > 0:
                best_ask = min(depth.sell_orders.keys())
                size = min(desired_size, buy_left)
                if size > 0:
                    orders.setdefault(symbol, []).append(Order(symbol, best_ask, size))

            elif score < 0 < sell_left:
                best_bid = max(depth.buy_orders.keys())
                size = min(desired_size, sell_left)
                if size > 0:
                    orders.setdefault(symbol, []).append(Order(symbol, best_bid, -size))"""

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        old_trader_data = json.loads(state.traderData) if state.traderData else {}
        new_trader_data: dict[str, JSON] = {}

        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        for strategy_key, strategy in self.strategies.items():
            if isinstance(strategy, StatefulStrategy) and strategy_key in old_trader_data:
                strategy.load(old_trader_data[strategy_key])

            strategy_orders, strategy_conversions = strategy.run(state)
            conversions += strategy_conversions

            if isinstance(strategy_orders, dict):
                for symbol, symbol_orders in strategy_orders.items():
                    if symbol_orders:
                        orders.setdefault(symbol, []).extend(symbol_orders)
            else:
                if strategy_orders:
                    orders.setdefault(strategy_key, []).extend(strategy_orders)

            if isinstance(strategy, StatefulStrategy):
                new_trader_data[strategy_key] = strategy.save()

        signal_by_symbol = self._extract_informed_flow_signal(state)

        option_orders = {}

        for option_symbol, signal in signal_by_symbol.items():
            partial_orders = self.option_insider.run_single_option(state, option_symbol, signal)

            for s, o in partial_orders.items():
                option_orders.setdefault(s, []).extend(o)

        for symbol, symbol_orders in option_orders.items():
            if symbol_orders:
                orders.setdefault(symbol, []).extend(symbol_orders)

        if signal_by_symbol:
            logger.print("Informed flow signals:", signal_by_symbol)

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    def trade(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        return self.run(state)

