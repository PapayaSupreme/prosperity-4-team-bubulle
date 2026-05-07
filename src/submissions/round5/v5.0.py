#FROM v3.7, v3.9, v3.10 - clean slate of hardcoded product trading
import json
import math
import statistics
from abc import abstractmethod
from collections import deque
from typing import Any

try:
    # Competition website import path.
    from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
except ImportError:
    # Local development fallback.
    from src.algorithms.datamodel import (
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

class AdaptiveStrategy(StatefulStrategy):
    """
    Starts every unknown product with generic adaptive market making.
    If the recent mid-price series looks anchored, permanently switches that product
    to AnchoredMarketMaker using the detected median anchor.
    """

    PRICE_WINDOW = 120
    ANCHOR_WINDOW = 80

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.prices: list[float] = []
        self.ema: float | None = None
        self.residuals: deque[float] = deque(maxlen=80)
        self.regime = "UNKNOWN"
        self.detected_anchor: float | None = None
        self.anchor_strategy: AnchoredMarketMaker | None = None

    def _make_anchor_strategy(self, anchor: float) -> AnchoredMarketMaker:
        strat = AnchoredMarketMaker(
            symbol=self.symbol,
            limit=self.limit,
            anchor=anchor,
            ema_alpha=0.08,
            residual_alpha=0.30,
            inventory_skew=0.10,
            take_edge=1.0,
        )
        # Warm start the anchored EMA from the adaptive EMA to avoid a cold switch.
        strat.ema_mid = self.ema
        return strat

    def classify(self) -> str:
        if len(self.prices) < 40:
            return "UNKNOWN"

        returns = [self.prices[i] - self.prices[i - 1] for i in range(1, len(self.prices))]
        if len(returns) < 5:
            return "UNKNOWN"

        try:
            r0 = returns[:-1]
            r1 = returns[1:]
            m0 = statistics.mean(r0)
            m1 = statistics.mean(r1)
            cov = sum((r0[i] - m0) * (r1[i] - m1) for i in range(len(r1)))
            denom0 = sum((x - m0) ** 2 for x in r0)
            denom1 = sum((x - m1) ** 2 for x in r1)
            autocorr1 = cov / math.sqrt(denom0 * denom1) if denom0 > 0 and denom1 > 0 else 0.0
        except Exception:
            autocorr1 = 0.0

        slope = (self.prices[-1] - self.prices[0]) / max(1, len(self.prices) - 1)

        if autocorr1 < -0.15:
            return "MEAN_REVERTING"
        if autocorr1 > 0.20 or abs(slope) > 0.50:
            return "TREND"
        return "RANDOM_WALK"

    def try_detect_anchor(self) -> float | None:
        if len(self.prices) < self.ANCHOR_WINDOW:
            return None

        window = self.prices[-self.ANCHOR_WINDOW:]
        anchor = statistics.median(window)
        if anchor <= 0:
            return None

        price_range = max(window) - min(window)
        range_pct = price_range / anchor
        slope = (window[-1] - window[0]) / max(1, len(window) - 1)
        slope_pct = abs(slope) / anchor

        crosses = 0
        for i in range(1, len(window)):
            if (window[i - 1] - anchor) * (window[i] - anchor) < 0:
                crosses += 1

        # Conservative enough to avoid instantly anchoring trends, but not so strict
        # that a genuinely bounded product never switches.
        if range_pct < 0.025 and slope_pct < 0.00008 and crosses >= 4:
            return anchor
        return None

    def _record_mid(self, mid: float) -> None:
        self.prices.append(mid)
        if len(self.prices) > self.PRICE_WINDOW:
            self.prices.pop(0)

        alpha = 0.10
        if self.ema is None:
            self.ema = mid
        else:
            self.ema = alpha * mid + (1.0 - alpha) * self.ema

        self.residuals.append(mid - self.ema)

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        best_bid, best_bid_volume = buy_orders[0]
        best_ask, best_ask_volume_raw = sell_orders[0]
        best_ask_volume = -best_ask_volume_raw
        mid = (best_bid + best_ask) / 2.0

        self._record_mid(mid)

        if self.anchor_strategy is not None:
            anchor_orders, anchor_conversions = self.anchor_strategy.run(state)
            self.orders.extend(anchor_orders)
            self.conversions += anchor_conversions
            return

        anchor = self.try_detect_anchor()
        if anchor is not None:
            self.detected_anchor = anchor
            self.anchor_strategy = self._make_anchor_strategy(anchor)
            anchor_orders, anchor_conversions = self.anchor_strategy.run(state)
            self.orders.extend(anchor_orders)
            self.conversions += anchor_conversions
            return

        residual = mid - (self.ema if self.ema is not None else mid)
        try:
            residual_mean = statistics.mean(self.residuals) if self.residuals else 0.0
            residual_std = statistics.pstdev(self.residuals) if len(self.residuals) > 1 else 0.0
            z_score = (residual - residual_mean) / residual_std if residual_std > 1e-8 else 0.0
        except Exception:
            z_score = 0.0

        self.regime = self.classify()
        position, buy_left, sell_left = self._get_position_capacities(state)

        # Use tiny sizes because every unknown product has limit 10.
        take_size = 2
        passive_size = 2

        if self.regime == "TREND":
            if len(self.prices) >= 5 and self.prices[-1] > self.prices[-5] and buy_left > 0:
                self.buy(best_ask, min(buy_left, take_size))
            elif sell_left > 0:
                self.sell(best_bid, min(sell_left, take_size))
            return

        if self.regime == "MEAN_REVERTING":
            fair = self.ema if self.ema is not None else mid
            if z_score > 2.0 and sell_left > 0:
                self.sell(best_bid, min(sell_left, take_size))
                sell_left -= min(sell_left, take_size)
            elif z_score < -2.0 and buy_left > 0:
                self.buy(best_ask, min(buy_left, take_size))
                buy_left -= min(buy_left, take_size)

            bid_quote = min(int(math.floor(fair - 1)), best_ask - 1)
            ask_quote = max(int(math.ceil(fair + 1)), best_bid + 1)
            if buy_left > 0:
                self.buy(bid_quote, min(buy_left, passive_size))
            if sell_left > 0:
                self.sell(ask_quote, min(sell_left, passive_size))
            return

        # UNKNOWN / RANDOM_WALK: passive only, small size.
        fair = mid
        total_top_depth = best_bid_volume + best_ask_volume
        imbalance = (best_bid_volume - best_ask_volume) / total_top_depth if total_top_depth > 0 else 0.0
        reservation = fair - 0.12 * position + 0.5 * imbalance
        bid_quote = min(int(math.floor(reservation - 2)), best_ask - 1)
        ask_quote = max(int(math.ceil(reservation + 2)), best_bid + 1)

        if buy_left > 0:
            self.buy(bid_quote, min(buy_left, passive_size))
        if sell_left > 0:
            self.sell(ask_quote, min(sell_left, passive_size))

    def save(self) -> JSON:
        return {
            "prices": [round(x, 4) for x in self.prices[-self.PRICE_WINDOW:]],
            "ema": self.ema,
            "residuals": [round(x, 4) for x in list(self.residuals)[-80:]],
            "regime": self.regime,
            "detected_anchor": self.detected_anchor,
            "anchor_strategy": self.anchor_strategy.save() if self.anchor_strategy is not None else None,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        prices = data.get("prices")
        if isinstance(prices, list):
            self.prices = [float(x) for x in prices if isinstance(x, (int, float))][-self.PRICE_WINDOW:]

        ema = data.get("ema")
        if isinstance(ema, (int, float)):
            self.ema = float(ema)

        residuals = data.get("residuals")
        if isinstance(residuals, list):
            self.residuals = deque((float(x) for x in residuals if isinstance(x, (int, float))), maxlen=80)

        regime = data.get("regime")
        if isinstance(regime, str):
            self.regime = regime

        detected_anchor = data.get("detected_anchor")
        if isinstance(detected_anchor, (int, float)):
            self.detected_anchor = float(detected_anchor)
            self.anchor_strategy = self._make_anchor_strategy(self.detected_anchor)
            anchor_data = data.get("anchor_strategy")
            if isinstance(anchor_data, dict):
                self.anchor_strategy.load(anchor_data)

class Trader:
    """
    Website entrypoint: run(state) -> (orders, conversions, trader_data)
    Trades up to 50 unknown products. Each product uses limit 10.
    """

    PRODUCT_LIMIT = 10
    MAX_PRODUCTS = 50

    def __init__(self) -> None:
        self.strategies: dict[Symbol, AdaptiveStrategy] = {}

    def _load_old_data(self, trader_data: str) -> dict[str, JSON]:
        if not trader_data:
            return {}
        try:
            data = json.loads(trader_data)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _get_symbols_to_trade(self, state: TradingState) -> list[Symbol]:
        # Prefer listings order when available, otherwise fall back to order_depths order.
        symbols = list(state.listings.keys()) if state.listings else list(state.order_depths.keys())
        symbols = [s for s in symbols if s in state.order_depths]
        return symbols[: self.MAX_PRODUCTS]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        old_trader_data = self._load_old_data(state.traderData)
        new_trader_data: dict[str, JSON] = {}

        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        for symbol in self._get_symbols_to_trade(state):
            if symbol not in self.strategies:
                self.strategies[symbol] = AdaptiveStrategy(symbol=symbol, limit=self.PRODUCT_LIMIT)

            strategy = self.strategies[symbol]
            saved = old_trader_data.get(symbol)
            if saved is not None:
                strategy.load(saved)

            strategy_orders, strategy_conversions = strategy.run(state)
            conversions += strategy_conversions

            if strategy_orders:
                orders[symbol] = strategy_orders

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    def trade(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        return self.run(state)
