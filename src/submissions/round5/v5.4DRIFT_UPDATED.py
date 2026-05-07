#FROM v5.1 - doesnt trade the unpredictable products (see blacklisted list)

# PnL :
# 5.3.0 (confidence): 28 353
# 5.3.1 (blacklist removed): bad, very bad (20K)
# 5.3.2 (confidence for regime change): 36 444
# 5.3.3 (anchor update): less good
# 5.3.4 (drift update):

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


class PassiveRandomWalkMM(StatefulStrategy):
    """
    Very conservative random-walk market maker.
    Assumption: no directional edge, so only try to capture spread when spread is wide enough.
    """

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.ema_mid: float | None = None
        self.alpha = 0.08

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        best_bid, best_bid_volume = buy_orders[0]
        best_ask, best_ask_volume_raw = sell_orders[0]
        best_ask_volume = -best_ask_volume_raw
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        if self.ema_mid is None:
            self.ema_mid = mid
        else:
            self.ema_mid = (1.0 - self.alpha) * self.ema_mid + self.alpha * mid

        position, buy_left, sell_left = self._get_position_capacities(state)

        # No edge if spread is too tight. Avoid getting picked off.
        if spread < 3:
            return

        total_top_depth = best_bid_volume + best_ask_volume
        imbalance = (best_bid_volume - best_ask_volume) / total_top_depth if total_top_depth > 0 else 0.0

        # Tiny inventory skew. With limit 10, this matters quickly.
        reservation = self.ema_mid - 0.15 * position + 0.4 * imbalance

        # Join/improve only when the spread allows it. Keep distance from mid.
        bid_quote = min(best_bid + 1, int(math.floor(reservation - 1)), best_ask - 1)
        ask_quote = max(best_ask - 1, int(math.ceil(reservation + 1)), best_bid + 1)

        size = 1
        if buy_left > 0:
            self.buy(bid_quote, min(size, buy_left))
        if sell_left > 0:
            self.sell(ask_quote, min(size, sell_left))

    def save(self) -> JSON:
        return {"ema_mid": self.ema_mid}

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return
        ema_mid = data.get("ema_mid")
        if isinstance(ema_mid, (int, float)):
            self.ema_mid = float(ema_mid)


class DriftMarketMaker(StatefulStrategy):
    """
    Intarian-style market maker around a fitted line: fair = intercept + slope * t.
    It is not pure momentum chasing; it quotes around the expected linear fair value.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        intercept: float,
        slope: float,
        start_tick: int,
        inventory_skew: float = 0.10,
        take_edge: float = 1.0,
    ) -> None:
        super().__init__(symbol, limit)
        self.intercept = intercept
        self.slope = slope
        self.start_tick = start_tick
        self.inventory_skew = inventory_skew
        self.take_edge = take_edge
        self.last_fair: float | None = None

    def _fair_value(self, state: TradingState) -> float:
        # Use relative ticks so fitted intercept remains meaningful across days/timestamps.
        t = max(0, state.timestamp - self.start_tick)
        return self.intercept + self.slope * t

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        best_bid, best_bid_volume = buy_orders[0]
        best_ask, best_ask_volume_raw = sell_orders[0]
        best_ask_volume = -best_ask_volume_raw
        spread = best_ask - best_bid

        position, buy_left, sell_left = self._get_position_capacities(state)

        fair = self._fair_value(state)
        self.last_fair = fair

        total_top_depth = best_bid_volume + best_ask_volume
        imbalance = (best_bid_volume - best_ask_volume) / total_top_depth if total_top_depth > 0 else 0.0
        reservation = fair - self.inventory_skew * position + 0.6 * imbalance

        # Take only clearly mispriced orders.
        for ask_price, ask_volume in sell_orders:
            if buy_left <= 0 or ask_price > reservation - self.take_edge:
                break
            size = min(buy_left, -ask_volume, 2)
            self.buy(ask_price, size)
            buy_left -= size
            position += size

        for bid_price, bid_volume in buy_orders:
            if sell_left <= 0 or bid_price < reservation + self.take_edge:
                break
            size = min(sell_left, bid_volume, 2)
            self.sell(bid_price, size)
            sell_left -= size
            position -= size

        buy_left = self.limit - position
        sell_left = self.limit + position
        if buy_left <= 0 and sell_left <= 0:
            return

        half_width = max(1.0, min(3.0, spread / 2.0))
        bid_quote = min(int(math.floor(reservation - half_width)), best_ask - 1)
        ask_quote = max(int(math.ceil(reservation + half_width)), best_bid + 1)

        size = 2
        if buy_left > 0:
            self.buy(bid_quote, min(size, buy_left))
        if sell_left > 0:
            self.sell(ask_quote, min(size, sell_left))

    def save(self) -> JSON:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "start_tick": self.start_tick,
            "last_fair": self.last_fair,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return
        intercept = data.get("intercept")
        slope = data.get("slope")
        start_tick = data.get("start_tick")
        last_fair = data.get("last_fair")
        if isinstance(intercept, (int, float)):
            self.intercept = float(intercept)
        if isinstance(slope, (int, float)):
            self.slope = float(slope)
        if isinstance(start_tick, int):
            self.start_tick = start_tick
        if isinstance(last_fair, (int, float)):
            self.last_fair = float(last_fair)


class ClassifierStrategy(StatefulStrategy):
    """
    Product router for unknown products.

    Flow:
    - observe-only at first
    - if anchor pattern is strong: switch to AnchoredMarketMaker
    - if linear drift is strong: switch to DriftMarketMaker
    - if random walk but spread is tradable: use PassiveRandomWalkMM
    - otherwise stay DISABLED and do not trade
    """

    PRICE_WINDOW = 180
    CLASSIFY_WINDOW = 100
    MIN_OBSERVE_TICKS = 70
    CONFIRM_TICKS = 8
    SWITCH_MARGIN = 0.08
    MIN_CONFIRM_CONFIDENCE = 0.55

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.prices: list[float] = []
        self.timestamps: list[int] = []
        self.mode = "OBSERVE"
        self.candidate_mode: str | None = None
        self.candidate_count = 0
        self.candidate_confidence = 0.0
        self.regime_confidence: dict[str, float] = {
            "ANCHOR": 0.0,
            "DRIFT": 0.0,
            "RANDOM_WALK": 0.0,
        }
        self.detected_anchor: float | None = None
        self.detected_slope: float | None = None
        self.detected_intercept: float | None = None
        self.child: StatefulStrategy | None = None

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _record_mid(self, timestamp: int, mid: float) -> None:
        self.timestamps.append(timestamp)
        self.prices.append(mid)
        if len(self.prices) > self.PRICE_WINDOW:
            self.prices.pop(0)
            self.timestamps.pop(0)

    def _returns_autocorr(self, window: list[float]) -> float:
        if len(window) < 8:
            return 0.0
        returns = [window[i] - window[i - 1] for i in range(1, len(window))]
        if len(returns) < 5:
            return 0.0
        try:
            r0 = returns[:-1]
            r1 = returns[1:]
            m0 = statistics.mean(r0)
            m1 = statistics.mean(r1)
            cov = sum((r0[i] - m0) * (r1[i] - m1) for i in range(len(r1)))
            denom0 = sum((x - m0) ** 2 for x in r0)
            denom1 = sum((x - m1) ** 2 for x in r1)
            return cov / math.sqrt(denom0 * denom1) if denom0 > 0 and denom1 > 0 else 0.0
        except Exception:
            return 0.0

    def _linear_fit(self, xs: list[int], ys: list[float]) -> tuple[float, float, float]:
        """Returns intercept, slope, r2 for y = intercept + slope * (x - xs[0])."""
        if len(xs) < 5 or len(xs) != len(ys):
            return 0.0, 0.0, 0.0

        x0 = xs[0]
        rel_x = [x - x0 for x in xs]
        mx = statistics.mean(rel_x)
        my = statistics.mean(ys)
        sxx = sum((x - mx) ** 2 for x in rel_x)
        if sxx <= 0:
            return ys[-1], 0.0, 0.0

        sxy = sum((rel_x[i] - mx) * (ys[i] - my) for i in range(len(ys)))
        slope = sxy / sxx
        intercept = my - slope * mx

        ss_tot = sum((y - my) ** 2 for y in ys)
        ss_res = sum((ys[i] - (intercept + slope * rel_x[i])) ** 2 for i in range(len(ys)))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        return intercept, slope, r2

    def _slope_tstat(self, xs: list[int], ys: list[float]) -> tuple[float, float, float, float, float]:
        """Return intercept, slope, r2, t_stat (slope / SE), and SE(slope)."""
        n = len(xs)
        if n < 5 or n != len(ys):
            return 0.0, 0.0, 0.0, 0.0, 0.0

        x0 = xs[0]
        rel_x = [x - x0 for x in xs]
        mx = statistics.mean(rel_x)
        my = statistics.mean(ys)
        sxx = sum((x - mx) ** 2 for x in rel_x)
        if sxx <= 0:
            return ys[-1], 0.0, 0.0, 0.0, 0.0

        sxy = sum((rel_x[i] - mx) * (ys[i] - my) for i in range(n))
        slope = sxy / sxx
        intercept = my - slope * mx

        ss_res = sum((ys[i] - (intercept + slope * rel_x[i])) ** 2 for i in range(n))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        denom = max(1, n - 2)
        sigma2 = ss_res / denom if denom > 0 else ss_res
        se_slope = math.sqrt(max(1e-12, sigma2 / sxx))
        t_stat = slope / se_slope if se_slope > 0 else 0.0

        return intercept, slope, r2, t_stat, se_slope

    def _anchor_signal(self) -> tuple[bool, float | None, float]:
        if len(self.prices) < self.CLASSIFY_WINDOW:
            return False, None, 0.0

        window = self.prices[-self.CLASSIFY_WINDOW:]
        anchor = statistics.median(window)
        if anchor <= 0:
            return False, None, 0.0

        price_range = max(window) - min(window)
        range_pct = price_range / anchor
        slope = (window[-1] - window[0]) / max(1, len(window) - 1)
        slope_pct = abs(slope) / anchor
        autocorr = self._returns_autocorr(window)

        crosses = 0
        for i in range(1, len(window)):
            if (window[i - 1] - anchor) * (window[i] - anchor) < 0:
                crosses += 1

        # Anchor: bounded, weak trend, repeated crossing, and some mean-reversion.
        is_anchor = (
            range_pct < 0.025
            and slope_pct < 0.00008
            and crosses >= 4
            and autocorr < 0.05
        )

        range_score = self._clamp01((0.025 - range_pct) / 0.025)
        slope_score = self._clamp01((0.00008 - slope_pct) / 0.00008)
        cross_score = self._clamp01(crosses / 8.0)
        autocorr_score = self._clamp01((0.05 - autocorr) / 0.10)
        confidence = 0.3 * range_score + 0.3 * slope_score + 0.2 * cross_score + 0.2 * autocorr_score

        return is_anchor, anchor if is_anchor else None, confidence

    def _drift_signal(self) -> tuple[bool, float | None, float | None, float, float]:
        """Return (is_drift, intercept, slope, t_stat, confidence).
        Improved: use slope t-stat (slope / SE) as a drift significance metric.
        """
        if len(self.prices) < self.CLASSIFY_WINDOW:
            return False, None, None, 0.0, 0.0

        ys = self.prices[-self.CLASSIFY_WINDOW:]
        xs = self.timestamps[-self.CLASSIFY_WINDOW:]
        intercept, slope, r2, t_stat, se_slope = self._slope_tstat(xs, ys)

        anchor = statistics.median(ys)
        if anchor <= 0:
            return False, None, None, 0.0, 0.0

        total_move = abs(ys[-1] - ys[0])
        noise = statistics.pstdev([ys[i] - ys[i - 1] for i in range(1, len(ys))]) if len(ys) > 2 else 0.0
        range_pct = (max(ys) - min(ys)) / anchor

        # Require some baseline linear explainability and meaningful move, plus t-stat significance.
        is_drift = r2 > 0.72 and total_move > max(3.0, 2.5 * noise) and range_pct > 0.002 and abs(t_stat) > 2.0

        r2_score = self._clamp01((r2 - 0.50) / (0.90 - 0.50))
        move_threshold = max(3.0, 2.5 * noise)
        move_score = self._clamp01(total_move / max(1e-9, move_threshold * 1.8))
        range_score = self._clamp01(range_pct / 0.01)

        # Map t_stat -> [0,1], t=2..8 -> 0..1
        t_score = self._clamp01((abs(t_stat) - 2.0) / 6.0)

        confidence = 0.35 * r2_score + 0.30 * move_score + 0.15 * range_score + 0.20 * t_score

        return is_drift, intercept if is_drift else None, slope if is_drift else None, t_stat if is_drift else 0.0, confidence

    def _random_walk_signal(self, state: TradingState) -> tuple[bool, float]:
        if len(self.prices) < self.MIN_OBSERVE_TICKS:
            return False, 0.0

        buy_orders, sell_orders = self._get_sorted_orders(state)
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        spread = best_ask - best_bid

        window = self.prices[-min(len(self.prices), self.CLASSIFY_WINDOW):]
        autocorr = self._returns_autocorr(window)
        slope = (window[-1] - window[0]) / max(1, len(window) - 1)

        # Only trade random walk if spread is wide enough to harvest.
        is_rw = abs(autocorr) < 0.12 and abs(slope) < 0.25 and spread >= 3

        autocorr_score = self._clamp01((0.12 - abs(autocorr)) / 0.12)
        slope_score = self._clamp01((0.25 - abs(slope)) / 0.25)
        spread_score = self._clamp01((spread - 2.0) / 4.0)
        confidence = 0.4 * autocorr_score + 0.3 * slope_score + 0.3 * spread_score

        return is_rw, confidence

    def _set_candidate(self, mode: str, confidence: float) -> None:
        if self.candidate_mode == mode:
            self.candidate_count += 1
            self.candidate_confidence = confidence
        else:
            self.candidate_mode = mode
            self.candidate_count = 1
            self.candidate_confidence = confidence

    def _build_child(self, mode: str, state: TradingState) -> None:
        if mode == "ANCHOR" and self.detected_anchor is not None:
            self.child = AnchoredMarketMaker(
                symbol=self.symbol,
                limit=self.limit,
                anchor=self.detected_anchor,
                ema_alpha=0.08,
                residual_alpha=0.30,
                inventory_skew=0.10,
                take_edge=1.0,
            )
        elif mode == "DRIFT" and self.detected_intercept is not None and self.detected_slope is not None:
            start_tick = self.timestamps[-self.CLASSIFY_WINDOW] if len(self.timestamps) >= self.CLASSIFY_WINDOW else state.timestamp
            self.child = DriftMarketMaker(
                symbol=self.symbol,
                limit=self.limit,
                intercept=self.detected_intercept,
                slope=self.detected_slope,
                start_tick=start_tick,
            )
        elif mode == "RANDOM_WALK":
            self.child = PassiveRandomWalkMM(symbol=self.symbol, limit=self.limit)
        else:
            self.child = None

    def _classify_and_maybe_switch(self, state: TradingState) -> None:
        if len(self.prices) < self.MIN_OBSERVE_TICKS:
            self.mode = "OBSERVE"
            return

        is_anchor, anchor, anchor_conf = self._anchor_signal()
        is_drift, intercept, slope, t_stat, drift_conf = self._drift_signal()
        is_rw, rw_conf = self._random_walk_signal(state)

        self.regime_confidence["ANCHOR"] = anchor_conf
        self.regime_confidence["DRIFT"] = drift_conf
        self.regime_confidence["RANDOM_WALK"] = rw_conf

        passed: list[tuple[str, float]] = []
        if is_anchor and anchor is not None:
            self.detected_anchor = anchor
            passed.append(("ANCHOR", anchor_conf))
        if is_drift and intercept is not None and slope is not None:
            self.detected_intercept = intercept
            self.detected_slope = slope
            passed.append(("DRIFT", drift_conf))
        if is_rw:
            passed.append(("RANDOM_WALK", rw_conf))

        if passed:
            mode, confidence = max(passed, key=lambda x: x[1])
            passed_scores = {m: c for m, c in passed}

            # Hysteresis: avoid switching away from current candidate unless new mode wins by margin.
            if self.candidate_mode in passed_scores and self.candidate_mode != mode:
                current_conf = passed_scores[self.candidate_mode]
                if confidence < current_conf + self.SWITCH_MARGIN:
                    mode = self.candidate_mode
                    confidence = current_conf

            if confidence >= self.MIN_CONFIRM_CONFIDENCE:
                self._set_candidate(mode, confidence)
            else:
                self._set_candidate("DISABLED", 0.0)
        else:
            self._set_candidate("DISABLED", 0.0)

        if self.candidate_count >= self.CONFIRM_TICKS and self.candidate_mode is not None:
            self.mode = self.candidate_mode
            self._build_child(self.mode, state)

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        mid = (best_bid + best_ask) / 2.0
        self._record_mid(state.timestamp, mid)

        # Once a strong child strategy exists, use it. Disabled means explicitly no trade.
        if self.child is not None:
            child_orders, child_conversions = self.child.run(state)
            self.orders.extend(child_orders)
            self.conversions += child_conversions
            return

        self._classify_and_maybe_switch(state)

        if self.child is not None:
            child_orders, child_conversions = self.child.run(state)
            self.orders.extend(child_orders)
            self.conversions += child_conversions

    def save(self) -> JSON:
        child_type = None
        if isinstance(self.child, AnchoredMarketMaker):
            child_type = "ANCHOR"
        elif isinstance(self.child, DriftMarketMaker):
            child_type = "DRIFT"
        elif isinstance(self.child, PassiveRandomWalkMM):
            child_type = "RANDOM_WALK"

        return {
            "prices": [round(x, 4) for x in self.prices[-self.PRICE_WINDOW:]],
            "timestamps": self.timestamps[-self.PRICE_WINDOW:],
            "mode": self.mode,
            "candidate_mode": self.candidate_mode,
            "candidate_count": self.candidate_count,
            "candidate_confidence": round(self.candidate_confidence, 6),
            "regime_confidence": {k: round(v, 6) for k, v in self.regime_confidence.items()},
            "detected_anchor": self.detected_anchor,
            "detected_intercept": self.detected_intercept,
            "detected_slope": self.detected_slope,
            "child_type": child_type,
            "child": self.child.save() if self.child is not None else None,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        prices = data.get("prices")
        if isinstance(prices, list):
            self.prices = [float(x) for x in prices if isinstance(x, (int, float))][-self.PRICE_WINDOW:]

        timestamps = data.get("timestamps")
        if isinstance(timestamps, list):
            self.timestamps = [int(x) for x in timestamps if isinstance(x, int)][-self.PRICE_WINDOW:]

        mode = data.get("mode")
        if isinstance(mode, str):
            self.mode = mode

        candidate_mode = data.get("candidate_mode")
        self.candidate_mode = candidate_mode if isinstance(candidate_mode, str) else None

        candidate_count = data.get("candidate_count")
        if isinstance(candidate_count, int):
            self.candidate_count = candidate_count

        candidate_confidence = data.get("candidate_confidence")
        if isinstance(candidate_confidence, (int, float)):
            self.candidate_confidence = float(candidate_confidence)

        regime_confidence = data.get("regime_confidence")
        if isinstance(regime_confidence, dict):
            for regime in ("ANCHOR", "DRIFT", "RANDOM_WALK"):
                value = regime_confidence.get(regime)
                if isinstance(value, (int, float)):
                    self.regime_confidence[regime] = float(value)

        detected_anchor = data.get("detected_anchor")
        if isinstance(detected_anchor, (int, float)):
            self.detected_anchor = float(detected_anchor)

        detected_intercept = data.get("detected_intercept")
        if isinstance(detected_intercept, (int, float)):
            self.detected_intercept = float(detected_intercept)

        detected_slope = data.get("detected_slope")
        if isinstance(detected_slope, (int, float)):
            self.detected_slope = float(detected_slope)

        child_type = data.get("child_type")
        child_data = data.get("child")
        self.child = None

        if child_type == "ANCHOR" and self.detected_anchor is not None:
            self.child = AnchoredMarketMaker(
                symbol=self.symbol,
                limit=self.limit,
                anchor=self.detected_anchor,
                ema_alpha=0.08,
                residual_alpha=0.30,
                inventory_skew=0.10,
                take_edge=1.0,
            )
        elif child_type == "DRIFT" and self.detected_intercept is not None and self.detected_slope is not None:
            start_tick = self.timestamps[-self.CLASSIFY_WINDOW] if len(self.timestamps) >= self.CLASSIFY_WINDOW else 0
            self.child = DriftMarketMaker(
                symbol=self.symbol,
                limit=self.limit,
                intercept=self.detected_intercept,
                slope=self.detected_slope,
                start_tick=start_tick,
            )
        elif child_type == "RANDOM_WALK":
            self.child = PassiveRandomWalkMM(symbol=self.symbol, limit=self.limit)

        if self.child is not None and isinstance(child_data, dict):
            self.child.load(child_data)

class Trader:
    """
    Website entrypoint: run(state) -> (orders, conversions, trader_data)
    Trades up to 50 unknown products. Each product uses limit 10 and is routed by ClassifierStrategy.
    """

    PRODUCT_LIMIT = 10
    MAX_PRODUCTS = 50

    def __init__(self) -> None:
        self.strategies: dict[Symbol, ClassifierStrategy] = {}

    def _load_old_data(self, trader_data: str) -> dict[str, JSON]:
        if not trader_data:
            return {}
        try:
            data = json.loads(trader_data)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _get_symbols_to_trade(self, state: TradingState) -> list[Symbol]:
        symbols = list(state.listings.keys()) if state.listings else list(state.order_depths.keys())
        symbols = [s for s in symbols if s in state.order_depths]

        # 🚫 FILTER HERE
        #blacklist = ["SLEEP_POD", "PEBBLES"]
        blacklist = []

        symbols = [
            s for s in symbols
            if not any(bad in s for bad in blacklist)
        ]

        return symbols[: self.MAX_PRODUCTS]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        old_trader_data = self._load_old_data(state.traderData)
        new_trader_data: dict[str, JSON] = {}

        orders: dict[Symbol, list[Order]] = {}
        conversions = 0

        for symbol in self._get_symbols_to_trade(state):
            if symbol not in self.strategies:
                self.strategies[symbol] = ClassifierStrategy(symbol=symbol, limit=self.PRODUCT_LIMIT)

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
