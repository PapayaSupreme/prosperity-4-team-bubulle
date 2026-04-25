# FROM v3.1 - implements options trading logic
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
OPTION_LIMIT = 300

DAYS_PER_YEAR = 365
DAY = 5

THR_OPEN, THR_CLOSE = 0.5, 0.0
LOW_VEGA_THR_ADJ = 0.5

THEO_NORM_WINDOW = 20
IV_SCALPING_THR = 0.7
IV_SCALPING_WINDOW = 100

underlying_mean_reversion_thr = 15
underlying_mean_reversion_window = 10
options_mean_reversion_thr = 5
options_mean_reversion_window = 30


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

    def _take_sell_levels(
        self,
        sell_orders: list[tuple[int, int]],
        buy_left: int,
        position: int,
        max_price: float,
    ) -> tuple[int, int]:
        for ask_price, ask_volume in sell_orders:
            if buy_left <= 0 or ask_price > max_price:
                break
            size = min(buy_left, -ask_volume)
            self.buy(ask_price, size)
            buy_left -= size
            position += size
        return buy_left, position

    def _take_buy_levels(
        self,
        buy_orders: list[tuple[int, int]],
        sell_left: int,
        position: int,
        min_price: float,
    ) -> tuple[int, int]:
        for bid_price, bid_volume in buy_orders:
            if sell_left <= 0 or bid_price < min_price:
                break
            size = min(sell_left, bid_volume)
            self.sell(bid_price, size)
            sell_left -= size
            position -= size
        return sell_left, position

    def _post_passive_orders(
        self,
        buy_left: int,
        sell_left: int,
        bid_quote: int,
        ask_quote: int,
        bid_size: int,
        ask_size: int,
    ) -> None:
        if buy_left > 0 and bid_size > 0:
            self.buy(bid_quote, min(buy_left, bid_size))
        if sell_left > 0 and ask_size > 0:
            self.sell(ask_quote, min(sell_left, ask_size))


class StatefulStrategy(Strategy):
    @abstractmethod
    def save(self) -> JSON:
        raise NotImplementedError()

    @abstractmethod
    def load(self, data: JSON) -> None:
        raise NotImplementedError()



class AdaptiveMarketMaker(StatefulStrategy):
    """Market maker algorithm intended for stocks mean-reversing around a semi-fixed anchor"""
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.fair_value: float | None = None
        self.residual_history: list[float] = []
        self.ema_alpha = 0.1
        self.residual_alpha = 0.25
        self.z_score_coeff = 1.0
        self.z_score_window = 16
        self.z_score_threshold = 0.5
        self.z_score = 0.0
        self.ema_anchor: float | None = None

    def _compute_zscore(self, residual: float) -> None: # Use once !!
        z = 0.0
        if len(self.residual_history) >= 4:
            mean_ = sum(self.residual_history) / len(self.residual_history)
            variance = sum((x - mean_) ** 2 for x in self.residual_history) / len(self.residual_history)
            std = variance ** 0.5

            if std >= 0.1:
                z = (residual - mean_) / std

        self.residual_history.append(residual)
        if len(self.residual_history) > self.z_score_window:
            self.residual_history.pop(0)

        self.z_score =  max(-2.0, min(2.0, z))

    def _estimate_fair_value(self, current_mid: float) -> float:
        prev_anchor = self.ema_anchor

        self._update_ema_anchor(current_mid)

        anchor = self.ema_anchor if prev_anchor is None else prev_anchor
        residual = current_mid - anchor

        return current_mid - self.residual_alpha * residual

    def _update_ema_anchor(self, current_mid: float) -> None:
        if self.ema_anchor is None:
            self.ema_anchor = current_mid
        else:
            self.ema_anchor = (
                self.ema_alpha * current_mid
                + (1 - self.ema_alpha) * self.ema_anchor
            )

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        position, buy_left, sell_left = self._get_position_capacities(state)

        # Calculate current mid-price from best bid/ask
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        current_mid = (best_bid + best_ask) / 2

        # Microprice weights prices by opposite-side top-of-book volume.
        best_bid_vol = max(0, buy_orders[0][1])
        best_ask_vol = max(0, -sell_orders[0][1])
        top_vol = best_bid_vol + best_ask_vol
        microprice = (
            (best_ask * best_bid_vol + best_bid * best_ask_vol) / top_vol
            if top_vol > 0
            else current_mid
        )

        self.fair_value = self._estimate_fair_value(current_mid)
        fair_value = self.fair_value if self.fair_value is not None else current_mid

        # ============================================================
        # 1) TAKE OBVIOUS MISPRICINGS AROUND ESTIMATED FAIR VALUE
        # ============================================================

        # Take thresholds around current fair value, taking into account z_score.
        take_buy_price = fair_value - 1  # THE HIGHER, THE EASIER. THIS IS THE PRICE I WANT TO BUY MY STOCK FOR
        take_sell_price = fair_value + 1  # THE LOWER, THE EASIER. THIS IS THE PRICE I WANT TO SELL MY STOCK FOR


        buy_left, position = self._take_sell_levels(sell_orders, buy_left, position, take_buy_price)
        sell_left, position = self._take_buy_levels(buy_orders, sell_left, position, take_sell_price)

        # Recompute remaining capacity.
        buy_left = self.limit - position
        sell_left = self.limit + position

        if buy_left <= 0 and sell_left <= 0:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        # ============================================================
        # 2) PASSIVE QUOTING
        # ============================================================

        fair_int = int(round(fair_value))
        bid_quote = min(best_bid + 1, fair_int - 1)
        ask_quote = max(best_ask - 1, fair_int + 1)

        # Safety: never cross.
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        base_bid_size = buy_left
        base_ask_size = sell_left

        if self.z_score == 2:
            # Absurdly Rich vs anchor -> lean only short
            bid_size = 0
            ask_size = base_ask_size

        elif self.z_score > 1.5:
            # Very Rich vs anchor -> lean very short
            bid_size = max(base_bid_size // 4, 2)
            ask_size = base_ask_size

        elif self.z_score > 1.0:
            # Rich vs anchor -> lean short
            bid_size = max(base_bid_size // 3, 4)
            ask_size = base_ask_size

        elif self.z_score > 0.5:
            # Mildly rich
            bid_size = max(base_bid_size // 2, 6)
            ask_size = base_ask_size

        elif self.z_score == -2:
            # Absurdly Cheap vs anchor -> lean only long
            bid_size = base_bid_size
            ask_size = 0

        elif self.z_score < -1.5:
            # Very Cheap vs anchor -> lean very long
            bid_size = base_bid_size
            ask_size = max(base_ask_size // 4, 2)

        elif self.z_score < -1.0:
            # Cheap vs anchor -> lean long
            bid_size = base_bid_size
            ask_size = max(base_ask_size // 3, 4)

        elif self.z_score < -0.5:
            # Mildly cheap
            bid_size = base_bid_size
            ask_size = max(base_ask_size // 2, 6)

        else:
            # Near anchor
            bid_size = base_bid_size
            ask_size = base_ask_size

        # ============================================================
        # 3) POST PASSIVE ORDERS
        # ============================================================

        self._post_passive_orders(buy_left, sell_left, bid_quote, ask_quote, bid_size, ask_size)

    def save(self) -> JSON:
        return {
            "fair_value": self.fair_value,
            "ema_anchor": self.ema_anchor,
            "residual_history": self.residual_history,
            "z_score": self.z_score
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        fair_value = data.get("fair_value")
        if isinstance(fair_value, (int, float)):
            self.fair_value = float(fair_value)

        residual_history = data.get("residual_history")
        if isinstance(residual_history, list):
            self.residual_history = [float(x) for x in residual_history]

        ema_anchor = data.get("ema_anchor")
        if isinstance(ema_anchor, (int, float)):
            self.ema_anchor = float(ema_anchor)

        z_score = data.get("z_score")
        if isinstance(z_score, (int, float)):
            self.z_score = float(z_score)


class OptionsPortfolioStrategy(StatefulStrategy):
    """Port of hedgehogs option logic using v3.6 state save/load and execution flow."""

    def __init__(self, underlying_symbol: Symbol, underlying_limit: int, option_limit: int) -> None:
        super().__init__(underlying_symbol, underlying_limit)
        self.option_limit = option_limit
        self.ema_store: dict[str, float] = {}

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol]

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def _extract_strike(self, symbol: Symbol) -> int | None:
        if not symbol.startswith(OPTION_PREFIX):
            return None
        try:
            return int(symbol.split("_")[-1])
        except ValueError:
            return None

    def _ema(self, key: str, window: int, value: float) -> float:
        old = self.ema_store.get(key, 0.0)
        alpha = 2.0 / (window + 1)
        new = alpha * value + (1.0 - alpha) * old
        self.ema_store[key] = new
        return new

    def _tte(self, timestamp: int) -> float:
        tte = 1.0 - (DAYS_PER_YEAR - 8 + DAY + (timestamp // 100) / 10_000) / DAYS_PER_YEAR
        return max(tte, 1e-6)

    def _get_option_values(self, spot: float, strike: float, tte: float) -> tuple[float, float, float]:
        m_t_k = math.log(strike / spot) / math.sqrt(tte)
        coeffs = [0.27362531, 0.01007566, 0.14876677]
        iv = max(0.05, coeffs[0] * m_t_k * m_t_k + coeffs[1] * m_t_k + coeffs[2])

        d1 = (math.log(spot / strike) + 0.5 * iv * iv * tte) / (iv * math.sqrt(tte))
        d2 = d1 - iv * math.sqrt(tte)

        call_value = spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)
        delta = self._norm_cdf(d1)
        vega = spot * self._norm_pdf(d1) * math.sqrt(tte)
        return call_value, delta, vega

    def _get_book(self, state: TradingState, symbol: Symbol) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        depth = state.order_depths[symbol]
        buy_orders = sorted(depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(depth.sell_orders.items())
        return buy_orders, sell_orders

    def _discover_option_symbols(self, state: TradingState) -> list[Symbol]:
        symbols = [s for s in state.order_depths.keys() if s.startswith(OPTION_PREFIX)]
        return sorted(symbols, key=lambda s: self._extract_strike(s) or 0)

    def _get_mid_and_best(
        self,
        buy_orders: list[tuple[int, int]],
        sell_orders: list[tuple[int, int]],
    ) -> tuple[float | None, int | None, int | None]:
        best_bid = buy_orders[0][0] if buy_orders else None
        best_ask = sell_orders[0][0] if sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0, best_bid, best_ask
        if best_ask is not None:
            return best_ask - 0.5, best_ask - 1, best_ask
        if best_bid is not None:
            return best_bid + 0.5, best_bid, best_bid + 1
        return None, None, None

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int]:
        orders_by_symbol: dict[Symbol, list[Order]] = {}

        if self.symbol not in state.order_depths:
            return orders_by_symbol, 0

        option_symbols = self._discover_option_symbols(state)
        if not option_symbols:
            return orders_by_symbol, 0

        try:
            u_buy, u_sell = self._get_book(state, self.symbol)
        except KeyError:
            return orders_by_symbol, 0

        underlying_mid, underlying_best_bid, underlying_best_ask = self._get_mid_and_best(u_buy, u_sell)
        if underlying_mid is None or underlying_best_bid is None or underlying_best_ask is None:
            return orders_by_symbol, 0

        indicators: dict[str, Any] = {
            "ema_u_dev": None,
            "ema_o_dev": None,
            "mean_theo_diffs": {},
            "current_theo_diffs": {},
            "switch_means": {},
            "vegas": {},
        }

        new_mean_price = self._ema("ema_u", underlying_mean_reversion_window, underlying_mid)
        indicators["ema_u_dev"] = underlying_mid - new_mean_price

        new_mean_price = self._ema("ema_o", options_mean_reversion_window, underlying_mid)
        indicators["ema_o_dev"] = underlying_mid - new_mean_price

        option_state: dict[Symbol, dict[str, float | int]] = {}

        for option_symbol in option_symbols:
            strike = self._extract_strike(option_symbol)
            if strike is None:
                continue

            buy_orders, sell_orders = self._get_book(state, option_symbol)
            wall_mid, best_bid, best_ask = self._get_mid_and_best(buy_orders, sell_orders)
            if wall_mid is None or best_bid is None or best_ask is None:
                continue

            tte = self._tte(state.timestamp)
            option_theo, _, option_vega = self._get_option_values(underlying_mid, float(strike), tte)
            option_theo_diff = wall_mid - option_theo

            indicators["current_theo_diffs"][option_symbol] = option_theo_diff
            indicators["vegas"][option_symbol] = option_vega

            mean_diff = self._ema(f"{option_symbol}_theo_diff", THEO_NORM_WINDOW, option_theo_diff)
            indicators["mean_theo_diffs"][option_symbol] = mean_diff

            avg_dev = self._ema(
                f"{option_symbol}_avg_devs",
                IV_SCALPING_WINDOW,
                abs(option_theo_diff - mean_diff),
            )
            indicators["switch_means"][option_symbol] = avg_dev

            pos = state.position.get(option_symbol, 0)
            option_state[option_symbol] = {
                "mid": wall_mid,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "position": pos,
                "buy_left": self.option_limit - pos,
                "sell_left": self.option_limit + pos,
            }

        if not option_state:
            return orders_by_symbol, 0

        def place_bid(symbol: Symbol, price: int, quantity: int) -> None:
            st = option_state[symbol]
            size = min(max(0, quantity), int(st["buy_left"]))
            if size <= 0:
                return
            orders_by_symbol.setdefault(symbol, []).append(Order(symbol, int(price), size))
            st["buy_left"] = int(st["buy_left"]) - size
            st["position"] = int(st["position"]) + size

        def place_ask(symbol: Symbol, price: int, quantity: int) -> None:
            st = option_state[symbol]
            size = min(max(0, quantity), int(st["sell_left"]))
            if size <= 0:
                return
            orders_by_symbol.setdefault(symbol, []).append(Order(symbol, int(price), -size))
            st["sell_left"] = int(st["sell_left"]) - size
            st["position"] = int(st["position"]) - size

        sorted_symbols = sorted(option_state.keys(), key=lambda s: self._extract_strike(s) or 0)
        iv_scalping_symbols = sorted_symbols[1:]
        mr_symbols = sorted_symbols[:1]

        if state.timestamp / 100 >= min(THEO_NORM_WINDOW, underlying_mean_reversion_window, options_mean_reversion_window):
            for option_symbol in iv_scalping_symbols:
                switch_mean = indicators["switch_means"].get(option_symbol)
                current_theo_diff = indicators["current_theo_diffs"].get(option_symbol)
                mean_theo_diff = indicators["mean_theo_diffs"].get(option_symbol)

                if switch_mean is None or current_theo_diff is None or mean_theo_diff is None:
                    continue

                st = option_state[option_symbol]
                best_bid = int(st["best_bid"])
                best_ask = int(st["best_ask"])
                wall_mid = float(st["mid"])

                if switch_mean >= IV_SCALPING_THR:
                    low_vega_adj = LOW_VEGA_THR_ADJ if indicators["vegas"].get(option_symbol, 0.0) <= 1 else 0.0

                    if (
                        current_theo_diff - wall_mid + best_bid - mean_theo_diff
                        >= (THR_OPEN + low_vega_adj)
                    ):
                        place_ask(option_symbol, best_bid, int(st["sell_left"]))

                    if (
                        current_theo_diff - wall_mid + best_bid - mean_theo_diff >= THR_CLOSE
                        and int(st["position"]) > 0
                    ):
                        place_ask(option_symbol, best_bid, int(st["position"]))

                    elif (
                        current_theo_diff - wall_mid + best_ask - mean_theo_diff
                        <= -(THR_OPEN + low_vega_adj)
                    ):
                        place_bid(option_symbol, best_ask, int(st["buy_left"]))

                    if (
                        current_theo_diff - wall_mid + best_ask - mean_theo_diff <= -THR_CLOSE
                        and int(st["position"]) < 0
                    ):
                        place_bid(option_symbol, best_ask, -int(st["position"]))

                else:
                    if int(st["position"]) > 0:
                        place_ask(option_symbol, best_bid, int(st["position"]))
                    elif int(st["position"]) < 0:
                        place_bid(option_symbol, best_ask, -int(st["position"]))

            for option_symbol in mr_symbols:
                current_theo_diff = indicators["current_theo_diffs"].get(option_symbol)
                mean_theo_diff = indicators["mean_theo_diffs"].get(option_symbol)
                ema_o_dev = indicators.get("ema_o_dev")

                if current_theo_diff is None or mean_theo_diff is None or ema_o_dev is None:
                    continue

                st = option_state[option_symbol]
                current_deviation = float(ema_o_dev) + (current_theo_diff - mean_theo_diff)

                if current_deviation > options_mean_reversion_thr:
                    place_ask(option_symbol, int(st["best_bid"]), int(st["sell_left"]))
                elif current_deviation < -options_mean_reversion_thr:
                    place_bid(option_symbol, int(st["best_ask"]), int(st["buy_left"]))

        if state.timestamp / 100 >= underlying_mean_reversion_window:
            ema_o_dev = indicators.get("ema_o_dev")
            if ema_o_dev is not None:
                u_pos = state.position.get(self.symbol, 0)
                u_buy_left = self.limit - u_pos
                u_sell_left = self.limit + u_pos

                if ema_o_dev > underlying_mean_reversion_thr and u_sell_left > 0:
                    price = int(underlying_best_bid + 1)
                    orders_by_symbol.setdefault(self.symbol, []).append(Order(self.symbol, price, -u_sell_left))
                elif ema_o_dev < -underlying_mean_reversion_thr and u_buy_left > 0:
                    price = int(underlying_best_ask - 1)
                    orders_by_symbol.setdefault(self.symbol, []).append(Order(self.symbol, price, u_buy_left))

        return orders_by_symbol, 0

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
        limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            "VELVETFRUIT_EXTRACT_VOUCHER": 300, # for each of the 10 vouchers
        }

        self.strategies: dict[str, Any] = {
            "HYDROGEL_PACK": AdaptiveMarketMaker("HYDROGEL_PACK", limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": AdaptiveMarketMaker("VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"]),
            "OPTIONS_PORTFOLIO": OptionsPortfolioStrategy(
                OPTION_UNDERLYING_SYMBOL,
                limits["VELVETFRUIT_EXTRACT"],
                limits["VELVETFRUIT_EXTRACT_VOUCHER"],
            ),
        }

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

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data

    def trade(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        return self.run(state)
