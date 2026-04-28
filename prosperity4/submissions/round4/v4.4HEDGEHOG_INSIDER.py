#FROM v4.11
#PnL: 7 930
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

# Insider trading detection
INFORMED_TRADER_ID = "Mark 14"
LONG, NEUTRAL, SHORT = 1, 0, -1


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
        enable_informed: bool = False,
    ) -> None:
        super().__init__(symbol, limit)
        self.anchor = anchor
        self.ema_alpha = ema_alpha
        self.residual_alpha = residual_alpha
        self.inventory_skew = inventory_skew
        self.take_edge = take_edge
        self.enable_informed = enable_informed

        self.ema_mid: float | None = None
        self.fair_value: float | None = None
        self.informed_direction: int = NEUTRAL
        self.informed_bought_ts: int | None = None
        self.informed_sold_ts: int | None = None
        self.informed_window: int = 500  # react for 500 timestamps after informed trade

    def _estimate_fair_value(self, mid: float) -> float:
        if self.ema_mid is None:
            self.ema_mid = mid
        else:
            self.ema_mid = (1.0 - self.ema_alpha) * self.ema_mid + self.ema_alpha * mid

        adaptive_anchor = 0.75 * self.anchor + 0.25 * self.ema_mid
        residual = mid - adaptive_anchor
        return mid - self.residual_alpha * residual

    def _check_for_informed(self, state: TradingState) -> None:
        """Update informed trader direction based on recent trades (hedgehogs-style)."""
        if not self.enable_informed:
            return

        trades = state.market_trades.get(self.symbol, []) + state.own_trades.get(self.symbol, [])

        for trade in trades:
            if trade.buyer == INFORMED_TRADER_ID:
                self.informed_bought_ts = trade.timestamp
            if trade.seller == INFORMED_TRADER_ID:
                self.informed_sold_ts = trade.timestamp

        # Determine direction based on last action
        if self.informed_bought_ts is None and self.informed_sold_ts is None:
            self.informed_direction = NEUTRAL
        elif self.informed_bought_ts is None:
            self.informed_direction = SHORT
        elif self.informed_sold_ts is None:
            self.informed_direction = LONG
        elif self.informed_bought_ts and self.informed_sold_ts:
            if self.informed_sold_ts > self.informed_bought_ts:
                self.informed_direction = SHORT
            elif self.informed_bought_ts > self.informed_sold_ts:
                self.informed_direction = LONG
            else:
                self.informed_direction = NEUTRAL

    def act(self, state: TradingState) -> None:
        if self.enable_informed:
            self._check_for_informed(state)

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

        # Informed trading boost: adjust reservation based on informed direction
        if self.enable_informed and self.informed_direction == LONG:
            if self.informed_bought_ts is not None and self.informed_bought_ts + self.informed_window >= state.timestamp:
                reservation -= 2.0  # Boost to buy more aggressively
        elif self.enable_informed and self.informed_direction == SHORT:
            if self.informed_sold_ts is not None and self.informed_sold_ts + self.informed_window >= state.timestamp:
                reservation += 2.0  # Boost to sell more aggressively

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
            "informed_bought_ts": self.informed_bought_ts,
            "informed_sold_ts": self.informed_sold_ts,
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

        informed_bought_ts = data.get("informed_bought_ts")
        if informed_bought_ts is not None and isinstance(informed_bought_ts, (int, float)):
            self.informed_bought_ts = int(informed_bought_ts)

        informed_sold_ts = data.get("informed_sold_ts")
        if informed_sold_ts is not None and isinstance(informed_sold_ts, (int, float)):
            self.informed_sold_ts = int(informed_sold_ts)


class OptionSmilePortfolio(StatefulStrategy):
    """Trade all VEV vouchers off a fitted IV smile with expiry-aware unwinds."""

    def __init__(self, underlying_symbol: Symbol, option_limit: int) -> None:
        super().__init__(underlying_symbol, limit=200)
        self.option_limit = option_limit
        self.ema_store: dict[str, float] = {}

        # Smile fit: iv = 3.000615 * m^2 + 1.171123 * m + 0.254329
        self.smile_a = 3.000615
        self.smile_b = 1.171123
        self.smile_c = 0.254329

        self.edge_open = 1.4
        self.edge_close = 0.3
        self.max_order_size = 35

        # Last ~2% of each day focuses on flattening voucher inventory.
        self.unwind_start_timestamp = 980_000

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol]

    def _discover_option_symbols(self, state: TradingState) -> list[Symbol]:
        symbols = [symbol for symbol in state.order_depths.keys() if symbol.startswith(OPTION_PREFIX)]

        def strike(sym: Symbol) -> int:
            return int(sym.split("_")[-1])

        return sorted(symbols, key=strike)

    def _extract_strike(self, symbol: Symbol) -> int | None:
        try:
            return int(symbol.split("_")[-1])
        except (ValueError, IndexError):
            return None

    def _book_mid(self, state: TradingState, symbol: Symbol) -> tuple[float | None, int | None, int | None]:
        if symbol not in state.order_depths:
            return None, None, None

        depth = state.order_depths[symbol]
        if not depth.buy_orders and not depth.sell_orders:
            return None, None, None

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0, best_bid, best_ask
        if best_ask is not None:
            return best_ask - 0.5, best_ask - 1, best_ask
        return best_bid + 0.5, best_bid, best_bid + 1

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_call_price(self, spot: float, strike: float, sigma: float, tte_years: float) -> float:
        if spot <= 0.0 or strike <= 0.0:
            return 0.0
        if tte_years <= 0.0:
            return max(spot - strike, 0.0)

        sigma = max(1e-9, sigma)
        sqrt_t = math.sqrt(tte_years)
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        value = spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)
        return max(0.0, value)

    def _time_to_expiry(self, timestamp: int) -> float:
        # Decay the 4-day tenor through the day; keep a small floor for numerical stability.
        progress = max(0.0, min(1.0, timestamp / 999_900.0))
        days_left = max(0.25, 5.0 * (1.0 - progress))
        return days_left / 365.0

    def _expected_fair_value(self, spot: float, strike: int, tte_years: float) -> float:
        moneyness = math.log(strike / spot)
        fitted_iv = self.smile_a * moneyness * moneyness + self.smile_b * moneyness + self.smile_c
        fitted_iv = max(0.05, min(3.0, fitted_iv))
        return self._bs_call_price(spot, float(strike), fitted_iv, tte_years)

    def _ema(self, key: str, window: int, value: float) -> float:
        alpha = 2.0 / (window + 1)
        prev = self.ema_store.get(key, value)
        current = alpha * value + (1.0 - alpha) * prev
        self.ema_store[key] = current
        return current

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int]:
        orders_by_symbol: dict[Symbol, list[Order]] = {}

        spot, _, _ = self._book_mid(state, self.symbol)
        if spot is None:
            return orders_by_symbol, 0

        tte_years = self._time_to_expiry(state.timestamp)
        option_symbols = self._discover_option_symbols(state)
        if not option_symbols:
            return orders_by_symbol, 0

        unwind_mode = state.timestamp >= self.unwind_start_timestamp

        for option_symbol in option_symbols:
            strike = self._extract_strike(option_symbol)
            if strike is None:
                continue

            option_mid, best_bid, best_ask = self._book_mid(state, option_symbol)
            if option_mid is None or best_bid is None or best_ask is None:
                continue

            position = state.position.get(option_symbol, 0)
            buy_left = self.option_limit - position
            sell_left = self.option_limit + position

            if unwind_mode:
                if position > 0:
                    size = min(position, self.max_order_size)
                    orders_by_symbol.setdefault(option_symbol, []).append(Order(option_symbol, best_bid, -size))
                elif position < 0:
                    size = min(-position, self.max_order_size)
                    orders_by_symbol.setdefault(option_symbol, []).append(Order(option_symbol, best_ask, size))
                continue

            theo = self._expected_fair_value(spot, strike, tte_years)
            edge = option_mid - theo

            edge_mean = self._ema(f"{option_symbol}_edge_mean", 24, edge)
            edge_dev = self._ema(f"{option_symbol}_edge_dev", 80, abs(edge - edge_mean))
            dynamic_open = self.edge_open + 0.2 * edge_dev

            # Mean-reversion around smile-theo edge with conservative size scaling.
            if edge >= dynamic_open and sell_left > 0:
                raw_size = int(min(self.max_order_size, 8 + 6 * (edge - dynamic_open)))
                size = min(raw_size, sell_left)
                if size > 0:
                    orders_by_symbol.setdefault(option_symbol, []).append(Order(option_symbol, best_bid, -size))

            elif edge <= -dynamic_open and buy_left > 0:
                raw_size = int(min(self.max_order_size, 8 + 6 * (-edge - dynamic_open)))
                size = min(raw_size, buy_left)
                if size > 0:
                    orders_by_symbol.setdefault(option_symbol, []).append(Order(option_symbol, best_ask, size))

            # Close toward flat when mispricing fades.
            if abs(edge) <= self.edge_close:
                if position > 0:
                    size = min(position, self.max_order_size)
                    orders_by_symbol.setdefault(option_symbol, []).append(Order(option_symbol, best_bid, -size))
                elif position < 0:
                    size = min(-position, self.max_order_size)
                    orders_by_symbol.setdefault(option_symbol, []).append(Order(option_symbol, best_ask, size))

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
        self.limits = {
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            "VELVETFRUIT_EXTRACT_VOUCHER": 300,
        }

        self.strategies: dict[str, Any] = {
            "HYDROGEL_PACK": AnchoredMarketMaker(
                symbol="HYDROGEL_PACK",
                limit=self.limits["HYDROGEL_PACK"],
                anchor=9_994.0,
                ema_alpha=0.08,
                residual_alpha=0.30,
                inventory_skew=0.08,
                take_edge=1.0,
                enable_informed=True,
            ),
            "VELVETFRUIT_EXTRACT": AnchoredMarketMaker(
                symbol="VELVETFRUIT_EXTRACT",
                limit=self.limits["VELVETFRUIT_EXTRACT"],
                anchor=5_250.0,
                ema_alpha=0.08,
                residual_alpha=0.33,
                inventory_skew=0.07,
                take_edge=1.0,
                enable_informed=False,
            ),
            #"OPTIONS_PORTFOLIO": OptionSmilePortfolio(
            #    underlying_symbol=OPTION_UNDERLYING_SYMBOL,
            #    option_limit=self.limits["VELVETFRUIT_EXTRACT_VOUCHER"],
            #),
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

