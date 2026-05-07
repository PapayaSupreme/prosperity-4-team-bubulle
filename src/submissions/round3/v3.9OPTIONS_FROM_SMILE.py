# FROM v3.7 - implements OptionTrader with a strat from the smile plot
# PnL : 1 583
# MC : 19 856 + 8 544
import json
import math
from abc import abstractmethod
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
        self.ema_anchor: float | None = None

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

        fair_value = self._estimate_fair_value(current_mid)

        self.fair_value = fair_value
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

        bid_size = buy_left
        ask_size = sell_left

        # ============================================================
        # 3) POST PASSIVE ORDERS
        # ============================================================

        self._post_passive_orders(buy_left, sell_left, bid_quote, ask_quote, bid_size, ask_size)

    def save(self) -> JSON:
        return {
            "fair_value": self.fair_value,
            "ema_anchor": self.ema_anchor,
            "residual_history": self.residual_history,
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

class OptionTrader(Strategy):
    """Simple VEV voucher trader.

    Logic:
    - Treat VEV_4000 / VEV_4500 as overpriced downside-vol products when their
      market price is high versus your fitted-smile fair value.
    - Sell them only when VELVETFRUIT_EXTRACT is close to its 5250 anchor and
      the underlying L1 imbalance is not strongly bearish.
    - Buy back shorts when the option becomes cheap, downside pressure appears,
      or the underlying moves too far away from anchor.

    This is deliberately conservative. It is not a full option market maker yet.
    """

    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        strike: int,
        underlying_symbol: Symbol = "VELVETFRUIT_EXTRACT",
    ) -> None:
        super().__init__(symbol, limit)
        self.strike = strike
        self.underlying_symbol = underlying_symbol

        self.anchor = 5250.0
        self.max_anchor_deviation = 45.0

        # Conservative first-pass thresholds. Tune after backtest.
        self.sell_edge = 1.0
        self.buyback_edge = -1.0
        self.max_order_size = 30

        # Your assumption from the corrected smile script.
        self.tte_years = 5.0 / 365.0
        # iv = a*m^2 + b*m + c
        self.smile_a = 2.473985
        self.smile_b = 1.152545
        self.smile_c = 0.276766

    def get_required_symbols(self) -> list[Symbol]:
        return [self.symbol, self.underlying_symbol]

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_call_price(self, spot: float, strike: float, sigma: float) -> float:
        """Black-Scholes call value, rate=0."""
        if spot <= 0.0 or strike <= 0.0:
            return 0.0
        if self.tte_years <= 0.0:
            return max(spot - strike, 0.0)

        sigma = max(sigma, 1e-9)
        sqrt_t = math.sqrt(self.tte_years)

        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * self.tte_years) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        value = spot * self._norm_cdf(d1) - strike * self._norm_cdf(d2)
        return max(0.0, value)

    def _expected_fair_price(self, spot: float) -> float:
        """Fair option value from your fitted IV smile."""
        moneyness = math.log(self.strike / spot)
        fitted_iv = (
            self.smile_a * moneyness * moneyness
            + self.smile_b * moneyness
            + self.smile_c
        )

        fitted_iv = max(0.05, min(2.0, fitted_iv))
        return self._bs_call_price(spot, self.strike, fitted_iv)

    def _book_mid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def _l1_imbalance(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        bid_vol = max(0, order_depth.buy_orders[best_bid])
        ask_vol = max(0, -order_depth.sell_orders[best_ask])
        total = bid_vol + ask_vol

        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def act(self, state: TradingState) -> None:
        option_bids, option_asks = self._get_sorted_orders(state)
        underlying_depth = state.order_depths[self.underlying_symbol]

        position, buy_left, sell_left = self._get_position_capacities(state)

        best_option_bid = option_bids[0][0]
        best_option_ask = option_asks[0][0]
        option_mid = (best_option_bid + best_option_ask) / 2.0

        spot = self._book_mid(underlying_depth)
        l1 = self._l1_imbalance(underlying_depth)

        fair = self._expected_fair_price(spot)
        edge = option_mid - fair

        near_anchor = abs(spot - self.anchor) <= self.max_anchor_deviation
        not_bearish = l1 > -0.20

        # Main trade:
        # If the voucher trades above fitted-smile fair, sell it.
        # This captures the "overpriced put / downside vol" idea.
        if sell_left > 0 and edge >= self.sell_edge and near_anchor and not_bearish:
            visible_bid_size = max(0, option_bids[0][1])
            size = min(sell_left, visible_bid_size, self.max_order_size)
            self.sell(best_option_bid, size)
            sell_left -= size
            position -= size

        # Risk reduction / buyback:
        # If already short and the option cheapens, or the underlying starts
        # showing downside pressure, buy some back.
        if position < 0 and buy_left > 0:
            should_buy_back = edge <= self.buyback_edge or l1 < -0.35 or not near_anchor
            if should_buy_back:
                visible_ask_size = max(0, -option_asks[0][1])
                size = min(buy_left, visible_ask_size, abs(position), self.max_order_size)
                self.buy(best_option_ask, size)


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

        option_symbols_to_trade: dict[Symbol, int] = {
            "VEV_4000": 4000,
            "VEV_4500": 4500,
        }

        self.strategies: dict[Symbol, Strategy] = {
            "HYDROGEL_PACK": AdaptiveMarketMaker("HYDROGEL_PACK", limits["HYDROGEL_PACK"]),
            "VELVETFRUIT_EXTRACT": AdaptiveMarketMaker("VELVETFRUIT_EXTRACT", limits["VELVETFRUIT_EXTRACT"]),
            **{
                symbol: OptionTrader(
                    symbol=symbol,
                    limit=limits["VELVETFRUIT_EXTRACT_VOUCHER"],
                    strike=strike,
                )
                for symbol, strike in option_symbols_to_trade.items()
            },
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
