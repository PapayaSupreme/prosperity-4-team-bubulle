# v1.0 - port of emerald logic but for Intarian pepper, with midprice being 12 000 + 0.001 * timestep
# 7 286
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


class IntarianMarketMaker(StatefulStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.fair_value: float = 12000.0
        self.soft_pos = 40
        self.hard_pos = 70

    def _update_fair_value(self) -> None:
        self.fair_value += 100

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        position, buy_left, sell_left = self._get_position_capacities(state)

        # Update non-resetting fair value
        self._update_fair_value()

        fair_int = int(self.fair_value)

        # Hard thresholds around fair
        take_buy_price = fair_int - 1
        take_sell_price = fair_int + 1

        # ============================================================
        # 1) TAKE OBVIOUS MISPRICINGS AROUND KNOWN FAIR
        # ============================================================

        buy_left, position = self._take_sell_levels(
            sell_orders, buy_left, position, take_buy_price
        )
        sell_left, position = self._take_buy_levels(
            buy_orders, sell_left, position, take_sell_price
        )

        # Recompute remaining capacity after aggressive fills
        buy_left = self.limit - position
        sell_left = self.limit + position

        if buy_left <= 0 and sell_left <= 0:
            return

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        # ============================================================
        # 2) PASSIVE QUOTING
        # ============================================================

        bid_quote = min(best_bid + 1, fair_int - 1)
        ask_quote = max(best_ask - 1, fair_int + 1)

        # Never cross
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        if abs(position) < self.soft_pos:
            bid_size = min(buy_left, 32)
            ask_size = min(sell_left, 32)

        elif abs(position) < self.hard_pos:
            bid_size = min(buy_left, 20)
            ask_size = min(sell_left, 20)

            if position > 0:
                # long -> bid less aggressively, ask more aggressively
                bid_quote = min(best_bid, fair_int - 1)
                ask_quote = max(best_bid + 1, fair_int + 1)
            elif position < 0:
                # short -> bid more aggressively, ask less aggressively
                bid_quote = min(best_ask - 1, fair_int - 1)
                ask_quote = max(best_ask, fair_int + 1)

        else:
            bid_size = min(buy_left, 12)
            ask_size = min(sell_left, 12)

            if position > 0:
                bid_size = 0
                ask_quote = max(best_bid + 1, fair_int)
                ask_size = min(sell_left, 40)

            elif position < 0:
                ask_size = 0
                bid_quote = min(best_ask - 1, fair_int)
                bid_size = min(buy_left, 40)

        # ============================================================
        # 3) OPTIONAL INVENTORY FLATTENING AT FAIR
        # ============================================================

        if position > 0 and sell_left > 0:
            for bid_price, bid_volume in buy_orders:
                if bid_price == fair_int:
                    size = min(position, sell_left, bid_volume)
                    if size > 0:
                        self.sell(bid_price, size)
                        sell_left -= size
                        position -= size
                    break

        elif position < 0 < buy_left:
            for ask_price, ask_volume in sell_orders:
                if ask_price == fair_int:
                    size = min(-position, buy_left, -ask_volume)
                    if size > 0:
                        self.buy(ask_price, size)
                        buy_left -= size
                        position += size
                    break

        # Recompute after flattening
        buy_left = self.limit - position
        sell_left = self.limit + position

        # ============================================================
        # 4) POST PASSIVE ORDERS
        # ============================================================

        self._post_passive_orders(
            buy_left, sell_left, bid_quote, ask_quote, bid_size, ask_size
        )

    def save(self) -> JSON:
        return {
            "fair_value": self.fair_value,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        fair_value = data.get("fair_value")
        if isinstance(fair_value, (int, float)):
            self.fair_value = float(fair_value)


class TomatoesAdaptiveMarketMaker(StatefulStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.fair_value: float | None = None
        self.prev_microprice: float | None = None
        self.prev_mid_prices = []
        self.microprice_alpha = 0.3
        self.ema_microprice: float | None = None
        self.ema_alpha = 0.6

    def _estimate_fair_value(self, microprice: float) -> float:
        if self.ema_microprice is None:
            self.ema_microprice = microprice
        else:
            self.ema_microprice = (
                self.ema_alpha * microprice
                + (1 - self.ema_alpha) * self.ema_microprice
            )
        # Mean-revert short-term ema microprice moves instead of raw mid moves.
        return self.ema_microprice

    def act(self, state: TradingState) -> None:
        buy_orders, sell_orders = self._get_sorted_orders(state)
        position, buy_left, sell_left = self._get_position_capacities(state)

        # Calculate current mid-price from best bid/ask
        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        current_mid = (best_bid + best_ask) / 2

        self.prev_mid_prices.append(current_mid)
        if len(self.prev_mid_prices) > 10:
            self.prev_mid_prices.pop(0)

        # Simple trend: last 4 mids increasing or decreasing
        trend_bias = 0
        if len(self.prev_mid_prices) >= 4:
            last_4 = self.prev_mid_prices[-4:]
            diffs = [last_4[i + 1] - last_4[i] for i in range(3)]

            pos_count = sum(d > 0 for d in diffs)
            neg_count = sum(d < 0 for d in diffs)

            if pos_count >= 2 or neg_count >= 2:
                trend_bias = sum(diffs) / 3

        # Microprice weights prices by opposite-side top-of-book volume.
        best_bid_vol = max(0, buy_orders[0][1])
        best_ask_vol = max(0, -sell_orders[0][1])
        top_vol = best_bid_vol + best_ask_vol
        microprice = (
            (best_ask * best_bid_vol + best_bid * best_ask_vol) / top_vol
            if top_vol > 0
            else current_mid
        )

        self.fair_value = self._estimate_fair_value(microprice)

        fair_value = self.fair_value if self.fair_value is not None else current_mid

        # ============================================================
        # 1) TAKE OBVIOUS MISPRICINGS AROUND ESTIMATED FAIR VALUE
        # ============================================================

        # Take thresholds around current fair value, taking into account trend bias.
        take_buy_price = fair_value - 1 # THE HIGHER, THE EASIER. THIS IS THE PRICE I WANT TO BUY MY STOCK FOR
        take_sell_price = fair_value + 1 # THE LOWER, THE EASIER. THIS IS THE PRICE I WANT TO SELL MY STOCK FOR
        if trend_bias > 0: # market is up, expect a reversion, harder to buy, bid price--
            take_buy_price -= 1
            take_sell_price -= 1
        elif trend_bias < 0: # market is down, expect a reversion, harder to sell, ask price++
            take_buy_price += 1
            take_sell_price += 1

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

        fair_int = int(fair_value - trend_bias)
        bid_quote = min(best_bid + 1, fair_int - 1)
        ask_quote = max(best_ask - 1, fair_int + 1)

        # Safety: never cross.
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)

        # Base quote size
        base_bid_size = buy_left
        base_ask_size = sell_left

        if trend_bias > 0:
            bid_size = max(base_bid_size // 2, 6)  # buy less
            ask_size = base_ask_size  # sell more
        elif trend_bias < 0:
            bid_size = base_bid_size  # buy more
            ask_size = max(base_ask_size // 2, 6)  # sell less
        else:
            bid_size = base_bid_size
            ask_size = base_ask_size

        # ============================================================
        # 3) POST PASSIVE ORDERS
        # ============================================================

        self._post_passive_orders(buy_left, sell_left, bid_quote, ask_quote, bid_size, ask_size)

    def save(self) -> JSON:
        return {
            "fair_value": self.fair_value,
            "ema_microprice": self.ema_microprice,
            "prev_mid_prices": self.prev_mid_prices,
        }

    def load(self, data: JSON) -> None:
        if not isinstance(data, dict):
            return

        fair_value = data.get("fair_value")
        if isinstance(fair_value, (int, float)):
            self.fair_value = float(fair_value)

        ema_microprice = data.get("ema_microprice")
        if isinstance(ema_microprice, (int, float)):
            self.ema_microprice = float(ema_microprice)
        else:
            # backward compatibility with older saved state
            prev_microprice = data.get("prev_microprice")
            if isinstance(prev_microprice, (int, float)):
                self.ema_microprice = float(prev_microprice)

        mid_history = data.get("prev_mid_prices")
        if isinstance(mid_history, list):
            self.prev_mid_prices = [float(x) for x in mid_history]



class Trader:
    """
    Website entrypoint: run(state) -> (orders, conversions, trader_data)
    Local compatibility alias: trade(state)
    """

    def __init__(self) -> None:
        limits = {
            "INTARIAN_PEPPER_ROOT": 80,
            "ASH_COATED_OSMIUM": 80,
        }

        self.strategies: dict[Symbol, Strategy] = {
            "INTARIAN_PEPPER_ROOT": IntarianMarketMaker("INTARIAN_PEPPER_ROOT", limits["INTARIAN_PEPPER_ROOT"]),
            #"ASH_COATED_OSMIUM": TomatoesAdaptiveMarketMaker("ASH_COATED_OSMIUM", limits["ASH_COATED_OSMIUM"]),
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
