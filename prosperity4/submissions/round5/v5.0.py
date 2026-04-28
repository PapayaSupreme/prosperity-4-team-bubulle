# === UNIFIED ADAPTIVE TRADER (v5) ===
import math

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
# =========================
# All products
# =========================
products = [
"GALAXY_SOUNDS_DARK_MATTER",
"GALAXY_SOUNDS_BLACK_HOLES",
"GALAXY_SOUNDS_PLANETARY_RINGS",
"GALAXY_SOUNDS_SOLAR_WINDS",
"GALAXY_SOUNDS_SOLAR_FLAMES",
"SLEEP_POD_SUEDE",
"SLEEP_POD_LAMB_WOOL",
"SLEEP_POD_POLYESTER",
"SLEEP_POD_NYLON",
"SLEEP_POD_COTTON",
"MICROCHIP_CIRCLE",
"MICROCHIP_OVAL",
"MICROCHIP_SQUARE",
"MICROCHIP_RECTANGLE",
"MICROCHIP_TRIANGLE",
"PEBBLES_XS",
"PEBBLES_S",
"PEBBLES_M",
"PEBBLES_L",
"PEBBLES_XL",
"ROBOT_VACUUMING",
"ROBOT_MOPPING",
"ROBOT_DISHES",
"ROBOT_LAUNDRY",
"ROBOT_IRONING",
"UV_VISOR_YELLOW",
"UV_VISOR_AMBER",
"UV_VISOR_ORANGE",
"UV_VISOR_RED",
"UV_VISOR_MAGENTA",
"TRANSLATOR_SPACE_GRAY",
"TRANSLATOR_ASTRO_BLACK",
"TRANSLATOR_ECLIPSE_CHARCOAL",
"TRANSLATOR_GRAPHITE_MIST",
"TRANSLATOR_VOID_BLUE",
"PANEL_1X2",
"PANEL_2X2",
"PANEL_1X4",
"PANEL_2X4",
"PANEL_4X4",
"OXYGEN_SHAKE_MORNING_BREATH",
"OXYGEN_SHAKE_EVENING_BREATH",
"OXYGEN_SHAKE_MINT",
"OXYGEN_SHAKE_CHOCOLATE",
"OXYGEN_SHAKE_GARLIC",
"SNACKPACK_CHOCOLATE",
"SNACKPACK_VANILLA",
"SNACKPACK_PISTACHIO",
"SNACKPACK_STRAWBERRY",
"SNACKPACK_RASPBERRY",]
# =========================
# Base Strategy Classes
# =========================

class Strategy:
    def __init__(self, symbol: Symbol, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.orders = []

    def run(self, state: TradingState):
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price, qty):
        if qty > 0:
            self.orders.append(Order(self.symbol, price, qty))

    def sell(self, price, qty):
        if qty > 0:
            self.orders.append(Order(self.symbol, price, -qty))


# =========================
# Mean Reversion (YOUR CODE)
# =========================

class AnchoredMarketMaker(Strategy):
    def __init__(self, symbol, limit, anchor):
        super().__init__(symbol, limit)
        self.anchor = anchor
        self.ema = None

    def act(self, state: TradingState):
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders:
            return

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2

        if self.ema is None:
            self.ema = mid
        else:
            self.ema = 0.9 * self.ema + 0.1 * mid

        fair = 0.75 * self.anchor + 0.25 * self.ema
        z = mid - fair

        pos = state.position.get(self.symbol, 0)
        buy_left = self.limit - pos
        sell_left = self.limit + pos

        # Take trades
        if z < -2 and buy_left > 0:
            self.buy(best_ask, min(20, buy_left))
        if z > 2 and sell_left > 0:
            self.sell(best_bid, min(20, sell_left))

        # Passive quotes
        self.buy(int(fair - 2), min(10, buy_left))
        self.sell(int(fair + 2), min(10, sell_left))


# =========================
# Adaptive Strategy
# =========================

class AdaptiveStrategy(Strategy):
    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)
        self.prices = []

    def classify(self):
        if len(self.prices) < 50:
            return "FLAT"

        returns = [self.prices[i] - self.prices[i-1] for i in range(1, len(self.prices))]
        mean = sum(returns) / len(returns)
        var = sum((x - mean) ** 2 for x in returns) / len(returns)
        std = math.sqrt(var)

        if abs(mean) > 0.25 * std:
            return "TREND"
        elif std < 2:
            return "FLAT"
        else:
            return "NOISY"

    def act(self, state: TradingState):
        od = state.order_depths[self.symbol]
        if not od.buy_orders or not od.sell_orders:
            return

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2

        self.prices.append(mid)
        if len(self.prices) > 200:
            self.prices.pop(0)

        regime = self.classify()

        pos = state.position.get(self.symbol, 0)
        buy_left = self.limit - pos
        sell_left = self.limit + pos

        # TREND
        if regime == "TREND":
            if self.prices[-1] > self.prices[-5]:
                self.buy(best_ask, min(15, buy_left))
            else:
                self.sell(best_bid, min(15, sell_left))

        # FLAT (market making)
        elif regime == "FLAT":
            fair = mid
            self.buy(int(fair - 2), min(10, buy_left))
            self.sell(int(fair + 2), min(10, sell_left))

        # NOISY → reduce risk
        else:
            fair = mid
            self.buy(int(fair - 3), min(5, buy_left))
            self.sell(int(fair + 3), min(5, sell_left))


# =========================
# Trader
# =========================

class Trader:
    def __init__(self):
        self.strategies = {}
        self.LIMIT_DEFAULT = 10

    def run(self, state: TradingState):
        result = {}

        for symbol in state.order_depths:
            if symbol not in self.strategies:
                self.strategies[symbol] = AdaptiveStrategy(symbol, self.LIMIT_DEFAULT)

            orders = self.strategies[symbol].run(state)

            if orders:
                result[symbol] = orders

        return result, 0, ""
