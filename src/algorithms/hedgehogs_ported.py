import json

try:
	# Competition website import path.
	from datamodel import OrderDepth, Order, Symbol, TradingState
except ImportError:
	# Local development fallback.
	from src.algorithms.datamodel import OrderDepth, Order, Symbol, TradingState

STATIC_SYMBOL = "EMERALDS"
DYNAMIC_SYMBOL = "TOMATOES"

POS_LIMITS = {STATIC_SYMBOL: 80, DYNAMIC_SYMBOL: 80}

LONG, NEUTRAL, SHORT = 1, 0, -1


class ProductTrader:
	def __init__(self, name, state, prints, new_trader_data, product_group=None):
		self.orders = []

		self.name = name
		self.state = state
		self.prints = prints
		self.new_trader_data = new_trader_data
		self.product_group = name if product_group is None else product_group

		self.last_traderData = self.get_last_traderData()

		self.position_limit = POS_LIMITS.get(self.name, 0)
		self.initial_position = self.state.position.get(self.name, 0)
		self.expected_position = self.initial_position

		self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
		self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
		self.best_bid, self.best_ask = self.get_best_bid_ask()

		self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
		self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

	def get_last_traderData(self):
		last_traderData = {}
		try:
			if self.state.traderData != "":
				last_traderData = json.loads(self.state.traderData)
		except Exception:
			self.log("ERROR", "td")

		return last_traderData

	def get_best_bid_ask(self):
		best_bid = best_ask = None

		try:
			if len(self.mkt_buy_orders) > 0:
				best_bid = max(self.mkt_buy_orders.keys())
			if len(self.mkt_sell_orders) > 0:
				best_ask = min(self.mkt_sell_orders.keys())
		except Exception:
			pass

		return best_bid, best_ask

	def get_walls(self):
		bid_wall = wall_mid = ask_wall = None

		try:
			bid_wall = min([x for x, _ in self.mkt_buy_orders.items()])
		except Exception:
			pass

		try:
			ask_wall = max([x for x, _ in self.mkt_sell_orders.items()])
		except Exception:
			pass

		try:
			wall_mid = (bid_wall + ask_wall) / 2
		except Exception:
			pass

		return bid_wall, wall_mid, ask_wall

	def get_total_market_buy_sell_volume(self):
		market_bid_volume = market_ask_volume = 0

		try:
			market_bid_volume = sum([v for _, v in self.mkt_buy_orders.items()])
			market_ask_volume = sum([v for _, v in self.mkt_sell_orders.items()])
		except Exception:
			pass

		return market_bid_volume, market_ask_volume

	def get_max_allowed_volume(self):
		max_allowed_buy_volume = self.position_limit - self.initial_position
		max_allowed_sell_volume = self.position_limit + self.initial_position
		return max_allowed_buy_volume, max_allowed_sell_volume

	def get_order_depth(self):
		order_depth, buy_orders, sell_orders = {}, {}, {}

		try:
			order_depth = self.state.order_depths[self.name]
		except Exception:
			pass
		try:
			buy_orders = {
				bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
			}
		except Exception:
			pass
		try:
			sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
		except Exception:
			pass

		return buy_orders, sell_orders

	def bid(self, price, volume, logging=True):
		abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
		order = Order(self.name, int(price), abs_volume)
		if logging:
			self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group="ORDERS")
		self.max_allowed_buy_volume -= abs_volume
		self.orders.append(order)

	def ask(self, price, volume, logging=True):
		abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
		order = Order(self.name, int(price), -abs_volume)
		if logging:
			self.log("SELLO", {"p": price, "s": self.name, "v": int(volume)}, product_group="ORDERS")
		self.max_allowed_sell_volume -= abs_volume
		self.orders.append(order)

	def log(self, kind, message, product_group=None):
		if product_group is None:
			product_group = self.product_group

		if product_group == "ORDERS":
			group = self.prints.get(product_group, [])
			group.append({kind: message})
		else:
			group = self.prints.get(product_group, {})
			group[kind] = message

		self.prints[product_group] = group

	def check_for_informed(self):
		informed_direction, informed_bought_ts, informed_sold_ts = NEUTRAL, None, None

		informed_bought_ts, informed_sold_ts = self.last_traderData.get(self.name, [None, None])

		trades = self.state.market_trades.get(self.name, []) + self.state.own_trades.get(self.name, [])

		"""for trade in trades:
			if trade.buyer == INFORMED_TRADER_ID:
				informed_bought_ts = trade.timestamp
			if trade.seller == INFORMED_TRADER_ID:
				informed_sold_ts = trade.timestamp"""

		self.new_trader_data[self.name] = [informed_bought_ts, informed_sold_ts]

		informed_sold = informed_sold_ts is not None
		informed_bought = informed_bought_ts is not None

		if not informed_bought and not informed_sold:
			informed_direction = NEUTRAL
		elif not informed_bought and informed_sold:
			informed_direction = SHORT
		elif informed_bought and not informed_sold:
			informed_direction = LONG
		elif informed_bought and informed_sold:
			if informed_sold_ts > informed_bought_ts:
				informed_direction = SHORT
			elif informed_sold_ts < informed_bought_ts:
				informed_direction = LONG
			else:
				informed_direction = NEUTRAL

		self.log("TD", self.new_trader_data[self.name])
		self.log("ID", informed_direction)

		return informed_direction, informed_bought_ts, informed_sold_ts

	def get_orders(self):
		return {}


class StaticTrader(ProductTrader):
	def __init__(self, state, prints, new_trader_data):
		super().__init__(STATIC_SYMBOL, state, prints, new_trader_data)

	def get_orders(self):
		if self.wall_mid is not None:
			for sp, sv in self.mkt_sell_orders.items():
				if sp <= self.wall_mid - 1:
					self.bid(sp, sv, logging=False)
				elif sp <= self.wall_mid and self.initial_position < 0:
					volume = min(sv, abs(self.initial_position))
					self.bid(sp, volume, logging=False)

			for bp, bv in self.mkt_buy_orders.items():
				if bp >= self.wall_mid + 1:
					self.ask(bp, bv, logging=False)
				elif bp >= self.wall_mid and self.initial_position > 0:
					volume = min(bv, self.initial_position)
					self.ask(bp, volume, logging=False)

			bid_price = int(self.bid_wall + 1)
			ask_price = int(self.ask_wall - 1)

			for bp, bv in self.mkt_buy_orders.items():
				overbidding_price = bp + 1
				if bv > 1 and overbidding_price < self.wall_mid:
					bid_price = max(bid_price, overbidding_price)
					break
				elif bp < self.wall_mid:
					bid_price = max(bid_price, bp)
					break

			for sp, sv in self.mkt_sell_orders.items():
				underbidding_price = sp - 1
				if sv > 1 and underbidding_price > self.wall_mid:
					ask_price = min(ask_price, underbidding_price)
					break
				elif sp > self.wall_mid:
					ask_price = min(ask_price, sp)
					break

			self.bid(bid_price, self.max_allowed_buy_volume)
			self.ask(ask_price, self.max_allowed_sell_volume)

		return {self.name: self.orders}


class DynamicTrader(ProductTrader):
	def __init__(self, state, prints, new_trader_data):
		super().__init__(DYNAMIC_SYMBOL, state, prints, new_trader_data)

		self.informed_direction, self.informed_bought_ts, self.informed_sold_ts = self.check_for_informed()

	def get_orders(self):
		if self.wall_mid is not None:
			bid_price = self.bid_wall + 1
			bid_volume = self.max_allowed_buy_volume

			if self.informed_bought_ts is not None and self.informed_bought_ts + 5_00 >= self.state.timestamp:
				if self.initial_position < 40:
					bid_price = self.ask_wall
					bid_volume = 40 - self.initial_position

			else:
				if self.wall_mid - bid_price < 1 and (self.informed_direction == SHORT and self.initial_position > -40):
					bid_price = self.bid_wall

			self.bid(bid_price, bid_volume)

			ask_price = self.ask_wall - 1
			ask_volume = self.max_allowed_sell_volume

			if self.informed_sold_ts is not None and self.informed_sold_ts + 5_00 >= self.state.timestamp:
				if self.initial_position > -40:
					ask_price = self.bid_wall
					ask_volume = 40 + self.initial_position

			if ask_price - self.wall_mid < 1 and (self.informed_direction == LONG and self.initial_position < 40):
				ask_price = self.ask_wall

			self.ask(ask_price, ask_volume)

		return {self.name: self.orders}


class Trader:
	def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
		prints = {}
		new_trader_data = {}

		orders: dict[Symbol, list[Order]] = {}

		static_orders = StaticTrader(state, prints, new_trader_data).get_orders()
		dynamic_orders = DynamicTrader(state, prints, new_trader_data).get_orders()

		orders.update(static_orders)
		orders.update(dynamic_orders)

		conversions = 0
		trader_data = json.dumps(new_trader_data, separators=(",", ":"))
		return orders, conversions, trader_data

	def trade(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
		return self.run(state)


