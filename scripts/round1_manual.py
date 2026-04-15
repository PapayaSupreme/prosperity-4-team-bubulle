from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class Order:
    side: str   # "buy" or "sell"
    price: int
    volume: int


@dataclass
class ProductConfig:
    name: str
    bids: Dict[int, int]   # price -> volume
    asks: Dict[int, int]   # price -> volume
    liquidation_price: float
    fee_per_traded_unit: float = 0.0


DRYLAND_FLAX = ProductConfig(
    name="DRYLAND_FLAX",
    bids={
        30: 30000,
        29: 5000,
        28: 12000,
        27: 28000,
    },
    asks={
        28: 40000,
        31: 20000,
        32: 20000,
        33: 30000,
    },
    liquidation_price=30.0,
    fee_per_traded_unit=0.0,
)

EMBER_MUSHROOM = ProductConfig(
    name="EMBER_MUSHROOM",
    bids={
        20: 43000,
        19: 17000,
        18: 6000,
        17: 5000,
        16: 10000,
        15: 5000,
        14: 10000,
        13: 7000,
    },
    asks={
        12: 20000,
        13: 25000,
        14: 35000,
        15: 6000,
        16: 5000,
        17: 0,
        18: 10000,
        19: 12000,
    },
    liquidation_price=20.0,
    fee_per_traded_unit=0.10,
)


def merge_book_with_order(
    bids: Dict[int, int],
    asks: Dict[int, int],
    my_order: Optional[Order],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    new_bids = dict(bids)
    new_asks = dict(asks)

    if my_order is None or my_order.volume <= 0:
        return new_bids, new_asks

    if my_order.side == "buy":
        new_bids[my_order.price] = new_bids.get(my_order.price, 0) + my_order.volume
    elif my_order.side == "sell":
        new_asks[my_order.price] = new_asks.get(my_order.price, 0) + my_order.volume
    else:
        raise ValueError("side must be 'buy' or 'sell'")

    return new_bids, new_asks


def cumulative_buy_volume(bids: Dict[int, int], price: int) -> int:
    return sum(vol for p, vol in bids.items() if p >= price)


def cumulative_sell_volume(asks: Dict[int, int], price: int) -> int:
    return sum(vol for p, vol in asks.items() if p <= price)


def auction_candidates(bids: Dict[int, int], asks: Dict[int, int]) -> List[int]:
    return sorted(set(bids.keys()) | set(asks.keys()))


def find_clearing_price(
    bids: Dict[int, int],
    asks: Dict[int, int],
) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    candidates = auction_candidates(bids, asks)
    diagnostics = []

    best_price = None
    best_matched = -1

    for price in candidates:
        cum_bid = cumulative_buy_volume(bids, price)
        cum_ask = cumulative_sell_volume(asks, price)
        matched = min(cum_bid, cum_ask)
        diagnostics.append((price, cum_bid, cum_ask, matched))

        if matched > best_matched:
            best_matched = matched
            best_price = price
        elif matched == best_matched and price > best_price:
            best_price = price

    return best_price, best_matched, diagnostics


def estimate_user_fill(
    original_bids: Dict[int, int],
    original_asks: Dict[int, int],
    my_order: Optional[Order],
    clearing_price: int,
) -> int:
    """
    Exact priority logic under:
    - better price fills before worse price
    - at same price, existing book fills before us because we are last

    Returns signed fill:
    +qty if we bought
    -qty if we sold
    """
    if my_order is None or my_order.volume <= 0:
        return 0

    side = my_order.side
    p = my_order.price
    q = my_order.volume

    if side == "buy":
        # Must be marketable at clearing price
        if p < clearing_price:
            return 0

        total_sell_up_to_clear = sum(vol for price, vol in original_asks.items() if price <= clearing_price)

        # Existing buy demand ahead of us:
        # - all better-priced buys
        # - all existing buys at our price if our price == clearing_price
        if p > clearing_price:
            existing_ahead = sum(vol for price, vol in original_bids.items() if price > p)
        else:
            existing_ahead = (
                sum(vol for price, vol in original_bids.items() if price > clearing_price)
                + original_bids.get(clearing_price, 0)
            )

        remaining_for_us = max(0, total_sell_up_to_clear - existing_ahead)
        fill = min(q, remaining_for_us)
        return fill

    elif side == "sell":
        if p > clearing_price:
            return 0

        total_buy_from_clear = sum(vol for price, vol in original_bids.items() if price >= clearing_price)

        if p < clearing_price:
            existing_ahead = sum(vol for price, vol in original_asks.items() if price < p)
        else:
            existing_ahead = (
                sum(vol for price, vol in original_asks.items() if price < clearing_price)
                + original_asks.get(clearing_price, 0)
            )

        remaining_for_us = max(0, total_buy_from_clear - existing_ahead)
        fill = min(q, remaining_for_us)
        return -fill

    else:
        raise ValueError("side must be 'buy' or 'sell'")


def pnl_to_liquidation(
    config: ProductConfig,
    clearing_price: int,
    signed_fill: int,
) -> float:
    qty = abs(signed_fill)
    gross = signed_fill * (config.liquidation_price - clearing_price)
    fees = qty * config.fee_per_traded_unit
    return gross - fees


def print_book(config: ProductConfig, my_order: Optional[Order] = None) -> None:
    bids, asks = merge_book_with_order(config.bids, config.asks, my_order)

    print(f"\n{'=' * 70}")
    print(config.name)
    print(f"{'=' * 70}")

    print("\nBIDS")
    for price in sorted(bids.keys(), reverse=True):
        tag = ""
        if my_order and my_order.side == "buy" and my_order.price == price:
            tag = "  <- includes your buy order"
        print(f"  Price {price:>3} | Volume {bids[price]:>6}{tag}")

    print("\nASKS")
    for price in sorted(asks.keys()):
        tag = ""
        if my_order and my_order.side == "sell" and my_order.price == price:
            tag = "  <- includes your sell order"
        print(f"  Price {price:>3} | Volume {asks[price]:>6}{tag}")


def run_simulation(config: ProductConfig, my_order: Optional[Order]) -> None:
    print_book(config, my_order)

    agg_bids, agg_asks = merge_book_with_order(config.bids, config.asks, my_order)
    clearing_price, traded_volume, diagnostics = find_clearing_price(agg_bids, agg_asks)

    user_fill = estimate_user_fill(
        original_bids=config.bids,
        original_asks=config.asks,
        my_order=my_order,
        clearing_price=clearing_price,
    )

    pnl = pnl_to_liquidation(config, clearing_price, user_fill)

    print("\nCandidate prices:")
    print("  price | cum_bids>=p | cum_asks<=p | matched")
    for price, cum_bid, cum_ask, matched in diagnostics:
        print(f"  {price:>5} | {cum_bid:>11} | {cum_ask:>11} | {matched:>7}")

    print("\nAuction result:")
    print(f"  Clearing price : {clearing_price}")
    print(f"  Matched volume : {traded_volume}")

    if my_order is None:
        print("  Your order      : none")
        return

    print(f"  Your order      : {my_order.side} {my_order.volume} @ {my_order.price}")
    print(f"  Your fill       : {user_fill}")
    print(f"  Est. PnL        : {pnl:.2f}")
    print(f"  Liquidation px  : {config.liquidation_price}")
    print(f"  Fee / traded u. : {config.fee_per_traded_unit}")


def parse_order_input(product_name: str) -> Optional[Order]:
    raw = input(
        f"\nEnter your order for {product_name} "
        f"(format: 'buy price volume', 'sell price volume', or 'none'): "
    ).strip().lower()

    if raw in ("", "none"):
        return None

    parts = raw.split()
    if len(parts) != 3:
        raise ValueError("Expected: side price volume")

    side = parts[0]
    price = int(parts[1])
    volume = int(parts[2])

    if side not in ("buy", "sell"):
        raise ValueError("Side must be 'buy' or 'sell'")
    if volume <= 0:
        raise ValueError("Volume must be positive")

    return Order(side=side, price=price, volume=volume)


def main() -> None:
    print("Manual Auction Simulator")
    print("Rule: maximize matched volume, then choose higher clearing price.")
    print("Allocation: price priority, then time priority.")
    print("You are last in line at any price level you join.\n")

    dryland_order = parse_order_input(DRYLAND_FLAX.name)
    ember_order = parse_order_input(EMBER_MUSHROOM.name)

    print("\nRESULTS")
    run_simulation(DRYLAND_FLAX, dryland_order)
    run_simulation(EMBER_MUSHROOM, ember_order)


if __name__ == "__main__":
    main()
