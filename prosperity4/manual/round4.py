import numpy as np
from math import sqrt

# =========================
# Challenge constants
# =========================

S0 = 50.0
SIGMA = 2.51  # 251% annualized vol
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY

CONTRACT_SIZE = 3000

N_SIMS = 100_000
SEED = 42


# =========================
# Time helpers
# =========================

def weeks_to_steps(weeks: float) -> int:
    return int(round(weeks * 5 * STEPS_PER_DAY))


def weeks_to_years(weeks: float) -> float:
    return (weeks * 5) / TRADING_DAYS_PER_YEAR


# =========================
# GBM simulation
# =========================

def simulate_gbm_paths(
    s0: float,
    sigma: float,
    n_steps: int,
    n_sims: int,
    seed: int = 42
) -> np.ndarray:
    """
    Simulates discrete GBM paths with zero drift.

    Returns:
        paths shape = (n_sims, n_steps + 1)
        paths[:, 0] = initial price
    """
    rng = np.random.default_rng(seed)

    dt = 1 / STEPS_PER_YEAR

    # Risk-neutral GBM:
    # S_{t+dt} = S_t * exp((-0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    z = rng.standard_normal((n_sims, n_steps))

    log_returns = (-0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z

    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))

    return paths


# =========================
# Payoff functions
# =========================

def call_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0)


def put_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - ST, 0)


def binary_put_payoff(ST: np.ndarray, K: float, payout: float) -> np.ndarray:
    """
    Pays fixed payout if ST < K.
    """
    return np.where(ST < K, payout, 0.0)


def knockout_put_payoff(paths: np.ndarray, K: float, barrier: float) -> np.ndarray:
    """
    Down-and-out put.

    If price ever goes below barrier at a discrete step,
    the option becomes worthless.

    Otherwise payoff = max(K - ST, 0).
    """
    ST = paths[:, -1]

    knocked_out = np.any(paths < barrier, axis=1)

    payoff = np.maximum(K - ST, 0)
    payoff[knocked_out] = 0.0

    return payoff


def chooser_payoff(paths: np.ndarray, K: float, choose_step: int) -> np.ndarray:
    """
    Chooser option:
    At choose_step, buyer chooses call or put depending on which is better.

    Since the choice is based on the underlying at T+14:
    - if S_choose >= K, choose call
    - if S_choose < K, choose put

    Then final payoff is based on final expiry ST.
    """
    S_choose = paths[:, choose_step]
    ST = paths[:, -1]

    call = np.maximum(ST - K, 0)
    put = np.maximum(K - ST, 0)

    return np.where(S_choose >= K, call, put)


# =========================
# Pricing utility
# =========================

def fair_value(payoffs: np.ndarray) -> float:
    return float(np.mean(payoffs))


def summarize_product(name: str, fair: float, bid: float, ask: float, max_volume: int):
    buy_edge = fair - ask
    sell_edge = bid - fair

    if buy_edge > sell_edge and buy_edge > 0:
        action = "BUY"
        edge = buy_edge
    elif sell_edge > 0:
        action = "SELL"
        edge = sell_edge
    else:
        action = "NO TRADE"
        edge = 0.0

    expected_pnl = edge * max_volume * CONTRACT_SIZE

    return {
        "product": name,
        "fair": fair,
        "bid": bid,
        "ask": ask,
        "action": action,
        "volume": max_volume if action != "NO TRADE" else 0,
        "edge_per_unit": edge,
        "expected_pnl": expected_pnl,
    }


# =========================
# Main analysis
# =========================

def run_analysis():
    steps_2w = weeks_to_steps(2)
    steps_3w = weeks_to_steps(3)

    paths_2w = simulate_gbm_paths(S0, SIGMA, steps_2w, N_SIMS, SEED)
    paths_3w = simulate_gbm_paths(S0, SIGMA, steps_3w, N_SIMS, SEED)

    ST_2w = paths_2w[:, -1]
    ST_3w = paths_3w[:, -1]

    products = []

    # -------------------------
    # Underlying
    # -------------------------
    products.append(
        summarize_product(
            name="AC",
            fair=S0,
            bid=49.975,
            ask=50.025,
            max_volume=200,
        )
    )

    # -------------------------
    # 3-week vanilla options
    # -------------------------
    products.append(
        summarize_product(
            "AC_50_P",
            fair_value(put_payoff(ST_3w, 50)),
            bid=12.00,
            ask=12.05,
            max_volume=50,
        )
    )

    products.append(
        summarize_product(
            "AC_50_C",
            fair_value(call_payoff(ST_3w, 50)),
            bid=12.00,
            ask=12.05,
            max_volume=50,
        )
    )

    products.append(
        summarize_product(
            "AC_35_P",
            fair_value(put_payoff(ST_3w, 35)),
            bid=4.33,
            ask=4.35,
            max_volume=50,
        )
    )

    products.append(
        summarize_product(
            "AC_40_P",
            fair_value(put_payoff(ST_3w, 40)),
            bid=6.50,
            ask=6.55,
            max_volume=50,
        )
    )

    products.append(
        summarize_product(
            "AC_45_P",
            fair_value(put_payoff(ST_3w, 45)),
            bid=9.05,
            ask=9.10,
            max_volume=50,
        )
    )

    products.append(
        summarize_product(
            "AC_60_C",
            fair_value(call_payoff(ST_3w, 60)),
            bid=8.80,
            ask=8.85,
            max_volume=50,
        )
    )

    # -------------------------
    # 2-week vanilla options
    # -------------------------
    products.append(
        summarize_product(
            "AC_50_P_2",
            fair_value(put_payoff(ST_2w, 50)),
            bid=9.70,
            ask=9.75,
            max_volume=50,
        )
    )

    products.append(
        summarize_product(
            "AC_50_C_2",
            fair_value(call_payoff(ST_2w, 50)),
            bid=9.70,
            ask=9.75,
            max_volume=50,
        )
    )

    # -------------------------
    # Chooser option
    # -------------------------
    products.append(
        summarize_product(
            "AC_50_CO",
            fair_value(chooser_payoff(paths_3w, K=50, choose_step=steps_2w)),
            bid=22.20,
            ask=22.30,
            max_volume=50,
        )
    )

    # -------------------------
    # Binary put
    # -------------------------
    BINARY_PAYOUT = 10.0

    products.append(
        summarize_product(
            "AC_40_BP",
            fair_value(binary_put_payoff(ST_3w, K=40, payout=BINARY_PAYOUT)),
            bid=5.00,
            ask=5.10,
            max_volume=50,
        )
    )

    # -------------------------
    # Knock-out put
    # Assumption:
    # strike = 45, barrier = 45
    # If the actual barrier differs, change barrier.
    # -------------------------
    products.append(
        summarize_product(
            "AC_45_KO",
            fair_value(knockout_put_payoff(paths_3w, K=45, barrier=35)),
            bid=0.15,
            ask=0.175,
            max_volume=500,
        )
    )

    products = sorted(products, key=lambda x: abs(x["expected_pnl"]), reverse=True)

    for p in products:
        print(
            f"{p['product']:10s} | "
            f"fair={p['fair']:8.4f} | "
            f"bid={p['bid']:7.3f} | "
            f"ask={p['ask']:7.3f} | "
            f"{p['action']:8s} | "
            f"vol={p['volume']:4d} | "
            f"edge={p['edge_per_unit']:8.4f} | "
            f"EV PnL={p['expected_pnl']:10.2f}"
        )


if __name__ == "__main__":
    run_analysis()
