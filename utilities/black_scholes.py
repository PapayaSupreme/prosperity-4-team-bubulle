import math
from scipy.stats import norm

def black_scholes(s, k, t, r, sigma) -> float:
    """
    Computes the Black-Scholes price of a European call option.
    :param s: Current spot price of the underlying asset
    :param k: Strike price of the option
    :param t: Time to maturity (in years)
    :param r: Risk-free interest rate (annualized, continuously compounded)
    :param sigma: Volatility of the underlying asset (annualized standard deviation)
    :return: Black-Scholes price rounded to the 4th decimal
    """
    if t <= 0:
        return max(s - k, 0.0)

    sqrt_t = math.sqrt(t)

    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
