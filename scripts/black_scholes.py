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
        return max(s - k, 0)
    d1 = (math.log(s / k) + (r + (sigma * 2 / 2))) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    result = s * n_d1 - k * math.exp(-0. - r * t) * n_d2
    return round(result, 4)