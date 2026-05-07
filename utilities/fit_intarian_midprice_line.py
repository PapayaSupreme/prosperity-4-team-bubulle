from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct execution: python scripts/fit_intarian_midprice_line.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.data import read_all_round1_prices

PRODUCT = "INTARIAN_PEPPER_ROOT"
TICKS_PER_DAY = 1_000_000


def build_intarian_midprice_series() -> pd.DataFrame:
    """Load, filter, and sort the round 1 mid-price series for INTARIAN_PEPPER_ROOT."""
    prices = read_all_round1_prices()
    if prices.empty:
        raise ValueError("No round 1 prices found in data/ROUND_1.")

    series = prices.loc[prices["product"] == PRODUCT, ["day", "timestamp", "mid_price"]].copy()
    if series.empty:
        raise ValueError(f"No rows found for product {PRODUCT}.")

    series["mid_price"] = pd.to_numeric(series["mid_price"], errors="coerce")
    series = series.dropna(subset=["mid_price"]).sort_values(["day", "timestamp"]).reset_index(drop=True)
    if series.empty:
        raise ValueError(f"All mid_price values are missing for {PRODUCT}.")

    day_min = int(series["day"].min())
    # Convert day+timestamp into one continuous time axis for a single line fit.
    series["time_index"] = (series["day"] - day_min) * TICKS_PER_DAY + series["timestamp"]
    return series


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, and R^2 for y ~= slope*x + intercept."""
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept

    residual_ss = float(np.sum((y - y_hat) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total_ss == 0 else 1.0 - residual_ss / total_ss
    return float(slope), float(intercept), r_squared


def evaluate_fit(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> dict[str, float]:
    """Return residual-based accuracy metrics for the fitted line."""
    y_hat = slope * x + intercept
    residuals = y - y_hat
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "first_actual": float(y[0]),
        "first_fitted": float(y_hat[0]),
        "last_actual": float(y[-1]),
        "last_fitted": float(y_hat[-1]),
    }


def report_fit(label: str, series: pd.DataFrame) -> None:
    """Fit and print the equation plus how close it is to actual observations."""
    x = series["time_index"].to_numpy(dtype=float)
    y = series["mid_price"].to_numpy(dtype=float)
    slope, intercept, r_squared = linear_fit(x, y)
    metrics = evaluate_fit(x, y, slope, intercept)

    print(f"--- {label} ---")
    print(f"Samples: {len(series)}")
    print(f"Equation: mid_price ~= {slope:.12f} * time_index + {intercept:.6f}")
    print(f"R^2: {r_squared:.6f}")
    print(
        "Errors: "
        f"MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"MAX_ABS={metrics['max_abs_error']:.4f}"
    )
    print(
        "First tick: "
        f"actual={metrics['first_actual']:.4f}, "
        f"fitted={metrics['first_fitted']:.4f}, "
        f"err={metrics['first_actual'] - metrics['first_fitted']:.4f}"
    )
    print(
        "Last tick:  "
        f"actual={metrics['last_actual']:.4f}, "
        f"fitted={metrics['last_fitted']:.4f}, "
        f"err={metrics['last_actual'] - metrics['last_fitted']:.4f}"
    )


def main() -> None:
    series = build_intarian_midprice_series()
    print("INTARIAN_PEPPER_ROOT round 1 mid-price linear fit")
    report_fit("Raw data (includes any zero mid_price rows)", series)

    cleaned = series[series["mid_price"] > 0].reset_index(drop=True)
    removed = len(series) - len(cleaned)
    print(f"\nRemoved {removed} zero mid_price rows for cleaned fit comparison.")
    report_fit("Cleaned data (mid_price > 0)", cleaned)


if __name__ == "__main__":
    main()


