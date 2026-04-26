from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from prosperity4.analysis.data import read_all_round4_prices
except ModuleNotFoundError:
    # Allow running this file directly from the repository root.
    from data import read_all_round4_prices


OUTPUT_DIR = Path(__file__).resolve().parent / "round4_outputs"
UNDERLYING_PRODUCT = "VELVETFRUIT_EXTRACT"
OPTION_PREFIX = "VEV_"


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, ttm_years: float, sigma: float, rate: float = 0.0) -> float:
    if ttm_years <= 0.0:
        return max(spot - strike, 0.0)

    sigma = max(sigma, 1e-12)
    sqrt_t = math.sqrt(ttm_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * ttm_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return spot * _norm_cdf(d1) - strike * math.exp(-rate * ttm_years) * _norm_cdf(d2)


def implied_vol_call(
    spot: float,
    strike: float,
    ttm_years: float,
    option_price: float,
    rate: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float | None:
    if spot <= 0.0 or strike <= 0.0 or ttm_years <= 0.0 or option_price <= 0.0:
        return None

    intrinsic = max(spot - strike * math.exp(-rate * ttm_years), 0.0)
    upper_bound = spot
    if option_price < intrinsic - 1e-9 or option_price > upper_bound + 1e-9:
        return None
    if abs(option_price - intrinsic) <= 1e-9:
        return 1e-6

    lo, hi = 1e-6, 5.0
    lo_price = bs_call_price(spot, strike, ttm_years, lo, rate)
    hi_price = bs_call_price(spot, strike, ttm_years, hi, rate)

    if option_price < lo_price - 1e-9:
        return 1e-6
    if option_price > hi_price + 1e-9:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_price = bs_call_price(spot, strike, ttm_years, mid, rate)
        if abs(mid_price - option_price) <= tol:
            return mid
        if mid_price < option_price:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def _best_mid(df: pd.DataFrame) -> pd.Series:
    bid = df["bid_price_1"]
    ask = df["ask_price_1"]
    top_mid = (bid + ask) / 2.0
    # Fallback to feed mid when top-of-book is incomplete.
    return top_mid.where(~(bid.isna() | ask.isna()), df["mid_price"])


def build_iv_dataset(sample_every: int) -> pd.DataFrame:
    raw = read_all_round4_prices()
    if raw.empty:
        return raw

    sample_every = max(sample_every, 1)

    timestamps_sorted = np.sort(raw["timestamp"].unique())
    sampled_timestamps = set(timestamps_sorted[::sample_every].tolist())
    sampled = raw[raw["timestamp"].isin(sampled_timestamps)].copy()

    sampled["mid"] = _best_mid(sampled)

    underlying = sampled[sampled["product"] == UNDERLYING_PRODUCT][["day", "timestamp", "mid"]].copy()
    underlying = underlying.rename(columns={"mid": "spot"})

    options = sampled[sampled["product"].str.startswith(OPTION_PREFIX)][["day", "timestamp", "product", "mid"]].copy()
    options = options.rename(columns={"mid": "option_mid"})
    options["strike"] = options["product"].str.extract(r"VEV_(\d+)").astype(float)

    merged = options.merge(underlying, on=["day", "timestamp"], how="inner")

    # VEV vouchers have fixed time to expiry assumption: 5 trading days.
    merged["ttm_years"] = 5.0 / 365.0
    merged["moneyness"] = np.log(merged["strike"] / merged["spot"]) # / np.sqrt(merged["spot"])

    merged["implied_vol"] = merged.apply(
        lambda row: implied_vol_call(
            spot=float(row["spot"]),
            strike=float(row["strike"]),
            ttm_years=float(row["ttm_years"]),
            option_price=float(row["option_mid"]),
            rate=0.0,
        ),
        axis=1,
    )

    merged = merged.dropna(subset=["implied_vol", "moneyness"])
    merged = merged[merged["implied_vol"] > 0.0]

    return merged.sort_values(["day", "timestamp", "strike"]).reset_index(drop=True)


def fit_smile(iv_df: pd.DataFrame) -> np.ndarray:
    return np.polyfit(iv_df["moneyness"], iv_df["implied_vol"], deg=2)


def save_outputs(iv_df: pd.DataFrame, coeffs: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    points_path = OUTPUT_DIR / "vev_iv_points.csv"
    coeffs_path = OUTPUT_DIR / "vev_smile_coeffs.json"
    plot_path = OUTPUT_DIR / "vev_iv_vs_moneyness_smile.html"

    iv_df.to_csv(points_path, index=False)

    payload = {
        "model": "iv = a*m^2 + b*m + c",
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
        "n_points": int(len(iv_df)),
    }
    coeffs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    x = iv_df["moneyness"].to_numpy()
    y = iv_df["implied_vol"].to_numpy()
    x_grid = np.linspace(float(x.min()), float(x.max()), 300)
    y_fit = coeffs[0] * x_grid * x_grid + coeffs[1] * x_grid + coeffs[2]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observed IV",
            marker={
                "size": 5,
                "opacity": 0.5,
                "color": iv_df["strike"],
                "colorscale": "Viridis",
                "colorbar": {"title": "Strike"},
            },
            customdata=np.stack([iv_df["strike"], iv_df["product"]], axis=-1),
            hovertemplate=(
                "Moneyness=%{x:.4f}<br>IV=%{y:.4f}<br>"
                "Strike=%{customdata[0]:.0f}<br>Option=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_fit,
            mode="lines",
            name="Quadratic smile fit",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="VEV Options: Implied Volatility vs Moneyness (Round 4)",
        xaxis_title="Moneyness = ln(K / S)",
        yaxis_title="Implied Volatility (annualized)",
        template="plotly_white",
    )
    fig.write_html(plot_path, include_plotlyjs="cdn")

    print(f"Saved IV points to: {points_path}")
    print(f"Saved smile coefficients to: {coeffs_path}")
    print(f"Saved plot to: {plot_path}")
    print(
        "Smile fit: "
        f"iv = {coeffs[0]:.6f} * m^2 + {coeffs[1]:.6f} * m + {coeffs[2]:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VEV IV vs moneyness and fit a quadratic smile for Round 4.")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=20,
        help="Use every Nth timestamp for speed (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iv_df = build_iv_dataset(sample_every=args.sample_every)
    if iv_df.empty:
        raise RuntimeError("No valid VEV option rows found to compute implied vols.")

    coeffs = fit_smile(iv_df)
    save_outputs(iv_df, coeffs)


if __name__ == "__main__":
    main()
