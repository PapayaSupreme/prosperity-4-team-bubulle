import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================
PRICES_CSV = "../data/tutorial_round/prices_round_0_day_-2.csv"   # replace with your file name
PRODUCT = "TOMATOES"


# ============================================================
# LOAD AND PREPARE PRICE DATA
# ============================================================
# Your file is semicolon-separated.
df = pd.read_csv(PRICES_CSV, sep=";")

# Keep only the chosen product and sort chronologically.
df = df[df["product"] == PRODUCT].copy()
df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)

# Mid-price series.
df["mid_price"] = df["mid_price"].astype(float)

# 1-step price change:
# delta_t = mid_t - mid_{t-1}
df["delta"] = df["mid_price"].diff()

# Move sign:
#  1 = up
# -1 = down
#  0 = flat
df["sign"] = np.sign(df["delta"]).astype("Int64")

# Previous move sign and next move sign.
df["prev_sign"] = df["sign"].shift(1)
df["next_sign"] = df["sign"].shift(-1)

# Keep rows where current sign exists.
valid = df.dropna(subset=["delta", "sign"]).copy()


# ============================================================
# 1) CONDITIONAL PROBABILITIES OF SAME-SIGN MOVE
# ============================================================
# We want:
# P(down_t | down_{t-1})
# P(up_t   | up_{t-1})
#
# Using prev_sign = sign at t-1, sign = sign at t.

down_after_down_base = valid[valid["prev_sign"] == -1]
up_after_up_base = valid[valid["prev_sign"] == 1]

p_down_after_down = (
    (down_after_down_base["sign"] == -1).mean()
    if len(down_after_down_base) > 0 else np.nan
)

p_up_after_up = (
    (up_after_up_base["sign"] == 1).mean()
    if len(up_after_up_base) > 0 else np.nan
)

print("=== 1-step sign persistence ===")
print(f"P(down_t | down_(t-1)) = {p_down_after_down:.4f}   ({p_down_after_down*100:.2f}%)")
print(f"P(up_t   | up_(t-1))   = {p_up_after_up:.4f}   ({p_up_after_up*100:.2f}%)")
print()


# ============================================================
# 2) 2x2 SIGN TRANSITION TABLE
# ============================================================
# Transition from previous sign to current sign, ignoring flat moves.

transitions = valid[(valid["prev_sign"].isin([-1, 1])) & (valid["sign"].isin([-1, 1]))].copy()

transition_counts = pd.crosstab(
    transitions["prev_sign"],
    transitions["sign"],
    rownames=["previous move"],
    colnames=["current move"],
    dropna=False
)

transition_probs = pd.crosstab(
    transitions["prev_sign"],
    transitions["sign"],
    rownames=["previous move"],
    colnames=["current move"],
    normalize="index",
    dropna=False
)

# Rename for readability.
rename_map = {-1: "down", 1: "up"}
transition_counts = transition_counts.rename(index=rename_map, columns=rename_map)
transition_probs = transition_probs.rename(index=rename_map, columns=rename_map)

print("=== Sign transition counts ===")
print(transition_counts)
print()
print("=== Sign transition probabilities ===")
print(transition_probs)
print()


# ============================================================
# 3) LAG-1 AUTOCORRELATION OF PRICE CHANGES
# ============================================================
# Positive => momentum
# Negative => mean reversion
# Near zero => little serial dependence

returns = valid["delta"].dropna()
autocorr_lag1 = returns.autocorr(lag=1)

print("=== Lag-1 autocorrelation of delta ===")
print(f"autocorr(delta, lag=1) = {autocorr_lag1:.6f}")
print()


# ============================================================
# 4) CONDITIONAL EXPECTED NEXT MOVE
# ============================================================
# E[delta_t | delta_{t-1} > 0]
# E[delta_t | delta_{t-1} < 0]

exp_after_down = (
    valid.loc[valid["prev_sign"] == -1, "delta"].mean()
    if (valid["prev_sign"] == -1).any() else np.nan
)

exp_after_up = (
    valid.loc[valid["prev_sign"] == 1, "delta"].mean()
    if (valid["prev_sign"] == 1).any() else np.nan
)

print("=== Conditional expected current move ===")
print(f"E[delta_t | delta_(t-1) was down] = {exp_after_down:.6f}")
print(f"E[delta_t | delta_(t-1) was up]   = {exp_after_up:.6f}")
print()


# ============================================================
# 5) STREAK ANALYSIS
# ============================================================
# Example:
# after 2 consecutive downs, what is the probability of another down?
# after 2 consecutive ups, what is the probability of another up?

df["sign_lag1"] = df["sign"].shift(1)
df["sign_lag2"] = df["sign"].shift(2)

# Current row t:
# sign_lag2 = sign at t-2
# sign_lag1 = sign at t-1
# sign      = sign at t

down2 = df[(df["sign_lag2"] == -1) & (df["sign_lag1"] == -1) & df["sign"].notna()]
up2 = df[(df["sign_lag2"] == 1) & (df["sign_lag1"] == 1) & df["sign"].notna()]

p_down_after_2downs = (down2["sign"] == -1).mean() if len(down2) > 0 else np.nan
p_up_after_2ups = (up2["sign"] == 1).mean() if len(up2) > 0 else np.nan

print("=== 2-step streak continuation ===")
print(f"P(down_t | down_(t-2), down_(t-1)) = {p_down_after_2downs:.4f}   ({p_down_after_2downs*100:.2f}%)")
print(f"P(up_t   | up_(t-2), up_(t-1))     = {p_up_after_2ups:.4f}   ({p_up_after_2ups*100:.2f}%)")
print()


# ============================================================
# 6) DIRECT ANSWER TO YOUR EXACT QUESTION
# ============================================================
# "When mid[-1] < mid[-2], does the next one have >50% chance to be even smaller?"
#
# In time indexing:
# if mid[t-1] < mid[t-2], then is mid[t] < mid[t-1] more likely than 50%?

exact_base = df[df["sign_lag1"] == -1].copy()
exact_prob = (exact_base["sign"] == -1).mean() if len(exact_base) > 0 else np.nan

print("=== Exact question ===")
print(f"P(mid_t < mid_(t-1) | mid_(t-1) < mid_(t-2)) = {exact_prob:.4f}   ({exact_prob*100:.2f}%)")
print()


# ============================================================
# 7) VERY SIMPLE INTERPRETATION
# ============================================================
print("=== Quick interpretation guide ===")
print("- Around 50%: no strong directional edge")
print("- Clearly above 50%: momentum / continuation")
print("- Clearly below 50%: mean reversion")
print("- Positive lag-1 autocorr: momentum")
print("- Negative lag-1 autocorr: mean reversion")
