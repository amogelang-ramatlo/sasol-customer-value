"""Domain-rule hurdle features, feasibility masks, and temporal sample weighting.

Implements:
1. Store Category Growth Hurdles: hurdle_rand_<cat> = max(20.0, 0.25 * spend_last_90d)
   and growth saturation flags.
2. Deterministic Feasibility Masking: Enforces competition rules (Rule 5 & 6)
   guaranteeing 0 impossible adoption or growth predictions.
3. Dual-Kernel Seasonal x Recency Sample Weighting for temporal snapshot stacking.
"""

import numpy as np
import pandas as pd

# Authoritative mapping between Opportunity classes and lifetime purchase flags
ADOPTION_TO_FLAG: dict[str, str] = {
    "New category adoption: Beverages": "cat_ever_bought_Beverages",
    "New category adoption: Braai & Ice": "cat_ever_bought_Braai_and_Ice",
    "New category adoption: Breads & rolls": "cat_ever_bought_Breads_and_rolls",
    "New category adoption: Confectionary": "cat_ever_bought_Confectionary",
    "New category adoption: Food & beverage": "cat_ever_bought_Food_and_beverage",
    "New category adoption: Groceries": "cat_ever_bought_Groceries",
    "New category adoption: Ice-cream": "cat_ever_bought_Ice-cream",
    "New category adoption: Lubricants": "cat_ever_bought_Lubricants",
    "New category adoption: Snacking": "cat_ever_bought_Snacking",
    "New category adoption: Tobacco": "cat_ever_bought_Tobacco",
}

GROWTH_TO_FLAG: dict[str, str] = {
    "Existing-category growth: Beverages": "cat_ever_bought_Beverages",
    "Existing-category growth: Food & beverage": "cat_ever_bought_Food_and_beverage",
    "Existing-category growth: Fuel": "cat_ever_bought_Fuel",
}


def add_store_category_hurdles(X: pd.DataFrame) -> pd.DataFrame:
    """Derive store category growth hurdles and saturation flags from 90d baseline spend.

    For each retail category with observed 90-day spend:
    - hurdle_rand_<cat>: Net rand increase required to satisfy the growth rule:
      hurdle = max(20.0, 0.25 * spend_last_90d).
    - is_saturated_<cat>: Binary flag indicating high Q1 spend requiring > R100 increase.
    """
    X_out = X.copy()
    for col in X.columns:
        if col.startswith("cat_spend_last_90d_"):
            cat = col.replace("cat_spend_last_90d_", "")
            spend = X[col].fillna(0.0).to_numpy(dtype=np.float64)
            hurdle = np.maximum(20.0, 0.25 * spend)
            X_out[f"hurdle_rand_{cat}"] = hurdle
            X_out[f"is_saturated_{cat}"] = (hurdle > 100.0).astype(np.float64)
    return X_out


def apply_feasibility_mask(
    log_scores: np.ndarray,
    X: pd.DataFrame,
    class_names: list[str],
) -> np.ndarray:
    """Hard-mask mathematically impossible Opportunity classes based on lifetime history.

    Rules enforced:
    - Adoption requires 0 lifetime prior purchases: mask if cat_ever_bought == 1.
    - Growth requires >= 1 historical purchase: mask if cat_ever_bought == 0.
    """
    masked_scores = log_scores.copy()
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for cls_name, flag_col in ADOPTION_TO_FLAG.items():
        resolved_col = (
            flag_col
            if flag_col in X.columns
            else flag_col.replace("-", "_")
            if flag_col.replace("-", "_") in X.columns
            else None
        )
        if cls_name in class_to_idx and resolved_col is not None:
            already_bought = X[resolved_col].to_numpy() > 0
            masked_scores[already_bought, class_to_idx[cls_name]] = -1e9

    for cls_name, flag_col in GROWTH_TO_FLAG.items():
        resolved_col = (
            flag_col
            if flag_col in X.columns
            else flag_col.replace("-", "_")
            if flag_col.replace("-", "_") in X.columns
            else None
        )
        if cls_name in class_to_idx and resolved_col is not None:
            never_bought = X[resolved_col].to_numpy() == 0
            masked_scores[never_bought, class_to_idx[cls_name]] = -1e9

    return masked_scores


def compute_snapshot_weights(
    cutoff_dates: list[str] | pd.Series,
    target_month_day: tuple[int, int] = (12, 1),
    kappa: float = 0.50,
    half_life_months: float = 12.0,
) -> np.ndarray:
    """Compute Dual-Kernel Seasonal (von Mises) x Recency sample weights for historical snapshots.

    Args:
        cutoff_dates: List or Series of snapshot cutoff strings ('YYYY-MM-DD').
        target_month_day: (month, day) of the target season anchor (default: Dec 1st for summer).
        kappa: Concentration parameter for circular seasonal kernel (higher = sharper focus on season).
        half_life_months: Exponential recency half-life in months.

    Returns:
        Normalized sample weights with mean = 1.0.
    """
    cutoffs = pd.Series(pd.to_datetime(cutoff_dates))
    max_cutoff = cutoffs.max()

    # 1. Circular seasonal distance in days to target anchor
    day_of_year = cutoffs.dt.dayofyear
    target_doy = pd.Timestamp(
        year=2024, month=target_month_day[0], day=target_month_day[1]
    ).dayofyear
    angle = 2.0 * np.pi * (day_of_year - target_doy) / 365.25
    seasonal_weight = np.exp(kappa * np.cos(angle))

    # 2. Macro recency decay relative to latest observed training cutoff
    months_lag = (max_cutoff - cutoffs).dt.days / 30.4375
    decay_rate = np.log(2.0) / half_life_months
    recency_weight = np.exp(-decay_rate * months_lag)

    weights = seasonal_weight * recency_weight
    return (weights / weights.mean()).to_numpy(dtype=np.float64)
