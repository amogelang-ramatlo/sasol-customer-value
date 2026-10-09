"""Label-aligned behavioural features (leak-free, calendar-month based).

The Opportunity label compares outcome-window spend against the *calendar* previous
quarter ``[cutoff - 3 months, cutoff)`` using signed ``TotalExclAmount`` per original
Level-2 category. These features reproduce that baseline exactly and add regularity,
loyalty and trend signals.

Conventions:
    * ``k`` = months before cutoff (k=1 is the last full calendar month before cutoff).
    * q1 = months 1-3 (the label's growth baseline), q2 = months 4-6,
      ly = months 10-12 (same calendar quarter as the outcome window, one year earlier).
    * Any window starting before the first observed transaction is set to NaN, so
      "no data" is never confused with "zero spend".
"""

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    ADJUSTMENT_ITEMS,
    CATEGORY_MAPPING,
    DATETIME_COL,
    ID_COL,
    TRANSACTION_ID_COL,
)

INCLUDED_CATEGORIES: list[str] = sorted(
    c for c, v in CATEGORY_MAPPING.items() if v.get("included", True)
)
_WINDOWS: dict[str, tuple[int, int]] = {"q1": (1, 3), "q2": (4, 6), "ly": (10, 12)}


def _safe_name(cat: str) -> str:
    return cat.replace(" ", "_").replace("&", "and").replace("-", "_")


def _month_index(times: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    """Months before cutoff (1 = calendar month immediately preceding cutoff)."""
    return (cutoff.year * 12 + cutoff.month) - (times.dt.year * 12 + times.dt.month)


def extract_behavioral_features(
    transactions_df: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    customer_ids: pd.Index,
) -> pd.DataFrame:
    """Extract label-aligned behavioural features using only data strictly before cutoff.

    Args:
        transactions_df: Loaded transactions (from ``load_transactions``).
        cutoff_date: Snapshot cutoff; must be the first day of a month.
        customer_ids: Customer IDs to align the output index to.

    Returns:
        DataFrame indexed by customer ID (string) with numeric feature columns.
    """
    cutoff = pd.Timestamp(cutoff_date)
    if cutoff != cutoff.normalize() or cutoff.day != 1:
        raise ValueError(f"cutoff_date must be the first day of a month, got {cutoff}")

    # 1. Anti-leakage guardrail: strictly filter transactions before cutoff (first op)
    time_col = "_time" if "_time" in transactions_df.columns else DATETIME_COL
    df = transactions_df[transactions_df[time_col] < cutoff]
    df = df[~df["ItemName"].isin(ADJUSTMENT_ITEMS)].copy()

    idx = pd.Index(customer_ids, name=ID_COL).astype(str)
    feats = pd.DataFrame(index=idx)
    if len(df) == 0:
        return feats

    data_start = df[time_col].min()
    df["_k"] = _month_index(df[time_col], cutoff)
    df["_amt"] = df["TotalExclAmount"].fillna(0.0)
    df["_qty"] = df["Quantity"].fillna(0.0)
    df = df[df[ID_COL].isin(idx)]

    def window_valid(k_hi: int) -> bool:
        # Window covering months k_lo..k_hi starts at cutoff - k_hi months.
        return bool(cutoff - pd.DateOffset(months=k_hi) >= data_start)

    feats["history_available_days"] = float((cutoff - data_start).days)

    in_cat = df["ItemCategoryLevel2"].isin(INCLUDED_CATEGORIES)
    is_fuel = (df["ItemCategoryLevel2"] == "Fuel") & (df["LineItemType"] == "Sale_Fuel")
    is_fuel_cat = df["ItemCategoryLevel2"] == "Fuel"
    is_store = in_cat & ~is_fuel_cat
    qualifying = (df["_qty"] > 0) & (df["_amt"] > 0)

    def agg(mask: pd.Series, col: str, how: str = "sum") -> pd.Series:
        sub = df[mask]
        if how == "nunique":
            return sub.groupby(ID_COL)[col].nunique()
        return sub.groupby(ID_COL)[col].sum()

    # 2. Window aggregates (fuel rands / litres, store rands, baskets)
    for w, (lo, hi) in _WINDOWS.items():
        in_w = df["_k"].between(lo, hi)
        cols = {
            f"fuel_rand_{w}": agg(in_w & is_fuel_cat & in_cat, "_amt"),
            f"fuel_litres_{w}": agg(in_w & is_fuel, "_qty"),
            f"store_rand_{w}": agg(in_w & is_store, "_amt"),
            f"baskets_{w}": agg(in_w, TRANSACTION_ID_COL, "nunique"),
            f"store_baskets_{w}": agg(in_w & is_store & qualifying, TRANSACTION_ID_COL, "nunique"),
        }
        valid = window_valid(hi)
        for name, s in cols.items():
            feats[name] = feats.index.map(s).fillna(0.0).astype(float) if valid else np.nan

    feats["fuel_price_q1"] = feats["fuel_rand_q1"] / feats["fuel_litres_q1"].where(
        feats["fuel_litres_q1"] > 0
    )
    for base in ["fuel_rand", "store_rand", "baskets"]:
        feats[f"{base}_q1_over_q2"] = np.log1p(feats[f"{base}_q1"].clip(lower=0)) - np.log1p(
            feats[f"{base}_q2"].clip(lower=0)
        )
        feats[f"{base}_q1_over_ly"] = np.log1p(feats[f"{base}_q1"].clip(lower=0)) - np.log1p(
            feats[f"{base}_ly"].clip(lower=0)
        )

    # Growth hurdle for fuel: rand increment needed next quarter for fuel growth label
    base_fuel = feats["fuel_rand_q1"].clip(lower=0)
    feats["fuel_growth_hurdle_rand"] = np.maximum(20.0, 0.25 * base_fuel)

    # Average quarterly fuel spend over the customer's observed tenure
    first_seen = df.groupby(ID_COL)[time_col].min()
    tenure_q = ((cutoff - feats.index.map(first_seen)).days / 91.3).to_numpy(dtype=float)
    tenure_q = np.clip(np.nan_to_num(tenure_q, nan=1.0), 1.0, None)
    fuel_life = feats.index.map(agg(is_fuel_cat & in_cat, "_amt")).fillna(0.0).to_numpy(dtype=float)
    store_life = feats.index.map(agg(is_store, "_amt")).fillna(0.0).to_numpy(dtype=float)
    feats["fuel_rand_per_quarter_avg"] = fuel_life / tenure_q
    feats["store_rand_per_quarter_avg"] = store_life / tenure_q
    feats["fuel_q1_vs_avg"] = np.log1p(base_fuel) - np.log1p(feats["fuel_rand_per_quarter_avg"].clip(lower=0))
    feats["store_q1_vs_avg"] = np.log1p(feats["store_rand_q1"].clip(lower=0)) - np.log1p(
        feats["store_rand_per_quarter_avg"].clip(lower=0)
    )

    # 3. Monthly series (last 6 months): fuel rands, store rands, baskets + trend slope
    month_ks = list(range(1, 7))
    for name, mask, col, how in [
        ("fuel_rand", is_fuel_cat & in_cat, "_amt", "sum"),
        ("store_rand", is_store, "_amt", "sum"),
        ("baskets", pd.Series(True, index=df.index), TRANSACTION_ID_COL, "nunique"),
    ]:
        sub = df[mask & df["_k"].between(1, 6)]
        if how == "nunique":
            piv = sub.groupby([ID_COL, "_k"])[col].nunique().unstack(fill_value=0)
        else:
            piv = sub.groupby([ID_COL, "_k"])[col].sum().unstack(fill_value=0.0)
        piv = piv.reindex(index=feats.index, columns=month_ks, fill_value=0).astype(float)
        for k in month_ks:
            if not window_valid(k):
                piv[k] = np.nan
            feats[f"{name}_m{k}"] = piv[k]
        # Slope of log1p(monthly) vs time (positive = accelerating); NaN-aware
        y = np.log1p(piv[month_ks].clip(lower=0).to_numpy())
        x = -np.array(month_ks, dtype=float)  # more recent = larger x
        valid_cols = ~np.isnan(y).all(axis=0)
        if valid_cols.sum() >= 3:
            yv, xv = y[:, valid_cols], x[valid_cols]
            xc = xv - xv.mean()
            feats[f"{name}_slope_6m"] = ((yv - yv.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc**2).sum()
        else:
            feats[f"{name}_slope_6m"] = np.nan
        feats[f"{name}_active_months_6m"] = (piv[month_ks] > 0).sum(axis=1).astype(float)

    # 4. Per-category: signed calendar-q1 spend (exact label baseline) + history baskets
    cat_df = df[in_cat]
    q1_spend = (
        cat_df[cat_df["_k"].between(1, 3)]
        .groupby([ID_COL, "ItemCategoryLevel2"])["_amt"].sum()
        .unstack(fill_value=0.0)
        .reindex(index=feats.index, columns=INCLUDED_CATEGORIES, fill_value=0.0)
    )
    hist_baskets = (
        cat_df[qualifying.loc[cat_df.index]]
        .groupby([ID_COL, "ItemCategoryLevel2"])[TRANSACTION_ID_COL].nunique()
        .unstack(fill_value=0)
        .reindex(index=feats.index, columns=INCLUDED_CATEGORIES, fill_value=0)
    )
    q1_valid = window_valid(3)
    for cat in INCLUDED_CATEGORIES:
        s = _safe_name(cat)
        feats[f"q1cat_{s}"] = q1_spend[cat].astype(float) if q1_valid else np.nan
        feats[f"histbaskets_{s}"] = hist_baskets[cat].astype(float)

    store_cats = [c for c in INCLUDED_CATEGORIES if c != "Fuel"]
    feats["n_store_cats_never_bought"] = (hist_baskets[store_cats] == 0).sum(axis=1).astype(float)
    feats["n_store_cats_bought_once"] = (hist_baskets[store_cats] == 1).sum(axis=1).astype(float)
    feats["n_cats_active_q1"] = (q1_spend > 0).sum(axis=1).astype(float)
    feats["max_store_cat_q1"] = q1_spend[store_cats].max(axis=1).astype(float)

    # 5. Visit regularity (gaps between baskets) and recency relative to habit
    bt = (
        df.groupby([ID_COL, TRANSACTION_ID_COL])[time_col].min().reset_index()
        .sort_values([ID_COL, time_col])
    )
    bt["_gap"] = bt.groupby(ID_COL)[time_col].diff().dt.total_seconds() / 86400.0
    g = bt.dropna(subset=["_gap"]).groupby(ID_COL)["_gap"]
    gap_stats = pd.DataFrame(
        {"gap_mean": g.mean(), "gap_std": g.std(), "gap_median": g.median(), "gap_max": g.max()}
    )
    gap_stats["gap_last"] = bt.dropna(subset=["_gap"]).groupby(ID_COL)["_gap"].last()
    for col in gap_stats.columns:
        feats[col] = feats.index.map(gap_stats[col]).astype(float)
    feats["gap_cv"] = feats["gap_std"] / feats["gap_mean"].clip(lower=0.5)
    last_visit = bt.groupby(ID_COL)[time_col].max()
    days_since = (cutoff - feats.index.map(last_visit)).total_seconds() / 86400.0
    feats["recency_over_median_gap"] = np.asarray(days_since, dtype=float) / feats["gap_median"].clip(lower=0.5)
    # Expected visits in a 91-day window given median gap
    feats["expected_visits_next_q"] = 91.0 / feats["gap_median"].clip(lower=0.5)

    # 6. Primary station loyalty
    loc = df.groupby([ID_COL, "LocationId"])[TRANSACTION_ID_COL].nunique()
    top = loc.groupby(level=0).max()
    tot = loc.groupby(level=0).sum()
    feats["primary_station_share"] = feats.index.map(top / tot).astype(float)

    # 7. Returns / negative lines behaviour
    feats["return_lines_share"] = feats.index.map(
        (df["_amt"] < 0).groupby(df[ID_COL]).mean()
    ).astype(float)

    return feats
