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
            f"store_baskets_{w}": agg(
                in_w & is_store & qualifying, TRANSACTION_ID_COL, "nunique"
            ),
        }
        valid = window_valid(hi)
        for name, s in cols.items():
            feats[name] = (
                feats.index.map(s).fillna(0.0).astype(float) if valid else np.nan
            )

    feats["fuel_price_q1"] = feats["fuel_rand_q1"] / feats["fuel_litres_q1"].where(
        feats["fuel_litres_q1"] > 0
    )
    for base in ["fuel_rand", "store_rand", "baskets"]:
        feats[f"{base}_q1_over_q2"] = np.log1p(
            feats[f"{base}_q1"].clip(lower=0)
        ) - np.log1p(feats[f"{base}_q2"].clip(lower=0))
        feats[f"{base}_q1_over_ly"] = np.log1p(
            feats[f"{base}_q1"].clip(lower=0)
        ) - np.log1p(feats[f"{base}_ly"].clip(lower=0))

    # Growth hurdle for fuel: rand increment needed next quarter for fuel growth label
    base_fuel = feats["fuel_rand_q1"].clip(lower=0)
    feats["fuel_growth_hurdle_rand"] = np.maximum(20.0, 0.25 * base_fuel)

    # Average quarterly fuel spend over the customer's observed tenure
    first_seen = df.groupby(ID_COL)[time_col].min()
    tenure_q = ((cutoff - feats.index.map(first_seen)).days / 91.3).to_numpy(
        dtype=float
    )
    tenure_q = np.clip(np.nan_to_num(tenure_q, nan=1.0), 1.0, None)
    fuel_life = (
        feats.index.map(agg(is_fuel_cat & in_cat, "_amt"))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    store_life = (
        feats.index.map(agg(is_store, "_amt")).fillna(0.0).to_numpy(dtype=float)
    )
    feats["fuel_rand_per_quarter_avg"] = fuel_life / tenure_q
    feats["store_rand_per_quarter_avg"] = store_life / tenure_q
    feats["fuel_q1_vs_avg"] = np.log1p(base_fuel) - np.log1p(
        feats["fuel_rand_per_quarter_avg"].clip(lower=0)
    )
    feats["store_q1_vs_avg"] = np.log1p(
        feats["store_rand_q1"].clip(lower=0)
    ) - np.log1p(feats["store_rand_per_quarter_avg"].clip(lower=0))

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
        piv = piv.reindex(index=feats.index, columns=month_ks, fill_value=0).astype(
            float
        )
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
            feats[f"{name}_slope_6m"] = (
                (yv - yv.mean(axis=1, keepdims=True)) * xc
            ).sum(axis=1) / (xc**2).sum()
        else:
            feats[f"{name}_slope_6m"] = np.nan
        feats[f"{name}_active_months_6m"] = (
            (piv[month_ks] > 0).sum(axis=1).astype(float)
        )

    # 4. Per-category: signed calendar-q1 spend (exact label baseline) + history baskets
    cat_df = df[in_cat]
    q1_spend = (
        cat_df[cat_df["_k"].between(1, 3)]
        .groupby([ID_COL, "ItemCategoryLevel2"])["_amt"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=feats.index, columns=INCLUDED_CATEGORIES, fill_value=0.0)
    )
    hist_baskets = (
        cat_df[qualifying.loc[cat_df.index]]
        .groupby([ID_COL, "ItemCategoryLevel2"])[TRANSACTION_ID_COL]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=feats.index, columns=INCLUDED_CATEGORIES, fill_value=0)
    )
    extra: dict[str, pd.Series | np.ndarray] = {}
    q1_valid = window_valid(3)
    for cat in INCLUDED_CATEGORIES:
        s = _safe_name(cat)
        extra[f"q1cat_{s}"] = (
            q1_spend[cat].astype(float) if q1_valid else pd.Series(np.nan, index=idx)
        )
        extra[f"histbaskets_{s}"] = hist_baskets[cat].astype(float)

    store_cats = [c for c in INCLUDED_CATEGORIES if c != "Fuel"]
    extra["n_store_cats_never_bought"] = (
        (hist_baskets[store_cats] == 0).sum(axis=1).astype(float)
    )
    extra["n_store_cats_bought_once"] = (
        (hist_baskets[store_cats] == 1).sum(axis=1).astype(float)
    )
    extra["n_cats_active_q1"] = (q1_spend > 0).sum(axis=1).astype(float)
    extra["max_store_cat_q1"] = q1_spend[store_cats].max(axis=1).astype(float)

    # 5. Visit regularity (gaps between baskets) and recency relative to habit
    bt = (
        df.groupby([ID_COL, TRANSACTION_ID_COL])[time_col]
        .min()
        .reset_index()
        .sort_values([ID_COL, time_col])
    )
    bt["_gap"] = bt.groupby(ID_COL)[time_col].diff().dt.total_seconds() / 86400.0
    g = bt.dropna(subset=["_gap"]).groupby(ID_COL)["_gap"]
    gap_stats = pd.DataFrame(
        {
            "gap_mean": g.mean(),
            "gap_std": g.std(),
            "gap_median": g.median(),
            "gap_max": g.max(),
        }
    )
    gap_stats["gap_last"] = bt.dropna(subset=["_gap"]).groupby(ID_COL)["_gap"].last()
    for col in gap_stats.columns:
        extra[col] = pd.Series(idx.map(gap_stats[col]), index=idx, dtype=float)
    gap_mean_s = pd.Series(extra["gap_mean"], index=idx, dtype=float)
    gap_std_s = pd.Series(extra["gap_std"], index=idx, dtype=float)
    gap_med_s = pd.Series(extra["gap_median"], index=idx, dtype=float)
    extra["gap_cv"] = (gap_std_s / gap_mean_s.clip(lower=0.5)).astype(float)
    last_visit = bt.groupby(ID_COL)[time_col].max()
    days_since = (cutoff - idx.map(last_visit)).total_seconds() / 86400.0
    extra["recency_over_median_gap"] = (
        np.asarray(days_since, dtype=float) / gap_med_s.clip(lower=0.5)
    ).astype(float)
    # Expected visits in a 91-day window given median gap
    extra["expected_visits_next_q"] = (91.0 / gap_med_s.clip(lower=0.5)).astype(float)

    # 6. Primary station loyalty
    loc = df.groupby([ID_COL, "LocationId"])[TRANSACTION_ID_COL].nunique()
    top = loc.groupby(level=0).max()
    tot = loc.groupby(level=0).sum()
    extra["primary_station_share"] = pd.Series(
        idx.map(top / tot), index=idx, dtype=float
    )

    # 7. Returns / negative lines behaviour
    ret_mean = (df["_amt"] < 0).groupby(df[ID_COL]).mean()
    extra["return_lines_share"] = pd.Series(idx.map(ret_mean), index=idx, dtype=float)

    # 8. Store affinity & zero-spend separation (Option A)
    df["_is_fuel"] = is_fuel
    df["_is_store"] = is_store
    df["_is_store_qual"] = is_store & qualifying
    df["_store_spend"] = np.where(is_store, df["_amt"], 0.0)
    df["_fuel_qty"] = np.where(is_fuel, df["_qty"], 0.0)

    # Basket-level flags
    basket_summary = (
        df.groupby([ID_COL, TRANSACTION_ID_COL])
        .agg(
            _t=(time_col, "min"),
            _has_fuel=("_is_fuel", "any"),
            _has_store=("_is_store_qual", "any"),
            _s_spend=("_store_spend", "sum"),
            _f_qty=("_fuel_qty", "sum"),
        )
        .reset_index()
    )

    tot_b = (
        basket_summary.groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
        .clip(lower=1.0)
    )
    store_b = (
        basket_summary[basket_summary["_has_store"]]
        .groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
    )
    fuel_only_b = (
        basket_summary[basket_summary["_has_fuel"] & ~basket_summary["_has_store"]]
        .groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
    )
    extra["store_basket_ratio"] = (store_b / tot_b).astype(float)
    extra["fuel_only_basket_ratio"] = (fuel_only_b / tot_b).astype(float)
    extra["is_never_store_shopper"] = (store_b == 0).astype(float)

    last_store = (
        basket_summary[basket_summary["_has_store"]]
        .groupby(ID_COL)["_t"]
        .max()
        .reindex(idx)
    )
    extra["days_since_last_store_basket"] = (
        ((cutoff - last_store).dt.total_seconds() / 86400.0).fillna(999.0).astype(float)
    )

    t_30 = cutoff - pd.Timedelta(days=30)
    t_90 = cutoff - pd.Timedelta(days=90)
    t_180 = cutoff - pd.Timedelta(days=180)

    s30 = (
        basket_summary[(basket_summary["_t"] >= t_30) & basket_summary["_has_store"]]
        .groupby(ID_COL)["_s_spend"]
        .sum()
        .reindex(idx)
        .fillna(0.0)
    )
    s90 = (
        basket_summary[(basket_summary["_t"] >= t_90) & basket_summary["_has_store"]]
        .groupby(ID_COL)["_s_spend"]
        .sum()
        .reindex(idx)
        .fillna(0.0)
    )
    extra["store_spend_30d_over_90d"] = (s30 / s90.clip(lower=1.0)).astype(float)
    extra["store_active_last_30d"] = (s30 > 0).astype(float)
    extra["store_active_last_90d"] = (s90 > 0).astype(float)

    # 9. Growth hurdles & category spend acceleration (Option B)
    f90 = (
        basket_summary[(basket_summary["_t"] >= t_90) & basket_summary["_has_fuel"]]
        .groupby(ID_COL)["_f_qty"]
        .sum()
        .reindex(idx)
        .fillna(0.0)
    )
    fprev = (
        basket_summary[
            (basket_summary["_t"] >= t_180)
            & (basket_summary["_t"] < t_90)
            & basket_summary["_has_fuel"]
        ]
        .groupby(ID_COL)["_f_qty"]
        .sum()
        .reindex(idx)
        .fillna(0.0)
    )
    extra["fuel_litres_accel_90_vs_prev90"] = (f90 - fprev).astype(float)
    extra["fuel_litres_ratio_90_vs_prev90"] = ((f90 + 1.0) / (fprev + 1.0)).astype(
        float
    )
    extra["fuel_candidate_growth_flag"] = ((f90 >= 1.25 * fprev) & (f90 > 0)).astype(
        float
    )

    sprev = (
        basket_summary[
            (basket_summary["_t"] >= t_180)
            & (basket_summary["_t"] < t_90)
            & basket_summary["_has_store"]
        ]
        .groupby(ID_COL)["_s_spend"]
        .sum()
        .reindex(idx)
        .fillna(0.0)
    )
    extra["store_spend_accel_90_vs_prev90"] = (s90 - sprev).astype(float)
    extra["store_candidate_growth_flag"] = (
        ((s90 - sprev) >= 20.0) & (s90 >= 1.25 * sprev) & (s90 > 0)
    ).astype(float)

    # 10. Commuter personas & temporal patterns (Option C)
    dow = basket_summary["_t"].dt.dayofweek
    hour = basket_summary["_t"].dt.hour
    we_b = (
        basket_summary[dow >= 5]
        .groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
    )
    mo_b = (
        basket_summary[hour.between(6, 9)]
        .groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
    )
    mi_b = (
        basket_summary[hour.between(10, 14)]
        .groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
    )
    ev_b = (
        basket_summary[hour.between(15, 19)]
        .groupby(ID_COL)[TRANSACTION_ID_COL]
        .nunique()
        .reindex(idx)
        .fillna(0.0)
    )
    extra["weekend_basket_share"] = (we_b / tot_b).astype(float)
    extra["morning_commute_share"] = (mo_b / tot_b).astype(float)
    extra["midday_basket_share"] = (mi_b / tot_b).astype(float)
    extra["evening_commute_share"] = (ev_b / tot_b).astype(float)

    # 11. Customer Shopper Personas (Fuel-Only, Store-Only, Dual/Both)
    tot_f_qty = df.groupby(ID_COL)["_fuel_qty"].sum().reindex(idx).fillna(0.0)
    tot_s_spend = df.groupby(ID_COL)["_store_spend"].sum().reindex(idx).fillna(0.0)
    has_life_f = tot_f_qty > 0
    has_life_s = tot_s_spend > 0

    extra["is_fuel_only_buyer_lifetime"] = (has_life_f & ~has_life_s).astype(float)
    extra["is_store_only_buyer_lifetime"] = (~has_life_f & has_life_s).astype(float)
    extra["is_dual_buyer_lifetime"] = (has_life_f & has_life_s).astype(float)

    has_90d_f = f90 > 0
    has_90d_s = s90 > 0
    extra["is_fuel_only_buyer_90d"] = (has_90d_f & ~has_90d_s).astype(float)
    extra["is_store_only_buyer_90d"] = (~has_90d_f & has_90d_s).astype(float)
    extra["is_dual_buyer_90d"] = (has_90d_f & has_90d_s).astype(float)

    # Integer segment: 0=Inactive/Neither, 1=Fuel-only, 2=Store-only, 3=Dual
    segment = np.zeros(len(idx), dtype=float)
    segment[has_90d_f & ~has_90d_s] = 1.0
    segment[~has_90d_f & has_90d_s] = 2.0
    segment[has_90d_f & has_90d_s] = 3.0
    extra["shopper_persona_segment_90d"] = segment

    extra_df = pd.DataFrame(extra, index=idx)
    return pd.concat([feats, extra_df], axis=1)
