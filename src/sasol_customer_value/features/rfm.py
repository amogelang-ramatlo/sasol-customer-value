"""Leakage-free RFM (Recency, Frequency, Monetary) and category feature extractor.

All aggregations are strictly parameterized by cutoff_date (t < t_cutoff).
Future transactions are filtered out as the very first operation.
"""

import pandas as pd

from sasol_customer_value.config import (
    ADJUSTMENT_ITEMS,
    CATEGORY_MAPPING,
    DATETIME_COL,
    EXCLUDED_CATEGORIES,
    ID_COL,
    TRANSACTION_ID_COL,
)


def extract_rfm_features(
    transactions_df: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    customer_ids: pd.Index | None = None,
) -> pd.DataFrame:
    """Extract leak-free customer-level features strictly using data before cutoff_date.

    Args:
        transactions_df: Prepared or raw transactions DataFrame.
        cutoff_date: Timestamp threshold. Only transactions strictly < cutoff_date are used.
        customer_ids: Target customer IDs to align the output index to.

    Returns:
        DataFrame with customer ID index and engineered numeric feature columns.
    """
    cutoff = pd.Timestamp(cutoff_date)

    # 1. Anti-leakage guardrail: strictly filter transactions before cutoff
    time_col = "_time" if "_time" in transactions_df.columns else DATETIME_COL
    df = transactions_df[transactions_df[time_col] < cutoff].copy()

    # 2. Filter out non-contributing adjustment items
    df = df[~df["ItemName"].isin(ADJUSTMENT_ITEMS)]

    if customer_ids is None:
        customer_ids = df[ID_COL].unique()

    # Feature dataframe indexed by customer ID
    feats = pd.DataFrame(index=pd.Index(customer_ids, name=ID_COL).astype(str))

    if len(df) == 0:
        return feats

    # 3. Recency features
    time_diff = (cutoff - df[time_col]).dt.total_seconds() / 86400.0
    df["_days_ago"] = time_diff

    last_visit = df.groupby(ID_COL)["_days_ago"].min()
    first_visit = df.groupby(ID_COL)["_days_ago"].max()
    feats["days_since_last_visit"] = feats.index.map(last_visit).fillna(999.0)
    feats["days_since_first_visit"] = feats.index.map(first_visit).fillna(999.0)
    feats["customer_tenure_days"] = (
        feats["days_since_first_visit"] - feats["days_since_last_visit"]
    ).clip(lower=0.0)

    # 4. Frequency features (Basket-level counts)
    baskets_lifetime = df.groupby(ID_COL)[TRANSACTION_ID_COL].nunique()
    feats["baskets_lifetime"] = feats.index.map(baskets_lifetime).fillna(0).astype(int)

    for days in [30, 60, 90]:
        recent_baskets = (
            df[df["_days_ago"] <= days].groupby(ID_COL)[TRANSACTION_ID_COL].nunique()
        )
        feats[f"baskets_last_{days}d"] = (
            feats.index.map(recent_baskets).fillna(0).astype(int)
        )

    # Ratio of recent activity to quarterly baseline
    feats["visit_momentum_30_to_90d"] = (
        feats["baskets_last_30d"] * 3.0 / feats["baskets_last_90d"].clip(lower=1.0)
    )

    # 5. Fuel Volume Features (Quantity where Level2 == 'Fuel' and LineItemType == 'Sale_Fuel')
    fuel_mask = (df["ItemCategoryLevel2"] == "Fuel") & (
        df["LineItemType"] == "Sale_Fuel"
    )
    df_fuel = df[fuel_mask]

    fuel_litres_lifetime = df_fuel.groupby(ID_COL)["Quantity"].sum()
    feats["fuel_litres_lifetime"] = feats.index.map(fuel_litres_lifetime).fillna(0.0)

    for days in [30, 60, 90]:
        recent_fuel = (
            df_fuel[df_fuel["_days_ago"] <= days].groupby(ID_COL)["Quantity"].sum()
        )
        feats[f"fuel_litres_last_{days}d"] = feats.index.map(recent_fuel).fillna(0.0)

    fuel_baskets = df_fuel.groupby(ID_COL)[TRANSACTION_ID_COL].nunique()
    feats["fuel_baskets_lifetime"] = feats.index.map(fuel_baskets).fillna(0).astype(int)
    feats["fuel_litres_per_visit"] = feats["fuel_litres_lifetime"] / feats[
        "fuel_baskets_lifetime"
    ].clip(lower=1.0)

    last_fuel = df_fuel.groupby(ID_COL)["_days_ago"].min()
    feats["days_since_last_fuel"] = feats.index.map(last_fuel).fillna(999.0)

    # 6. Non-Fuel Spend Features (TotalExclAmount outside Fuel & Excluded Categories)
    nonfuel_mask = (~df["ItemCategoryLevel2"].isin(["Fuel"] + EXCLUDED_CATEGORIES)) & (
        df["TotalExclAmount"].notna()
    )
    df_nonfuel = df[nonfuel_mask]

    nonfuel_spend_lifetime = df_nonfuel.groupby(ID_COL)["TotalExclAmount"].sum()
    feats["nonfuel_spend_lifetime"] = feats.index.map(nonfuel_spend_lifetime).fillna(
        0.0
    )

    for days in [30, 60, 90]:
        recent_spend = (
            df_nonfuel[df_nonfuel["_days_ago"] <= days]
            .groupby(ID_COL)["TotalExclAmount"]
            .sum()
        )
        feats[f"nonfuel_spend_last_{days}d"] = feats.index.map(recent_spend).fillna(0.0)

    nonfuel_baskets = df_nonfuel.groupby(ID_COL)[TRANSACTION_ID_COL].nunique()
    feats["nonfuel_baskets_lifetime"] = (
        feats.index.map(nonfuel_baskets).fillna(0).astype(int)
    )
    feats["nonfuel_spend_per_visit"] = feats["nonfuel_spend_lifetime"] / feats[
        "nonfuel_baskets_lifetime"
    ].clip(lower=1.0)

    last_nonfuel = df_nonfuel.groupby(ID_COL)["_days_ago"].min()
    feats["days_since_last_nonfuel"] = feats.index.map(last_nonfuel).fillna(999.0)

    # 7. Cross-Category Interactions & Ratios
    feats["fuel_to_total_baskets_ratio"] = feats["fuel_baskets_lifetime"] / feats[
        "baskets_lifetime"
    ].clip(lower=1.0)
    feats["nonfuel_to_total_baskets_ratio"] = feats["nonfuel_baskets_lifetime"] / feats[
        "baskets_lifetime"
    ].clip(lower=1.0)
    feats["has_both_fuel_and_store"] = (
        (feats["fuel_baskets_lifetime"] > 0) & (feats["nonfuel_baskets_lifetime"] > 0)
    ).astype(int)

    # Momentum metrics (acceleration/deceleration into future)
    feats["fuel_momentum_30_to_90d"] = (
        feats["fuel_litres_last_30d"]
        * 3.0
        / feats["fuel_litres_last_90d"].clip(lower=1.0)
    )
    feats["nonfuel_momentum_30_to_90d"] = (
        feats["nonfuel_spend_last_30d"]
        * 3.0
        / feats["nonfuel_spend_last_90d"].clip(lower=1.0)
    )

    # Visit cadence & churn risk
    feats["mean_days_between_baskets"] = feats["customer_tenure_days"] / (
        feats["baskets_lifetime"] - 1
    ).clip(lower=1.0)
    feats["recency_to_cadence_ratio"] = feats["days_since_last_visit"] / (
        feats["mean_days_between_baskets"] + 1.0
    )
    feats["churn_risk_flag_60d"] = (feats["days_since_last_visit"] > 60.0).astype(int)

    # Basket spending depth
    total_spend = df.groupby(ID_COL)["TotalExclAmount"].sum()
    feats["total_spend_lifetime"] = feats.index.map(total_spend).fillna(0.0)
    feats["avg_spend_per_basket"] = feats["total_spend_lifetime"] / feats[
        "baskets_lifetime"
    ].clip(lower=1.0)

    # Station / Location loyalty
    distinct_locs = df.groupby(ID_COL)["LocationId"].nunique()
    feats["distinct_locations_visited"] = (
        feats.index.map(distinct_locs).fillna(0).astype(int)
    )

    # Weekend shopping habit
    is_weekend = df[time_col].dt.dayofweek >= 5
    weekend_baskets = df[is_weekend].groupby(ID_COL)[TRANSACTION_ID_COL].nunique()
    feats["weekend_baskets_count"] = (
        feats.index.map(weekend_baskets).fillna(0).astype(int)
    )
    feats["weekend_baskets_ratio"] = feats["weekend_baskets_count"] / feats[
        "baskets_lifetime"
    ].clip(lower=1.0)

    # 8. Category-Level Presence & Baseline Window Spend (Last 90d comparison)
    # Checks each in-scope category for historical adoption and previous-quarter spend
    in_scope_cats = sorted(
        c for c, p in CATEGORY_MAPPING.items() if p.get("included", True)
    )

    # Filter to transactions with strictly positive quantity & amount
    qualifying_lines = df[(df["Quantity"] > 0) & (df["TotalExclAmount"] > 0)]

    for cat in in_scope_cats:
        cat_safe = cat.replace(" ", "_").replace("&", "and")
        df_cat = qualifying_lines[qualifying_lines["ItemCategoryLevel2"] == cat]

        # Has customer ever bought this category? (Binary flag)
        ever_bought = df_cat.groupby(ID_COL).size()
        feats[f"cat_ever_bought_{cat_safe}"] = (
            feats.index.map(ever_bought).fillna(0) > 0
        ).astype(int)

        # Baseline window spend (last 90 days) - critical for growth comparison!
        recent_cat_spend = (
            df_cat[df_cat["_days_ago"] <= 90].groupby(ID_COL)["TotalExclAmount"].sum()
        )
        feats[f"cat_spend_last_90d_{cat_safe}"] = feats.index.map(
            recent_cat_spend
        ).fillna(0.0)

    # Count of distinct merchandise categories ever tried
    cat_counts = qualifying_lines.groupby(ID_COL)["ItemCategoryLevel2"].nunique()
    feats["distinct_categories_bought"] = (
        feats.index.map(cat_counts).fillna(0).astype(int)
    )

    # 9. 1-Year Seasonal Lookback Features
    # Captures customer behavior in the corresponding 3-month season of the previous year
    t_season_start = cutoff - pd.DateOffset(years=1)
    t_season_end = t_season_start + pd.DateOffset(months=3)
    seasonal_slice = df[
        (df[time_col] >= t_season_start) & (df[time_col] < t_season_end)
    ]

    if len(seasonal_slice) > 0:
        seasonal_baskets = seasonal_slice.groupby(ID_COL)[TRANSACTION_ID_COL].nunique()
        feats["same_season_prior_year_baskets"] = (
            feats.index.map(seasonal_baskets).fillna(0).astype(int)
        )
        feats["same_season_prior_year_active"] = (
            feats["same_season_prior_year_baskets"] > 0
        ).astype(int)

        fuel_mask = (seasonal_slice["ItemCategoryLevel2"] == "Fuel") & (
            seasonal_slice["LineItemType"] == "Sale_Fuel"
        )
        seasonal_fuel = seasonal_slice[fuel_mask].groupby(ID_COL)["Quantity"].sum()
        feats["same_season_prior_year_fuel_litres"] = feats.index.map(
            seasonal_fuel
        ).fillna(0.0)

        nonfuel_mask = (
            ~seasonal_slice["ItemCategoryLevel2"].isin(["Fuel"] + EXCLUDED_CATEGORIES)
        ) & (seasonal_slice["TotalExclAmount"].notna())
        seasonal_nonfuel = (
            seasonal_slice[nonfuel_mask].groupby(ID_COL)["TotalExclAmount"].sum()
        )
        feats["same_season_prior_year_nonfuel_spend"] = feats.index.map(
            seasonal_nonfuel
        ).fillna(0.0)
    else:
        feats["same_season_prior_year_baskets"] = 0
        feats["same_season_prior_year_active"] = 0
        feats["same_season_prior_year_fuel_litres"] = 0.0
        feats["same_season_prior_year_nonfuel_spend"] = 0.0

    feats = feats.fillna(0.0)

    # If the lookback window starts before the first observed transaction, the
    # seasonal values are unknown (not zero): mark them missing.
    if t_season_start < df[time_col].min():
        seasonal_cols = [
            "same_season_prior_year_baskets",
            "same_season_prior_year_active",
            "same_season_prior_year_fuel_litres",
            "same_season_prior_year_nonfuel_spend",
        ]
        feats[seasonal_cols] = float("nan")

    return feats
