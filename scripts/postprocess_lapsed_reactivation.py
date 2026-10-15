"""Post-processing heuristic: Summer Road-Trip Lapsed Customer Reactivation.

Restores natural holiday season prevalence for high-tenure loyalty members
who were temporarily inactive in spring (Sept-Nov) but have high predicted fuel spend
and confirmed prior summer holiday activity.

Input:  submissions/sub_champion_v5.csv
Output: submissions/sub_champion_v5_reactivated.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    PROCESSED_DATA_DIR,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
    TARGET_FUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.data.snapshots import build_features
from sasol_customer_value.utils.helpers import validate_submission_format


def get_or_build_test_features() -> pd.DataFrame:
    """Retrieve cached test features or generate them dynamically."""
    cache_path = PROCESSED_DATA_DIR / "snapshots" / "X_test_2025-12-01.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df_trans = load_transactions()
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))
    X_test = build_features(
        df_trans, cutoff=PUBLIC_HISTORY_END_EXCLUSIVE, customer_ids=pd.Index(test_ids)
    ).reindex(test_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(cache_path)
    return X_test


def apply_lapsed_reactivation(
    base_submission_path: Path,
    output_path: Path,
    top_k: int = 100,
) -> pd.DataFrame:
    """Score lapsed customers and reclassify top K from Inactivity to Fuel Growth."""
    if not base_submission_path.exists():
        raise FileNotFoundError(f"Base submission not found: {base_submission_path}")

    sub = pd.read_csv(base_submission_path)
    X_test = get_or_build_test_features().reindex(sub[ID_COL].astype(str))

    # Eligible pool:
    # 1. Currently predicted as Inactivity
    # 2. 0 fuel litres in last 90 days (Sept-Nov)
    # 3. Established loyalty history (lifetime >= 40 litres)
    # 4. Verified active during prior summer holiday season (Dec-Feb)
    is_lapsed_mask = (
        (sub[TARGET_OPPORTUNITY].values == "Inactivity")
        & (X_test["fuel_litres_last_90d"].values == 0)
        & (X_test["fuel_litres_lifetime"].values >= 40)
        & (X_test["same_season_prior_year_active"].values == 1.0)
    )

    indices = np.where(is_lapsed_mask)[0]

    df_cand = pd.DataFrame(
        {
            ID_COL: sub.iloc[indices][ID_COL].values,
            "CLV_fuel": sub.iloc[indices][TARGET_FUEL].values,
            "days_since_last_visit": X_test.iloc[indices][
                "days_since_last_visit"
            ].values,
            "fuel_litres_lifetime": X_test.iloc[indices]["fuel_litres_lifetime"].values,
            "baskets_active_months_6m": X_test.iloc[indices][
                "baskets_active_months_6m"
            ].values,
        }
    )

    # Composite holiday reactivation score:
    # Heavy positive weight on Model F predicted fuel spend,
    # bonus for active months, slight penalty for older dormancy
    df_cand["reactivation_score"] = (
        df_cand["CLV_fuel"] * 2.0
        + (df_cand["baskets_active_months_6m"] / 6.0) * 0.3
        - (df_cand["days_since_last_visit"] / 365.0) * 0.2
    )

    df_cand = df_cand.sort_values("reactivation_score", ascending=False)
    selected_ids = set(df_cand.head(top_k)[ID_COL])

    print("=" * 80)
    print("APPLYING SUMMER ROAD-TRIP REACTIVATION")
    print(f"Base Submission:        {base_submission_path.name}")
    print(f"Target Flip Count (K):  {top_k}")
    print(f"Eligible Candidate Pool:{len(df_cand)} lapsed holiday members")
    print("-" * 80)
    print(
        f"Reactivation Score Cutoff: {df_cand.iloc[top_k - 1]['reactivation_score']:.4f}"
    )
    print(f"Min CLV_fuel in flipped:   {df_cand.head(top_k)['CLV_fuel'].min():.4f}")
    print(f"Mean CLV_fuel in flipped:  {df_cand.head(top_k)['CLV_fuel'].mean():.4f}")
    print(
        f"Mean Lifetime Litres:      {df_cand.head(top_k)['fuel_litres_lifetime'].mean():.1f} L"
    )
    print(
        f"Mean Days Since Visit:     {df_cand.head(top_k)['days_since_last_visit'].mean():.1f} days"
    )
    print("=" * 80)

    # Apply flip
    sub_out = sub.copy()
    flip_mask = sub_out[ID_COL].isin(selected_ids)
    sub_out.loc[flip_mask, TARGET_OPPORTUNITY] = "Existing-category growth: Fuel"

    # Strict compliance check
    validate_submission_format(sub_out, expected_ids=list(sub[ID_COL].astype(str)))
    sub_out.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Reactivated submission saved to: {output_path}")
    print("\nUpdated Opportunity Distribution (Top 5):")
    print(sub_out[TARGET_OPPORTUNITY].value_counts().head(5))
    print(
        f"\nFuel Growth Share: {((sub_out[TARGET_OPPORTUNITY] == 'Existing-category growth: Fuel').mean() * 100):.2f}% "
        "(Historical Summer Benchmark: 27.30%)"
    )
    return sub_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply summer road-trip reactivation post-processing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SUBMISSIONS_DIR / "sub_champion_v5.csv",
        help="Path to base submission file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SUBMISSIONS_DIR / "sub_champion_v5_reactivated.csv",
        help="Path for output submission file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top lapsed customers to flip to Fuel Growth",
    )
    args = parser.parse_args()

    apply_lapsed_reactivation(
        base_submission_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
    )
