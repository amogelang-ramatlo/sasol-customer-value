"""Post-processing heuristic: Summer Road-Trip Lapsed Customer Reactivation.

Restores natural holiday season prevalence for high-tenure loyalty members
who were temporarily inactive in spring (Sept-Nov) but have high predicted fuel spend
and confirmed prior summer holiday activity.

Input:  submissions/sub_champion_v5.csv
Output: submissions/sub_champion_v5_reactivated.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    PROCESSED_DATA_DIR,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
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

    from sasol_customer_value.models.opportunity import apply_lapsed_fuel_reactivation

    sub = pd.read_csv(base_submission_path)
    X_test = get_or_build_test_features().reindex(sub[ID_COL].astype(str))

    sub_out = apply_lapsed_fuel_reactivation(sub=sub, X_test=X_test, top_k=top_k)

    print("=" * 80)
    print("APPLYING SUMMER ROAD-TRIP REACTIVATION")
    print(f"Base Submission:        {base_submission_path.name}")
    print(f"Target Flip Count (K):  {top_k}")
    print("=" * 80)

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
