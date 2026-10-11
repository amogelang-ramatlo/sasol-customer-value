"""Unified Submission Assembler.

Assembles and validates production submissions from target specialists:
- --config best_zindi -> submissions/sub_model_best1.csv (Live Zindi Leaderboard Champion: 0.31076 - 0.31144)
- --config best_local -> submissions/sub_model_best2.csv (Local Out-Of-Time Validation Champion: 0.28926 / 0.27737)
"""

import argparse
from pathlib import Path

import pandas as pd
from train import train_production_models

from sasol_customer_value.config import (
    ID_COL,
    SUBMISSIONS_DIR,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)
from sasol_customer_value.data.loader import load_test_customers
from sasol_customer_value.utils.helpers import validate_submission_format


def assemble_submission(config_name: str, output_path: Path | None = None) -> Path:
    """Train specialists, assemble submission, and validate compliance."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        if config_name == "best_zindi":
            output_path = SUBMISSIONS_DIR / "sub_champion_zindi.csv"
        elif config_name == "best_local":
            output_path = SUBMISSIONS_DIR / "sub_champion_local.csv"
        else:
            output_path = SUBMISSIONS_DIR / f"sub_{config_name}.csv"

    print("=" * 80)
    print(f"ASSEMBLING SUBMISSION: {config_name.upper()} -> {output_path.name}")
    print("=" * 80)

    # 1. Train full-history specialists and gather test predictions
    preds = train_production_models(config_name=config_name, target="all")

    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))

    sub = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_FUEL: preds[TARGET_FUEL].reindex(test_ids).values,
            TARGET_NONFUEL: preds[TARGET_NONFUEL].reindex(test_ids).values,
            TARGET_OPPORTUNITY: preds[TARGET_OPPORTUNITY].reindex(test_ids).values,
        }
    )

    # 2. Automated 5-point submission compliance check
    validate_submission_format(sub, expected_ids=test_ids)
    sub.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Submission assembled and verified at: {output_path}")
    print("\n--- Diagnostic Scorecard ---")
    print(f"Rows:               {len(sub):,}")
    print(f"Mean CLV_fuel:      {sub['CLV_fuel'].mean():.4f} (std: {sub['CLV_fuel'].std():.4f})")
    print(f"Mean CLV_nonfuel:   {sub['CLV_nonfuel'].mean():.4f} (std: {sub['CLV_nonfuel'].std():.4f})")
    print("\nOpportunity Class Distribution (Top 8):")
    print(sub["Opportunity"].value_counts().head(8))
    print("=" * 80)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble production submissions")
    parser.add_argument(
        "--config",
        choices=["best_zindi", "best_local"],
        default="best_zindi",
        help="Model configuration to assemble (best_zindi -> sub_model_best1.csv, best_local -> sub_model_best2.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output CSV path",
    )
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    assemble_submission(config_name=args.config, output_path=out)
