"""Unified Submission Assembler.

Assembles and validates production submissions from target specialists:
- --config best_zindi -> submissions/sub_model_best1.csv (Live Zindi Leaderboard Champion: 0.31076 - 0.31144)
- --config best_local -> submissions/sub_model_best2.csv (Local Out-Of-Time Validation Champion: 0.28926 / 0.27737)
"""

import argparse
from pathlib import Path

import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    PROCESSED_DATA_DIR,
    PUBLIC_HISTORY_END_EXCLUSIVE,
    SUBMISSIONS_DIR,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
    ModelVariant,
)
from sasol_customer_value.data.loader import load_test_customers, load_transactions
from sasol_customer_value.models.opportunity import apply_lapsed_fuel_reactivation
from sasol_customer_value.models.pipeline import train_production_models
from sasol_customer_value.utils.helpers import validate_submission_format


def assemble_submission(
    config_name: ModelVariant, output_path: Path | None = None
) -> Path:
    """Train specialists, assemble submission, and validate compliance."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        if config_name in ("1", "champion_triple_crown"):
            output_path = SUBMISSIONS_DIR / "sub_champion_triple_crown.csv"
        elif config_name in ("2", "champion_v5_reactivated"):
            output_path = SUBMISSIONS_DIR / "sub_champion_v5_reactivated.csv"
        elif config_name == "best_zindi":
            output_path = SUBMISSIONS_DIR / "sub_champion_zindi.csv"
        elif config_name == "best_local":
            output_path = SUBMISSIONS_DIR / "sub_champion_local.csv"
        elif config_name == "champion_v2":
            output_path = SUBMISSIONS_DIR / "sub_champion_v2.csv"
        elif config_name == "champion_v3":
            output_path = SUBMISSIONS_DIR / "sub_champion_v3.csv"
        elif config_name == "champion_v4":
            output_path = SUBMISSIONS_DIR / "sub_champion_v4.csv"
        elif config_name == "champion_v5":
            output_path = SUBMISSIONS_DIR / "sub_champion_v5.csv"
        elif config_name == "champion_v6":
            output_path = SUBMISSIONS_DIR / "sub_champion_v6.csv"
        else:
            output_path = SUBMISSIONS_DIR / f"sub_{config_name}.csv"

    if config_name in ("2", "champion_v5_reactivated"):
        v5_path = SUBMISSIONS_DIR / "sub_champion_v5.csv"
        if not v5_path.exists():
            print("[INFO] Building base Champion V5 submission first...")
            assemble_submission(config_name="champion_v5", output_path=v5_path)

        cache_path = PROCESSED_DATA_DIR / "snapshots" / "X_test_2025-12-01.parquet"
        if cache_path.exists():
            X_test = pd.read_parquet(cache_path)
        else:
            from sasol_customer_value.data.snapshots import build_features

            df_trans = load_transactions()
            df_test = load_test_customers()
            test_ids = list(df_test[ID_COL].astype(str))
            X_test = build_features(
                df_trans,
                cutoff=PUBLIC_HISTORY_END_EXCLUSIVE,
                customer_ids=pd.Index(test_ids),
            ).reindex(test_ids)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            X_test.to_parquet(cache_path)

        sub_v5 = pd.read_csv(v5_path)
        sub_out = apply_lapsed_fuel_reactivation(sub=sub_v5, X_test=X_test, top_k=100)
        validate_submission_format(
            sub_out, expected_ids=list(sub_out[ID_COL].astype(str))
        )
        sub_out.to_csv(output_path, index=False)
        print(
            f"\n[SUCCESS] Selection 2 (V5 Reactivated) assembled and verified at: {output_path}"
        )
        print("\nOpportunity Class Distribution (Top 8):")
        print(sub_out[TARGET_OPPORTUNITY].value_counts().head(8))
        print("=" * 80)
        return output_path

    if config_name in ("1", "champion_triple_crown"):
        v14_act_path = SUBMISSIONS_DIR / "sub_v14_all_star_reactivated.csv"
        v5_path = SUBMISSIONS_DIR / "sub_champion_v5.csv"
        if not v14_act_path.exists():
            v14_base = SUBMISSIONS_DIR / "sub_v14_all_star_synthesis.csv"
            if v14_base.exists():
                cache_path = (
                    PROCESSED_DATA_DIR / "snapshots" / "X_test_2025-12-01.parquet"
                )
                X_test = pd.read_parquet(cache_path)
                df_v14 = pd.read_csv(v14_base)
                sub_v14_act = apply_lapsed_fuel_reactivation(
                    df_v14, X_test=X_test, top_k=100
                )
                sub_v14_act.to_csv(v14_act_path, index=False)
            else:
                assemble_submission(
                    config_name="champion_v5_reactivated", output_path=v14_act_path
                )

        if not v5_path.exists():
            assemble_submission(config_name="champion_v5", output_path=v5_path)

        sub_v14 = pd.read_csv(v14_act_path)
        sub_v5 = pd.read_csv(v5_path)

        sub_tc = pd.DataFrame(
            {
                ID_COL: sub_v14[ID_COL].values,
                TARGET_FUEL: sub_v14[TARGET_FUEL].values,
                TARGET_NONFUEL: sub_v5[TARGET_NONFUEL].values,
                TARGET_OPPORTUNITY: sub_v14[TARGET_OPPORTUNITY].values,
            }
        )
        validate_submission_format(
            sub_tc, expected_ids=list(sub_tc[ID_COL].astype(str))
        )
        sub_tc.to_csv(output_path, index=False)
        print(
            f"\n[SUCCESS] Selection 1 (Triple Crown) assembled and verified at: {output_path}"
        )
        print("\nOpportunity Class Distribution (Top 8):")
        print(sub_tc[TARGET_OPPORTUNITY].value_counts().head(8))
        print("=" * 80)
        return output_path

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
    print(
        f"Mean CLV_fuel:      {sub['CLV_fuel'].mean():.4f} (std: {sub['CLV_fuel'].std():.4f})"
    )
    print(
        f"Mean CLV_nonfuel:   {sub['CLV_nonfuel'].mean():.4f} (std: {sub['CLV_nonfuel'].std():.4f})"
    )
    print("\nOpportunity Class Distribution (Top 8):")
    print(sub["Opportunity"].value_counts().head(8))
    print("=" * 80)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble production submissions")
    parser.add_argument(
        "--config",
        choices=[
            "1",
            "2",
            "champion_triple_crown",
            "champion_v5_reactivated",
            "champion_v5",
            "champion_v6",
            "champion_v4",
            "champion_v3",
            "champion_v2",
            "best_zindi",
            "best_local",
        ],
        default="1",
        help="Model configuration to assemble: '1' / 'champion_triple_crown' (0.313971), '2' / 'champion_v5_reactivated' (0.313745)",
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
