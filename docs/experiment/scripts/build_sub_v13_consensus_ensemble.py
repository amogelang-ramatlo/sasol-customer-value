"""Standalone script to build Flagship Submission v13 (Consensus Ensemble).

Winning Synthesis:
1. Continuous Spend (CLV_fuel & CLV_nonfuel):
   Retains the 5-seed bagged multi-model regression from sub_v8, which proved to be
   our highest-scoring continuous spend on the Zindi Leaderboard (overall 0.30994).
2. Opportunity Classification:
   Committee consensus / majority vote combining our top 3 validated opportunity models:
   - sub_v8 (LB Opportunity: 0.538125 - #1 Performer)
   - sub_v4 (LB Opportunity: 0.536735 - #2 Performer)
   - sub_v9 (Shopper Personas Multi-Seed - #3 Performer)
   This eliminates soft-probability dilution (which hurt v12) while correcting
   isolated classification edge errors across the top models.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from sasol_customer_value.config import ID_COL, SUBMISSIONS_DIR
from sasol_customer_value.data.loader import load_test_customers
from sasol_customer_value.utils.helpers import validate_submission_format


def build_sub_v13(output_dir: Path = SUBMISSIONS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))

    path_v8 = output_dir / "sub_v8_multiseed_ensemble.csv"
    path_v4 = output_dir / "sub_v4_calibrated_ensemble.csv"
    path_v9 = output_dir / "sub_v9_shopper_personas.csv"

    for p in [path_v8, path_v4, path_v9]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required submission dependency at: {p}")

    df_v8 = pd.read_csv(path_v8)
    df_v4 = pd.read_csv(path_v4)
    df_v9 = pd.read_csv(path_v9)

    df_v8[ID_COL] = df_v8[ID_COL].astype(str)
    df_v4[ID_COL] = df_v4[ID_COL].astype(str)
    df_v9[ID_COL] = df_v9[ID_COL].astype(str)

    df_v8 = df_v8.set_index(ID_COL).reindex(test_ids).reset_index()
    df_v4 = df_v4.set_index(ID_COL).reindex(test_ids).reset_index()
    df_v9 = df_v9.set_index(ID_COL).reindex(test_ids).reset_index()

    # 1. Committee Majority Voting on Opportunity
    consensus_opp = []
    for i in range(len(test_ids)):
        c8 = df_v8.at[i, "Opportunity"]
        c4 = df_v4.at[i, "Opportunity"]
        c9 = df_v9.at[i, "Opportunity"]

        counts = Counter([c8, c4, c9])
        # If there is a 2-vote majority, pick it.
        # If all 3 disagree (a 3-way tie), default to c8 (the highest-scoring LB model)
        top_cand, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            consensus_opp.append(top_cand)
        else:
            consensus_opp.append(c8)

    # 2. Assemble Final Submission
    sub_v13 = pd.DataFrame(
        {
            ID_COL: test_ids,
            "CLV_fuel": df_v8["CLV_fuel"].values,
            "CLV_nonfuel": df_v8["CLV_nonfuel"].values,
            "Opportunity": consensus_opp,
        }
    )

    out_path = output_dir / "sub_v13_consensus_ensemble.csv"
    validate_submission_format(sub_v13, expected_ids=test_ids)
    sub_v13.to_csv(out_path, index=False)
    print(f"[SUCCESS] sub_v13 generated and verified at: {out_path}")

    print("\nOpportunity Class Distribution in sub_v13:")
    print(sub_v13["Opportunity"].value_counts())

    return out_path


if __name__ == "__main__":
    build_sub_v13()
