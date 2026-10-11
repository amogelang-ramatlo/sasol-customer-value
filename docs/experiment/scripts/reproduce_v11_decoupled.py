"""Standalone reproduction script for Version 11 (Decoupled Hybrid Submission).

Combines:
1. CLV_fuel: Best seasonal regression from v3 (0.59720 RMSE)
2. CLV_nonfuel: Best seasonal regression from v3 (0.72776 RMSE)
3. Opportunity: Calibrated multi-seed classification from v8 / v9 (>51.5% F1)
"""

from pathlib import Path

import pandas as pd

from sasol_customer_value.config import ID_COL, SUBMISSIONS_DIR
from sasol_customer_value.data.loader import load_test_customers
from sasol_customer_value.utils.helpers import validate_submission_format


def build_v11_submissions(output_dir: Path = SUBMISSIONS_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_test = load_test_customers()
    test_ids = list(df_test[ID_COL].astype(str))

    path_v3 = output_dir / "sub_v3_seasonal.csv"
    path_v8 = output_dir / "sub_v8_multiseed_ensemble.csv"
    path_v9 = output_dir / "sub_v9_shopper_personas.csv"

    if not path_v3.exists():
        raise FileNotFoundError(f"Missing base seasonal submission at {path_v3}")
    if not path_v8.exists():
        raise FileNotFoundError(f"Missing base multiseed submission at {path_v8}")
    if not path_v9.exists():
        raise FileNotFoundError(f"Missing base persona submission at {path_v9}")

    v3 = pd.read_csv(path_v3)
    v8 = pd.read_csv(path_v8)
    v9 = pd.read_csv(path_v9)

    v3["ID"] = v3["ID"].astype(str)
    v8["ID"] = v8["ID"].astype(str)
    v9["ID"] = v9["ID"].astype(str)

    v3 = v3.set_index("ID").reindex(test_ids).reset_index()
    v8 = v8.set_index("ID").reindex(test_ids).reset_index()
    v9 = v9.set_index("ID").reindex(test_ids).reset_index()

    results = {}

    # 1. Option A: Pure v3 regression + v8 Opportunity
    p_11a = output_dir / "sub_v11_seasonal_reg_v8_opp.csv"
    sub_11a = pd.DataFrame(
        {
            "ID": test_ids,
            "CLV_fuel": v3["CLV_fuel"],
            "CLV_nonfuel": v3["CLV_nonfuel"],
            "Opportunity": v8["Opportunity"],
        }
    )
    validate_submission_format(sub_11a, expected_ids=test_ids)
    sub_11a.to_csv(p_11a, index=False)
    results["11a"] = p_11a

    # 2. Option B: Pure v3 regression + v9 Opportunity (shopper personas)
    p_11b = output_dir / "sub_v11_seasonal_reg_v9_opp.csv"
    sub_11b = pd.DataFrame(
        {
            "ID": test_ids,
            "CLV_fuel": v3["CLV_fuel"],
            "CLV_nonfuel": v3["CLV_nonfuel"],
            "Opportunity": v9["Opportunity"],
        }
    )
    validate_submission_format(sub_11b, expected_ids=test_ids)
    sub_11b.to_csv(p_11b, index=False)
    results["11b"] = p_11b

    # 3. Option C: 50/50 Continuous Blend (v3 + v8) + v9 Opportunity
    p_11c = output_dir / "sub_v11_blend_reg_v9_opp.csv"
    sub_11c = pd.DataFrame(
        {
            "ID": test_ids,
            "CLV_fuel": 0.5 * v3["CLV_fuel"] + 0.5 * v8["CLV_fuel"],
            "CLV_nonfuel": 0.5 * v3["CLV_nonfuel"] + 0.5 * v8["CLV_nonfuel"],
            "Opportunity": v9["Opportunity"],
        }
    )
    validate_submission_format(sub_11c, expected_ids=test_ids)
    sub_11c.to_csv(p_11c, index=False)
    results["11c"] = p_11c

    print(f"[REPRODUCED] All v11 submissions built and verified in: {output_dir}")
    return results


if __name__ == "__main__":
    build_v11_submissions()
