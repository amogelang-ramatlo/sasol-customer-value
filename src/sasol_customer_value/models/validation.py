"""Authoritative Leak-Free Rolling-Origin Validation Harness with Strict 3-Month Outcome Embargo.

Design Principles:
1. Strict Embargo: To evaluate validation cutoff T, training snapshots must satisfy:
   snapshot_cutoff + 3 months <= T.
   This guarantees ZERO overlap between training outcome windows and validation outcome windows.
2. Multi-Fold Evaluation:
   - Fold 1: Val Cutoff 2025-03-01 (Train cutoffs <= 2024-12-01)
   - Fold 2: Val Cutoff 2025-06-01 (Train cutoffs <= 2025-03-01)
   - Fold 3: Val Cutoff 2025-09-01 (Train cutoffs <= 2025-06-01)
3. Evaluates and tracks:
   - CLV_fuel RMSE
   - CLV_nonfuel RMSE
   - Opportunity Weighted F1
   - Official Composite Evaluation Score:
     Overall = 0.60 + 0.40 * F1 - 0.40540558 * RMSE_fuel - 0.36764644 * RMSE_nonfuel
"""

from collections.abc import Generator
from dataclasses import dataclass

import pandas as pd

from sasol_customer_value.data.snapshots import cached_paths, load_stacked_snapshots

# All available monthly snapshot cutoffs in the released dataset
ALL_CUTOFFS: list[str] = [
    str(d.date()) for d in pd.date_range("2024-07-01", "2025-09-01", freq="MS")
]

# Benchmark validation cutoffs (quarterly evaluation points matching test window length)
DEFAULT_VAL_CUTOFFS: list[str] = ["2025-03-01", "2025-06-01", "2025-09-01"]


def get_embargoed_folds(
    all_cutoffs: list[str] = ALL_CUTOFFS,
    val_cutoffs: list[str] = DEFAULT_VAL_CUTOFFS,
    horizon_months: int = 3,
) -> Generator[tuple[list[str], str], None, None]:
    """Yield (train_cutoffs, val_cutoff) tuples enforcing strict 3-month outcome embargo."""
    for val_c in val_cutoffs:
        val_time = pd.Timestamp(val_c)
        train_cutoffs = [
            c
            for c in all_cutoffs
            if pd.Timestamp(c) + pd.DateOffset(months=horizon_months) <= val_time
        ]
        yield train_cutoffs, val_c


@dataclass
class FoldResult:
    val_cutoff: str
    n_train_rows: int
    n_val_rows: int
    rmse_fuel: float
    rmse_nonfuel: float
    f1_opportunity: float
    lb_score: float
    y_pred: pd.DataFrame


def load_fold_data(
    train_cutoffs: list[str], val_cutoff: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached feature and label parquets for the given train cutoffs and val cutoff."""
    X_train, y_train = load_stacked_snapshots(train_cutoffs)

    xp_val, yp_val = cached_paths(val_cutoff)
    if not xp_val.exists() or not yp_val.exists():
        raise FileNotFoundError(f"Missing cached validation snapshot for {val_cutoff}")
    X_val = pd.read_parquet(xp_val)
    y_val = pd.read_parquet(yp_val)

    return X_train, y_train, X_val, y_val


def summarize_validation_results(
    fold_results: list[FoldResult], model_name: str = "Model"
) -> pd.DataFrame:
    """Print and return a clean comparison summary of validation performance across all folds."""
    rows = []
    for f in fold_results:
        rows.append(
            {
                "Model": model_name,
                "Val Cutoff": f.val_cutoff,
                "Train N": f.n_train_rows,
                "Val N": f.n_val_rows,
                "Fuel RMSE": f.rmse_fuel,
                "Nonfuel RMSE": f.rmse_nonfuel,
                "Opp F1": f.f1_opportunity,
                "Composite Score": f.lb_score,
            }
        )

    df_summary = pd.DataFrame(rows)
    # Add mean row across folds
    mean_row = {
        "Model": model_name,
        "Val Cutoff": "MEAN",
        "Train N": int(df_summary["Train N"].mean()),
        "Val N": int(df_summary["Val N"].mean()),
        "Fuel RMSE": df_summary["Fuel RMSE"].mean(),
        "Nonfuel RMSE": df_summary["Nonfuel RMSE"].mean(),
        "Opp F1": df_summary["Opp F1"].mean(),
        "Composite Score": df_summary["Composite Score"].mean(),
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([mean_row])], ignore_index=True)
    return df_summary
