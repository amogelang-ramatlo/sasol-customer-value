"""Snapshot construction: leak-free features + ground-truth labels for a cutoff.

A snapshot pairs features computed strictly before ``cutoff`` with labels from the
3-month window starting at ``cutoff``. Snapshots are cached as parquet in
``data/processed/snapshots/`` because label generation and feature extraction are
the slowest steps of every experiment.
"""

from pathlib import Path

import pandas as pd

from sasol_customer_value.config import ID_COL, PROCESSED_DATA_DIR
from sasol_customer_value.data.labels import (
    generate_raw_labels,
    load_label_config,
    normalize_targets,
    prepare_transactions_for_labels,
)
from sasol_customer_value.features.behavioral import extract_behavioral_features
from sasol_customer_value.features.rfm import extract_rfm_features

SNAPSHOT_DIR = PROCESSED_DATA_DIR / "snapshots"
FEATURE_VERSION = "v6"


def build_features(
    transactions_df: pd.DataFrame,
    cutoff: str,
    customer_ids: pd.Index,
) -> pd.DataFrame:
    """Full feature matrix (RFM + behavioural) for the given customers at cutoff."""
    ids = pd.Index(customer_ids).astype(str)
    rfm = extract_rfm_features(transactions_df, cutoff_date=cutoff, customer_ids=ids)
    beh = extract_behavioral_features(
        transactions_df, cutoff_date=cutoff, customer_ids=ids
    )
    X = pd.concat([rfm, beh], axis=1)
    X.index.name = ID_COL
    return X


def build_snapshot(
    transactions_df: pd.DataFrame,
    cutoff: str,
    prepared_df: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, y) for a labelled cutoff; y has ID, CLV_fuel, CLV_nonfuel, Opportunity.

    X is indexed by customer ID in the same order as y's rows.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    x_path = SNAPSHOT_DIR / f"X_{FEATURE_VERSION}_{cutoff}.parquet"
    y_path = SNAPSHOT_DIR / f"y_{cutoff}.parquet"

    if use_cache and x_path.exists() and y_path.exists():
        X = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)
        return X, y

    if y_path.exists() and use_cache:
        y = pd.read_parquet(y_path)
    else:
        prepared = (
            prepared_df
            if prepared_df is not None
            else prepare_transactions_for_labels(transactions_df)
        )
        config = load_label_config()
        raw = generate_raw_labels(prepared, cutoff, config)
        y = normalize_targets(raw, config)
        y[ID_COL] = y[ID_COL].astype(str)
        y.to_parquet(y_path, index=False)

    X = build_features(transactions_df, cutoff, pd.Index(y[ID_COL]))
    X.to_parquet(x_path)
    return X, y


def cached_paths(cutoff: str) -> tuple[Path, Path]:
    """Paths of the cached feature/label parquet files for a cutoff."""
    return (
        SNAPSHOT_DIR / f"X_{FEATURE_VERSION}_{cutoff}.parquet",
        SNAPSHOT_DIR / f"y_{cutoff}.parquet",
    )


def load_stacked_snapshots(
    cutoffs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and stack multiple cached snapshot parquets into unified training matrices.

    Args:
        cutoffs: List of snapshot cutoff date strings (e.g. ['2024-07-01', ...]).

    Returns:
        Tuple of (X_stacked, y_stacked) containing combined feature and label rows.
    """
    x_list: list[pd.DataFrame] = []
    y_list: list[pd.DataFrame] = []
    df_trans = None
    for c in cutoffs:
        xp, yp = cached_paths(c)
        if not xp.exists() or not yp.exists():
            if df_trans is None:
                from sasol_customer_value.data.loader import load_transactions

                df_trans = load_transactions()
            xc, yc = build_snapshot(df_trans, cutoff=c, use_cache=True)
            x_list.append(xc)
            y_list.append(yc)
        else:
            x_list.append(pd.read_parquet(xp))
            y_list.append(pd.read_parquet(yp))
    X_stacked = pd.concat(x_list, ignore_index=True)
    y_stacked = pd.concat(y_list, ignore_index=True)
    return X_stacked, y_stacked
