"""Ground-truth label generator for Sasol customer value and opportunity targets.

Wraps and enforces the authoritative competition business logic specified in
docs/label_config.json and docs/LABEL_RULES.md.
"""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from sasol_customer_value.config import (
    ID_COL,
    LABEL_CONFIG_PATH,
    SCALE,
    TARGET_FUEL,
    TARGET_NONFUEL,
    TARGET_OPPORTUNITY,
)


def load_label_config(config_path: str | Path | None = None) -> dict:
    """Load authoritative label configuration JSON."""
    path = Path(config_path) if config_path is not None else LABEL_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_transactions_for_labels(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw transaction DataFrame with Decimal-scaled integer amounts.

    Ensures zero floating-point drift when determining winning categories and increments.
    """
    d = raw_df.copy()
    if "_time" not in d.columns:
        d["_time"] = pd.to_datetime(d.DateTimeZA, format="mixed", errors="raise")

    for name, key in [("Quantity", "_quantity"), ("TotalExclAmount", "_amount")]:
        if key not in d.columns:
            values = {}
            for v in d[name].dropna().unique():
                scaled = Decimal(str(v)) * SCALE
                if scaled != scaled.to_integral_value():
                    raise ValueError(
                        f"{name} has precision beyond six decimal places: {v}"
                    )
                values[v] = int(scaled)
            d[key] = d[name].map(values).fillna(0).astype("int64")
    return d


def get_eligible_ids(
    df: pd.DataFrame, cutoff: str, minimum_transactions: int = 3
) -> pd.Index:
    """Extract IDs of customers with at least minimum_transactions strictly before cutoff."""
    t = pd.Timestamp(cutoff)
    hist = df[df._time < t]
    counts = hist.groupby(ID_COL).TransactionId.nunique()
    return counts[counts >= minimum_transactions].index.sort_values()


def generate_raw_labels(
    prepared_df: pd.DataFrame,
    cutoff: str,
    config: dict | None = None,
) -> pd.DataFrame:
    """Generate un-normalized ground truth labels for a 3-month window starting at cutoff.

    Args:
        prepared_df: DataFrame with _time, _quantity, and _amount columns.
        cutoff: Snapshot timestamp string (e.g. '2025-09-01').
        config: Loaded label_config.json dict. Defaults to loading from config.

    Returns:
        DataFrame with columns: [ID, 'fuel_litres', 'nonfuel_rand', 'Opportunity',
                                 'winning_source_category', 'winning_increment_rand']
    """
    if config is None:
        config = load_label_config()

    t = pd.Timestamp(cutoff)
    end = t + pd.DateOffset(months=3)
    ids = get_eligible_ids(
        prepared_df, cutoff, config["minimum_historical_transactions"]
    )

    # Exclude adjustment items
    p = prepared_df[~prepared_df.ItemName.isin(config["adjustment_items"])]
    h = p[p._time < t]
    recent = h[h._time >= t - pd.DateOffset(months=3)]
    future = p[(p._time >= t) & (p._time < end)]

    mapping = config["category_mapping"]
    cats = sorted(c for c, v in mapping.items() if v["included"])
    unknown = set(p.ItemCategoryLevel2) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped source categories: {sorted(unknown)}")

    def matrix(frame: pd.DataFrame, count: bool = False) -> pd.DataFrame:
        frame = frame[frame.ItemCategoryLevel2.isin(cats)]
        if count:
            frame = frame[(frame._quantity > 0) & (frame._amount > 0)]
        g = frame.groupby([ID_COL, "ItemCategoryLevel2"])
        s = g.TransactionId.nunique() if count else g._amount.sum()
        return (
            s.unstack(fill_value=0)
            .reindex(index=ids, columns=cats, fill_value=0)
            .astype("int64")
        )

    historical = matrix(h, True)
    future_purchases = matrix(future, True)
    previous_spend = matrix(recent)
    future_spend = matrix(future)
    delta = future_spend - previous_spend

    # Growth criteria
    grow = historical.gt(0) & future_purchases.ge(config["growth_min_purchases"])
    grow &= delta.ge(int(Decimal(str(config["growth_min_rand"])) * SCALE))
    if config["growth_min_relative"] is not None:
        grow &= future_spend.gt(
            previous_spend.clip(lower=0) * (1 + config["growth_min_relative"])
        )

    # Adoption criteria
    adopt = (
        historical.eq(0)
        & future_purchases.ge(config["adoption_min_purchases"])
        & future_spend.gt(0)
    )
    threshold = int(Decimal(str(config["adoption_min_rand"])) * SCALE)
    adopt &= future_spend.ge(threshold)

    av = future_spend.where(adopt, 0).to_numpy()
    gv = delta.where(grow, 0).to_numpy()
    am, gm = av.max(axis=1), gv.max(axis=1)

    is_adopt = (am > 0) & (am >= gm)
    is_growth = (gm > 0) & ~is_adopt

    labels = np.full(len(ids), "Stable", dtype=object)
    winner = np.full(len(ids), "", dtype=object)
    increment = np.zeros(len(ids), dtype="int64")

    for mask, values, kind in [
        (is_adopt, av, "New category adoption"),
        (is_growth, gv, "Existing-category growth"),
    ]:
        cat = np.array(cats)[values.argmax(axis=1)]
        winner[mask] = cat[mask]
        increment[mask] = values.max(axis=1)[mask]
        labels[mask] = [kind + ": " + mapping[c][kind] for c in cat[mask]]

    # Inactivity: no qualifying purchases across any category in outcome window
    labels[future_purchases.sum(axis=1).eq(0)] = "Inactivity"

    # Fuel regression litres
    fuel = (
        future[
            future.ItemCategoryLevel2.eq("Fuel") & future.LineItemType.eq("Sale_Fuel")
        ]
        .groupby(ID_COL)
        ._quantity.sum()
        .reindex(ids, fill_value=0)
    )

    # Non-fuel regression rand (excludes Fuel)
    nonfuel = future_spend.drop(columns=["Fuel"]).sum(axis=1)

    result = pd.DataFrame(
        {
            ID_COL: ids,
            "fuel_litres": fuel.to_numpy() / SCALE,
            "nonfuel_rand": nonfuel.to_numpy() / SCALE,
            TARGET_OPPORTUNITY: labels,
            "winning_source_category": winner,
            "winning_increment_rand": increment / SCALE,
        }
    )

    if (result[["fuel_litres", "nonfuel_rand"]] < 0).any().any():
        raise ValueError(
            "Negative net customer target detected; requires explicit policy"
        )

    invalid = set(labels) - set(config["opportunity_labels"])
    if invalid:
        raise ValueError(f"Unexpected opportunity labels: {invalid}")

    return result


def normalize_targets(
    raw_labels_df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Normalize raw labels to match competition submission schema z = ln(1 + y) / s.

    Args:
        raw_labels_df: Output from generate_raw_labels.
        config: Label config containing 'normalization' constants.

    Returns:
        DataFrame with columns: [ID, CLV_fuel, CLV_nonfuel, Opportunity]
    """
    if config is None:
        config = load_label_config()

    out = raw_labels_df[[ID_COL]].copy()
    for target, source in [
        (TARGET_FUEL, "fuel_litres"),
        (TARGET_NONFUEL, "nonfuel_rand"),
    ]:
        s = config["normalization"][target]
        out[target] = np.log1p(raw_labels_df[source].to_numpy()) / s
    out[TARGET_OPPORTUNITY] = raw_labels_df[TARGET_OPPORTUNITY].to_numpy()
    return out


def generate_cutoff_labels(
    transactions_df: pd.DataFrame,
    cutoff: str,
    save_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both raw and normalized ground-truth labels for a given cutoff.

    Args:
        transactions_df: Loaded transaction DataFrame.
        cutoff: Snapshot cutoff date (e.g. '2025-09-01').
        save_path: Optional path to save normalized labels as parquet/csv.

    Returns:
        Tuple of (raw_labels_df, normalized_labels_df).
    """
    prepared = prepare_transactions_for_labels(transactions_df)
    config = load_label_config()
    raw = generate_raw_labels(prepared, cutoff, config)
    normalized = normalize_targets(raw, config)

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".parquet":
            normalized.to_parquet(p, index=False)
        else:
            normalized.to_csv(p, index=False)

    return raw, normalized
