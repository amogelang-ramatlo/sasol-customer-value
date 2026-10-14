"""Unit tests for Target Specialist models (FuelModel, NonfuelModel, OpportunityModel)."""

import numpy as np
import pandas as pd
import pytest

from sasol_customer_value.models.fuel import FuelModel
from sasol_customer_value.models.nonfuel import NonfuelModel
from sasol_customer_value.models.opportunity import OpportunityModel


@pytest.fixture
def mock_data():
    np.random.seed(42)
    n_rows = 50
    n_cols = 230
    X = pd.DataFrame(
        np.random.randn(n_rows, n_cols),
        columns=[f"feat_{i}" for i in range(n_cols)],
    )
    y_fuel = np.clip(np.random.exponential(scale=1.5, size=n_rows), 0, 5)
    y_nonfuel = np.clip(np.random.exponential(scale=0.8, size=n_rows), 0, 4)
    opp_classes = [
        "Inactivity",
        "Stable",
        "Existing-category growth: Fuel",
        "Existing-category growth: Beverages",
    ]
    y_opp = np.random.choice(opp_classes, size=n_rows)
    return X, y_fuel, y_nonfuel, y_opp


def test_fuel_model_fit_predict(mock_data):
    X, y_fuel, _, _ = mock_data
    model = FuelModel(variant="best_zindi", seed=42)
    model.fit(X, y_fuel)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0.0).all(), "Fuel predictions must be non-negative"


def test_nonfuel_model_fit_predict(mock_data):
    X, _, y_nonfuel, _ = mock_data
    model = NonfuelModel(variant="best_local", seeds=[42])
    model.fit(X, y_nonfuel)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 0.0).all(), "Nonfuel predictions must be non-negative"


def test_opportunity_model_fit_predict(mock_data):
    X, _, _, y_opp = mock_data
    model = OpportunityModel(variant="best_local")
    model.fit(X, y_opp)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert isinstance(preds[0], str)


def test_champion_v2_models_fit_predict(mock_data):
    X, y_fuel, y_nonfuel, y_opp = mock_data
    # Fast test with mock data for champion_v2
    fm = FuelModel(variant="champion_v2")
    fm.seeds = [42]  # single seed for fast test
    fm.fit(X, y_fuel)
    p_f = fm.predict(X)
    assert len(p_f) == len(X)
    assert (p_f >= 0.0).all()

    nfm = NonfuelModel(variant="champion_v2")
    nfm.seeds = [42]
    nfm.fit(X, y_nonfuel)
    p_nf = nfm.predict(X)
    assert len(p_nf) == len(X)
    assert (p_nf >= 0.0).all()

    om = OpportunityModel(variant="champion_v2")
    om.seeds_A = [42]
    om.seeds_B = [42]
    om.seeds_C = [42]
    om.fit(X, y_opp)
    p_opp = om.predict(X)
    assert len(p_opp) == len(X)
    assert isinstance(p_opp[0], str)


def test_champion_v3_models_fit_predict(mock_data):
    X, y_fuel, y_nonfuel, y_opp = mock_data
    # Add dummy category spend and purchase flag columns
    X_rich = X.copy()
    X_rich["cat_spend_last_90d_Beverages"] = np.random.uniform(0, 100, size=len(X))
    X_rich["cat_ever_bought_Beverages"] = np.random.choice([0, 1], size=len(X))

    # Fast test for champion_v3 with sample_weight on Fuel
    weights = np.ones(len(X), dtype=float)
    fm = FuelModel(variant="champion_v3")
    fm.seeds = [42]
    fm.fit(X_rich, y_fuel, sample_weight=weights)
    p_f = fm.predict(X_rich)
    assert len(p_f) == len(X)
    assert (p_f >= 0.0).all()

    nfm = NonfuelModel(variant="champion_v3")
    nfm.seeds = [42]
    nfm.fit(X_rich, y_nonfuel)
    p_nf = nfm.predict(X_rich)
    assert len(p_nf) == len(X)
    assert (p_nf >= 0.0).all()

    om = OpportunityModel(variant="champion_v3")
    om.seeds_A = [42]
    om.seeds_B = [42]
    om.seeds_C = [42]
    om.fit(X_rich, y_opp)
    p_opp = om.predict(X_rich)
    assert len(p_opp) == len(X)
    assert isinstance(p_opp[0], str)


def test_hurdles_and_masking():
    from sasol_customer_value.features.hurdles import (
        add_store_category_hurdles,
        apply_feasibility_mask,
        compute_snapshot_weights,
    )

    # 1. Test hurdle additions
    df = pd.DataFrame(
        {
            "cat_spend_last_90d_Beverages": [0.0, 50.0, 500.0],
            "cat_ever_bought_Beverages": [0, 1, 1],
        }
    )
    df_h = add_store_category_hurdles(df)
    assert "hurdle_rand_Beverages" in df_h.columns
    assert "is_saturated_Beverages" in df_h.columns
    assert np.isclose(df_h["hurdle_rand_Beverages"].iloc[0], 20.0)
    assert np.isclose(df_h["hurdle_rand_Beverages"].iloc[1], 20.0)
    assert np.isclose(df_h["hurdle_rand_Beverages"].iloc[2], 125.0)
    assert df_h["is_saturated_Beverages"].iloc[2] == 1.0

    # 2. Test feasibility masking
    scores = np.zeros((3, 2))
    classes = [
        "New category adoption: Beverages",
        "Existing-category growth: Beverages",
    ]
    masked = apply_feasibility_mask(scores, df, classes)
    # Customer 0: ever_bought=0 -> cannot grow -> growth score masked to -1e9
    assert masked[0, 1] < -1e8
    # Customer 1 & 2: ever_bought=1 -> cannot adopt -> adoption score masked to -1e9
    assert masked[1, 0] < -1e8
    assert masked[2, 0] < -1e8

    # 3. Test weights computation
    dates = ["2024-07-01", "2024-12-01", "2025-06-01"]
    w = compute_snapshot_weights(dates, target_month_day=(12, 1))
    assert len(w) == 3
    assert np.isclose(w.mean(), 1.0)
    # Dec 1st should have higher seasonal weight than July 1st
    assert w[1] > w[0]


def test_champion_v4_models_fit_predict(mock_data):
    X, y_fuel, y_nonfuel, y_opp = mock_data
    # Fast test for champion_v4 tri-family Fuel model
    fm = FuelModel(variant="champion_v4")
    fm.seeds = [42]
    fm.fit(X, y_fuel)
    p_f = fm.predict(X)
    assert len(p_f) == len(X)
    assert (p_f >= 0.0).all()

    # Fast test for champion_v4 Nonfuel model
    nfm = NonfuelModel(variant="champion_v4")
    nfm.seeds = [42]
    nfm.fit(X, y_nonfuel)
    p_nf = nfm.predict(X)
    assert len(p_nf) == len(X)
    assert (p_nf >= 0.0).all()

    # Fast test for champion_v4 Opportunity model
    om = OpportunityModel(variant="champion_v4")
    om.seeds_A = [42]
    om.seeds_B = [42]
    om.seeds_C = [42]
    om.fit(X, y_opp)
    p_opp = om.predict(X)
    assert len(p_opp) == len(X)
    assert isinstance(p_opp[0], str)


def test_champion_v5_models_fit_predict(mock_data):
    X, y_fuel, y_nonfuel, y_opp = mock_data
    # Fast test for champion_v5 tri-family Fuel model
    fm = FuelModel(variant="champion_v5")
    fm.seeds = [42]
    fm.fit(X, y_fuel)
    p_f = fm.predict(X)
    assert len(p_f) == len(X)
    assert (p_f >= 0.0).all()

    # Fast test for champion_v5 CatBoost-dominated Nonfuel model
    nfm = NonfuelModel(variant="champion_v5")
    nfm.seeds = [42]
    nfm.fit(X, y_nonfuel)
    p_nf = nfm.predict(X)
    assert len(p_nf) == len(X)
    assert (p_nf >= 0.0).all()

    # Fast test for champion_v5 Opportunity model
    om = OpportunityModel(variant="champion_v5")
    om.seeds_A = [42]
    om.seeds_B = [42]
    om.seeds_C = [42]
    om.fit(X, y_opp)
    p_opp = om.predict(X)
    assert len(p_opp) == len(X)
    assert isinstance(p_opp[0], str)


def test_champion_v6_models_fit_predict(mock_data):
    X, y_fuel, y_nonfuel, y_opp = mock_data
    # Fast test for champion_v6 Fuel model
    fm = FuelModel(variant="champion_v6")
    fm.seeds = [42]
    fm.fit(X, y_fuel)
    p_f = fm.predict(X)
    assert len(p_f) == len(X)
    assert (p_f >= 0.0).all()

    # Fast test for champion_v6 Nonfuel model
    nfm = NonfuelModel(variant="champion_v6")
    nfm.seeds = [42]
    nfm.fit(X, y_nonfuel)
    p_nf = nfm.predict(X)
    assert len(p_nf) == len(X)
    assert (p_nf >= 0.0).all()

    # Fast test for champion_v6 Opportunity model with CatBoost ensembling
    om = OpportunityModel(variant="champion_v6")
    om.seeds_A = [42]
    om.seeds_B = [42]
    om.seeds_C = [42]
    om.fit(X, y_opp)
    p_opp = om.predict(X)
    assert len(p_opp) == len(X)
    assert isinstance(p_opp[0], str)
    assert om.cb_model_B is not None
    assert om.cb_model_C is not None
