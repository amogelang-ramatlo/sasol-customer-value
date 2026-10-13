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
