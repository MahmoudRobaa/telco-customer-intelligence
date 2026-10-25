import numpy as np
import pandas as pd
import pytest

from telco.model import (ADDONS, ID, TARGET, add_features, best_threshold,
                         lgbm_pipeline, logistic_pipeline, retention_value, split)


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        ID: [f"id{i}" for i in range(n)],
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        TARGET: rng.choice([0, 1], n, p=[0.73, 0.27]),
    })
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2)
    for c in ADDONS:
        df[c] = rng.choice(["Yes", "No", "No internet service"], n)
    return df


def test_split_is_disjoint_stratified_and_has_no_target_or_id():
    df = toy()
    Xtr, Xte, ytr, yte = split(df)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert len(Xtr) + len(Xte) == len(df)
    assert TARGET not in Xtr.columns and ID not in Xtr.columns
    assert abs(ytr.mean() - yte.mean()) < 0.02


def test_add_features_counts_addons_and_handles_tenure_zero():
    df = toy().iloc[:2].copy()
    df[ADDONS] = "No"
    df.loc[df.index[0], ["OnlineSecurity", "TechSupport"]] = "Yes"
    df["tenure"] = [10, 0]
    df["TotalCharges"] = [500.0, 0.0]
    df["MonthlyCharges"] = [60.0, 40.0]
    out = add_features(df)
    assert out["n_addons"].tolist() == [2, 0]
    assert out["monthly_vs_avg"].tolist() == [10.0, 0.0]  # 60 - 500/10 ; new customer -> 0
    assert not out[["n_addons", "monthly_vs_avg"]].isna().any().any()


@pytest.mark.parametrize("make", [logistic_pipeline, lgbm_pipeline])
def test_pipeline_fits_and_survives_unseen_category(make):
    df = add_features(toy())
    Xtr, Xte, ytr, _ = split(df)
    model = make(Xtr).fit(Xtr, ytr)
    Xte = Xte.copy()
    Xte.iloc[0, Xte.columns.get_loc("Contract")] = "Three year"  # never seen in training
    p = model.predict_proba(Xte)[:, 1]
    assert p.shape == (len(Xte),) and ((p >= 0) & (p <= 1)).all()


def test_retention_value_hand_computed():
    y = [1, 1, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1]
    m = [100, 100, 100, 100]
    # contact first and third: saved = 1 churner * 0.3 * 12 * 100 = 360; cost = 2 * 20
    assert retention_value(y, p, m, threshold=0.5) == pytest.approx(320.0)
    assert retention_value(y, p, m, threshold=1.01) == 0.0  # contact nobody


def test_best_threshold_ignores_customers_not_worth_the_cost():
    y = [1] * 10 + [0] * 90
    p = [0.9] * 10 + [0.1] * 90
    t, v = best_threshold(y, p, [100] * 100)
    assert 0.1 < t <= 0.9 and v == pytest.approx(10 * 0.3 * 12 * 100 - 10 * 20)
