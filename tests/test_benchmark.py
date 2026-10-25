import pytest

from telco.benchmark import candidates
from telco.model import add_features, split
from test_model import toy


def test_every_candidate_fits_and_returns_valid_probabilities():
    Xtr, Xte, ytr, _ = split(add_features(toy(300)))
    for name, (pipe, space) in candidates(Xtr).items():
        p = pipe.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name
        assert all(k.startswith("clf__") for k in space), name


def test_candidate_set_has_the_requested_families():
    Xtr, *_ = split(add_features(toy(100)))
    assert {"XGBoost", "LightGBM", "random forest", "logistic regression"} <= set(candidates(Xtr))
