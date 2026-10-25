import pandas as pd
import pytest

from telco.cleaning import clean


def raw(**overrides):
    base = {
        "customerID": ["a", "b", "c"],
        "tenure": [0, 5, 12],
        "TotalCharges": [" ", "100.5", "800"],
        "Churn": ["No", "Yes", "No"],
    }
    return pd.DataFrame({**base, **overrides})


def test_totalcharges_becomes_float_and_blank_new_customer_is_zero():
    out = clean(raw())
    assert out["TotalCharges"].dtype == float
    assert out["TotalCharges"].tolist() == [0.0, 100.5, 800.0]


def test_no_missing_values_remain():
    assert clean(raw()).isna().sum().sum() == 0


def test_blank_totalcharges_with_tenure_raises():
    with pytest.raises(ValueError, match="blank TotalCharges"):
        clean(raw(TotalCharges=[" ", " ", "800"]))


def test_churn_is_zero_one_and_unknown_label_raises():
    assert clean(raw())["Churn"].tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="Churn"):
        clean(raw(Churn=["No", "Maybe", "No"]))


def test_input_is_not_modified_and_rows_are_kept():
    df = raw()
    out = clean(df)
    assert df["Churn"].tolist() == ["No", "Yes", "No"]
    assert len(out) == len(df)
