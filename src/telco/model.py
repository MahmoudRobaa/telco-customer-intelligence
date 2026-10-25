import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID = "customerID"
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only: nothing is learned from the data, so this is safe before the split."""
    out = df.copy()
    out["n_addons"] = (out[ADDONS] == "Yes").sum(axis=1)
    avg_paid = (out["TotalCharges"] / out["tenure"].where(out["tenure"] > 0)).fillna(out["MonthlyCharges"])
    out["monthly_vs_avg"] = out["MonthlyCharges"] - avg_paid  # >0: paying more now than on average so far
    return out


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split. The test set is for one final evaluation only."""
    X = df.drop(columns=[TARGET, ID])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = list(X.select_dtypes(include="number").columns)
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def logistic_pipeline(X: pd.DataFrame, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X, scale=True)),
        ("clf", LogisticRegression(C=C, max_iter=3000)),
    ])


def lgbm_pipeline(X: pd.DataFrame, **params) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(X, scale=False)),
        ("clf", LGBMClassifier(random_state=42, verbose=-1, **params)),
    ])


def retention_value(y_true, proba, monthly, threshold, save_rate=0.3, cost=20.0, months=12) -> float:
    """Net value of contacting every customer with proba >= threshold.

    A contacted churner is saved with probability `save_rate`, worth `months` of
    their monthly charge. Every contact costs `cost`. All three numbers are
    assumptions, not facts from the data.
    """
    y_true, proba, monthly = map(np.asarray, (y_true, proba, monthly))
    contacted = proba >= threshold
    saved_value = (contacted & (y_true == 1)) * save_rate * months * monthly
    return float(saved_value.sum() - cost * contacted.sum())


def best_threshold(y_true, proba, monthly, **assumptions):
    """Threshold on the grid 0.01..0.99 that maximises retention_value."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    values = [retention_value(y_true, proba, monthly, t, **assumptions) for t in grid]
    i = int(np.argmax(values))
    return float(grid[i]), values[i]
