import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from telco.model import make_preprocessor


def candidates(X: pd.DataFrame, seed: int = 42) -> dict:
    """name -> (pipeline, search space). Every model gets the same preprocessing family and a tuning space."""
    def pipe(clf, scale):
        return Pipeline([("prep", make_preprocessor(X, scale=scale)), ("clf", clf)])

    return {
        "dummy (prior)": (pipe(DummyClassifier(strategy="prior"), False), {}),
        "logistic regression": (pipe(LogisticRegression(max_iter=3000), True), {"clf__C": loguniform(0.01, 10)}),
        "k-nearest neighbours": (pipe(KNeighborsClassifier(), True),
                                 {"clf__n_neighbors": randint(15, 100), "clf__weights": ["uniform", "distance"]}),
        "MLP (neural net)": (pipe(MLPClassifier(max_iter=500, early_stopping=True, random_state=seed), True),
                             {"clf__hidden_layer_sizes": [(32,), (64,), (32, 16)], "clf__alpha": loguniform(1e-4, 1e-1),
                              "clf__learning_rate_init": loguniform(1e-3, 1e-2)}),
        "random forest": (pipe(RandomForestClassifier(random_state=seed, n_jobs=-1), False),
                          {"clf__n_estimators": randint(200, 500), "clf__max_depth": [4, 6, 8, 12, None],
                           "clf__min_samples_leaf": randint(5, 40), "clf__max_features": ["sqrt", 0.3, 0.5]}),
        "extra trees": (pipe(ExtraTreesClassifier(random_state=seed, n_jobs=-1), False),
                        {"clf__n_estimators": randint(200, 500), "clf__max_depth": [4, 6, 8, 12, None],
                         "clf__min_samples_leaf": randint(5, 40), "clf__max_features": ["sqrt", 0.3, 0.5]}),
        "HistGradientBoosting (sklearn)": (pipe(HistGradientBoostingClassifier(random_state=seed), False),
                                           {"clf__learning_rate": loguniform(0.01, 0.2), "clf__max_leaf_nodes": randint(4, 32),
                                            "clf__min_samples_leaf": randint(20, 100), "clf__l2_regularization": loguniform(0.01, 10),
                                            "clf__max_iter": randint(80, 400)}),
        "XGBoost": (pipe(XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1), False),
                    {"clf__n_estimators": randint(80, 500), "clf__learning_rate": loguniform(0.01, 0.1), "clf__max_depth": randint(2, 6),
                     "clf__min_child_weight": randint(1, 10), "clf__subsample": uniform(0.6, 0.4),
                     "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(0.1, 30)}),
        "LightGBM": (pipe(LGBMClassifier(random_state=seed, verbose=-1), False),
                     {"clf__n_estimators": randint(80, 500), "clf__learning_rate": loguniform(0.01, 0.1), "clf__num_leaves": randint(4, 32),
                      "clf__min_child_samples": randint(20, 120), "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1],
                      "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(0.1, 30)}),
    }
