"""
conftest.py
===========
Shared pytest fixtures for the XAI Student Performance Prediction test suite.

These fixtures provide lightweight in-memory objects so tests run fast without
requiring the heavy results/artefacts.pkl file to be present.
"""

import os
import pickle
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
DATA_DIR      = os.path.join(ROOT, "data")
RESULTS_DIR   = os.path.join(ROOT, "results")
ARTEFACTS_PKL = os.path.join(RESULTS_DIR, "artefacts.pkl")


# ── Raw data fixture ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def raw_math_df():
    """Load the Mathematics dataset from data/student-mat.csv."""
    path = os.path.join(DATA_DIR, "student-mat.csv")
    if not os.path.exists(path):
        pytest.skip("student-mat.csv not found — skipping data-dependent tests.")
    return pd.read_csv(path, sep=";")


@pytest.fixture(scope="session")
def raw_por_df():
    """Load the Portuguese dataset from data/student-por.csv."""
    path = os.path.join(DATA_DIR, "student-por.csv")
    if not os.path.exists(path):
        pytest.skip("student-por.csv not found — skipping data-dependent tests.")
    return pd.read_csv(path, sep=";")


# ── Minimal synthetic dataset ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def synthetic_df():
    """
    A tiny, fully controlled DataFrame with the same schema as the UCI dataset.
    Used for unit tests that do not need real data.
    """
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "school":     np.random.choice(["GP", "MS"], n),
        "sex":        np.random.choice(["F", "M"], n),
        "age":        np.random.randint(15, 22, n),
        "address":    np.random.choice(["U", "R"], n),
        "famsize":    np.random.choice(["GT3", "LE3"], n),
        "Pstatus":    np.random.choice(["T", "A"], n),
        "Medu":       np.random.randint(0, 5, n),
        "Fedu":       np.random.randint(0, 5, n),
        "Mjob":       np.random.choice(["teacher", "health", "services", "at_home", "other"], n),
        "Fjob":       np.random.choice(["teacher", "health", "services", "at_home", "other"], n),
        "reason":     np.random.choice(["home", "reputation", "course", "other"], n),
        "guardian":   np.random.choice(["mother", "father", "other"], n),
        "traveltime": np.random.randint(1, 5, n),
        "studytime":  np.random.randint(1, 5, n),
        "failures":   np.random.randint(0, 4, n),
        "schoolsup":  np.random.choice(["yes", "no"], n),
        "famsup":     np.random.choice(["yes", "no"], n),
        "paid":       np.random.choice(["yes", "no"], n),
        "activities": np.random.choice(["yes", "no"], n),
        "nursery":    np.random.choice(["yes", "no"], n),
        "higher":     np.random.choice(["yes", "no"], n),
        "internet":   np.random.choice(["yes", "no"], n),
        "romantic":   np.random.choice(["yes", "no"], n),
        "famrel":     np.random.randint(1, 6, n),
        "freetime":   np.random.randint(1, 6, n),
        "goout":      np.random.randint(1, 6, n),
        "Dalc":       np.random.randint(1, 6, n),
        "Walc":       np.random.randint(1, 6, n),
        "health":     np.random.randint(1, 6, n),
        "absences":   np.random.randint(0, 30, n),
        "G3":         np.random.randint(0, 21, n),
        "subject":    np.random.choice(["mathematics", "portuguese"], n),
    })


# ── Minimal preprocessor + model fixture ───────────────────────────────────
@pytest.fixture(scope="session")
def minimal_artefacts(synthetic_df):
    """
    Build a tiny Logistic Regression pipeline on the synthetic dataset.
    Returns a dict with the same keys as the real artefacts.pkl so that
    dashboard utility functions can be tested without the full pipeline.
    """
    df = synthetic_df.copy()
    df["pass"] = (df["G3"] >= 10).astype(int)

    num_feats = ["age", "Medu", "Fedu", "traveltime", "studytime",
                 "failures", "famrel", "freetime", "goout",
                 "Dalc", "Walc", "health", "absences"]
    cat_feats = ["school", "sex", "address", "famsize", "Pstatus",
                 "Mjob", "Fjob", "reason", "guardian",
                 "schoolsup", "famsup", "paid", "activities",
                 "nursery", "higher", "internet", "romantic", "subject"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])

    eng_feats = [
        "study_efficiency", "academic_risk_score", "studytime_x_failures",
        "social_engagement", "parent_edu_avg", "support_index", "health_x_absences",
    ]

    def make_eng(d):
        return pd.DataFrame({
            "study_efficiency":     d["studytime"] / (d["failures"] + 1),
            "academic_risk_score":  d["failures"] * 2 + d["absences"] / 10 + (5 - d["studytime"]),
            "studytime_x_failures": d["studytime"] * d["failures"],
            "social_engagement":    d["goout"] + d["freetime"] - d["Dalc"] - d["Walc"],
            "parent_edu_avg":       (d["Medu"] + d["Fedu"]) / 2,
            "support_index":        (d["schoolsup"] == "yes").astype(int)
                                  + (d["famsup"] == "yes").astype(int)
                                  + (d["paid"] == "yes").astype(int),
            "health_x_absences":    d["health"] * d["absences"],
        })

    X_raw  = preprocessor.fit_transform(df[num_feats + cat_feats])
    eng_sc = StandardScaler()
    X_eng  = eng_sc.fit_transform(make_eng(df))
    X_full = np.hstack([X_raw, X_eng])
    y      = df["pass"].values

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)

    feature_names = (
        preprocessor.get_feature_names_out().tolist() + eng_feats
    )

    metrics_df = pd.DataFrame({
        "Accuracy":  [0.82],
        "F1":        [0.81],
        "ROC-AUC":   [0.88],
        "Precision": [0.80],
        "Recall":    [0.82],
    }, index=["Logistic_Regression"])

    return {
        "models":              {"Logistic_Regression": model},
        "best_model_name":     "Logistic_Regression",
        "preprocessor":        preprocessor,
        "eng_scaler":          eng_sc,
        "numeric_features":    num_feats,
        "categorical_features": cat_feats,
        "feature_names":       feature_names,
        "eng_features":        eng_feats,
        "metrics_df":          metrics_df,
        "X_train_full":        X_tr,
        "X_test_full":         X_te,
        "y_train":             y_tr,
        "y_test":              y_te,
    }
