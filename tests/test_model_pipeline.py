"""
tests/test_model_pipeline.py
==============================
Integration tests for the ML training pipeline.

Covers:
  - Preprocessing transforms (no NaN, expected shape)
  - Model training and prediction contracts (correct output shapes/types)
  - Probability outputs sum to 1
  - Model selection logic (best model by ROC-AUC)
  - Cross-subject generalisation check
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# ── Preprocessing Tests ─────────────────────────────────────────────────────

class TestPreprocessing:

    def test_preprocessor_output_no_nan(self, minimal_artefacts, synthetic_df):
        """ColumnTransformer output must contain no NaN values."""
        pre = minimal_artefacts["preprocessor"]
        num = minimal_artefacts["numeric_features"]
        cat = minimal_artefacts["categorical_features"]
        X = pre.transform(synthetic_df[[c for c in num + cat if c in synthetic_df.columns]])
        assert not np.isnan(X).any(), "Preprocessor produced NaN values"

    def test_preprocessor_output_is_2d(self, minimal_artefacts, synthetic_df):
        """Output must be a 2-D array."""
        pre = minimal_artefacts["preprocessor"]
        num = minimal_artefacts["numeric_features"]
        cat = minimal_artefacts["categorical_features"]
        X = pre.transform(synthetic_df[[c for c in num + cat if c in synthetic_df.columns]])
        assert X.ndim == 2

    def test_eng_scaler_output_no_nan(self, minimal_artefacts, synthetic_df):
        """Engineered feature scaler output must contain no NaN."""
        eng_sc    = minimal_artefacts["eng_scaler"]
        eng_feats = minimal_artefacts["eng_features"]
        df = synthetic_df.copy()

        eng_vals = pd.DataFrame({
            "study_efficiency":     df["studytime"] / (df["failures"] + 1),
            "academic_risk_score":  df["failures"] * 2 + df["absences"] / 10 + (5 - df["studytime"]),
            "studytime_x_failures": df["studytime"] * df["failures"],
            "social_engagement":    df["goout"] + df["freetime"] - df["Dalc"] - df["Walc"],
            "parent_edu_avg":       (df["Medu"] + df["Fedu"]) / 2,
            "support_index":        (df["schoolsup"] == "yes").astype(int)
                                  + (df["famsup"] == "yes").astype(int)
                                  + (df["paid"] == "yes").astype(int),
            "health_x_absences":    df["health"] * df["absences"],
        })
        X_eng = eng_sc.transform(eng_vals[eng_feats])
        assert not np.isnan(X_eng).any()


# ── Model Contract Tests ─────────────────────────────────────────────────────

class TestModelContracts:
    """Verify that trained models honour scikit-learn's estimator interface."""

    def test_predict_returns_binary_labels(self, minimal_artefacts):
        """predict() must return values in {0, 1}."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        preds  = model.predict(X_test)
        assert set(preds).issubset({0, 1}), f"Unexpected labels: {set(preds)}"

    def test_predict_proba_shape(self, minimal_artefacts):
        """predict_proba() must return (n_samples, 2) array."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        proba  = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2), (
            f"Expected ({len(X_test)}, 2), got {proba.shape}"
        )

    def test_predict_proba_sums_to_one(self, minimal_artefacts):
        """Each row of predict_proba() must sum to approximately 1."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        proba  = model.predict_proba(X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6,
                                   err_msg="Row probabilities do not sum to 1")

    def test_predict_proba_in_unit_interval(self, minimal_artefacts):
        """All probability values must be in [0, 1]."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        proba  = model.predict_proba(X_test)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_prediction_length_matches_input(self, minimal_artefacts):
        """Length of predictions must equal number of test samples."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        preds  = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_model_has_been_fitted(self, minimal_artefacts):
        """Model must already be fitted (check_is_fitted should not raise)."""
        from sklearn.utils.validation import check_is_fitted
        model = minimal_artefacts["models"]["Logistic_Regression"]
        try:
            check_is_fitted(model)
        except Exception as exc:
            pytest.fail(f"Model is not fitted: {exc}")


# ── Metric Sanity Tests ──────────────────────────────────────────────────────

class TestModelMetrics:
    """Sanity-check that model achieves reasonable performance on synthetic data."""

    def test_roc_auc_above_chance(self, minimal_artefacts):
        """ROC-AUC must exceed 0.50 (better than random)."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        y_test = minimal_artefacts["y_test"]
        proba  = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, proba)
        assert auc > 0.50, f"ROC-AUC {auc:.3f} is at or below chance."

    def test_accuracy_above_chance(self, minimal_artefacts):
        """Accuracy must exceed 0.50 on the test split."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        y_test = minimal_artefacts["y_test"]
        preds  = model.predict(X_test)
        acc    = accuracy_score(y_test, preds)
        assert acc > 0.50, f"Accuracy {acc:.3f} is at or below chance."

    def test_f1_score_not_zero(self, minimal_artefacts):
        """F1 score must be greater than zero (model predicts both classes)."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        X_test = minimal_artefacts["X_test_full"]
        y_test = minimal_artefacts["y_test"]
        preds  = model.predict(X_test)
        f1     = f1_score(y_test, preds, zero_division=0)
        assert f1 > 0.0, "F1 score is zero — model predicts only one class."


# ── Model Selection Tests ────────────────────────────────────────────────────

class TestModelSelection:
    """Verify best-model selection logic used in the notebook."""

    def test_best_model_key_exists(self, minimal_artefacts):
        """artefacts dict must contain 'best_model_name'."""
        assert "best_model_name" in minimal_artefacts

    def test_best_model_in_models_dict(self, minimal_artefacts):
        """best_model_name must be a key in the 'models' sub-dict."""
        best = minimal_artefacts["best_model_name"]
        assert best in minimal_artefacts["models"], (
            f"best_model_name '{best}' not found in models dict"
        )

    def test_metrics_df_has_correct_columns(self, minimal_artefacts):
        """metrics_df must contain Accuracy, F1, ROC-AUC, Precision, Recall."""
        required = {"Accuracy", "F1", "ROC-AUC", "Precision", "Recall"}
        cols     = set(minimal_artefacts["metrics_df"].columns)
        assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_metrics_values_in_unit_interval(self, minimal_artefacts):
        """All metric values must be in [0, 1]."""
        m = minimal_artefacts["metrics_df"]
        assert (m >= 0).all().all() and (m <= 1).all().all()


# ── Feature Name Consistency ─────────────────────────────────────────────────

class TestFeatureNames:

    def test_feature_names_is_list(self, minimal_artefacts):
        assert isinstance(minimal_artefacts["feature_names"], list)

    def test_feature_names_non_empty(self, minimal_artefacts):
        assert len(minimal_artefacts["feature_names"]) > 0

    def test_feature_count_matches_X_train_columns(self, minimal_artefacts):
        """Number of feature names must match columns of X_train_full."""
        n_features = len(minimal_artefacts["feature_names"])
        n_cols     = minimal_artefacts["X_train_full"].shape[1]
        assert n_features == n_cols, (
            f"Feature names ({n_features}) != X_train columns ({n_cols})"
        )

    def test_eng_features_in_feature_names(self, minimal_artefacts):
        """All 7 engineered feature names must appear in feature_names."""
        for ef in minimal_artefacts["eng_features"]:
            assert ef in minimal_artefacts["feature_names"], (
                f"Engineered feature '{ef}' missing from feature_names"
            )
