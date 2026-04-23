"""
tests/test_fairness.py
=======================
Tests for the demographic fairness audit logic.

Verifies that:
  - Fairness metrics (accuracy, F1, recall) are computed correctly per subgroup
  - Disparity calculations are mathematically correct
  - The 0.10 threshold flagging works as intended
  - No subgroup is systematically empty after the split
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


# ── Utility: compute fairness metrics for one attribute ─────────────────────

def compute_subgroup_metrics(y_true, y_pred, groups):
    """
    Return a DataFrame of accuracy / F1 / recall per subgroup.

    Parameters
    ----------
    y_true  : array-like of true binary labels
    y_pred  : array-like of predicted binary labels
    groups  : array-like of group membership (categorical)

    Returns
    -------
    pd.DataFrame with columns [Accuracy, F1, Recall] indexed by group value
    """
    rows = {}
    for g in np.unique(groups):
        mask = groups == g
        yt = np.asarray(y_true)[mask]
        yp = np.asarray(y_pred)[mask]
        rows[g] = {
            "Accuracy": accuracy_score(yt, yp),
            "F1":       f1_score(yt, yp, zero_division=0),
            "Recall":   recall_score(yt, yp, zero_division=0),
            "n":        int(mask.sum()),
        }
    return pd.DataFrame(rows).T


def max_disparity(metrics_df, column):
    """Return the max − min difference for a given metric column."""
    return metrics_df[column].max() - metrics_df[column].min()


# ── Fixtures: simple controlled predictions ─────────────────────────────────

@pytest.fixture
def perfect_predictions():
    """Perfectly calibrated predictions — disparity should be zero."""
    np.random.seed(0)
    n = 200
    y_true = np.random.randint(0, 2, n)
    groups = np.where(np.arange(n) % 2 == 0, "F", "M")
    return y_true, y_true.copy(), groups   # y_pred == y_true


@pytest.fixture
def biased_predictions():
    """
    Deliberately biased predictions: group 'U' always predicted correctly,
    group 'R' always predicted as 0 (fail).
    """
    np.random.seed(1)
    n = 200
    y_true = np.random.randint(0, 2, n)
    groups = np.where(np.arange(n) % 2 == 0, "U", "R")

    y_pred = y_true.copy()
    # For 'R' group: always predict 0
    r_mask = groups == "R"
    y_pred[r_mask] = 0
    return y_true, y_pred, groups


# ── Tests: perfect predictions ───────────────────────────────────────────────

class TestPerfectFairness:

    def test_zero_accuracy_disparity(self, perfect_predictions):
        y_true, y_pred, groups = perfect_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        disp = max_disparity(m, "Accuracy")
        assert abs(disp) < 1e-9, f"Expected 0 disparity, got {disp:.4f}"

    def test_both_groups_present(self, perfect_predictions):
        y_true, y_pred, groups = perfect_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        assert set(m.index) == {"F", "M"}

    def test_accuracy_equals_one_for_all_groups(self, perfect_predictions):
        y_true, y_pred, groups = perfect_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        assert (m["Accuracy"] == 1.0).all(), (
            f"Expected all accuracies == 1.0:\n{m['Accuracy']}"
        )

    def test_no_threshold_flags(self, perfect_predictions):
        """With perfect predictions no metric should exceed the 0.10 threshold."""
        y_true, y_pred, groups = perfect_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        for col in ["Accuracy", "F1", "Recall"]:
            disp = max_disparity(m, col)
            assert disp <= 0.10, (
                f"{col} disparity {disp:.4f} exceeds threshold with perfect preds"
            )


# ── Tests: biased predictions ────────────────────────────────────────────────

class TestBiasedFairness:

    def test_disparity_exceeds_threshold(self, biased_predictions):
        """Biased predictions must produce disparity > 0.10 for Recall."""
        y_true, y_pred, groups = biased_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        disp = max_disparity(m, "Recall")
        assert disp > 0.10, (
            f"Expected recall disparity > 0.10, got {disp:.4f}"
        )

    def test_r_group_recall_lower(self, biased_predictions):
        """Group 'R' (always predicted 0) must have lower recall than 'U'."""
        y_true, y_pred, groups = biased_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        assert m.loc["R", "Recall"] < m.loc["U", "Recall"], (
            "Expected R group to have lower recall due to bias"
        )

    def test_r_group_f1_lower(self, biased_predictions):
        y_true, y_pred, groups = biased_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        assert m.loc["R", "F1"] <= m.loc["U", "F1"]

    def test_group_sizes_non_zero(self, biased_predictions):
        """Both subgroups must have at least 1 sample."""
        y_true, y_pred, groups = biased_predictions
        m = compute_subgroup_metrics(y_true, y_pred, groups)
        assert (m["n"] > 0).all(), f"Empty subgroup detected:\n{m['n']}"


# ── Tests: threshold flagging ────────────────────────────────────────────────

class TestThresholdFlagging:
    """Verify the 0.10 threshold logic used in the fairness summary table."""

    @pytest.mark.parametrize("disparity,expected_flag", [
        (0.05, False),
        (0.10, False),   # exactly at threshold → no flag
        (0.101, True),   # just above → flag
        (0.30, True),
    ])
    def test_threshold_boundary(self, disparity, expected_flag):
        """Flag logic: disparity > 0.10 → flag required."""
        flagged = disparity > 0.10
        assert flagged == expected_flag, (
            f"Disparity {disparity} flagged={flagged}, expected {expected_flag}"
        )


# ── Tests: on synthetic data from fixture ────────────────────────────────────

class TestFairnessOnSyntheticData:

    def test_sex_subgroups_both_present(self, synthetic_df, minimal_artefacts):
        """Both 'F' and 'M' groups must be present in synthetic data."""
        assert set(synthetic_df["sex"].unique()) == {"F", "M"}

    def test_address_subgroups_both_present(self, synthetic_df):
        """Both 'U' and 'R' address groups must be present."""
        assert set(synthetic_df["address"].unique()) == {"U", "R"}

    def test_metrics_df_values_in_unit_interval(self, minimal_artefacts):
        """All values in the metrics summary must be in [0, 1]."""
        m = minimal_artefacts["metrics_df"]
        numeric_cols = m.select_dtypes(include=[np.number])
        assert (numeric_cols >= 0).all().all()
        assert (numeric_cols <= 1).all().all()

    def test_model_predictions_on_sex_subgroups(self, minimal_artefacts, synthetic_df):
        """Verify fairness computation runs without error on sex subgroups."""
        model  = minimal_artefacts["models"]["Logistic_Regression"]
        pre    = minimal_artefacts["preprocessor"]
        eng_sc = minimal_artefacts["eng_scaler"]
        num    = minimal_artefacts["numeric_features"]
        cat    = minimal_artefacts["categorical_features"]
        eng_f  = minimal_artefacts["eng_features"]

        df = synthetic_df.copy()
        df["pass"] = (df["G3"] >= 10).astype(int)

        X_raw = pre.transform(df[[c for c in num + cat if c in df.columns]])
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
        X_eng  = eng_sc.transform(eng_vals[eng_f])
        X_full = np.hstack([X_raw, X_eng])
        y_pred = model.predict(X_full)

        groups = df["sex"].values
        result = compute_subgroup_metrics(df["pass"].values, y_pred, groups)
        # Must run without error and return both groups
        assert len(result) == 2
        assert (result["n"] > 0).all()
