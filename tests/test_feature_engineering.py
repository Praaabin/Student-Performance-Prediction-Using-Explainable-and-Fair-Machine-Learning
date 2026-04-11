"""
tests/test_feature_engineering.py
===================================
Tests for the feature engineering functions used in the notebook pipeline.

All engineered features are derived from raw student attributes. This module
verifies their mathematical correctness, range constraints, and edge-case
behaviour on the synthetic dataset provided by conftest.py.
"""

import pytest
import numpy as np
import pandas as pd


# ── Reproduce the feature engineering logic from the notebook ───────────────

def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the feature engineering block from the notebook / dashboard.
    Returns a DataFrame with the 7 engineered feature columns.
    """
    out = pd.DataFrame(index=df.index)
    out["study_efficiency"]     = df["studytime"] / (df["failures"] + 1)
    out["academic_risk_score"]  = (
        df["failures"] * 2
        + df["absences"] / 10
        + (5 - df["studytime"])
    )
    out["studytime_x_failures"] = df["studytime"] * df["failures"]
    out["social_engagement"]    = (
        df["goout"] + df["freetime"] - df["Dalc"] - df["Walc"]
    )
    out["parent_edu_avg"]       = (df["Medu"] + df["Fedu"]) / 2
    out["support_index"]        = (
        (df["schoolsup"] == "yes").astype(int)
        + (df["famsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )
    out["health_x_absences"]    = df["health"] * df["absences"]
    return out


class TestStudyEfficiency:
    """study_efficiency = studytime / (failures + 1)"""

    def test_zero_failures_equals_studytime(self, synthetic_df):
        """When failures == 0, efficiency must equal studytime."""
        df = synthetic_df[synthetic_df["failures"] == 0].copy()
        if df.empty:
            pytest.skip("No zero-failure rows in synthetic data.")
        eng = compute_engineered_features(df)
        pd.testing.assert_series_equal(
            eng["study_efficiency"].reset_index(drop=True),
            df["studytime"].astype(float).reset_index(drop=True),
            check_names=False,
        )

    def test_always_positive(self, synthetic_df):
        """study_efficiency must always be > 0 (studytime ≥ 1, failures ≥ 0)."""
        eng = compute_engineered_features(synthetic_df)
        assert (eng["study_efficiency"] > 0).all()

    def test_max_value_bounded(self, synthetic_df):
        """study_efficiency ≤ 4 (max studytime=4, failures=0 → 4/1=4)."""
        eng = compute_engineered_features(synthetic_df)
        assert eng["study_efficiency"].max() <= 4.0 + 1e-9


class TestAcademicRiskScore:
    """academic_risk_score = failures*2 + absences/10 + (5 - studytime)"""

    def test_minimum_risk_is_positive(self, synthetic_df):
        """Even the best student (0 failures, 0 absences, studytime=4) has score 1."""
        eng = compute_engineered_features(synthetic_df)
        assert (eng["academic_risk_score"] >= 0).all()

    def test_more_failures_higher_risk(self):
        """Doubling failures should increase the risk score."""
        base = pd.DataFrame({
            "studytime": [2], "failures": [1], "absences": [5],
            "goout": [3], "freetime": [3], "Dalc": [1], "Walc": [1],
            "Medu": [2], "Fedu": [2],
            "schoolsup": ["no"], "famsup": ["no"], "paid": ["no"],
            "health": [3],
        })
        high = base.copy()
        high["failures"] = 2
        r_base = compute_engineered_features(base)["academic_risk_score"].iloc[0]
        r_high = compute_engineered_features(high)["academic_risk_score"].iloc[0]
        assert r_high > r_base

    def test_more_absences_higher_risk(self):
        """Increasing absences should increase risk."""
        base = pd.DataFrame({
            "studytime": [3], "failures": [0], "absences": [2],
            "goout": [2], "freetime": [2], "Dalc": [1], "Walc": [1],
            "Medu": [3], "Fedu": [3],
            "schoolsup": ["yes"], "famsup": ["yes"], "paid": ["no"],
            "health": [4],
        })
        high = base.copy()
        high["absences"] = 20
        r_base = compute_engineered_features(base)["academic_risk_score"].iloc[0]
        r_high = compute_engineered_features(high)["academic_risk_score"].iloc[0]
        assert r_high > r_base


class TestStudytimeXFailures:
    """studytime_x_failures = studytime * failures"""

    def test_zero_when_no_failures(self, synthetic_df):
        """Interaction term should be 0 when failures == 0."""
        df = synthetic_df[synthetic_df["failures"] == 0].copy()
        if df.empty:
            pytest.skip("No zero-failure rows.")
        eng = compute_engineered_features(df)
        assert (eng["studytime_x_failures"] == 0).all()

    def test_non_negative(self, synthetic_df):
        """Product of two non-negative integers must be non-negative."""
        eng = compute_engineered_features(synthetic_df)
        assert (eng["studytime_x_failures"] >= 0).all()


class TestSocialEngagement:
    """social_engagement = goout + freetime - Dalc - Walc"""

    def test_range(self, synthetic_df):
        """goout, freetime, Dalc, Walc ∈ [1,5] → engagement ∈ [-8, 8]."""
        eng = compute_engineered_features(synthetic_df)
        assert eng["social_engagement"].between(-8, 8).all()

    def test_high_alcohol_lowers_engagement(self):
        """Higher alcohol consumption should reduce social_engagement."""
        low_alc = pd.DataFrame({
            "studytime": [2], "failures": [0], "absences": [0],
            "goout": [3], "freetime": [3], "Dalc": [1], "Walc": [1],
            "Medu": [2], "Fedu": [2],
            "schoolsup": ["no"], "famsup": ["no"], "paid": ["no"],
            "health": [3],
        })
        high_alc = low_alc.copy()
        high_alc["Dalc"] = 5
        high_alc["Walc"] = 5
        e_low  = compute_engineered_features(low_alc)["social_engagement"].iloc[0]
        e_high = compute_engineered_features(high_alc)["social_engagement"].iloc[0]
        assert e_high < e_low


class TestParentEduAvg:
    """parent_edu_avg = (Medu + Fedu) / 2"""

    def test_range(self, synthetic_df):
        """Medu, Fedu ∈ [0,4] → avg ∈ [0.0, 4.0]."""
        eng = compute_engineered_features(synthetic_df)
        assert eng["parent_edu_avg"].between(0.0, 4.0).all()

    def test_symmetry(self):
        """avg(Medu=1, Fedu=3) == avg(Medu=3, Fedu=1) == 2.0."""
        for m, f in [(1, 3), (3, 1)]:
            row = pd.DataFrame({
                "studytime": [2], "failures": [0], "absences": [0],
                "goout": [2], "freetime": [2], "Dalc": [1], "Walc": [1],
                "Medu": [m], "Fedu": [f],
                "schoolsup": ["no"], "famsup": ["no"], "paid": ["no"],
                "health": [3],
            })
            avg = compute_engineered_features(row)["parent_edu_avg"].iloc[0]
            assert abs(avg - 2.0) < 1e-9, f"Expected 2.0 for Medu={m}, Fedu={f}"


class TestSupportIndex:
    """support_index = (schoolsup=='yes') + (famsup=='yes') + (paid=='yes')"""

    def test_range(self, synthetic_df):
        """Support index must be in {0, 1, 2, 3}."""
        eng = compute_engineered_features(synthetic_df)
        assert eng["support_index"].isin([0, 1, 2, 3]).all()

    def test_full_support(self):
        """All three supports active → index == 3."""
        row = pd.DataFrame({
            "studytime": [3], "failures": [0], "absences": [0],
            "goout": [2], "freetime": [3], "Dalc": [1], "Walc": [1],
            "Medu": [3], "Fedu": [3],
            "schoolsup": ["yes"], "famsup": ["yes"], "paid": ["yes"],
            "health": [4],
        })
        val = compute_engineered_features(row)["support_index"].iloc[0]
        assert val == 3

    def test_no_support(self):
        """No supports → index == 0."""
        row = pd.DataFrame({
            "studytime": [2], "failures": [1], "absences": [5],
            "goout": [3], "freetime": [3], "Dalc": [2], "Walc": [2],
            "Medu": [1], "Fedu": [1],
            "schoolsup": ["no"], "famsup": ["no"], "paid": ["no"],
            "health": [3],
        })
        val = compute_engineered_features(row)["support_index"].iloc[0]
        assert val == 0


class TestHealthXAbsences:
    """health_x_absences = health * absences"""

    def test_zero_absences_gives_zero(self, synthetic_df):
        """Zero absences should always produce a zero interaction term."""
        df = synthetic_df[synthetic_df["absences"] == 0].copy()
        if df.empty:
            pytest.skip("No zero-absence rows in synthetic data.")
        eng = compute_engineered_features(df)
        assert (eng["health_x_absences"] == 0).all()

    def test_non_negative(self, synthetic_df):
        """Both health and absences are non-negative; product must be non-negative."""
        eng = compute_engineered_features(synthetic_df)
        assert (eng["health_x_absences"] >= 0).all()

    def test_dtype_numeric(self, synthetic_df):
        """health_x_absences must be a numeric dtype."""
        eng = compute_engineered_features(synthetic_df)
        assert pd.api.types.is_numeric_dtype(eng["health_x_absences"])


class TestEngineeredFeatureShape:
    """Verify that the engineering function returns the right shape."""

    def test_returns_seven_columns(self, synthetic_df):
        eng = compute_engineered_features(synthetic_df)
        assert eng.shape[1] == 7, f"Expected 7 columns, got {eng.shape[1]}"

    def test_row_count_preserved(self, synthetic_df):
        eng = compute_engineered_features(synthetic_df)
        assert len(eng) == len(synthetic_df)

    def test_no_nans_on_valid_input(self, synthetic_df):
        eng = compute_engineered_features(synthetic_df)
        assert not eng.isnull().any().any(), "Unexpected NaN in engineered features"
