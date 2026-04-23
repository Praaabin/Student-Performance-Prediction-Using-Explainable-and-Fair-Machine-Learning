"""
tests/test_data_loading.py
===========================
Tests for raw dataset integrity and schema validation.

Verifies that both CSV files load correctly, contain expected columns,
have no critical nulls, and respect documented value ranges.
"""

import pytest
import pandas as pd
import numpy as np


# Expected columns present in both datasets (UCI schema)
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3",
]

BINARY_COLS = ["schoolsup", "famsup", "paid", "activities",
               "nursery", "higher", "internet", "romantic"]


class TestMathDataset:
    """Validate the Mathematics (student-mat.csv) dataset."""

    def test_loads_successfully(self, raw_math_df):
        """Dataset loads into a non-empty DataFrame."""
        assert isinstance(raw_math_df, pd.DataFrame)
        assert len(raw_math_df) > 0

    def test_row_count(self, raw_math_df):
        """UCI documentation records 395 students in the Maths dataset."""
        assert len(raw_math_df) == 395, (
            f"Expected 395 rows, got {len(raw_math_df)}"
        )

    def test_expected_columns_present(self, raw_math_df):
        """All expected UCI schema columns must be present."""
        missing = [c for c in EXPECTED_COLUMNS if c not in raw_math_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_null_values(self, raw_math_df):
        """Raw data should have no missing values."""
        null_counts = raw_math_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"Unexpected nulls found:\n{cols_with_nulls}"
        )

    def test_grade_range(self, raw_math_df):
        """G1, G2, G3 grades must be in [0, 20] as per UCI docs."""
        for col in ["G1", "G2", "G3"]:
            assert raw_math_df[col].between(0, 20).all(), (
                f"Column {col} contains values outside [0, 20]"
            )

    def test_age_range(self, raw_math_df):
        """Student ages should be between 15 and 22."""
        assert raw_math_df["age"].between(15, 22).all(), (
            "age contains values outside [15, 22]"
        )

    def test_binary_columns(self, raw_math_df):
        """Binary yes/no columns must only contain 'yes' or 'no'."""
        for col in BINARY_COLS:
            if col in raw_math_df.columns:
                unique = set(raw_math_df[col].unique())
                assert unique <= {"yes", "no"}, (
                    f"Column {col} has unexpected values: {unique}"
                )

    def test_failures_range(self, raw_math_df):
        """Failures column must be in {0, 1, 2, 3} per UCI schema."""
        assert raw_math_df["failures"].isin([0, 1, 2, 3]).all(), (
            "failures column contains values outside {0,1,2,3}"
        )

    def test_sex_values(self, raw_math_df):
        """Sex column must only contain 'F' or 'M'."""
        assert set(raw_math_df["sex"].unique()) <= {"F", "M"}

    def test_absences_non_negative(self, raw_math_df):
        """Absences cannot be negative."""
        assert (raw_math_df["absences"] >= 0).all()


class TestPortugueseDataset:
    """Validate the Portuguese (student-por.csv) dataset."""

    def test_loads_successfully(self, raw_por_df):
        assert isinstance(raw_por_df, pd.DataFrame)
        assert len(raw_por_df) > 0

    def test_row_count(self, raw_por_df):
        """UCI documentation records 649 students in the Portuguese dataset."""
        assert len(raw_por_df) == 649, (
            f"Expected 649 rows, got {len(raw_por_df)}"
        )

    def test_expected_columns_present(self, raw_por_df):
        missing = [c for c in EXPECTED_COLUMNS if c not in raw_por_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_null_values(self, raw_por_df):
        null_counts = raw_por_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, f"Unexpected nulls:\n{cols_with_nulls}"

    def test_grade_range(self, raw_por_df):
        for col in ["G1", "G2", "G3"]:
            assert raw_por_df[col].between(0, 20).all(), (
                f"Column {col} contains values outside [0, 20]"
            )

    def test_binary_columns(self, raw_por_df):
        for col in BINARY_COLS:
            if col in raw_por_df.columns:
                unique = set(raw_por_df[col].unique())
                assert unique <= {"yes", "no"}, (
                    f"Column {col} has unexpected values: {unique}"
                )


class TestCombinedDataset:
    """Tests on the combined (Mathematics + Portuguese) merged dataset."""

    def test_combined_row_count(self, raw_math_df, raw_por_df):
        """Combined dataset should have 1044 rows (395 + 649)."""
        import pandas as pd
        mat = raw_math_df.copy()
        por = raw_por_df.copy()
        mat["subject"] = "mathematics"
        por["subject"] = "portuguese"
        combined = pd.concat([mat, por], ignore_index=True)
        assert len(combined) == 1044, (
            f"Expected 1044 combined rows, got {len(combined)}"
        )

    def test_target_label_creation(self, raw_math_df):
        """Pass/fail label derived from G3 >= 10 should be binary."""
        df = raw_math_df.copy()
        df["pass"] = (df["G3"] >= 10).astype(int)
        assert set(df["pass"].unique()) <= {0, 1}
        assert df["pass"].dtype in [np.int32, np.int64, int]

    def test_pass_rate_plausible(self, raw_math_df):
        """Overall pass rate should be between 40% and 85%."""
        pass_rate = (raw_math_df["G3"] >= 10).mean()
        assert 0.40 <= pass_rate <= 0.85, (
            f"Unusual pass rate: {pass_rate:.2%}"
        )
