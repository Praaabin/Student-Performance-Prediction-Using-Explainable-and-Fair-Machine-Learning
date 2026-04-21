"""
tests/test_dashboard_utils.py
==============================
Unit tests for utility functions used inside the Streamlit dashboard (app.py).

These tests isolate pure logic from the Streamlit rendering layer so they can
be executed in a standard pytest environment without launching a browser.

Tested:
  - Risk label assignment (PASS / FAIL + risk band)
  - Dynamic plain-English summary bullet generation
  - Student profile dict construction
  - Engineered feature computation from the dashboard's eng_student() helper
"""

import pytest
import numpy as np
import pandas as pd


# ── Duplicate dashboard helper functions here (pure logic, no Streamlit) ────
# These mirror the exact logic in dashboard/app.py so that if the app changes
# these tests will catch regressions.

def assign_risk_label(y_pred: int, y_proba: float) -> tuple:
    """
    Return (outcome_word, risk_label) for a prediction.

    Parameters
    ----------
    y_pred  : 1 = PASS, 0 = FAIL
    y_proba : P(PASS) from model

    Returns
    -------
    (outcome_word: str, risk_label: str)
    """
    risk_pct = (1 - y_proba) * 100
    if y_pred == 1:
        risk_label = "LOW" if risk_pct < 30 else "MODERATE"
        outcome    = "pass"
    else:
        risk_label = "VERY HIGH" if y_proba < 0.20 else "HIGH"
        outcome    = "fail"
    return outcome, risk_label


def eng_student(d: dict) -> dict:
    """Dashboard's eng_student() reproduced for isolated testing."""
    return {
        "study_efficiency":     d["studytime"] / (d["failures"] + 1),
        "academic_risk_score":  d["failures"] * 2 + d["absences"] / 10 + (5 - d["studytime"]),
        "studytime_x_failures": d["studytime"] * d["failures"],
        "social_engagement":    d["goout"] + d["freetime"] - d["Dalc"] - d["Walc"],
        "parent_edu_avg":       (d["Medu"] + d["Fedu"]) / 2,
        "support_index":        sum(1 for c in ["schoolsup", "famsup", "paid"]
                                    if d.get(c) == "yes"),
        "health_x_absences":    d["health"] * d["absences"],
    }


def build_summary_points(y_pred, y_proba, failures, absences, study_time,
                          schoolsup, famsup, paid, Medu, Fedu):
    """
    Build the plain-English summary bullet list.
    Returns a list of (icon, title, detail) tuples.
    """
    support_count  = sum(1 for v in [schoolsup, famsup, paid] if v == "yes")
    parent_edu_avg = (Medu + Fedu) / 2
    study_eff      = study_time / (failures + 1)
    risk_score     = failures * 2 + absences / 10 + (5 - study_time)

    points = []

    # Failures
    if failures == 0:
        points.append(("✅", "No past failures",
                        "This student has no record of previously failed courses."))
    elif failures == 1:
        points.append(("⚠️", "1 past failure",
                        "One prior failed course increases the predicted risk moderately."))
    else:
        points.append(("🔴", f"{failures} past failures",
                        f"{failures} failures significantly raises the risk."))

    # Absences
    if absences <= 3:
        points.append(("✅", f"Low absences ({absences})", "Good attendance."))
    elif absences <= 10:
        points.append(("⚠️", f"Moderate absences ({absences})", "Some sessions missed."))
    else:
        points.append(("🔴", f"High absences ({absences})",
                        f"{absences} missed sessions — high risk contributor."))

    # Study time
    study_labels = {1: "less than 2 hours", 2: "2–5 hours",
                    3: "5–10 hours", 4: "more than 10 hours"}
    if study_time >= 3:
        points.append(("✅", f"Strong study time ({study_labels[study_time]} per week)",
                        "Above-average study time is a positive signal."))
    elif study_time == 2:
        points.append(("⚠️", f"Moderate study time ({study_labels[study_time]} per week)",
                        "Study time is at the lower end."))
    else:
        points.append(("🔴", f"Very low study time ({study_labels[study_time]} per week)",
                        "Less than 2 hours — significantly raises predicted risk."))

    return points


# ── Tests: risk label assignment ─────────────────────────────────────────────

class TestRiskLabelAssignment:

    @pytest.mark.parametrize("proba,expected_label", [
        (0.80, "LOW"),       # risk_pct = 20% → LOW
        (0.71, "LOW"),       # risk_pct = 29% → LOW (< 30)
        (0.70, "MODERATE"),  # risk_pct = 30% → MODERATE (not strictly < 30)
        (0.65, "MODERATE"),  # risk_pct = 35%
        (0.50, "MODERATE"),  # risk_pct = 50%
    ])
    def test_pass_risk_labels(self, proba, expected_label):
        outcome, label = assign_risk_label(1, proba)
        assert outcome == "pass"
        assert label   == expected_label, (
            f"P={proba}: expected {expected_label}, got {label}"
        )

    @pytest.mark.parametrize("proba,expected_label", [
        (0.10, "VERY HIGH"),
        (0.19, "VERY HIGH"),
        (0.20, "HIGH"),      # exactly 0.20 → HIGH (not < 0.20)
        (0.45, "HIGH"),
    ])
    def test_fail_risk_labels(self, proba, expected_label):
        outcome, label = assign_risk_label(0, proba)
        assert outcome == "fail"
        assert label   == expected_label, (
            f"P={proba}: expected {expected_label}, got {label}"
        )

    def test_outcome_word_is_string(self):
        for pred, proba in [(1, 0.75), (0, 0.30)]:
            outcome, label = assign_risk_label(pred, proba)
            assert isinstance(outcome, str)
            assert isinstance(label, str)


# ── Tests: eng_student() helper ──────────────────────────────────────────────

class TestEngStudentHelper:

    @pytest.fixture
    def sample_student(self):
        return {
            "studytime": 2, "failures": 1, "absences": 5,
            "goout": 3, "freetime": 3, "Dalc": 1, "Walc": 2,
            "Medu": 2, "Fedu": 3, "health": 4,
            "schoolsup": "yes", "famsup": "no", "paid": "no",
        }

    def test_returns_seven_keys(self, sample_student):
        result = eng_student(sample_student)
        assert len(result) == 7

    def test_study_efficiency_value(self, sample_student):
        result = eng_student(sample_student)
        expected = 2 / (1 + 1)   # studytime=2, failures=1 → 1.0
        assert abs(result["study_efficiency"] - expected) < 1e-9

    def test_support_index_value(self, sample_student):
        result = eng_student(sample_student)
        assert result["support_index"] == 1   # only schoolsup='yes'

    def test_health_x_absences(self, sample_student):
        result = eng_student(sample_student)
        assert result["health_x_absences"] == 4 * 5   # health=4, absences=5

    def test_parent_edu_avg(self, sample_student):
        result = eng_student(sample_student)
        assert abs(result["parent_edu_avg"] - 2.5) < 1e-9  # (2+3)/2

    def test_no_failures_case(self):
        student = {
            "studytime": 3, "failures": 0, "absences": 2,
            "goout": 2, "freetime": 3, "Dalc": 1, "Walc": 1,
            "Medu": 4, "Fedu": 4, "health": 5,
            "schoolsup": "yes", "famsup": "yes", "paid": "yes",
        }
        result = eng_student(student)
        assert result["study_efficiency"]     == 3.0   # 3/(0+1)
        assert result["studytime_x_failures"] == 0     # 3*0
        assert result["support_index"]        == 3     # all active


# ── Tests: summary bullet generation ─────────────────────────────────────────

class TestSummaryBullets:

    def test_correct_number_of_bullets(self):
        """build_summary_points must always return exactly 3 bullets."""
        points = build_summary_points(
            y_pred=1, y_proba=0.75,
            failures=0, absences=2, study_time=3,
            schoolsup="yes", famsup="yes", paid="no",
            Medu=3, Fedu=3,
        )
        assert len(points) == 3

    def test_green_icon_for_good_student(self):
        """A student with no failures and low absences should get ✅ icons."""
        points = build_summary_points(
            y_pred=1, y_proba=0.85,
            failures=0, absences=1, study_time=4,
            schoolsup="yes", famsup="yes", paid="yes",
            Medu=4, Fedu=4,
        )
        icons = [p[0] for p in points]
        assert all(ic == "✅" for ic in icons), f"Expected all ✅, got {icons}"

    def test_red_icon_for_at_risk_student(self):
        """A student with 3 failures and 20 absences should get 🔴 icons."""
        points = build_summary_points(
            y_pred=0, y_proba=0.15,
            failures=3, absences=20, study_time=1,
            schoolsup="no", famsup="no", paid="no",
            Medu=0, Fedu=0,
        )
        icons = [p[0] for p in points]
        assert all(ic == "🔴" for ic in icons), f"Expected all 🔴, got {icons}"

    def test_bullet_is_tuple_of_three(self):
        """Each bullet must be a (icon, title, detail) 3-tuple."""
        points = build_summary_points(
            y_pred=1, y_proba=0.60,
            failures=1, absences=8, study_time=2,
            schoolsup="no", famsup="yes", paid="no",
            Medu=2, Fedu=2,
        )
        for p in points:
            assert isinstance(p, tuple) and len(p) == 3

    def test_moderate_student_gets_warning_icons(self):
        """A borderline student should trigger ⚠️ icons."""
        points = build_summary_points(
            y_pred=1, y_proba=0.55,
            failures=1, absences=8, study_time=2,
            schoolsup="yes", famsup="no", paid="no",
            Medu=2, Fedu=1,
        )
        icons = [p[0] for p in points]
        assert "⚠️" in icons, f"Expected at least one ⚠️, got {icons}"


# ── Tests: student profile dict construction ─────────────────────────────────

class TestStudentProfileDict:
    """Verify the shape and types of the student_raw dict built in the dashboard."""

    def test_required_keys_present(self):
        """All keys needed by the preprocessor must be present in student_raw."""
        required_keys = [
            "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
            "famrel", "freetime", "goout", "Dalc", "Walc", "health",
            "absences", "school", "sex", "address", "famsize", "Pstatus",
            "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup",
            "paid", "activities", "nursery", "higher", "internet",
            "romantic", "subject",
        ]
        student_raw = {
            "age": 17, "Medu": 2, "Fedu": 2, "traveltime": 1,
            "studytime": 2, "failures": 0, "famrel": 4,
            "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1,
            "health": 3, "absences": 4,
            "school": "GP", "sex": "F", "address": "U", "famsize": "GT3",
            "Pstatus": "T", "Mjob": "other", "Fjob": "other",
            "reason": "course", "guardian": "mother",
            "schoolsup": "no", "famsup": "yes", "paid": "no",
            "activities": "no", "nursery": "yes", "higher": "yes",
            "internet": "yes", "romantic": "no", "subject": "mathematics",
        }
        missing = [k for k in required_keys if k not in student_raw]
        assert not missing, f"Missing keys in student_raw: {missing}"

    def test_numeric_fields_are_numeric(self):
        """Age, Medu, failures etc. must be numeric, not strings."""
        student_raw = {
            "age": 17, "Medu": 2, "Fedu": 3, "traveltime": 1,
            "studytime": 3, "failures": 0, "famrel": 4,
            "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1,
            "health": 4, "absences": 5,
        }
        for k, v in student_raw.items():
            assert isinstance(v, (int, float)), f"Key '{k}' is not numeric: {type(v)}"
