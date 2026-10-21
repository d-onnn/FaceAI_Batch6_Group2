"""Phase 3c: each accuracy standard applied in its own form."""

from __future__ import annotations

import numpy as np
import pytest

from faceai import standards as S


def _criterion(target: str) -> dict:
    return next(c for c in S.CRITERIA if c["target"] == target)


def test_relative_share_uses_a_window_proportional_to_the_true_value():
    c = _criterion("hba1c")  # +-5 %
    y = np.array([5.0, 10.0])
    # Both are 0.4 points off: inside 5 % of 10 (0.5), outside 5 % of 5 (0.25).
    assert S.statistic(c, y + 0.4, y)["share_within"] == 0.5


def test_absolute_share_and_arms():
    rr = _criterion("respiratory_rate")
    y = np.array([16.0, 18.0, 20.0, 22.0])
    assert S.statistic(rr, y + np.array([0, 3, -3, 3.5]), y)["share_within"] == 0.75
    sp = _criterion("spo2")
    assert S.statistic(sp, y + np.array([3, -3, 3, -3]), y)["arms"] == pytest.approx(3)


def test_artery_grades_at_their_boundaries():
    c = _criterion("arterial_stiffness")
    grade = lambda m, sd: S.verdict(c, {"mean_difference": m, "sd_difference": sd})
    assert grade(0.5, 0.8) == "excellent"
    assert grade(-0.6, 0.8) == "acceptable"
    assert grade(0.9, 1.5) == "acceptable"
    assert grade(1.0, 1.0) == "poor"
    assert grade(0.0, 1.6) == "poor"


def test_mean_difference_is_reference_minus_test():
    c = _criterion("arterial_stiffness")
    y = np.array([8.0, 9.0, 10.0])
    assert S.statistic(c, y + 1, y)["mean_difference"] == pytest.approx(-1)


def test_implied_mae_matches_errors_that_just_meet_the_rule():
    # Unbiased normal errors with exactly the SD the NGSP rule allows: 90 % land within
    # +-5 % of the true value, and the MAE equals the implied MAE.
    c = _criterion("hba1c")
    rng = np.random.default_rng(0)
    y = rng.uniform(4.5, 7.0, 400_000)
    sd = c["tolerance"] / 1.6448536269514722  # z(0.95)
    pred = y + rng.normal(0, sd * y)
    assert S.statistic(c, pred, y)["share_within"] == pytest.approx(0.9, abs=0.002)
    imp = S.implied(c, y)
    assert np.mean(np.abs(pred - y)) == pytest.approx(imp["mae"], rel=0.01)
    r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)
    assert r2 == pytest.approx(imp["required_r2"], abs=0.005)


def test_rules_without_a_required_share_have_no_verdict_or_implied_mae():
    for target in ("respiratory_rate", "temperature"):
        c = _criterion(target)
        assert S.verdict(c, {"share_within": 1.0}) == "no pass rule"
        assert S.implied(c, np.array([1.0, 2.0])) is None


def test_required_r2_is_negative_when_a_constant_already_meets_the_rule():
    c = _criterion("spo2")  # ARMS <= 4 %
    y = np.array([96.0, 97.0, 98.0, 99.0])  # SD ~1.1, far below 4
    assert S.implied(c, y)["required_r2"] < 0
    assert S.verdict(c, S.statistic(c, np.full(4, y.mean()), y)) == "meets"
