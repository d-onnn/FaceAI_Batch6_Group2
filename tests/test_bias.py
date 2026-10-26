"""Phase 3d: bias measures and de-biasing that never sees the test labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faceai import bias as B


def test_slope_is_minus_one_for_a_constant_and_zero_for_a_perfect_predictor():
    y = np.array([3.0, 5.0, 6.0, 9.0])
    assert B.slope(y, np.full(4, y.mean()) - y) == pytest.approx(-1)
    assert B.slope(y, np.zeros(4)) == pytest.approx(0)


def test_limits_of_agreement_are_mean_plus_minus_196_sd():
    err = np.array([1.0, -1.0, 2.0, 0.0])
    a = B.agreement(err, np.arange(4), np.random.default_rng(0))
    sd = np.std(err, ddof=1)
    assert a["bias"] == pytest.approx(0.5)
    assert a["loa"] == pytest.approx([0.5 - 1.96 * sd, 0.5 + 1.96 * sd])
    assert a["mae"] == pytest.approx(1.0)


def test_tails_report_the_mean_error_of_the_lowest_and_highest_tenth():
    y = np.arange(100.0)
    p = B.proportional(y, -0.5 * (y - y.mean()), [])
    assert p["low_n"] == p["high_n"] == 10
    assert p["low_bias"] > 0 > p["high_bias"]
    assert p["slope"] == pytest.approx(-0.5)


def test_slope_correction_removes_regression_to_the_mean():
    rng = np.random.default_rng(1)
    y = rng.normal(10, 3, 5000)
    shrunk = 7 + 0.3 * y + rng.normal(0, 0.2, y.size)  # follows 30 % of differences
    fit = B.fit_corrections(shrunk[:2500], y[:2500])
    fixed = B.correct(shrunk[2500:], fit, "slope")
    assert B.slope(y[2500:], shrunk[2500:] - y[2500:]) == pytest.approx(-0.7, abs=0.02)
    assert B.slope(y[2500:], fixed - y[2500:]) == pytest.approx(0, abs=0.03)


def test_slope_correction_worsens_mae_when_the_signal_is_weak():
    # A least-squares predictor from a noisy feature (R^2 = 0.2) regresses to the mean
    # because it is uncertain; stretching it back amplifies the feature's noise.
    rng = np.random.default_rng(3)
    y = rng.normal(10, 3, 6000)
    x = y + rng.normal(0, 6, y.size)
    k, c = np.polyfit(x[:3000], y[:3000], 1)
    pred = c + k * x
    fit = B.fit_corrections(pred[:3000], y[:3000])
    fixed = B.correct(pred[3000:], fit, "slope")
    before = np.abs(pred[3000:] - y[3000:]).mean()
    after = np.abs(fixed - y[3000:]).mean()
    assert fit["b"] == pytest.approx(0.2, abs=0.03)
    assert after > 1.8 * before


def test_slope_correction_is_skipped_when_there_is_no_signal_to_stretch():
    fit = {"offset": 0.0, "a": 5.0, "b": B.MIN_SLOPE, "alpha": 0.0, "beta": 0.0}
    pred = np.array([5.1, 4.9])
    assert np.array_equal(B.correct(pred, fit, "slope"), pred)


def test_valid_corrections_ignore_the_test_labels_and_the_invalid_one_does_not():
    fit = B.fit_corrections(
        np.array([1.0, 2.0, 2.5, 4.0]), np.array([1.0, 3.0, 2.0, 5.0])
    )
    pred = np.array([2.0, 3.0])
    for method in ("offset", "slope"):
        a = B.correct(pred, fit, method, np.array([0.0, 0.0]))
        b = B.correct(pred, fit, method, np.array([99.0, -99.0]))
        assert np.array_equal(a, b), method
    leaky = [
        B.correct(pred, fit, "uses test label", y)
        for y in (np.zeros(2), np.full(2, 9.0))
    ]
    assert not np.array_equal(*leaky)


def test_inner_fit_never_reads_the_outer_fold():
    rng = np.random.default_rng(2)
    n = 120
    t = pd.DataFrame(
        {
            "patient_id": np.repeat(np.arange(n // 2), 2),
            "fold": np.repeat(np.arange(n // 2) % 5, 2),
            "age": rng.uniform(18, 70, n),
            "sex_m": rng.integers(0, 2, n),
            "bmi": rng.uniform(17, 35, n),
        }
    )
    t["hba1c"] = 4.5 + 0.02 * t["age"] + rng.normal(0, 0.3, n)
    before = B._inner_fit(t, "hba1c", 0, "Demographics", {})
    scrambled = t.copy()
    scrambled.loc[scrambled["fold"] == 0, "hba1c"] = (
        99.0  # the outer test fold's labels
    )
    after = B._inner_fit(scrambled, "hba1c", 0, "Demographics", {})
    assert before == after
