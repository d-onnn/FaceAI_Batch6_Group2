"""Phase 3e: gray-box dT, kriging leave-one-out calibration, space-filling design."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from faceai import design as D
from faceai import graybox as G
from faceai import surrogate as S


def _beat(dia_at: float, dia_amp: float, dia_width: float = 0.05) -> np.ndarray:
    t = np.arange(0, 0.8, 0.001)
    g = lambda at, a, w: a * np.exp(-((t - at) ** 2) / (2 * w**2))
    return g(0.15, 1.0, 0.05) + g(dia_at, dia_amp, dia_width)


# ------------------------------------------------------------------------- gray box
def test_delta_t_finds_a_separate_diastolic_peak():
    dt, kind = G.delta_t(_beat(0.40, 0.5))
    assert kind == "peak"
    assert dt == pytest.approx(0.25, abs=0.01)


def test_delta_t_falls_back_to_the_inflection_when_there_is_no_second_peak():
    dt, kind = G.delta_t(_beat(0.27, 0.35, 0.07))  # a shoulder, not a peak
    assert kind == "inflection"
    assert 0.08 < dt < 0.25


def test_delta_t_grows_with_the_reflected_wave_delay():
    assert G.delta_t(_beat(0.35, 0.5))[0] < G.delta_t(_beat(0.45, 0.5))[0]


def test_rho_ci_and_band_difference_on_known_data():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(100), 2)
    a = rng.normal(size=200)
    r = G.rho_ci(a, 2 * a + rng.normal(0, 0.1, 200), subjects, rng)
    assert r["rho"] > 0.95 and r["ci"][0] > 0.9
    age = np.where(subjects < 20, 40.0, 20.0)
    v = np.where(age >= 25, 3.0, 1.0)
    b = G.band_difference(v, age, subjects, rng)
    assert b["older_minus_younger"] == pytest.approx(2.0)


# --------------------------------------------------------------------- kriging LOO
def test_closed_form_loo_matches_refitting_without_each_point():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (15, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, 15)
    kernel = 1.0 * RBF([1.0, 1.0]) + WhiteKernel(0.01)
    gp = GaussianProcessRegressor(kernel, optimizer=None, normalize_y=False).fit(X, y)
    z = S.loo_standardised_residuals(gp)
    brute = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        m = GaussianProcessRegressor(kernel, optimizer=None, normalize_y=False).fit(
            X[keep], y[keep]
        )
        mu, sd = m.predict(X[i : i + 1], return_std=True)
        brute.append((y[i] - mu[0]) / sd[0])
    np.testing.assert_allclose(z, brute, rtol=1e-4)


def test_error_table_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 8.0])
    m = S.errors_table(pred - y, y, pred)
    assert m["mae"] == pytest.approx(1.0)
    assert m["median_ae"] == pytest.approx(0.0)
    assert m["max_ae"] == pytest.approx(4.0)
    assert m["r2"] == pytest.approx(1 - 16 / 5)


# ----------------------------------------------------------------------- design
def test_kennard_stone_starts_with_the_farthest_pair_and_is_maximin():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(60, 3))
    order = D.kennard_stone(Z, 10)
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    assert set(order[:2]) == set(np.unravel_index(np.argmax(d), d.shape))
    for k in range(2, 10):  # each pick is the farthest from those already chosen
        nearest = d[:, order[:k]].min(axis=1)
        nearest[order[:k]] = -1
        assert order[k] == int(np.argmax(nearest))
    assert len(set(order)) == 10


def test_every_strategy_returns_n_distinct_subjects_from_the_pool():
    rng = np.random.default_rng(3)
    n_sub = 120
    pool = pd.DataFrame(
        {
            "patient_id": np.arange(1000, 1000 + n_sub),
            "fold": 0,
            "sex_m": rng.integers(0, 2, n_sub),
            **{c: rng.normal(size=n_sub) for c in D.TARGETS if c != "age"},
            "age": np.where(rng.random(n_sub) < 0.1, 50.0, 20.0),
        }
    )
    for strategy in D.STRATEGIES:
        chosen = D.select(strategy, pool, 30, rng)
        assert len(chosen) == len(set(chosen)) == 30, strategy
        assert set(chosen) <= set(pool["patient_id"]), strategy


def test_space_filling_reaches_the_extremes_random_choice_misses():
    rng = np.random.default_rng(4)
    n_sub = 300
    age = np.where(
        rng.random(n_sub) < 0.08, rng.uniform(40, 80, n_sub), rng.uniform(18, 24, n_sub)
    )
    pool = pd.DataFrame(
        {
            "patient_id": np.arange(n_sub),
            "fold": 0,
            "sex_m": rng.integers(0, 2, n_sub),
            **{c: rng.normal(size=n_sub) for c in D.TARGETS if c != "age"},
            "age": age,
        }
    )
    ks = D.select("space-filling", pool, 30, rng)
    assert pool.set_index("patient_id").loc[ks, "age"].max() == pytest.approx(age.max())
