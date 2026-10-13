"""Phase 3b: camera-timed sampling, the quality gate, clinical metrics and provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faceai import clinical
from faceai import features as FT
from faceai.robustness import gate_predictions


# ------------------------------------------------------------------ camera timing
def test_sampling_at_frames_reproduces_a_slow_wave():
    t_ppg = np.arange(0, 60, 0.01)
    v = np.sin(2 * np.pi * 1.2 * t_ppg)  # 72 bpm
    frames = np.arange(0, 60, 1 / 30)
    out = FT.sample_at_frames(t_ppg, v, frames)
    grid = np.arange(frames[0], frames[-1], 1 / FT.FS)
    assert len(out) == len(grid)
    assert np.max(np.abs(out - np.sin(2 * np.pi * 1.2 * grid))) < 0.03


def test_sampling_drops_frames_outside_the_signal_and_non_increasing_times():
    t_ppg = np.arange(0, 10, 0.01)
    v = t_ppg.copy()  # a ramp, so interpolation is exact
    frames = np.array([-1.0, 0.5, 1.0, 1.0, 0.9, 2.0, 11.0])
    out = FT.sample_at_frames(t_ppg, v, frames)
    grid = np.arange(0.5, 2.0, 1 / FT.FS)
    np.testing.assert_allclose(out, grid, atol=1e-9)


def test_camera_tables_cover_the_same_sessions_as_clean():
    if not FT.features_path(camera=FT.CAMERAS[0]).exists():
        pytest.skip("camera features not built - run `make features`")
    clean = FT.load_features()
    for cam in FT.CAMERAS:
        f = FT.load_features(camera=cam)
        assert f[["patient_id", "step"]].equals(clean[["patient_id", "step"]])
        assert f["readable"].equals(clean["readable"])
        ok = f[f["readable"]]
        assert (ok["systolic_frac"] < 0.5).all()  # same polarity flip as contact


# --------------------------------------------------------------------- quality gate
def test_gate_passes_good_quality_and_falls_back_otherwise():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    sqi = np.array([0.95, 0.80, np.nan, 0.90])
    gated, ok = gate_predictions(pred, sqi, tau=0.90, fallback=-1.0)
    np.testing.assert_array_equal(ok, [True, False, False, True])
    np.testing.assert_array_equal(gated, [1.0, -1.0, -1.0, 4.0])


# ------------------------------------------------------------------ clinical metrics
def test_auroc_ci_on_a_perfect_and_a_useless_score():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1] * 25)
    subjects = np.repeat(np.arange(50), 2)
    draws = clinical._subject_draws(subjects, rng)
    perfect = clinical.auroc_ci(y, y.astype(float), draws)
    assert perfect["auroc"] == 1.0 and perfect["ci"] == [1.0, 1.0]
    flat = clinical.auroc_ci(y, np.zeros(len(y)), draws)
    assert flat["auroc"] == 0.5


def test_subject_draws_keep_both_sessions_together():
    subjects = np.repeat(np.arange(30), 2)
    for rows in clinical._subject_draws(subjects, np.random.default_rng(1))[:20]:
        counts = pd.Series(subjects[rows]).value_counts()
        assert (counts % 2 == 0).all()


def test_change_rho_and_direction_ignore_signals_that_did_not_move():
    dy = np.array([1.0, 2.0, 3.0, -1.0, 0.0, 4.0])
    dx = np.array([0.5, 1.0, 1.5, 0.0, 0.2, 2.0])
    r = clinical._rho_ci(dx, dy, np.random.default_rng(0))
    assert r["signal_unchanged"] == 1
    assert r["label_moved"] == 5
    assert r["direction_agreement"] == 1.0  # the unmoved dx is excluded, not a miss


# ----------------------------------------------------------------------- provenance
def test_provenance_records_commit_split_and_inputs():
    from faceai.data.hub import ROOT
    from faceai.provenance import record

    p = record([ROOT / "configs" / "targets.yaml"])
    assert len(p["git_commit"]) == 40
    assert p["splits_assignment_sha256"].startswith("08f6fb63")
    assert list(p["inputs_sha256"]) == ["configs/targets.yaml"]


# ------------------------------------------------------------------ recording length
def test_shortened_recordings_cannot_give_respiration_under_40_s():
    nk = pytest.importorskip("neurokit2")
    x = np.asarray(
        nk.ppg_simulate(duration=180, sampling_rate=100, heart_rate=80, random_state=3)
    )
    short, longer = FT.extract(x[: 30 * 100]), FT.extract(x[: 60 * 100])
    assert np.isnan(short["resp_rate_fused"])  # 30 s < RESP_MIN_S, by design
    assert np.isfinite(short["hr_mean_bpm"])  # 40 beats at 80 bpm is enough for HR
    assert np.isfinite(longer["resp_rate_fused"])


def test_duration_tables_cover_the_same_sessions_as_clean():
    if not FT.features_path(duration_s=FT.DURATIONS_S[0]).exists():
        pytest.skip("duration features not built - run `make features`")
    clean = FT.load_features()
    for d in FT.DURATIONS_S:
        f = FT.load_features(duration_s=d)
        assert f[["patient_id", "step"]].equals(clean[["patient_id", "step"]])


# --------------------------------------------------------------------- explanations
def test_grouped_importance_finds_the_only_input_that_matters():
    from sklearn.linear_model import LinearRegression

    from faceai import explain

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({c: rng.normal(size=n) for c in FT.MODEL_FEATURES})
    X["age"], X["sex_m"], X["bmi"] = rng.normal(size=n), rng.integers(0, 2, n), 0.0
    y = 3.0 * X["apg_b_a"].to_numpy() + rng.normal(scale=0.1, size=n)
    est = LinearRegression().fit(X, y)
    imp = explain.grouped_importance(est, X, y, np.random.default_rng(1))
    assert max(imp, key=imp.get) == "pulse shape"
    assert all(abs(v) < 0.05 for g, v in imp.items() if g != "pulse shape")


def test_explanation_groups_partition_the_ppg_features():
    from faceai import explain

    cols = [c for g in explain.PPG_GROUPS for c in explain.GROUPS[g]]
    assert sorted(cols) == sorted(FT.MODEL_FEATURES)
    assert len(cols) == len(set(cols))
