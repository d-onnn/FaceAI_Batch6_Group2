"""Phase 3 feature extraction: correctness on synthetic pulses, then the real artifact."""

from __future__ import annotations

import numpy as np
import pytest

from faceai import features as FT


@pytest.fixture(scope="module")
def pulse():
    nk = pytest.importorskip("neurokit2")
    return np.asarray(
        nk.ppg_simulate(duration=180, sampling_rate=100, heart_rate=72, random_state=42)
    )


def test_heart_rate_recovered(pulse):
    f = FT.extract(pulse)
    assert abs(f["hr_mean_bpm"] - 72) < 2
    # The Welch peak is the coarser estimator: 0.94 bpm bins, and the simulator adds
    # beat-to-beat variability (on real dev data it sits 2.2 bpm MAE from the IBI mean).
    assert abs(f["hr_spectral_bpm"] - 72) < 3


def test_every_model_feature_is_finite_on_a_clean_pulse(pulse):
    f = FT.extract(pulse)
    bad = [k for k in FT.MODEL_FEATURES if not np.isfinite(f[k])]
    assert not bad, bad


def test_physiological_polarity_and_shape(pulse):
    f = FT.extract(pulse)
    assert f["systolic_frac"] < 0.5  # fast upstroke: the peak sits early in the beat
    assert 0.05 < f["rise_time_s"] < 0.35
    assert f["apg_b_a"] < 0  # b wave is negative


def test_inverting_the_pulse_flips_the_shape(pulse):
    """Why load_session_ppg inverts the raw .PW: shape features depend on polarity."""
    assert FT.extract(-pulse)["systolic_frac"] > FT.extract(pulse)["systolic_frac"]


def test_too_short_signal_gives_nan_not_garbage():
    nk = pytest.importorskip("neurokit2")
    x = np.asarray(
        nk.ppg_simulate(duration=10, sampling_rate=100, heart_rate=72, random_state=1)
    )
    f = FT.extract(x)
    assert all(np.isnan(f[k]) for k in FT.MODEL_FEATURES)


@pytest.mark.parametrize("snr_db", FT.SNR_LEVELS_DB)
def test_noise_hits_the_stated_in_band_snr(pulse, snr_db):
    noisy = FT.add_noise(pulse, snr_db, FT.noise_rng(1234, "before", snr_db))
    noise = noisy - pulse
    measured = 10 * np.log10(np.var(FT._pulsatile(pulse)) / np.var(noise))
    assert abs(measured - snr_db) < 0.01


def test_noise_is_reproducible(pulse):
    a = FT.add_noise(pulse, -5, FT.noise_rng(1234, "after", -5))
    b = FT.add_noise(pulse, -5, FT.noise_rng(1234, "after", -5))
    c = FT.add_noise(pulse, -5, FT.noise_rng(1234, "before", -5))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------- built artifact
@pytest.fixture(scope="module")
def table():
    if not FT.features_path().exists():
        pytest.skip("features not built - run `make features`")
    return FT.load_features()


def test_artifact_covers_dev_sessions_only(table):
    from faceai.splits import load_splits

    splits = load_splits()
    test_subjects = set(splits.loc[splits["split"] == "test", "patient_id"])
    assert not test_subjects & set(table["patient_id"])
    assert len(table) == len(FT.dev_sessions())


def test_only_the_known_corrupt_file_is_unreadable(table):
    bad = table.loc[~table["readable"], ["patient_id", "step"]].values.tolist()
    assert bad == [[8706, "after"]]


def test_real_pulses_have_physiological_polarity(table):
    ok = table[table["readable"]]
    assert (ok["systolic_frac"] < 0.5).all()
