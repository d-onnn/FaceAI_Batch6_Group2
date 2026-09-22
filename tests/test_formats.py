"""Format readers, checked against real files.

These tests pin the quirks found by inspection on 2026-09-23. If one starts failing,
the data changed (or the pin moved) - investigate before "fixing" the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceai.data import formats as F
from faceai.data.hub import LOCAL_DIR

SUBJECT, STATE, CAMERA = "1020", "after", "FullHDwebcam"
ECG = LOCAL_DIR / f"ecg/{SUBJECT}_{STATE}.json"
PW = LOCAL_DIR / f"ppg/{SUBJECT}_{STATE}.PW"
META = LOCAL_DIR / f"meta/{SUBJECT}_{CAMERA}_{STATE}.txt"
SYNC = LOCAL_DIR / f"ppg_sync/{SUBJECT}_{CAMERA}_{STATE}.txt"


def _require(path):
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    return path


@pytest.fixture(scope="module")
def ecg():
    return F.read_ecg(_require(ECG))


@pytest.fixture(scope="module")
def meta():
    return F.read_meta(_require(META))


@pytest.fixture(scope="module")
def pw():
    return F.read_pw(_require(PW))


# ------------------------------------------------------------------ ECG
def test_ecg_is_500hz_12_lead_30s(ecg):
    assert ecg.frequency == 500
    assert len(ecg.leads) == 12
    assert set(ecg.leads) == {
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        *(f"V{i}" for i in range(1, 7)),
    }
    assert 29.0 <= ecg.time[-1] <= 30.0


def test_lead_I_is_short_and_matrix_truncates_safely(ecg):
    """Lead I has 14900 samples, the other 11 have 15000 - stacking must truncate."""
    assert ecg.ragged
    assert ecg.lead_lengths["I"] < ecg.lead_lengths["II"]
    matrix, time = ecg.matrix()
    assert matrix.shape == (12, ecg.common_length)
    assert len(time) == ecg.common_length
    assert np.isfinite(matrix).all()


# ----------------------------------------------------------------- meta
def test_meta_is_180s_indexed_from_one(meta):
    assert meta.frame_index[0] == 1
    assert 179.0 <= meta.duration_s <= 182.0
    assert 20.0 < meta.fps_mean < 32.0


def test_timestamps_parse_with_and_without_microseconds(tmp_path):
    """Regression: some real timestamps land exactly on the second.

    pandas infers the format from the first row, so a file whose first timestamp has
    microseconds used to raise partway through when it hit one that did not.
    """
    path = tmp_path / "meta.txt"
    path.write_text(
        "\n1\t2023-10-23 13:38:01.500000\n2\t2023-10-23 13:38:02\n"
        "3\t2023-10-23 13:38:02.500000\n",
        encoding="utf-8",
    )
    parsed = F.read_meta(path)
    assert parsed.n_frames == 3
    assert parsed.duration_s == pytest.approx(1.0)


def test_meta_rate_is_not_constant(meta):
    """Mean and median fps disagree because frames are unevenly spaced."""
    assert abs(meta.fps_mean - meta.fps_median) > 0.5


# ------------------------------------------------------------------- PW
def test_pw_is_8bit_and_100hz_on_average(pw):
    assert pw.value.min() >= F.CLIP_LOW
    assert pw.value.max() <= F.CLIP_HIGH
    assert 99.0 <= pw.mean_rate_hz <= 101.0
    assert 179.0 <= pw.duration_s <= 183.0
    assert pw.clipped_fraction < 0.05  # measured ~0.31% across 30 files


def test_pw_resamples_onto_a_uniform_grid(pw):
    grid, values = pw.resample(100.0)
    assert np.allclose(np.diff(grid), 0.01)
    assert np.isfinite(values).all()
    assert len(grid) == pytest.approx(pw.duration_s * 100, rel=0.01)


def test_pw_and_meta_share_a_wall_clock(pw, meta):
    """Alignment of PPG to video depends on this; a second apart would break it."""
    offset = abs((pw.timestamp[0] - meta.timestamp[0]) / np.timedelta64(1, "s"))
    assert offset < 1.0


# ------------------------------------------------------------- ppg_sync
def test_ppg_sync_has_fewer_rows_than_frames_and_good_sync(meta):
    sync = F.read_ppg_sync(_require(SYNC))
    # Measured over 250 recordings: meta has 0-71 more rows (median 20), never fewer.
    assert sync.n_rows <= meta.n_frames, "sync rows should never exceed meta frames"
    assert meta.n_frames - sync.n_rows <= 120
    assert sync.value.min() >= F.CLIP_LOW
    assert sync.value.max() <= F.CLIP_HIGH
    assert sync.median_sync_error_s < 0.01  # measured ~3.5-4 ms
