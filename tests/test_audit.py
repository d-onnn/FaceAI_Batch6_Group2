"""Integrity-audit behaviour on the real db.csv."""

from __future__ import annotations

import pandas as pd
import pytest

from faceai.data import audit
from faceai.data.hub import LOCAL_DIR


@pytest.fixture(scope="module")
def db():
    if not (LOCAL_DIR / "db.csv").exists():
        pytest.skip("db.csv not downloaded")
    return audit.load_db()


def test_exclusions_drop_exactly_the_known_defects(db):
    kept, summary = audit.apply_exclusions(db)
    # subject 4952 (6 rows) + subject 7412's `after` session (3 rows)
    assert summary["rows_dropped"] == 9
    assert summary["subjects_after"] == summary["subjects_before"] - 1
    assert audit.SUSPECT_SUBJECT not in set(kept["patient_id"])
    # 7412 keeps its healthy `before` session
    remaining = kept[kept["patient_id"] == 7412]
    assert len(remaining) == 3
    assert set(remaining["step"]) == {"before"}


def test_every_exclusion_carries_a_reason():
    for reason in (
        *audit.EXCLUDED_SUBJECTS.values(),
        *audit.EXCLUDED_SESSIONS.values(),
    ):
        assert len(reason) > 40, "an exclusion without evidence is a silent data change"


def test_path_columns_disagree_only_for_the_suspect_subject(db):
    paths = audit.check_path_consistency(db)
    by_column = paths.set_index("column")["rows_disagreeing_with_filename_convention"]
    assert by_column["video"] == 0
    assert by_column["meta"] == 0
    assert by_column["ppg_sync"] == 0
    assert by_column["ppg"] == 6  # the 6 rows of subject 4952
    assert by_column["ecg"] == 6


def test_labels_are_constant_across_the_three_cameras(db):
    """If this ever fails, the analysis unit is no longer the session."""
    from faceai.targets import load

    result = audit.check_labels_constant_across_cameras(db, load())
    assert result["sessions"] == 1200
    assert result["sessions_with_varying_labels"] == 0


def test_hr_estimator_tracks_ecg_better_than_the_pulse_label(db):
    """The finding that justifies never using `pulse` as an HR reference."""
    pytest.importorskip("neurokit2")
    from faceai.eda import hr_reference_check

    kept, _ = audit.apply_exclusions(db)
    result = hr_reference_check(kept, n=20)
    assert result["PPG vs ECG"]["MAE_bpm"] < result["ECG vs pulse label"]["MAE_bpm"]


def test_out_of_range_is_empty_for_every_target(db):
    from faceai.targets import load

    ranges = audit.check_out_of_range(db, load())
    assert (ranges["below"] == 0).all()
    assert (ranges["above"] == 0).all()


def test_degenerate_scan_finds_the_flat_line_files():
    """Subject 4952 (6 files) and subject 7412's `after` session (3 files)."""
    if not (LOCAL_DIR / "ppg_sync").exists():
        pytest.skip("ppg_sync not downloaded")
    found = audit.check_degenerate_signals("ppg_sync/4952*.txt")
    assert len(found) == 6
    assert set(found["subject"]) == {4952}
    assert isinstance(found, pd.DataFrame)
