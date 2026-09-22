"""THE blocking test. No model may be trained until this is green.

With 6 recordings per subject, a random split puts the same person in train and test and
produces beautiful, meaningless numbers. These tests assert the guarantee from several
independent directions, including that the guard itself rejects a known-bad split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from faceai.data import audit
from faceai.data.hub import LOCAL_DIR
from faceai.splits import (
    N_FOLDS,
    TEST_FOLD,
    LeakageError,
    assert_no_leakage,
    assignment_hash,
    fold_indices,
    load_splits,
    make_splits,
)

EXPECTED_SUBJECTS = 599
EXPECTED_RECORDINGS = 3591


@pytest.fixture(scope="module")
def db():
    if not (LOCAL_DIR / "db.csv").exists():
        pytest.skip("db.csv not downloaded")
    return audit.load_db()


@pytest.fixture(scope="module")
def splits(db):
    return make_splits(db)


# --------------------------------------------------------------- the core guarantee
def test_no_subject_is_in_both_test_and_dev(splits):
    test_ids = set(splits.loc[splits["split"] == "test", "patient_id"])
    dev_ids = set(splits.loc[splits["split"] == "dev", "patient_id"])
    assert test_ids, "the held-out test set must not be empty"
    assert test_ids & dev_ids == set()


def test_no_subject_is_in_both_train_and_val_in_any_fold(splits):
    for fold in range(N_FOLDS):
        train, val = fold_indices(splits, fold)
        train_ids = set(splits.loc[train, "patient_id"])
        val_ids = set(splits.loc[val, "patient_id"])
        assert val_ids, f"fold {fold} has no validation subjects"
        assert train_ids & val_ids == set(), f"fold {fold} leaks subjects"


def test_every_recording_of_a_subject_shares_one_split_and_fold(splits):
    """The 6-recordings-per-subject rule: 2 states x 3 cameras stay together."""
    spans = splits.groupby("patient_id")[["split", "fold"]].nunique()
    assert (spans["split"] == 1).all()
    assert (spans["fold"] == 1).all()


def test_folds_are_disjoint_and_cover_dev_exactly(splits):
    dev = splits[splits["split"] == "dev"]
    assert set(dev["fold"]) == set(range(N_FOLDS))
    per_fold = [set(g["patient_id"]) for _, g in dev.groupby("fold")]
    union = set().union(*per_fold)
    assert sum(len(f) for f in per_fold) == len(union), "folds overlap"
    assert union == set(dev["patient_id"])


def test_test_rows_carry_no_cv_fold(splits):
    assert set(splits.loc[splits["split"] == "test", "fold"]) == {TEST_FOLD}


# ------------------------------------------------------------ the guard guards itself
def test_assert_no_leakage_rejects_a_deliberately_leaky_split(splits):
    """If this passes, the assertions above prove nothing."""
    leaky = splits.copy()
    dev_subject = leaky.loc[leaky["split"] == "dev", "patient_id"].iloc[0]
    # put one of that subject's recordings into the test set
    row = leaky.index[leaky["patient_id"] == dev_subject][0]
    leaky.loc[row, "split"] = "test"
    leaky.loc[row, "fold"] = TEST_FOLD
    with pytest.raises(LeakageError):
        assert_no_leakage(leaky)


def test_assert_no_leakage_rejects_a_subject_split_across_two_folds(splits):
    leaky = splits.copy()
    dev = leaky[leaky["split"] == "dev"]
    subject = dev["patient_id"].iloc[0]
    rows = leaky.index[leaky["patient_id"] == subject]
    current = int(leaky.loc[rows[0], "fold"])
    leaky.loc[rows[0], "fold"] = (current + 1) % N_FOLDS
    with pytest.raises(LeakageError):
        assert_no_leakage(leaky)


# ------------------------------------------------------------------- content + counts
def test_excluded_subjects_and_sessions_are_absent(splits):
    assert set(splits["patient_id"]).isdisjoint(audit.EXCLUDED_SUBJECTS)
    for subject, step in audit.EXCLUDED_SESSIONS:
        present = splits[(splits["patient_id"] == subject) & (splits["step"] == step)]
        assert len(present) == 0


def test_counts_match_the_post_exclusion_cohort(splits, db):
    assert splits["patient_id"].nunique() == EXPECTED_SUBJECTS
    assert len(splits) == EXPECTED_RECORDINGS
    assert set(splits["patient_id"]) <= set(db["patient_id"])


def test_every_recording_appears_exactly_once(splits):
    assert not splits.duplicated(subset=["patient_id", "step", "camera"]).any()


# ------------------------------------------------------------------------ determinism
def test_regenerating_gives_an_identical_assignment(db):
    first, second = make_splits(db), make_splits(db)
    assert assignment_hash(first) == assignment_hash(second)
    pd.testing.assert_frame_equal(first, second)


def test_committed_split_file_matches_regeneration(db):
    """Guards against a stale or hand-edited artifacts/splits.parquet."""
    try:
        stored = load_splits()
    except FileNotFoundError:
        pytest.skip("artifacts/splits.parquet not built yet - run `make splits`")
    assert assignment_hash(stored) == assignment_hash(make_splits(db))
