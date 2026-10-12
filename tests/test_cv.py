"""Phase 3 CV harness: data assembly, baselines and the bootstrap. No model is fitted here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faceai import cv
from faceai.features import features_path


@pytest.fixture(scope="module")
def table():
    if not features_path().exists():
        pytest.skip("features not built - run `make features`")
    return cv.session_table()


def test_no_test_subject_reaches_the_harness(table):
    from faceai.splits import load_splits

    splits = load_splits()
    test_subjects = set(splits.loc[splits["split"] == "test", "patient_id"])
    assert not test_subjects & set(table["patient_id"])
    assert table["fold"].between(0, 4).all()


def test_each_subject_sits_in_exactly_one_fold(table):
    assert (table.groupby("patient_id")["fold"].nunique() == 1).all()


def test_expected_size_and_complete_labels(table):
    assert len(table) == 956  # 957 dev sessions minus the corrupt 8706_after.PW
    assert table["patient_id"].nunique() == 479
    assert not table[cv.TARGETS].isna().any().any()


def test_demographic_baseline_never_sees_its_own_target():
    for target in cv.TARGETS:
        cols = cv.demographic_columns(target)
        assert target not in cols
    assert cv.demographic_columns("age") == ["sex_m", "bmi"]
    assert cv.demographic_columns("hba1c") == ["age", "sex_m", "bmi"]


def _oof(errors: dict[str, float], n_subjects: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_subjects):
        for step in ("before", "after"):
            y = rng.normal(5, 1)
            rows.append(
                {
                    "patient_id": s,
                    "step": step,
                    "target": "x",
                    "y": y,
                    **{m: y + errors.get(m, 1.0) for m in cv.MODELS},
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_point_estimates_and_ordering():
    oof = _oof({"LightGBM": 0.0, "Mean": 1.0})
    res = cv.bootstrap(oof, "x", n_boot=200)
    assert res["models"]["LightGBM"]["mae"] == 0.0
    assert res["models"]["LightGBM"]["ci"] == [0.0, 0.0]
    assert res["models"]["Mean"]["mae"] == pytest.approx(1.0)
    assert res["n_subjects"] == 50 and res["n_sessions"] == 100


def test_verdict_needs_non_overlapping_cis():
    res = {"models": {m: {"ci": [1.0, 1.2]} for m in cv.MODELS}}
    res["models"]["LightGBM"]["ci"] = [0.5, 0.9]  # clear of both baselines
    res["models"]["Ridge (L2)"]["ci"] = [0.9, 1.05]  # overlaps
    v = cv.verdict(res)
    assert v["LightGBM"]["beats_both"]
    assert not v["Ridge (L2)"]["beats_both"]
