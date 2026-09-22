"""Subject-level splits. The ONLY valid grouping key is `patient_id`.

Project rule: never random-split, never split by recording. Every recording of a subject
(2 states x 3 cameras) sits in the same partition, because a subject appearing on both
sides of a split leaks identity and produces beautiful, meaningless numbers.

    python -m faceai.splits        # or: make splits

The held-out test set is SEALED until Phase 10. Nothing before then may read it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from faceai.data import audit
from faceai.data.hub import LOCAL_DIR, ROOT, git_blob_id

SEED = 42
N_FOLDS = 5
TEST_FRACTION = 0.2
OLDER_THAN = 25  # age band boundary, used ONLY to stratify - never as a model input
TEST_FOLD = -1  # test rows carry no CV fold

SPLITS = ROOT / "artifacts" / "splits.parquet"
PROVENANCE = ROOT / "artifacts" / "splits.json"
REPORT = ROOT / "reports" / "02_splits.md"
KEY = ["patient_id", "step", "camera"]


class LeakageError(AssertionError):
    """Raised when a split would put one subject on both sides of a boundary."""


def subject_frame(db: pd.DataFrame) -> pd.DataFrame:
    """One row per subject. Splitting happens here, then maps back to recordings."""
    return (
        db.drop_duplicates(subset=["patient_id"])[["patient_id", "age", "sex"]]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def stratum(subjects: pd.DataFrame) -> pd.Series:
    """sex x (age >= 25).

    Age is bimodal in this cohort (median 20, p95 52): only 55 of 599 subjects are 25+.
    Without stratifying, a fold can draw almost no older subjects, which makes `age` -
    a Tier A target - unstable across folds. Noted in the report: stratifying on age
    makes CV slightly more optimistic than a purely random split.
    """
    return subjects["sex"].astype(str) + np.where(
        subjects["age"] >= OLDER_THAN, "_old", "_young"
    )


def make_splits(db: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deterministic recording-level split assignment.

    Exclusions are applied first, so subject 4952 and 7412's `after` session never
    reach the splits.
    """
    if db is None:
        db = audit.load_db()
    kept, _ = audit.apply_exclusions(db)
    subjects = subject_frame(kept)
    strata = stratum(subjects)

    holdout = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=SEED
    )
    _, test_idx = next(holdout.split(subjects, strata))
    test_ids = set(subjects.iloc[test_idx]["patient_id"])

    out = kept[KEY].copy().sort_values(KEY).reset_index(drop=True)
    out["split"] = np.where(out["patient_id"].isin(test_ids), "test", "dev")
    out["fold"] = TEST_FOLD

    dev = out[out["split"] == "dev"]
    dev_strata = dev["patient_id"].map(
        dict(zip(subjects["patient_id"], strata, strict=True))
    )
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val) in enumerate(
        folds.split(dev, dev_strata, groups=dev["patient_id"])
    ):
        out.loc[dev.index[val], "fold"] = fold

    assert_no_leakage(out)
    return out


def assert_no_leakage(splits: pd.DataFrame) -> None:
    """Raise unless the split is airtight. Call this before ANY model sees the data.

    Checks are independent of how the split was produced, so this also catches a
    hand-edited or stale file - not only a bug in `make_splits`.
    """
    spans = splits.groupby("patient_id")[["split", "fold"]].nunique()
    straddling = spans[(spans > 1).any(axis=1)]
    if len(straddling):
        raise LeakageError(
            f"{len(straddling)} subject(s) span more than one split/fold: "
            f"{list(straddling.index[:5])}"
        )

    by_split = splits.groupby("split")["patient_id"].apply(set)
    test_ids = by_split.get("test", set())
    dev_ids = by_split.get("dev", set())
    if shared := test_ids & dev_ids:
        raise LeakageError(
            f"{len(shared)} subject(s) in BOTH test and dev: {sorted(shared)[:5]}"
        )

    dev = splits[splits["split"] == "dev"]
    per_fold = {int(k): set(g["patient_id"]) for k, g in dev.groupby("fold")}
    if set(per_fold) != set(range(N_FOLDS)):
        raise LeakageError(
            f"expected folds {list(range(N_FOLDS))}, found {sorted(per_fold)}"
        )
    for fold, val_ids in per_fold.items():
        train_ids = set().union(*(v for f, v in per_fold.items() if f != fold))
        if shared := val_ids & train_ids:
            raise LeakageError(
                f"fold {fold}: {len(shared)} subject(s) in both train and val: "
                f"{sorted(shared)[:5]}"
            )
        if shared := val_ids & test_ids:
            raise LeakageError(
                f"fold {fold} overlaps the held-out test set: {sorted(shared)[:5]}"
            )

    stray = set(splits.loc[splits["split"] == "test", "fold"]) - {TEST_FOLD}
    if stray:
        raise LeakageError(
            f"test rows must carry fold {TEST_FOLD}, found {sorted(stray)}"
        )


def assignment_hash(splits: pd.DataFrame) -> str:
    """Stable fingerprint of the assignment, so a changed split is detectable."""
    ordered = splits.sort_values(KEY)[[*KEY, "split", "fold"]]
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def provenance(splits: pd.DataFrame) -> dict:
    return {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "test_fraction": TEST_FRACTION,
        "stratified_by": f"sex x (age >= {OLDER_THAN})",
        "sklearn": sklearn.__version__,
        "excluded_subjects": list(audit.EXCLUDED_SUBJECTS),
        "excluded_sessions": [list(k) for k in audit.EXCLUDED_SESSIONS],
        "db_csv_git_blob_id": git_blob_id(LOCAL_DIR / "db.csv"),
        "n_subjects": int(splits["patient_id"].nunique()),
        "n_recordings": len(splits),
        "assignment_sha256": assignment_hash(splits),
    }


def write_splits(splits: pd.DataFrame) -> None:
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(SPLITS, index=False)
    PROVENANCE.write_text(
        json.dumps(provenance(splits), indent=2) + "\n", encoding="utf-8"
    )


def load_splits() -> pd.DataFrame:
    """Read the committed split, re-checking it for leakage on the way in."""
    if not SPLITS.exists():
        raise FileNotFoundError(f"{SPLITS} is missing - build it with `make splits`")
    splits = pd.read_parquet(SPLITS)
    assert_no_leakage(splits)
    return splits


def fold_indices(splits: pd.DataFrame, fold: int) -> tuple[pd.Index, pd.Index]:
    """(train, val) row indices for one CV fold. Test rows are never returned."""
    dev = splits[splits["split"] == "dev"]
    return dev.index[dev["fold"] != fold], dev.index[dev["fold"] == fold]


def summary_tables(splits: pd.DataFrame, db: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = splits.merge(db[[*KEY, "age", "sex"]], on=KEY, how="left", validate="1:1")
    by_split = (
        merged.groupby("split")
        .agg(subjects=("patient_id", "nunique"), recordings=("patient_id", "size"))
        .reset_index()
    )
    by_camera = (
        merged.groupby(["split", "camera"])["patient_id"]
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    by_state = (
        merged.groupby(["split", "step"])["patient_id"]
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    dev = merged[merged["split"] == "dev"]
    per_subject = dev.drop_duplicates("patient_id")
    by_fold = (
        per_subject.groupby("fold")
        .agg(
            subjects=("patient_id", "nunique"),
            female=("sex", lambda s: int((s == "F").sum())),
            male=("sex", lambda s: int((s == "M").sum())),
            age_ge_25=("age", lambda a: int((a >= OLDER_THAN).sum())),
            age_median=("age", "median"),
            age_max=("age", "max"),
        )
        .reset_index()
    )
    by_fold["recordings"] = (
        dev.groupby("fold").size().reindex(by_fold["fold"]).to_numpy()
    )
    return {
        "by_split": by_split,
        "by_camera": by_camera,
        "by_state": by_state,
        "by_fold": by_fold,
    }


def main() -> None:
    from faceai.eda import md_table

    db = audit.load_db()
    splits = make_splits(db)
    write_splits(splits)
    tables = summary_tables(splits, db)
    meta = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    lines = [
        "# 02 — Subject-level splits",
        "",
        f"Generated {meta['generated']} by `python -m faceai.splits`.",
        "",
        (
            "**Splitting is by `patient_id` only.** All recordings of a subject sit in the same "
            "partition. The held-out test set is sealed until Phase 10 — no phase before it may "
            "read those rows."
        ),
        "",
        "## Provenance",
        "",
        *[f"- `{k}`: {v}" for k, v in meta.items()],
        "",
        "## Subjects and recordings per split",
        "",
        *md_table(tables["by_split"]),
        "",
        "### Per camera",
        "",
        *md_table(tables["by_camera"]),
        "",
        "### Per state",
        "",
        *md_table(tables["by_state"]),
        "",
        "## Cross-validation folds (dev only)",
        "",
        *md_table(tables["by_fold"]),
        "",
        (
            f"Folds are `StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, "
            f"random_state={SEED})` grouped on `patient_id` and stratified on "
            f"sex x (age >= {OLDER_THAN})."
        ),
        "",
        (
            "Stratifying on age keeps older subjects present in every fold — only "
            f"{int((subject_frame(audit.apply_exclusions(db)[0])['age'] >= OLDER_THAN).sum())} "
            "of 599 subjects are 25 or older, so an unstratified draw could leave a fold with "
            "almost none, making `age` and `arterial_stiffness` unstable. **The trade-off: CV is "
            "slightly more optimistic than a purely random split**, because every fold is "
            "guaranteed a representative age and sex mix."
        ),
        "",
        "## Comparability with the paper",
        "",
        (
            "The authors split the same way — `train_test_split` on unique `patient_id`, "
            "`test_size=0.2`, `random_state=43` — so their Table 5 numbers are genuine "
            "subject-level results. We match their 20% test fraction, but their exact split "
            "cannot be reproduced: they had 600 subjects, we have 599 after the Phase 1 "
            "exclusions."
        ),
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    for name, table in tables.items():
        print(f"=== {name} ===")
        print(table.to_string(index=False))
        print()
    print(f"assignment sha256: {meta['assignment_sha256']}")
    print(
        f"wrote {SPLITS.relative_to(ROOT)}, {PROVENANCE.relative_to(ROOT)}, "
        f"{REPORT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
