"""configs/targets.yaml must account for the real db.csv, and honour the project rules."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from faceai.data import hub
from faceai.targets import load

DB_CSV = hub.LOCAL_DIR / "db.csv"

# The Group-2 list is FIXED by the brief. This test fails if anyone adds or drops one.
EXPECTED = {
    "spo2",
    "temperature",
    "glucose",
    "hba1c",
    "cholesterol",
    "respiratory_rate",
    "arterial_stiffness",
    "stress_psm25",
    "age",
    "sex",
    "bmi",
}
TIER_C = {"glucose", "hba1c", "cholesterol", "temperature"}


@pytest.fixture(scope="module")
def cfg():
    return load()


@pytest.fixture(scope="module")
def db():
    if not DB_CSV.exists():
        pytest.skip(f"{DB_CSV} not downloaded")
    return pd.read_csv(DB_CSV)


def test_target_list_is_exactly_the_brief(cfg):
    assert {t.name for t in cfg.targets} == EXPECTED


def test_glucose_is_unavailable(cfg):
    """db.csv has no glucose column; it must never be modelled."""
    glucose = cfg.by_name("glucose")
    assert glucose.available is False
    assert glucose.column is None


def test_tier_c_is_not_clinically_usable(cfg):
    for name in TIER_C:
        assert cfg.by_name(name).tier == "C"
        assert cfg.by_name(name).clinically_usable is False


def test_blood_pressure_is_not_a_target(cfg):
    excluded = {e.column for e in cfg.excluded}
    assert {"upper_ap", "lower_ap"} <= excluded
    assert {t.column for t in cfg.targets}.isdisjoint({"upper_ap", "lower_ap"})


def test_every_db_column_is_accounted_for(cfg, db):
    assert cfg.unaccounted(list(db.columns)) == set()


def test_mapped_columns_exist_in_db(cfg, db):
    for t in cfg.modellable():
        assert t.column in db.columns, f"{t.name} maps to a missing column {t.column}"


def rounding_tolerance(value: float) -> float:
    """Half a unit in the last printed decimal place.

    Paper Table 2 prints rounded bounds (BMI "15.39"), but the real minimum is
    15.389350..., which sits *below* the printed bound purely because of rounding.
    Comparing raw data against a rounded bound needs this slack - anything larger
    would be silently widening the range to make the check pass.
    """
    exponent = Decimal(str(value)).as_tuple().exponent
    return 0.5 * 10 ** int(exponent)


def test_observed_values_are_inside_the_declared_range(cfg, db):
    """The paper's Table 2 ranges must actually hold in the released data."""
    for t in cfg.modellable():
        if t.clinical_range is None:
            continue
        lo, hi = t.clinical_range
        col = db[t.column]
        lo_tol, hi_tol = rounding_tolerance(lo), rounding_tolerance(hi)
        assert col.min() >= lo - lo_tol, (
            f"{t.name}: observed min {col.min()} below declared {lo} by more than rounding"
        )
        assert col.max() <= hi + hi_tol, (
            f"{t.name}: observed max {col.max()} above declared {hi} by more than rounding"
        )
