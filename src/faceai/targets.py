"""The canonical target list, loaded from configs/targets.yaml.

Later phases must go through this module rather than hard-coding column names, so a
target cannot be silently renamed, typo'd or added. The fixed Group-2 list is a project
rule, not a convention.

    from faceai.targets import load
    cfg = load()
    [t.name for t in cfg.modellable()]   # what may actually be fitted
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "targets.yaml"

Tier = Literal["A", "B", "C"]


class Target(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    column: str | None
    unit: str
    tier: Tier
    task: Literal["regression", "classification"]
    available: bool
    clinical_range: tuple[float, float] | None
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> Target:
        if self.available and not self.column:
            raise ValueError(f"{self.name}: available targets need a column")
        if not self.available and self.column:
            raise ValueError(f"{self.name}: unavailable targets must not name a column")
        if self.clinical_range and self.clinical_range[0] >= self.clinical_range[1]:
            raise ValueError(f"{self.name}: clinical_range must be (min, max)")
        return self

    @property
    def clinically_usable(self) -> bool:
        """Tier C must never be presented as clinically usable (project rule)."""
        return self.tier != "C"


class Excluded(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    reason: str


class TargetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: tuple[Target, ...]
    excluded: tuple[Excluded, ...]

    @model_validator(mode="after")
    def _unique(self) -> TargetConfig:
        names = [t.name for t in self.targets]
        if len(set(names)) != len(names):
            raise ValueError("duplicate target names")
        columns = [t.column for t in self.targets if t.column] + [
            e.column for e in self.excluded
        ]
        if len(set(columns)) != len(columns):
            raise ValueError("a column is mapped more than once")
        return self

    def modellable(self) -> tuple[Target, ...]:
        """Targets that actually have labels. Anything else cannot be fitted."""
        return tuple(t for t in self.targets if t.available)

    def by_name(self, name: str) -> Target:
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(
            f"unknown target {name!r}; known: {[t.name for t in self.targets]}"
        )

    def known_columns(self) -> set[str]:
        return {t.column for t in self.targets if t.column} | {
            e.column for e in self.excluded
        }

    def unaccounted(self, columns: list[str]) -> set[str]:
        """db.csv columns this config does not mention - must always be empty."""
        return set(columns) - self.known_columns()


def load(path: Path = CONFIG) -> TargetConfig:
    return TargetConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
