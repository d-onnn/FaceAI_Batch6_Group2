"""Run provenance: which code, split, data and environment produced an artifact.

Every Phase 3 results file carries this block, so any number in a report can be traced to
the exact commit, split assignment, dataset file and pinned environment behind it - and
`git_dirty_paths > 0` says plainly when a result came from uncommitted code.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from faceai.data.hub import LOCAL_DIR, ROOT, git_blob_id


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def record(inputs: list[Path]) -> dict:
    """Provenance for a run that read `inputs` (paths under the repo root)."""
    splits = json.loads(
        (ROOT / "artifacts" / "splits.json").read_text(encoding="utf-8")
    )
    return {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_paths": len(_git("status", "--porcelain").splitlines()),
        "splits_assignment_sha256": splits["assignment_sha256"],
        "db_csv_git_blob": git_blob_id(LOCAL_DIR / "db.csv"),
        "lockfile_sha256": sha256(ROOT / "env" / "lockfile.txt"),
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs},
    }
