"""Pinned access to the MCD-rPPG dataset on the Hugging Face Hub.

Every call is pinned to REVISION, so a download made today and one made next month
are the same bytes. See reports/00_access.md for why this repo and this commit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import DatasetInfo, RepoSibling

ROOT = Path(__file__).resolve().parents[3]
LOCAL_DIR = ROOT / "data" / "mcd_rppg"  # gitignored

REPO_TYPE = "dataset"
REPO_ID = "wengziheng/mcd_rppg"  # gated=auto; the copy the team was given
REVISION = "91b967a5f2bab735d35f81722ff95e629358753b"
MIRROR_REPO_ID = (
    "Shubh782004/mcd_rppg"  # ungated copy; used ONLY for the identity check
)
CANONICAL_REPO_ID = (
    "kyegorov/mcd_rppg"  # named by the dataset card; HTTP 401 as of 2026-09-22
)

# Project rule: ask before any download > 2 GB. Enforced here, not just documented.
MAX_BYTES_WITHOUT_APPROVAL = 2_000_000_000


def info(repo_id: str = REPO_ID, revision: str | None = REVISION) -> DatasetInfo:
    """Repo metadata including per-file size and hash (no file contents downloaded)."""
    return HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)


def file_hash(sibling: RepoSibling) -> str | None:
    """LFS files carry a sha256; small git-stored files carry a git blob id (sha1)."""
    return sibling.lfs.sha256 if sibling.lfs else sibling.blob_id


def git_blob_id(path: Path) -> str:
    """The id git gives a file's content - verifies a non-LFS download byte-for-byte."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(
    filenames: list[str],
    *,
    allow_video: bool = False,
    max_bytes: int = MAX_BYTES_WITHOUT_APPROVAL,
) -> list[Path]:
    """Download named files at the pinned revision into data/mcd_rppg/.

    Refuses video unless allow_video=True, and refuses more than max_bytes in total.
    Raising either limit is a decision for the project lead, not a default.
    """
    if not allow_video and (bad := [f for f in filenames if f.startswith("video/")]):
        raise PermissionError(
            f"refusing to fetch video without allow_video=True: {bad[:3]}"
        )
    sizes = {s.rfilename: s.size or 0 for s in info().siblings or []}
    if missing := [f for f in filenames if f not in sizes]:
        raise FileNotFoundError(f"not in {REPO_ID}@{REVISION[:8]}: {missing[:5]}")
    total = sum(sizes[f] for f in filenames)
    if total > max_bytes:
        raise PermissionError(
            f"{total / 1e9:.2f} GB requested exceeds the {max_bytes / 1e9:.2f} GB limit; "
            "get approval, then pass max_bytes explicitly"
        )
    return [
        Path(
            hf_hub_download(
                REPO_ID, f, repo_type=REPO_TYPE, revision=REVISION, local_dir=LOCAL_DIR
            )
        )
        for f in filenames
    ]
