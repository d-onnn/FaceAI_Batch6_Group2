"""Record the exact installed environment - read back from the live env, never from intent.

Writes:
  env/lockfile.txt   `pip freeze` + `conda list --explicit` (the authoritative record)
  requirements.txt   pip pins, with the PyTorch cu130 index line
  environment.yml    conda-forge interpreter/tooling + the same pip pins

Re-run after EVERY install (project rule: pin what you install):
    python -m faceai.lockfile        # or: make lock
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
# conda-forge packages installed on purpose; their dependencies resolve automatically.
CONDA_SPECS = ("python", "pip", "make")


def _conda_exe() -> str:
    if os.environ.get("CONDA_EXE"):
        return os.environ["CONDA_EXE"]
    root = Path(sys.prefix).parents[1]  # <root>/envs/faceai -> <root>
    return str(root / ("Scripts/conda.exe" if os.name == "nt" else "bin/conda"))


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def pip_pins() -> list[str]:
    pins = []
    for line in _run(
        sys.executable, "-m", "pip", "freeze", "--exclude-editable"
    ).splitlines():
        line = line.strip()
        # Skip blanks, pip's comment lines, and this project itself (installed
        # editable; it is not on PyPI, so pinning it would break `pip install -r`).
        if not line or line.startswith("#") or line.lower().startswith("faceai"):
            continue
        if " @ " in line:  # conda-installed: freezes as an unusable file:// URL
            name = line.split(" @ ")[0].strip()
            line = f"{name}=={md.version(name)}"
        pins.append(line)
    return sorted(pins, key=str.lower)


def conda_versions() -> dict[str, str]:
    out = _run(_conda_exe(), "list", "-p", sys.prefix, "--json")
    return {
        p["name"]: p["version"] for p in json.loads(out) if p.get("channel") != "pypi"
    }


def main() -> None:
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    pins = pip_pins()
    conda = conda_versions()
    explicit = _run(_conda_exe(), "list", "-p", sys.prefix, "--explicit")
    py = platform.python_version()
    where = f"{platform.system()} {platform.release()} {platform.machine()}"

    (ROOT / "env").mkdir(exist_ok=True)
    (ROOT / "env" / "lockfile.txt").write_text(
        "\n".join(
            [
                f"# FaceAI environment lockfile - generated {stamp} by `python -m faceai.lockfile`",
                f"# prefix: {sys.prefix}",
                f"# python: {py} | {where}",
                "#",
                "# ===== pip freeze =====",
                *pins,
                "",
                "# ===== conda list --explicit =====",
                explicit.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    (ROOT / "requirements.txt").write_text(
        "\n".join(
            [
                "# FaceAI - exact pip pins, generated from the real environment",
                f"# by `python -m faceai.lockfile` on {stamp[:10]}.",
                f"# Python {py} | {where}",
                "#",
                "# The torch/torchvision/torchaudio pins carry the +cu130 local version and",
                "# resolve ONLY from the PyTorch index below. Install with:",
                "#     pip install -r requirements.txt",
                "",
                f"--extra-index-url {TORCH_INDEX}",
                "",
                *pins,
                "",
            ]
        ),
        encoding="utf-8",
    )

    missing = [s for s in CONDA_SPECS if s not in conda]
    if missing:
        sys.exit(f"conda packages expected but not installed: {missing}")
    (ROOT / "environment.yml").write_text(
        "\n".join(
            [
                "# FaceAI conda environment - FALLBACK manifest.",
                "# pyproject.toml is the primary source of truth; this file is generated",
                f"# from the same real environment by `python -m faceai.lockfile` on {stamp[:10]}.",
                "#",
                "# Recreate with:",
                "#     mamba env create -f environment.yml",
                "#",
                "# Conda supplies ONLY the interpreter and tooling (pip, make); every Python",
                "# package is installed from PyPI, because the CUDA 13.0 PyTorch build for",
                "# sm_120 Blackwell is published as a wheel, not a conda package.",
                "name: faceai",
                "channels:",
                "  - conda-forge          # conda-forge ONLY - no Anaconda defaults channel",
                "dependencies:",
                *(f"  - {s}={conda[s]}" for s in CONDA_SPECS),
                "  - pip:",
                f"    - --extra-index-url {TORCH_INDEX}",
                *(f"    - {p}" for p in pins),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"wrote env/lockfile.txt, requirements.txt, environment.yml ({len(pins)} pip pins)"
    )


if __name__ == "__main__":
    main()
