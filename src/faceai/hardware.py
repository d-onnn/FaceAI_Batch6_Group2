"""Hardware detection and the project device policy: CUDA > MPS > CPU.

    python -m faceai.hardware        # writes env/hardware.json

Every training/inference entry point should take its device from get_device().
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "env" / "hardware.json"
DISK_HEADROOM_GB = 50  # never plan a download that leaves less than this free


def get_device() -> str:
    """CUDA > MPS > CPU. Falling back to CPU is loud, never silent."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    warnings.warn(
        "No CUDA/MPS accelerator available - running on CPU. On a laptop with an NVIDIA "
        "GPU this usually means it is on battery and the dGPU is powered off: plug in AC.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


def _powershell(cmd: str) -> str:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # empty output is handled by the caller
    ).stdout.strip()


def _nvidia_smi() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "error": "nvidia-smi not on PATH"}
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,  # a failure (e.g. dGPU detached) is recorded, not raised
    )
    if q.returncode != 0:
        return {"available": False, "error": (q.stdout + q.stderr).strip()}
    name, driver, mem = (x.strip() for x in q.stdout.splitlines()[0].split(","))
    header = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    ).stdout
    cuda_max = re.search(r"CUDA Version:\s*([\d.]+)", header)
    return {
        "available": True,
        "name": name,
        "driver": driver,
        "memory_total": mem,
        # the driver's MAXIMUM supported CUDA runtime - not an installed toolkit
        "driver_max_cuda_runtime": cuda_max.group(1) if cuda_max else None,
    }


def detect() -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    battery = psutil.sensors_battery()
    on_windows = os.name == "nt"

    cpu_name = platform.processor()
    display_devices = None
    if on_windows:
        cpu_name = _powershell("(Get-CimInstance Win32_Processor).Name") or cpu_name
        raw = _powershell(
            "Get-PnpDevice -Class Display | Select-Object FriendlyName,Status | ConvertTo-Json"
        )
        parsed = json.loads(raw) if raw else []
        display_devices = [parsed] if isinstance(parsed, dict) else parsed

    t = {
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None))
        and torch.backends.mps.is_available(),
    }
    if t["cuda_available"]:
        cap = torch.cuda.get_device_capability(0)
        t |= {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{cap[0]}{cap[1]}",
            "arch_list": torch.cuda.get_arch_list(),
            "vram_gib": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            ),
        }

    return {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": {
            "name": cpu_name,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram_gib": round(vm.total / 1024**3, 1),
        # decimal GB, the same unit the Hub reports dataset sizes in
        "disk": {
            "path": str(ROOT),
            "free_gb": round(disk.free / 1e9, 1),
            "total_gb": round(disk.total / 1e9, 1),
        },
        "power": None
        if battery is None
        else {
            "ac_plugged": battery.power_plugged,
            "battery_percent": round(battery.percent),
        },
        "display_devices": display_devices,
        "nvidia_smi": _nvidia_smi(),
        "torch": t,
    }


def subset_plan(free_gb: float) -> dict:
    """--subset N sizing from the pinned repo's file metadata (no downloads)."""
    from faceai.data import hub

    video = collections.defaultdict(int)
    subjects, nonvideo = set(), 0
    for s in hub.info().siblings or []:
        if s.rfilename.startswith("video/"):
            subject, camera, _ = s.rfilename[6:].rsplit(".", 1)[0].split("_")
            subjects.add(subject)
            video[camera] += s.size or 0
        else:
            nonvideo += s.size or 0
    n = len(subjects)
    per_subject_all = sum(video.values()) / n / 1e9
    per_subject_frontal = video["FullHDwebcam"] / n / 1e9

    def row(n_subj: int) -> dict:
        all_cams = nonvideo / 1e9 + per_subject_all * n_subj
        return {
            "subjects": n_subj,
            "gb_all_cameras": round(all_cams, 1),
            "gb_frontal_only": round(nonvideo / 1e9 + per_subject_frontal * n_subj, 1),
            "fits_with_headroom": free_gb - all_cams >= DISK_HEADROOM_GB,
        }

    return {
        "source": f"{hub.REPO_ID}@{hub.REVISION}",
        "subjects_total": n,
        "nonvideo_gb_all_subjects": round(nonvideo / 1e9, 2),
        "video_gb_per_subject_all_cameras": round(per_subject_all, 3),
        "video_gb_per_subject_frontal": round(per_subject_frontal, 3),
        "disk_headroom_gb": DISK_HEADROOM_GB,
        # Sizes include ALL non-video data for every subject: the contact-PPG work
        # needs no video, so only video is ever subset.
        "tiers": {"smoke": row(25), "dev": row(100), "full": row(n)},
        "selection_rule": "subject-level only; the exact subset is fixed with the Phase 2 splits",
    }


def main() -> None:
    hw = detect()
    hw["device_policy"] = {"order": ["cuda", "mps", "cpu"], "selected": get_device()}
    hw["subset_plan"] = subset_plan(hw["disk"]["free_gb"])

    problems = []
    nvidia_present = any(
        "NVIDIA" in (d.get("FriendlyName") or "") for d in hw["display_devices"] or []
    )
    if nvidia_present and not hw["torch"]["cuda_available"]:
        problems.append(
            "NVIDIA GPU present but CUDA unavailable - dGPU detached (on battery?). "
            "This record does NOT describe the machine's real capability."
        )
    hw["warnings"] = problems

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(hw, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(hw, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if problems:
        print("\n!!! " + "\n!!! ".join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
