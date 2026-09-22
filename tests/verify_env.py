"""FaceAI environment verifier.

Checks the full rPPG stack end to end and prints a PASS/FAIL line per item.
Exit code 0 only if every check passes.

Usage:
    python tests/verify_env.py
    python tests/verify_env.py --no-camera   # skip webcam + landmarker (headless/CI)
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

REQUIRED_PY = (3, 12)

# module import name -> attribute holding the version string
PACKAGES = [
    ("numpy", "__version__"),
    ("scipy", "__version__"),
    ("pandas", "__version__"),
    ("matplotlib", "__version__"),
    ("sklearn", "__version__"),
    ("cv2", "__version__"),
    ("mediapipe", "__version__"),
    ("torch", "__version__"),
    ("heartpy", "__version__"),
    ("neurokit2", "__version__"),
    ("av", "__version__"),
    ("imageio", "__version__"),
    # Group 2 stack (added Phase 0, 2026-09-22)
    ("pyarrow", "__version__"),
    ("lightgbm", "__version__"),
    ("xgboost", "__version__"),
    ("catboost", "__version__"),
    ("mapie", "__version__"),
    ("gradio", "__version__"),
    ("mlflow", "__version__"),
    ("huggingface_hub", "__version__"),
    ("pydantic", "__version__"),
    ("yaml", "__version__"),
    ("pytest", "__version__"),
]

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# --------------------------------------------------------------------------
# 1. Python version
# --------------------------------------------------------------------------
def check_python() -> None:
    section("1. Python")
    v = sys.version_info
    ok = (v.major, v.minor) == REQUIRED_PY
    record(
        f"Python {REQUIRED_PY[0]}.{REQUIRED_PY[1]}",
        ok,
        f"running {platform.python_version()} ({sys.executable})",
    )
    in_env = "envs" in sys.prefix or sys.prefix != sys.base_prefix
    record("Running inside an isolated environment", in_env, f"prefix={sys.prefix}")


# --------------------------------------------------------------------------
# 2. Imports
# --------------------------------------------------------------------------
def check_imports() -> dict[str, str]:
    section("2. Package imports")
    versions: dict[str, str] = {}
    for mod_name, attr in PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, attr, "unknown")
            versions[mod_name] = str(ver)
            record(f"import {mod_name}", True, str(ver))
        except Exception as exc:  # noqa: BLE001 - we want the class name in the report
            versions[mod_name] = "MISSING"
            record(f"import {mod_name}", False, f"{type(exc).__name__}: {exc}")

    # Guard against the classic breakage: more than one OpenCV variant installed.
    try:
        from importlib.metadata import distributions

        variants = sorted(
            d.metadata["Name"]
            for d in distributions()
            if (d.metadata["Name"] or "").lower().startswith("opencv")
        )
        record(
            "Exactly one OpenCV variant installed",
            len(variants) == 1,
            ", ".join(variants) if variants else "none found",
        )
    except Exception as exc:  # noqa: BLE001
        record("OpenCV variant check", False, f"{type(exc).__name__}: {exc}")

    return versions


# --------------------------------------------------------------------------
# 3. Accelerator
# --------------------------------------------------------------------------
def check_accelerator() -> None:
    section("3. Accelerator")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        record("torch available for accelerator check", False, str(exc))
        return

    cuda = torch.cuda.is_available()
    mps = (
        bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    )
    print(f"  torch.cuda.is_available()         = {cuda}")
    print(f"  torch.backends.mps.is_available() = {mps}")
    print(f"  torch.version.cuda                = {torch.version.cuda}")

    if cuda:
        idx = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        arch_list = torch.cuda.get_arch_list()
        print(
            f"  device name                       = {torch.cuda.get_device_name(idx)}"
        )
        print(f"  compute capability                = sm_{cap[0]}{cap[1]}")
        print(f"  arch list in this build           = {arch_list}")
        record(
            "This GPU architecture is compiled into the wheel",
            f"sm_{cap[0]}{cap[1]}" in arch_list,
            f"sm_{cap[0]}{cap[1]}",
        )

    dev = "cuda" if cuda else ("mps" if mps else "cpu")
    record("Accelerator detected", dev != "cpu", f"using device {dev!r}")

    try:
        x = torch.rand(3, 3, device=dev)
        y = x @ x
        if dev == "cuda":
            torch.cuda.synchronize()
        finite = bool(torch.isfinite(y).all())
        record(f"Tensor matmul on {dev!r}", finite, f"result shape {tuple(y.shape)}")
        print(f"  x @ x =\n{y}")
    except Exception as exc:  # noqa: BLE001
        record(f"Tensor matmul on {dev!r}", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# 4. Webcam
# --------------------------------------------------------------------------
def check_webcam():
    """Return a BGR frame from device 0, or None."""
    section("4. Webcam capture")
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        record("cv2 available for webcam check", False, str(exc))
        return None

    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("default", cv2.CAP_ANY),
    ]
    for label, backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            cap.release()
            print(f"  {label}: could not open device 0")
            continue
        # Discard a few warm-up frames; the first are often black or malformed.
        frame = None
        for _ in range(10):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()
        if frame is not None:
            record(
                "Webcam device 0 -> one frame",
                True,
                f"backend {label}, shape {frame.shape}",
            )
            return frame
        print(f"  {label}: opened but returned no frame")

    record(
        "Webcam device 0 -> one frame",
        False,
        "no backend produced a frame (camera absent, in use, or blocked by privacy settings)",
    )
    return None


# --------------------------------------------------------------------------
# 5. MediaPipe Face Landmarker (Tasks API)
# --------------------------------------------------------------------------
def check_landmarker(frame) -> None:
    section("5. MediaPipe Face Landmarker")
    if not MODEL_PATH.exists():
        record("face_landmarker.task present", False, f"missing at {MODEL_PATH}")
        return
    record(
        "face_landmarker.task present",
        True,
        f"{MODEL_PATH.stat().st_size / 1e6:.2f} MB",
    )

    if frame is None:
        record("Face detected in webcam frame", False, "no frame available from step 4")
        return

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

        n_faces = len(result.face_landmarks)
        if n_faces:
            pts = result.face_landmarks[0]
            record(
                "Face detected in webcam frame",
                True,
                f"{n_faces} face, {len(pts)} landmarks "
                f"(first: x={pts[0].x:.3f}, y={pts[0].y:.3f}, z={pts[0].z:.3f})",
            )
        else:
            record(
                "Face detected in webcam frame",
                False,
                "landmarker ran cleanly but saw no face -- sit in frame with light on your face",
            )
    except Exception as exc:  # noqa: BLE001
        record("Face Landmarker (Tasks API)", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the FaceAI environment.")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="skip the webcam and landmarker checks (headless machines / CI)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("FaceAI environment verification")
    print(f"{platform.platform()} | {platform.machine()}")
    print(f"project root: {PROJECT_ROOT}")
    print("=" * 72)

    check_python()
    check_imports()
    check_accelerator()

    if args.no_camera:
        section("4-5. Webcam + Face Landmarker")
        print("  SKIPPED (--no-camera)")
    else:
        frame = check_webcam()
        check_landmarker(frame)

    section("SUMMARY")
    failed = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        print(
            f"  {'PASS' if ok else 'FAIL'}  {name}"
            + (f"  ({detail})" if detail else "")
        )
    print()
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\n  FAILED:")
        for name in failed:
            print(f"    - {name}")
        return 1
    print("\n  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
