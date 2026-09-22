"""Phase 1 integrity checks on the released data.

Each function returns structured results so `faceai.eda` can render them and the tests
can assert on them. Nothing here silently repairs data: problems are reported, and the
one known defect (subject 4952) is resolved by measurement, never by assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy import signal

from faceai.data import formats as F
from faceai.data.hub import LOCAL_DIR
from faceai.targets import TargetConfig

# db.csv lists `ppg`/`ecg` for this subject against the OPPOSITE session to its own
# `step`, in all 6 of its rows. Every other subject is self-consistent.
SUSPECT_SUBJECT = 4952

# Dropped from every analysis, with the evidence. Later phases MUST apply these
# before splitting. Found by scanning all 3600 ppg_sync files, not a sample.
EXCLUDED_SUBJECTS: dict[int, str] = {
    4952: (
        "Two independent defects. (1) All 6 rows point `ppg`/`ecg` at the opposite "
        "session to their own `step`, uniquely among 600 subjects. (2) All 6 of its "
        "ppg_sync files are CONSTANT (150 before, 38 after; std=0), i.e. the synced "
        "PPG is a flat line carrying no pulse signal at all. The correct before/after "
        "pairing could not be established: the `pulse` label agrees with simultaneous "
        "ECG only to ~9 bpm MAE, which cannot adjudicate the ~6 bpm difference between "
        "the two candidate files. Excluded rather than guessed."
    )
}

# Session-level defects: the subject's OTHER session is fine and is kept.
EXCLUDED_SESSIONS: dict[tuple[int, str], str] = {
    (7412, "after"): (
        "All 3 ppg_sync files for this session are CONSTANT (value 8.0, std=0) - no "
        "pulse signal. The `before` session is healthy and is retained."
    )
}


def apply_exclusions(db: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop known-bad subjects and sessions. Always call this before splitting."""
    before = len(db)
    keep = ~db["patient_id"].isin(EXCLUDED_SUBJECTS)
    for subject, step in EXCLUDED_SESSIONS:
        keep &= ~((db["patient_id"] == subject) & (db["step"] == step))
    out = db[keep]
    return out, {
        "rows_before": before,
        "rows_after": len(out),
        "rows_dropped": before - len(out),
        "subjects_before": db["patient_id"].nunique(),
        "subjects_after": out["patient_id"].nunique(),
        "excluded_subjects": list(EXCLUDED_SUBJECTS),
        "excluded_sessions": [list(k) for k in EXCLUDED_SESSIONS],
    }


LABEL_EXCLUDED_COLUMNS = {
    "step",
    "camera",
    "view",
    "ecg",
    "ppg",
    "video",
    "meta",
    "ppg_sync",
}


def load_db() -> pd.DataFrame:
    return pd.read_csv(LOCAL_DIR / "db.csv")


def rounding_tolerance(value: float) -> float:
    """Half a unit in the last printed decimal: Table 2 bounds are rounded."""
    return 0.5 * 10 ** int(Decimal(str(value)).as_tuple().exponent)


# --------------------------------------------------------------------------------
HR_BAND_HZ = (0.7, 4.0)  # 42-240 bpm, the usual band for HR from PPG


def estimate_hr_bpm(pw: F.PulseWave, fs: float = 100.0) -> float:
    """Heart rate from a contact-PPG waveform via the Welch spectral peak.

    Resampling first is essential: .PW samples arrive in bursts, so treating them as
    evenly spaced would shift the spectral peak.
    """
    _, v = pw.resample(fs)
    v = signal.detrend(v)
    lo, hi = HR_BAND_HZ
    b, a = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    v = signal.filtfilt(b, a, v)
    freqs, power = signal.welch(v, fs=fs, nperseg=int(min(len(v), fs * 20)))
    band = (freqs >= lo) & (freqs <= hi)
    return float(60.0 * freqs[band][np.argmax(power[band])])


@dataclass
class PairingEvidence:
    """Which contact-PPG file belongs to a session, judged against its `pulse` label.

    An earlier attempt correlated the .PW waveform against ppg_sync directly. It
    FAILED on known-good control subjects (r ~ 0.04-0.55, mostly inconclusive):
    ppg_sync carries no timestamps, so the series can only be stretched onto a shared
    timeline, and sub-second drift destroys the correlation of a ~1 Hz waveform.
    Heart rate is insensitive to that alignment, and `pulse` is an INDEPENDENT label
    measured with a tonometer, so it can adjudicate.
    """

    subject: int
    step: str
    pulse_label: float
    hr_same_step: float
    hr_opposite_step: float

    @property
    def err_same(self) -> float:
        return abs(self.hr_same_step - self.pulse_label)

    @property
    def err_opposite(self) -> float:
        return abs(self.hr_opposite_step - self.pulse_label)

    @property
    def verdict(self) -> str:
        if abs(self.err_same - self.err_opposite) < 5.0:  # bpm
            return "inconclusive"
        return "filename" if self.err_same < self.err_opposite else "db.csv path"


def check_session_pairing(db: pd.DataFrame, subject: int) -> list[PairingEvidence]:
    """Decide empirically which contact-PPG file belongs to each session.

    `hr_same_step` is the HR of the file the FILENAME convention implies
    ({subject}_{step}); `hr_opposite_step` is the other session's file, which is what
    db.csv's `ppg` column points at for the suspect subject. Whichever HR is closer to
    the session's `pulse` label wins.
    """
    out = []
    sessions = db[db["patient_id"] == subject].drop_duplicates(subset=["step"])
    for _, row in sessions.iterrows():
        step = row["step"]
        other = "before" if step == "after" else "after"
        same = LOCAL_DIR / f"ppg/{subject}_{step}.PW"
        opposite = LOCAL_DIR / f"ppg/{subject}_{other}.PW"
        if not (same.exists() and opposite.exists()):
            continue
        out.append(
            PairingEvidence(
                subject=subject,
                step=step,
                pulse_label=float(row["pulse"]),
                hr_same_step=estimate_hr_bpm(F.read_pw(same)),
                hr_opposite_step=estimate_hr_bpm(F.read_pw(opposite)),
            )
        )
    return out


def check_degenerate_signals(pattern: str = "ppg_sync/*.txt") -> pd.DataFrame:
    """Find flat-line signals. All six of subject 4952's ppg_sync files are constant."""
    rows = []
    for path in sorted(LOCAL_DIR.glob(pattern)):
        values = np.loadtxt(path, usecols=0)
        if values.std() == 0 or len(np.unique(values)) < 10:
            rows.append(
                {
                    "file": path.name,
                    "subject": int(path.name.split("_")[0]),
                    "distinct_values": len(np.unique(values)),
                    "constant_value": float(values[0]) if values.std() == 0 else None,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------
def check_path_consistency(db: pd.DataFrame) -> pd.DataFrame:
    """Do db.csv's path columns agree with the filename convention, and exist?"""
    rows = []
    for col, template in (
        ("video", "{p}_{c}_{s}"),
        ("meta", "{p}_{c}_{s}"),
        ("ppg_sync", "{p}_{c}_{s}"),
        ("ppg", "{p}_{s}"),
        ("ecg", "{p}_{s}"),
    ):
        expected = db.apply(
            lambda r, t=template: t.format(
                p=r["patient_id"], c=r["camera"], s=r["step"]
            ),
            axis=1,
        )
        agree = np.array(
            [e in v for e, v in zip(expected, db[col].astype(str), strict=True)]
        )
        present = db[col].dropna().unique()
        on_disk = sum((LOCAL_DIR / p).exists() for p in present)
        rows.append(
            {
                "column": col,
                "rows_disagreeing_with_filename_convention": int((~agree).sum()),
                "unique_paths": len(present),
                "present_on_disk": on_disk,
            }
        )
    return pd.DataFrame(rows)


def check_labels_constant_across_cameras(db: pd.DataFrame, cfg: TargetConfig) -> dict:
    """The 3 camera rows of one session must carry identical labels.

    If they do, the analysis unit is the session (1200), not the recording (3600),
    and computing statistics over 3600 rows would inflate n threefold.
    """
    label_cols = [
        c for c in db.columns if c not in LABEL_EXCLUDED_COLUMNS | {"patient_id"}
    ]
    grouped = db.groupby(["patient_id", "step"])[label_cols].nunique()
    varying = grouped[(grouped > 1).any(axis=1)]
    return {
        "sessions": len(grouped),
        "label_columns_checked": len(label_cols),
        "sessions_with_varying_labels": len(varying),
        "columns_that_vary": sorted(grouped.columns[(grouped > 1).any()].tolist()),
    }


def check_out_of_range(db: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """Observed values against the declared Table 2 ranges, allowing for rounding."""
    rows = []
    for t in cfg.modellable():
        if t.clinical_range is None:
            continue
        lo, hi = t.clinical_range
        col = db[t.column]
        below = int((col < lo - rounding_tolerance(lo)).sum())
        above = int((col > hi + rounding_tolerance(hi)).sum())
        rows.append(
            {
                "target": t.name,
                "column": t.column,
                "declared": f"{lo} .. {hi}",
                "observed": f"{col.min():.4g} .. {col.max():.4g}",
                "below": below,
                "above": above,
            }
        )
    return pd.DataFrame(rows)


def check_durations(db: pd.DataFrame, n: int = 25, seed: int = 0) -> pd.DataFrame:
    """Sample recordings and compare signal lengths against each other."""
    rng = np.random.default_rng(seed)
    sample = db.iloc[rng.choice(len(db), size=min(n, len(db)), replace=False)]
    rows = []
    for _, row in sample.iterrows():
        meta_p, sync_p = LOCAL_DIR / row["meta"], LOCAL_DIR / row["ppg_sync"]
        pw_p, ecg_p = LOCAL_DIR / row["ppg"], LOCAL_DIR / row["ecg"]
        if not all(p.exists() for p in (meta_p, sync_p, pw_p, ecg_p)):
            continue
        meta, sync = F.read_meta(meta_p), F.read_ppg_sync(sync_p)
        pw, ecg = F.read_pw(pw_p), F.read_ecg(ecg_p)
        rows.append(
            {
                "camera": row["camera"],
                "frames": meta.n_frames,
                "video_s": meta.duration_s,
                "fps_mean": meta.fps_mean,
                "sync_rows": sync.n_rows,
                "frames_minus_sync": meta.n_frames - sync.n_rows,
                "ppg_s": pw.duration_s,
                "ppg_hz": pw.mean_rate_hz,
                "ppg_clipped_pct": 100 * pw.clipped_fraction,
                "ecg_s": float(ecg.time[-1]),
                "ecg_leads": len(ecg.leads),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the integrity checks. The full write-up is `python -m faceai.eda`."""
    from faceai.targets import load

    cfg, db = load(), load_db()
    kept, summary = apply_exclusions(db)
    print("=== db.csv paths vs the filename convention ===")
    print(check_path_consistency(db).to_string(index=False))
    print("\n=== labels constant across the 3 cameras of a session ===")
    print(check_labels_constant_across_cameras(db, cfg))
    print("\n=== observed values vs paper Table 2 ranges ===")
    print(check_out_of_range(db, cfg).to_string(index=False))
    print("\n=== degenerate (flat-line) ppg_sync files ===")
    degenerate = check_degenerate_signals()
    print(degenerate.to_string(index=False) if len(degenerate) else "none")
    print("\n=== exclusions ===")
    for subject, reason in EXCLUDED_SUBJECTS.items():
        print(f"  subject {subject}: {reason}")
    for (subject, step), reason in EXCLUDED_SESSIONS.items():
        print(f"  subject {subject} [{step}]: {reason}")
    print(f"\n  {summary}")
    print("\n=== signal durations, 25 sampled recordings ===")
    durations = check_durations(kept).describe().loc[["mean", "min", "max"]]
    print(durations.round(2).to_string())


if __name__ == "__main__":
    main()
