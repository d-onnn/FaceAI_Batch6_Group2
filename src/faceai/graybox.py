"""Phase 3e (a) - a gray-box stiffness index from the contact PPG, and what it says about the label.

    python -m faceai.graybox        # or: make graybox

Maaz Ahmad's surrogate-modelling talk (MoSPSyDe group, NTU, 7 Oct 2026) separates white-box
(first-principles), gray-box (hybrid) and black-box (data-driven) models. Our Phase 3 models of
arterial stiffness are black boxes. The gray-box alternative is the digital-volume-pulse
stiffness index of Millasseau et al. (Clin Sci 2002;103:371-377, doi:10.1042/cs1030371):
SI = height / dT, where dT is the time from the systolic peak of the finger pulse to its
diastolic peak or, where there is no second peak, its inflection point. Stiffer arteries
return the reflected wave sooner, so SI rises with age.

dT is read from the same onset-aligned ensemble beat (upsampled to 1 kHz) that
`faceai.features.extract` builds: after the systolic peak, the first local maximum of the
first derivative marks the diastolic wave. If the derivative there is positive the wave is a
true peak, taken where the derivative falls back through zero; otherwise the inflection
itself. Dev sessions only; the held-out test set is not read.

Writes `artifacts/features/graybox.parquet` (dT, SI per session) and `artifacts/cv/graybox.json`:
  - whether dT behaves like a pulse-transit time (longer with height, shorter with age);
  - how the arterial-stiffness label behaves against age, heart rate and SI;
  - whether dT and SI improve the stiffness and age models (same folds, same tuning).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import interpolate, signal, stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from faceai import features as FT
from faceai.clinical import _subject_draws
from faceai.cv import OUT_DIR, build_model, inputs, session_table
from faceai.data import audit
from faceai.provenance import record
from faceai.robustness import _cluster_ci
from faceai.splits import N_FOLDS, OLDER_THAN, SEED

GRAYBOX_PATH = FT.OUT_DIR / "graybox.parquet"
GRAYBOX_FEATURES = ["dt_s", "si_mps"]
DIASTOLIC_WINDOW_MS = (50, 600)  # search after the systolic peak
TEST_TARGETS = ["arterial_stiffness", "age"]
TEST_MODELS = ["Ridge (L2)", "LightGBM"]


def beat_template(x: np.ndarray, fs: float = FT.FS) -> np.ndarray | None:
    """The ensemble-averaged beat of `features.extract`, upsampled to 1 kHz, or None."""
    import neurokit2 as nk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = FT._pulsatile(x, fs)
        try:
            peaks = np.asarray(
                nk.ppg_findpeaks(clean, sampling_rate=fs, method="elgendi")[
                    "PPG_Peaks"
                ],
                dtype=int,
            )
        except Exception:  # noqa: BLE001 - a failed detection is a missing value, not a crash
            return None
    if len(peaks) < FT.MIN_BEATS + 1:
        return None
    ibi = np.diff(peaks) / fs
    ok = FT._valid_ibis(ibi)
    if ok.sum() < FT.MIN_BEATS:
        return None
    w = int(FT.ONSET_SEARCH_S * fs)
    onsets = np.full(len(peaks), -1)
    for i, p in enumerate(peaks):
        lo = max(0, p - w, peaks[i - 1] + 1 if i > 0 else 0)
        if p - lo >= 2:
            onsets[i] = lo + int(np.argmin(clean[lo:p]))
    beat_ok = np.zeros(len(peaks), dtype=bool)
    beat_ok[1:] = ok
    beat_ok &= onsets >= 0
    length = int(np.rint(np.median(ibi[ok]) * fs))
    segs = [
        (clean[o : o + length] - clean[o]) / (clean[p] - clean[o])
        for p, o in zip(peaks[beat_ok], onsets[beat_ok], strict=True)
        if o + length <= len(clean) and clean[p] > clean[o]
    ]
    if len(segs) < FT.MIN_BEATS:
        return None
    t = np.arange(length) / fs
    return interpolate.CubicSpline(t, np.vstack(segs).mean(axis=0))(
        np.arange(0.0, t[-1], 0.001)
    )


def delta_t(fine: np.ndarray) -> tuple[float, str]:
    """(dT in seconds, 'peak' | 'inflection' | 'none') from a 1 kHz beat template."""
    smooth = signal.savgol_filter(fine, window_length=31, polyorder=3)
    d1 = np.gradient(smooth)
    ipk = int(np.argmax(smooth))
    lo, hi = (
        ipk + DIASTOLIC_WINDOW_MS[0],
        min(len(smooth) - 2, ipk + DIASTOLIC_WINDOW_MS[1]),
    )
    if hi - lo < 10:
        return np.nan, "none"
    seg = d1[lo:hi]
    k = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))
    if not len(k):
        return np.nan, "none"
    i = lo + 1 + int(k[0])
    if d1[i] > 0:  # a true diastolic peak: where the derivative returns through zero
        after = np.flatnonzero(d1[i:hi] <= 0)
        if not len(after):
            return np.nan, "none"
        return (i + int(after[0]) - ipk) / 1000.0, "peak"
    return (i - ipk) / 1000.0, "inflection"


def _session(patient_id: int, step: str) -> dict:
    fine = beat_template(FT.load_session_ppg(patient_id, step))
    dt, kind = (np.nan, "no template") if fine is None else delta_t(fine)
    return {"patient_id": patient_id, "step": step, "dt_s": dt, "dt_kind": kind}


def build(n_jobs: int = 12) -> pd.DataFrame:
    sessions = session_table()[["patient_id", "step"]]  # dev sessions with readable PPG
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_session)(p, s)
        for p, s in zip(sessions["patient_id"], sessions["step"], strict=True)
    )
    g = pd.DataFrame(rows)
    height = audit.load_db().drop_duplicates(["patient_id", "step"])[
        ["patient_id", "step", "height"]
    ]
    g = g.merge(height, on=["patient_id", "step"], how="left", validate="1:1")
    g["si_mps"] = g["height"] / 100.0 / g["dt_s"]
    return g


def load_graybox() -> pd.DataFrame:
    if not GRAYBOX_PATH.exists():
        raise FileNotFoundError(
            f"{GRAYBOX_PATH} missing - run `python -m faceai.graybox`"
        )
    return pd.read_parquet(GRAYBOX_PATH)


# --------------------------------------------------------------------------- analyses
def rho_ci(
    a: np.ndarray, b: np.ndarray, subjects: np.ndarray, rng: np.random.Generator
) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, subjects = a[ok], b[ok], subjects[ok]
    boot = [
        stats.spearmanr(a[i], b[i]).statistic for i in _subject_draws(subjects, rng)
    ]
    return {
        "rho": float(stats.spearmanr(a, b).statistic),
        "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
        "n_sessions": int(ok.sum()),
    }


def band_difference(
    v: np.ndarray, age: np.ndarray, subjects: np.ndarray, rng: np.random.Generator
) -> dict:
    """Mean of v for age >= OLDER_THAN minus age < OLDER_THAN, subject-cluster bootstrap CI."""
    ok = np.isfinite(v)
    v, older, subjects = v[ok], age[ok] >= OLDER_THAN, subjects[ok]
    point = float(v[older].mean() - v[~older].mean())
    boot = []
    for i in _subject_draws(subjects, rng):
        o = older[i]
        if 0 < o.sum() < len(i):
            boot.append(v[i][o].mean() - v[i][~o].mean())
    return {
        "older_minus_younger": point,
        "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
        "older_mean": float(v[older].mean()),
        "younger_mean": float(v[~older].mean()),
    }


def _fit_fold(
    t: pd.DataFrame, target: str, fold: int, model: str, extra: list[str]
) -> pd.DataFrame:
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    est, uses_ppg = build_model(model)
    cols = [*inputs(t, target, uses_ppg), *extra] if uses_ppg else [*extra]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if isinstance(est, GridSearchCV):
            est.fit(tr[cols], tr[target], groups=tr["patient_id"])
        else:
            est.fit(tr[cols], tr[target])
    return pd.DataFrame(
        {
            "patient_id": va["patient_id"].to_numpy(),
            "step": va["step"].to_numpy(),
            "fold": fold,
            "y": va[target].to_numpy(),
            "pred": est.predict(va[cols]),
        }
    )


def predictive(
    t: pd.DataFrame, oof: pd.DataFrame, rng: np.random.Generator, n_jobs: int = 12
) -> list[dict]:
    """Phase 3 model with dT and SI added, against the same model without them."""
    jobs = [
        (tg, m, k) for tg in TEST_TARGETS for m in TEST_MODELS for k in range(N_FOLDS)
    ]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(t, tg, k, m, GRAYBOX_FEATURES) for tg, m, k in jobs
    )
    out = []
    for (tg, m), group in pd.concat(
        [
            p.assign(target=tg, model=m)
            for p, (tg, m, _) in zip(parts, jobs, strict=True)
        ]
    ).groupby(["target", "model"]):
        base = oof[oof["target"] == tg][["patient_id", "step", m, "y"]].rename(
            columns={m: "base"}
        )
        d = group.merge(base, on=["patient_id", "step", "y"], validate="1:1")
        e_new, e_old = (
            np.abs(d["pred"] - d["y"]).to_numpy(),
            np.abs(d["base"] - d["y"]).to_numpy(),
        )
        ids = d["patient_id"].to_numpy()
        out.append(
            {
                "target": tg,
                "model": m,
                "mae_phase3": float(e_old.mean()),
                "mae_with_graybox": float(e_new.mean()),
                "change": float((e_new - e_old).mean()),
                "change_ci": _cluster_ci(e_new - e_old, ids, rng),
            }
        )
    return out


def analyse(t: pd.DataFrame, g: pd.DataFrame, oof: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    d = t.merge(g, on=["patient_id", "step"], how="left", validate="1:1")
    ids = d["patient_id"].to_numpy()
    col = lambda c: d[c].to_numpy(dtype=float)
    validity = {
        "dt_vs_height": rho_ci(col("dt_s"), col("height"), ids, rng),
        "dt_vs_age": rho_ci(col("dt_s"), col("age"), ids, rng),
        "si_vs_age": rho_ci(col("si_mps"), col("age"), ids, rng),
        "si_by_age_band": band_difference(col("si_mps"), col("age"), ids, rng),
    }
    label = {
        "label_vs_age": rho_ci(col("arterial_stiffness"), col("age"), ids, rng),
        "label_vs_heart_rate": rho_ci(
            col("arterial_stiffness"), col("hr_mean_bpm"), ids, rng
        ),
        "label_vs_si": rho_ci(col("arterial_stiffness"), col("si_mps"), ids, rng),
        "label_vs_dt": rho_ci(col("arterial_stiffness"), col("dt_s"), ids, rng),
        "label_by_age_band": band_difference(
            col("arterial_stiffness"), col("age"), ids, rng
        ),
    }
    w = d.pivot_table(
        index="patient_id",
        columns="step",
        values=["arterial_stiffness", "si_mps", "hr_mean_bpm"],
    ).dropna()
    change = lambda c: (w[(c, "after")] - w[(c, "before")]).to_numpy()
    sid = w.index.to_numpy()
    within = {
        "dlabel_vs_dsi": rho_ci(
            change("arterial_stiffness"), change("si_mps"), sid, rng
        ),
        "dlabel_vs_dhr": rho_ci(
            change("arterial_stiffness"), change("hr_mean_bpm"), sid, rng
        ),
    }
    return {
        "sessions": len(g),
        "dt_kind_counts": g["dt_kind"].value_counts().to_dict(),
        "dt_summary_s": {
            k: float(v)
            for k, v in g["dt_s"].describe()[["mean", "50%", "min", "max"]].items()
        },
        "validity": validity,
        "label": label,
        "within_person": within,
        "predictive": predictive(
            t.merge(
                g[["patient_id", "step", *GRAYBOX_FEATURES]],
                on=["patient_id", "step"],
                how="left",
            ),
            oof,
            rng,
        ),
    }


def main() -> None:
    g = build()
    GRAYBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(GRAYBOX_PATH, index=False)
    t = session_table()
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    out = {
        **analyse(t, g, oof),
        "provenance": record([GRAYBOX_PATH, OUT_DIR / "oof.parquet"]),
    }
    (OUT_DIR / "graybox.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "dt_kind_counts",
                    "validity",
                    "label",
                    "within_person",
                    "predictive",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
