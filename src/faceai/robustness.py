"""Phase 3 - robustness tests on the committed 5 folds.

    python -m faceai.robustness        # or: make robustness

Robustness asks whether a result survives conditions it was not fitted under. Seven tests,
all on dev folds (the held-out test set is not touched):

  noise injection   models fitted on CLEAN training-fold features score validation folds
                    whose PPG was corrupted with pulse-band noise (in-band SNR +20 .. -10 dB,
                    see `faceai.features`). A proxy for the contact-PPG -> camera-rPPG gap.
  state shift       models fitted on REST sessions only score rest and post-exercise
                    sessions of unseen subjects: a distribution shift in heart rate and
                    pulse shape with (for most targets) the same label.
  subgroups         pooled out-of-fold error by sex, age band (<25 / >=25) and state, with
                    subject-cluster bootstrap CIs. Small groups get wide CIs - reported as is.
  stability         HbA1c and cholesterol were recorded once per person (identical before
                    and after exercise for >99.7% of subjects), so a prediction that moves
                    between a person's two sessions is tracking the session, not the label.
  camera sampling   the contact pulse read at each camera's real frame times (see
                    `faceai.features`): models fitted on clean features score camera-sampled
                    ones, and are also refitted on camera-sampled features. Isolates the
                    camera's temporal sampling only - not its optics.
  recording length  the same two comparisons on features from only the first 30, 60 or
                    120 s of each recording: how long must a user hold still?
  quality gate      "verify every recording": a model predicts only when the recording's
                    template-match quality is at least the 5th percentile of its own training
                    fold, and otherwise falls back to the training-fold mean (abstains).
                    Scored at every noise level and camera variant, with coverage reported,
                    plus a sweep of that percentile for the error-coverage trade-off.

Adversarial testing (crafted worst-case inputs) is not run: it is meaningful for models on
raw signals and is deferred to the Phase 8 deep models.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV

from faceai.cv import (
    OUT_DIR,
    TARGETS,
    build_model,
    inputs,
    mae,
    session_table,
)
from faceai.features import (
    CAMERAS,
    DURATIONS_S,
    MODEL_FEATURES,
    SNR_LEVELS_DB,
    features_path,
    load_features,
)
from faceai.provenance import record
from faceai.splits import N_FOLDS, OLDER_THAN, SEED

ROBUST_MODELS = ["Mean", "Ridge (L2)", "LightGBM"]
STABLE_TARGETS = ["hba1c", "cholesterol"]
N_BOOT = 2000
REFIT_MODELS = ["Ridge (L2)", "LightGBM"]
GATE_SQI = "sqi_template_r"
GATE_QUANTILE = 0.05  # headline: abstain below the 5th percentile of training quality
# The threshold sweep behind the error-coverage trade-off. The headline above was fixed
# before any gate result was seen; the sweep is for choosing a deployment setting later.
GATE_SWEEP = (0.0, 0.01, 0.05, 0.10, 0.25)


def _fit(name: str, tr: pd.DataFrame, target: str):
    est, uses_ppg = build_model(name)
    cols = inputs(tr, target, uses_ppg)
    if isinstance(est, GridSearchCV):
        est.fit(tr[cols], tr[target], groups=tr["patient_id"])
    else:
        est.fit(tr[cols], tr[target])
    return est, cols


def noisy_tables() -> dict[str, pd.DataFrame]:
    """Validation features per SNR level, keyed like `session_table` rows."""
    out = {}
    for snr in SNR_LEVELS_DB:
        f = load_features(snr)
        out[f"{snr:+d} dB"] = f[f["readable"]].set_index(["patient_id", "step"])[
            MODEL_FEATURES
        ]
    return out


def camera_tables() -> dict[str, pd.DataFrame]:
    """Camera-sampled features per camera, keyed like `session_table` rows."""
    out = {}
    for cam in CAMERAS:
        f = load_features(camera=cam)
        out[f"camera {cam}"] = f[f["readable"]].set_index(["patient_id", "step"])[
            MODEL_FEATURES
        ]
    return out


def duration_tables() -> dict[str, pd.DataFrame]:
    """Shortened-recording features, keyed like `session_table` rows."""
    out = {}
    for d in DURATIONS_S:
        f = load_features(duration_s=d)
        out[f"first {d} s"] = f[f["readable"]].set_index(["patient_id", "step"])[
            MODEL_FEATURES
        ]
    return out


def _with_features(rows: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """`rows` with its PPG features replaced by the same sessions' rows of `feats`."""
    key = pd.MultiIndex.from_frame(rows[["patient_id", "step"]])
    out = rows.copy()
    out[MODEL_FEATURES] = feats.loc[key].to_numpy()
    return out


def gate_predictions(
    pred: np.ndarray, sqi: np.ndarray, tau: float, fallback: float
) -> tuple[np.ndarray, np.ndarray]:
    """Prediction where quality >= tau, else the fallback. NaN quality never passes."""
    with np.errstate(invalid="ignore"):
        ok = np.asarray(sqi, dtype=float) >= tau
    return np.where(ok, pred, fallback), ok


def _noise_fold(t: pd.DataFrame, noisy: dict, target: str, fold: int) -> list[dict]:
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    key = pd.MultiIndex.from_frame(va[["patient_id", "step"]])
    rows = []
    for name in ROBUST_MODELS:
        est, cols = _fit(name, tr, target)
        rows.append(
            {
                "target": target,
                "model": name,
                "fold": fold,
                "level": "clean",
                "val_mae": mae(va[target], est.predict(va[cols])),
            }
        )
        for level, feats in noisy.items():
            vn = va.copy()
            vn[MODEL_FEATURES] = feats.loc[key].to_numpy()
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "fold": fold,
                    "level": level,
                    "val_mae": mae(vn[target], est.predict(vn[cols])),
                }
            )
    return rows


def _shift_fold(t: pd.DataFrame, target: str, fold: int) -> list[dict]:
    tr = t[(t["fold"] != fold) & (t["step"] == "before")]
    va = t[t["fold"] == fold]
    rows = []
    for name in ROBUST_MODELS:
        est, cols = _fit(name, tr, target)
        for step in ["before", "after"]:
            v = va[va["step"] == step]
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "fold": fold,
                    "trained_on": "rest only",
                    "scored_on": step,
                    "n": len(v),
                    "val_mae": mae(v[target], est.predict(v[cols])),
                }
            )
    return rows


def _variant_fold(t: pd.DataFrame, cams: dict, target: str, fold: int) -> list[dict]:
    """Clean-fitted models scored on each variant, plus models refitted on that variant."""
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    rows = []
    for name in ROBUST_MODELS:  # fitted on clean contact features
        est, cols = _fit(name, tr, target)
        for level, feats in [("clean", None), *cams.items()]:
            v = va if feats is None else _with_features(va, feats)
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "fold": fold,
                    "trained_on": "clean",
                    "level": level,
                    "val_mae": mae(v[target], est.predict(v[cols])),
                }
            )
    for level, feats in cams.items():  # refitted on the same variant
        tc = _with_features(t, feats)
        trc, vac = tc[tc["fold"] != fold], tc[tc["fold"] == fold]
        for name in REFIT_MODELS:
            with warnings.catch_warnings():
                # Expected for the 30-s variant: the respiration features are NaN by
                # design (RESP_MIN_S = 40 s) and the imputer drops all-empty columns.
                warnings.filterwarnings(
                    "ignore", message="Skipping features without any observed values"
                )
                est, cols = _fit(name, trc, target)
                pred = est.predict(vac[cols])
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "fold": fold,
                    "trained_on": "same variant",
                    "level": level,
                    "val_mae": mae(vac[target], pred),
                }
            )
    return rows


def _gate_fold(t: pd.DataFrame, variants: dict, target: str, fold: int) -> list[dict]:
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    fallback = float(tr[target].mean())
    rows = []
    for name in REFIT_MODELS:
        est, cols = _fit(name, tr, target)
        for level, feats in variants.items():
            v = va if feats is None else _with_features(va, feats)
            y, pred = v[target].to_numpy(), est.predict(v[cols])
            for q in GATE_SWEEP:
                tau = float(np.nanquantile(tr[GATE_SQI], q))
                gated, ok = gate_predictions(
                    pred, v[GATE_SQI].to_numpy(), tau, fallback
                )
                rows.append(
                    {
                        "target": target,
                        "model": name,
                        "fold": fold,
                        "level": level,
                        "quantile": q,
                        "headline": q == GATE_QUANTILE,
                        "n": len(v),
                        "tau": tau,
                        "coverage": float(ok.mean()),
                        "mae_ungated": mae(y, pred),
                        "mae_gated": mae(y, gated),
                        "mae_constant": mae(y, np.full(len(y), fallback)),
                        "mae_covered": mae(y[ok], pred[ok]) if ok.any() else np.nan,
                    }
                )
    return rows


def _cluster_ci(
    err: np.ndarray, subject: np.ndarray, rng: np.random.Generator
) -> list[float]:
    codes, uniq = pd.factorize(subject)
    s_err = np.bincount(codes, weights=err)
    s_n = np.bincount(codes).astype(float)
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    boot = s_err[draws].sum(axis=1) / s_n[draws].sum(axis=1)
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def subgroups(t: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    meta = t[["patient_id", "step", "sex", "age"]].rename(columns={"age": "age_years"})
    d = oof.merge(meta, on=["patient_id", "step"], how="left", validate="m:1")
    d["age band"] = np.where(
        d["age_years"] >= OLDER_THAN, f">= {OLDER_THAN}", f"< {OLDER_THAN}"
    )
    d["state"] = d["step"].map({"before": "rest", "after": "post-exercise"})
    rng = np.random.default_rng(SEED)
    rows = []
    for target in TARGETS:
        dt_ = d[d["target"] == target]
        for factor in ["sex", "age band", "state"]:
            for level, g in dt_.groupby(factor):
                for m in [
                    "Mean",
                    "Demographics",
                    "Ridge (L2)",
                    "Lasso (L1)",
                    "LightGBM",
                ]:
                    err = np.abs(g[m].to_numpy() - g["y"].to_numpy())
                    rows.append(
                        {
                            "target": target,
                            "factor": factor,
                            "level": level,
                            "model": m,
                            "n_sessions": len(g),
                            "n_subjects": g["patient_id"].nunique(),
                            "mae": float(err.mean()),
                            "ci": _cluster_ci(err, g["patient_id"].to_numpy(), rng),
                        }
                    )
    return pd.DataFrame(rows)


def stability(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in STABLE_TARGETS:
        d = oof[oof["target"] == target]
        wide = d.pivot(
            index="patient_id",
            columns="step",
            values=["y", "Ridge (L2)", "Lasso (L1)", "LightGBM"],
        )
        both = wide.dropna()
        label_shift = (both[("y", "after")] - both[("y", "before")]).abs()
        for m in ["Ridge (L2)", "Lasso (L1)", "LightGBM"]:
            shift = (both[(m, "after")] - both[(m, "before")]).abs()
            rows.append(
                {
                    "target": target,
                    "model": m,
                    "subjects": len(both),
                    "label_identical_pct": float(100 * np.mean(label_shift == 0)),
                    "pred_shift_median": float(shift.median()),
                    "pred_shift_p90": float(shift.quantile(0.9)),
                    "pred_shift_max": float(shift.max()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    t = session_table()
    noisy = noisy_tables()
    jobs = [(target, k) for target in TARGETS for k in range(N_FOLDS)]
    res = Parallel(n_jobs=12)(delayed(_noise_fold)(t, noisy, tg, k) for tg, k in jobs)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(
        OUT_DIR / "noise.csv", index=False
    )
    res = Parallel(n_jobs=12)(delayed(_shift_fold)(t, tg, k) for tg, k in jobs)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(
        OUT_DIR / "shift.csv", index=False
    )
    cams, durs = camera_tables(), duration_tables()
    for name, var in [("camera", cams), ("duration", durs)]:
        res = Parallel(n_jobs=12)(
            delayed(_variant_fold)(t, var, tg, k) for tg, k in jobs
        )
        pd.DataFrame([r for rows in res for r in rows]).to_csv(
            OUT_DIR / f"{name}.csv", index=False
        )
    variants = {"clean": None, **noisy, **cams, **durs}
    res = Parallel(n_jobs=12)(delayed(_gate_fold)(t, variants, tg, k) for tg, k in jobs)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(
        OUT_DIR / "gate.csv", index=False
    )
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    subgroups(t, oof).to_json(OUT_DIR / "subgroups.json", orient="records", indent=1)
    stability(oof).to_csv(OUT_DIR / "stability.csv", index=False)
    inputs = [features_path(), *(features_path(s) for s in SNR_LEVELS_DB)]
    inputs += [features_path(camera=c) for c in CAMERAS]
    inputs += [features_path(duration_s=d) for d in DURATIONS_S]
    inputs.append(OUT_DIR / "oof.parquet")
    (OUT_DIR / "robustness_provenance.json").write_text(
        json.dumps(record(inputs), indent=2) + "\n", encoding="utf-8"
    )
    print(
        "noise, shift, camera, duration, gate, subgroups, stability + provenance ->",
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
