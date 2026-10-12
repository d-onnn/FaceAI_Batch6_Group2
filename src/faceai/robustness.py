"""Phase 3 - robustness tests on the committed 5 folds.

    python -m faceai.robustness        # or: make robustness

Robustness asks whether a result survives conditions it was not fitted under. Four tests,
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

Adversarial testing (crafted worst-case inputs) is not run: it is meaningful for models on
raw signals and is deferred to the Phase 8 deep models.
"""

from __future__ import annotations

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
from faceai.features import MODEL_FEATURES, SNR_LEVELS_DB, load_features
from faceai.splits import N_FOLDS, OLDER_THAN, SEED

ROBUST_MODELS = ["Mean", "Ridge (L2)", "LightGBM"]
STABLE_TARGETS = ["hba1c", "cholesterol"]
N_BOOT = 2000


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
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    subgroups(t, oof).to_json(OUT_DIR / "subgroups.json", orient="records", indent=1)
    stability(oof).to_csv(OUT_DIR / "stability.csv", index=False)
    print("noise.csv, shift.csv, subgroups.json, stability.csv ->", OUT_DIR)


if __name__ == "__main__":
    main()
