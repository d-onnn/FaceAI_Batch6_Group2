"""Phase 3 - 5-fold subject-level cross-validation on contact-PPG features.

    python -m faceai.cv        # or: make cv

Uses the COMMITTED folds (`faceai.splits.load_splits`), never a new split. Every model is
fitted on four folds and scored on the fifth, so each dev subject is predicted exactly
once by a model that never saw them. The held-out test set is not touched.

Models, per target:
  Mean / Median    constant predictors fitted on the training folds. The mean is the
                   brief's baseline; the median is the MAE-optimal constant, which is
                   what the paper's "naive baseline" amounts to under an MAE metric.
  Demographics     linear regression on age + sex + BMI (minus whichever is the target).
  Ridge (L2), Lasso (L1), LightGBM
                   on the PPG features plus the same demographics. Hyper-parameters are
                   chosen by an INNER 5-fold GroupKFold inside each training fold, so the
                   outer score is never used to tune the model that it scores.

Uncertainty: 95% CIs from a subject-cluster bootstrap (resampling subjects, not sessions,
because each subject contributes two sessions with near-identical labels). Verdict rule
from the brief: a model "beats" a baseline only if their CIs do not overlap.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from faceai.data import audit
from faceai.data.hub import ROOT
from faceai.features import MODEL_FEATURES, dev_sessions, load_features
from faceai.splits import N_FOLDS, SEED
from faceai.targets import load

OUT_DIR = ROOT / "artifacts" / "cv"
N_BOOT = 2000
INNER_FOLDS = 5

# The eight allocated targets that have labels, plus age as a POSITIVE CONTROL: age is
# known to be recoverable from pulse shape, so if the pipeline cannot find it, the
# pipeline - not the biomarker - is the problem.
TARGETS = [
    "hba1c",
    "cholesterol",
    "spo2",
    "stress_psm25",
    "temperature",
    "respiratory_rate",
    "arterial_stiffness",
    "bmi",
    "age",
]
DEMOGRAPHICS = {"age": "age", "sex": "sex_m", "bmi": "bmi"}  # target name -> column
MODELS = ["Mean", "Median", "Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
BASELINES = ["Mean", "Median", "Demographics"]

# Table 5 of the dataset paper (MAE; the authors' video model on THEIR split). Context
# only - a different split, test set and input (video), so never a like-for-like number.
PAPER_TABLE5 = {
    "hba1c": (0.43, 0.41),
    "cholesterol": (0.66, 0.60),
    "respiratory_rate": (1.36, 1.20),
    "arterial_stiffness": (2.19, 2.04),
    "age": (5.71, 3.91),
    "bmi": (3.18, 3.37),
    "stress_psm25": (1.20, 1.07),
    "spo2": (0.98, 0.98),
}

# Grids were widened after a first run chose their edges (Ridge alpha=1e4 for SpO2 in every
# fold; LightGBM's smallest model almost everywhere): an optimum on the edge is not one.
RIDGE_ALPHAS = np.logspace(-2, 6, 17)
LASSO_ALPHAS = np.logspace(-4, 1, 16)
LGBM_GRID = {"num_leaves": [2, 4, 8, 16], "n_estimators": [25, 50, 100, 300, 600]}


# ---------------------------------------------------------------------------- data
def session_table() -> pd.DataFrame:
    """One row per readable DEV session: fold, PPG features, demographics and labels."""
    cfg = load()
    cols = {t: cfg.by_name(t).column for t in TARGETS}
    db, _ = audit.apply_exclusions(audit.load_db())
    # Labels are identical across a session's three cameras (Phase 1: 0/1200 vary).
    sess = db.drop_duplicates(subset=["patient_id", "step"])
    labels = sess[["patient_id", "step", "sex", *sorted(set(cols.values()))]].rename(
        columns={v: k for k, v in cols.items()}
    )
    feats = load_features()
    feats = feats[feats["readable"]].drop(columns="readable")
    t = dev_sessions().merge(
        feats, on=["patient_id", "step"], how="inner", validate="1:1"
    )
    t = t.merge(labels, on=["patient_id", "step"], how="left", validate="1:1")
    t["sex_m"] = (t["sex"] == "M").astype(float)
    missing = t[TARGETS].isna().sum()
    if missing.any():
        raise ValueError(
            f"labels missing after merge: {missing[missing > 0].to_dict()}"
        )
    return t


def demographic_columns(target: str) -> list[str]:
    return [col for name, col in DEMOGRAPHICS.items() if name != target]


# -------------------------------------------------------------------------- models
def _inner_cv(estimator, grid) -> GridSearchCV:
    return GridSearchCV(
        estimator,
        grid,
        cv=GroupKFold(n_splits=INNER_FOLDS),
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )


def lgbm(**params) -> LGBMRegressor:
    base = {
        "learning_rate": 0.05,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": SEED,
        "n_jobs": 1,
        "verbose": -1,
    }
    return LGBMRegressor(**{**base, **params})


def build_model(name: str):
    """(estimator, uses_ppg_features). Grid-searched models are wrapped in an inner CV."""
    if name == "Mean":
        return DummyRegressor(strategy="mean"), False
    if name == "Median":
        return DummyRegressor(strategy="median"), False
    if name == "Demographics":
        return make_pipeline(StandardScaler(), LinearRegression()), False
    impute_scale = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if name == "Ridge (L2)":
        pipe = Pipeline([*impute_scale, ("model", Ridge())])
        return _inner_cv(pipe, {"model__alpha": RIDGE_ALPHAS}), True
    if name == "Lasso (L1)":
        pipe = Pipeline([*impute_scale, ("model", Lasso(max_iter=100_000))])
        return _inner_cv(pipe, {"model__alpha": LASSO_ALPHAS}), True
    if name == "LightGBM":  # trees handle NaN natively; no imputation or scaling
        return _inner_cv(lgbm(), LGBM_GRID), True
    raise KeyError(name)


def inputs(t: pd.DataFrame, target: str, uses_ppg: bool) -> list[str]:
    return [*(MODEL_FEATURES if uses_ppg else []), *demographic_columns(target)]


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _fit_fold(
    t: pd.DataFrame, target: str, fold: int
) -> tuple[list[dict], pd.DataFrame]:
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    rows, preds = [], {"patient_id": va["patient_id"].values, "step": va["step"].values}
    for name in MODELS:
        est, uses_ppg = build_model(name)
        cols = inputs(t, target, uses_ppg)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if isinstance(est, GridSearchCV):
                est.fit(tr[cols], tr[target], groups=tr["patient_id"])
            else:
                est.fit(tr[cols], tr[target])
        p_tr, p_va = est.predict(tr[cols]), est.predict(va[cols])
        preds[name] = p_va
        rows.append(
            {
                "target": target,
                "model": name,
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(va),
                "train_mae": mae(tr[target], p_tr),
                "val_mae": mae(va[target], p_va),
                "best_params": json.dumps(
                    getattr(est, "best_params_", {}), default=float
                ),
                "convergence_warnings": sum(
                    issubclass(w.category, ConvergenceWarning) for w in caught
                ),
            }
        )
    oof = pd.DataFrame(preds).assign(target=target, fold=fold)
    oof["y"] = va[target].values
    return rows, oof


def run_cv(t: pd.DataFrame, n_jobs: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(target, k) for target in TARGETS for k in range(N_FOLDS)]
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(t, target, k) for target, k in jobs
    )
    folds = pd.DataFrame([r for rows, _ in out for r in rows])
    oof = pd.concat([o for _, o in out], ignore_index=True)
    return folds, oof


# ----------------------------------------------------------------------- bootstrap
def bootstrap(oof: pd.DataFrame, target: str, n_boot: int = N_BOOT) -> dict:
    """Pooled out-of-fold MAE per model, with subject-cluster bootstrap CIs.

    Also returns paired CIs for (model - baseline): the same resampled subjects are
    used for both, so the difference CI is tighter than comparing two marginal CIs.
    """
    d = oof[oof["target"] == target]
    err = np.abs(d[MODELS].to_numpy() - d["y"].to_numpy()[:, None])
    codes, subjects = pd.factorize(d["patient_id"])
    s_err = np.zeros((len(subjects), len(MODELS)))
    np.add.at(s_err, codes, err)
    s_n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
    boot = (
        s_err[draws].sum(axis=1) / s_n[draws].sum(axis=1)[:, None]
    )  # (n_boot, models)
    point = err.mean(axis=0)
    res = {"n_sessions": len(d), "n_subjects": len(subjects), "models": {}}
    for j, m in enumerate(MODELS):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        res["models"][m] = {"mae": float(point[j]), "ci": [float(lo), float(hi)]}
    for j, m in enumerate(MODELS):
        res["models"][m]["vs"] = {}
        for b in BASELINES:
            k = MODELS.index(b)
            diff = boot[:, j] - boot[:, k]
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            # Percent change in MAE vs the baseline (negative = better), so targets in
            # different units can share one axis. Its CI comes from the same draws.
            pct = 100.0 * (boot[:, j] / boot[:, k] - 1.0)
            plo, phi = np.percentile(pct, [2.5, 97.5])
            res["models"][m]["vs"][b] = {
                "diff": float(point[j] - point[k]),
                "ci": [float(dlo), float(dhi)],
                "pct": float(100.0 * (point[j] / point[k] - 1.0)),
                "pct_ci": [float(plo), float(phi)],
            }
    return res


def verdict(res: dict) -> dict:
    """Brief rule: beats a baseline only with NON-OVERLAPPING CIs (stricter than paired)."""
    mean_ci = res["models"]["Mean"]["ci"]
    demo_ci = res["models"]["Demographics"]["ci"]
    out = {}
    for m in ["Ridge (L2)", "Lasso (L1)", "LightGBM"]:
        hi = res["models"][m]["ci"][1]
        out[m] = {
            "beats_mean": bool(hi < mean_ci[0]),
            "beats_demographics": bool(hi < demo_ci[0]),
        }
        out[m]["beats_both"] = out[m]["beats_mean"] and out[m]["beats_demographics"]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = session_table()
    print(
        f"{len(t)} dev sessions, {t['patient_id'].nunique()} subjects, {N_FOLDS} folds"
    )
    folds, oof = run_cv(t)
    folds.to_csv(OUT_DIR / "folds.csv", index=False)
    oof.to_parquet(OUT_DIR / "oof.parquet", index=False)
    results = {}
    for target in TARGETS:
        res = bootstrap(oof, target)
        res["verdict"] = verdict(res)
        res["paper_table5"] = PAPER_TABLE5.get(target)
        results[target] = res
        m = res["models"]
        print(
            f"{target:>20}: mean {m['Mean']['mae']:.3f}  demo {m['Demographics']['mae']:.3f}  "
            f"ridge {m['Ridge (L2)']['mae']:.3f}  lasso {m['Lasso (L1)']['mae']:.3f}  "
            f"lgbm {m['LightGBM']['mae']:.3f}"
        )
    meta = {
        "sessions": len(t),
        "subjects": int(t["patient_id"].nunique()),
        "features": MODEL_FEATURES,
        "n_boot": N_BOOT,
        "inner_folds": INNER_FOLDS,
        "seed": SEED,
        "convergence_warnings": int(folds["convergence_warnings"].sum()),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps({"meta": meta, "targets": results}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
