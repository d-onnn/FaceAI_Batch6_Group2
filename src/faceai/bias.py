"""Phase 3d - bias in our errors, and whether it can be removed without using the answer.

    python -m faceai.bias        # or: make bias

Reads `artifacts/cv/oof.parquet`, `artifacts/cv/folds.csv` and the session table (dev folds
only; the held-out test set is not touched) and writes `artifacts/cv/bias.json`.

Error is prediction minus label, so positive means an overestimate. Four views:
  1. Systematic bias: the mean error, with a subject-cluster bootstrap CI, and the
     Bland-Altman 95 % limits of agreement (mean +- 1.96 SD of the errors). This is how
     device validations (ISO 81060-2, the ARTERY Society guidelines) judge bias.
  2. Proportional bias (regression to the mean): the OLS slope of error on the true label.
     A constant predictor scores exactly -1 and a perfect one 0. Also the mean error in the
     lowest and highest 10 % of labels.
  3. Subgroup bias: mean error by sex, age band (< 25 / >= 25) and state (rest or
     post-exercise).
  4. De-biasing, learned without the test labels. In each outer fold the model is refitted,
     with the hyperparameters that fold's inner search chose, on 5 inner folds of the
     training subjects. Its inner out-of-fold predictions fit the correction, which is then
     applied to the untouched outer fold:
       offset - subtract the mean inner error;
       slope  - invert pred = a + b*y fitted on the inner folds: pred' = (pred - a) / b. This
                is the training-set "age-bias correction" of the brain-age literature. It is
                skipped in a fold where b <= MIN_SLOPE, because the inverse then multiplies
                the noise more than twentyfold.
     For reference only: the correction that uses the TEST label,
     pred' = pred - (alpha + beta * y_test), which some brain-age papers have reported. It
     puts the answer back into the prediction and cannot be used in practice.
The hyperparameters were chosen by a search over the whole outer training set, so the inner
predictions are slightly optimistic; the outer fold, on which everything is scored, is not.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, GroupKFold

from faceai.clinical import _subject_draws
from faceai.cv import INNER_FOLDS, OUT_DIR, TARGETS, build_model, inputs, session_table
from faceai.provenance import record
from faceai.robustness import _cluster_ci
from faceai.splits import N_FOLDS, OLDER_THAN, SEED

BIAS_MODELS = ["Mean", "Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
FIT_MODELS = ["Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
PPG_MODELS = ["Ridge (L2)", "Lasso (L1)", "LightGBM"]
MIN_SLOPE = 0.05
TAIL = 0.10


def slope(y: np.ndarray, err: np.ndarray) -> float:
    """OLS slope of the error on the true label."""
    yc = y - y.mean()
    return float(yc @ (err - err.mean()) / (yc @ yc))


def agreement(err: np.ndarray, subjects: np.ndarray, rng: np.random.Generator) -> dict:
    sd = float(err.std(ddof=1))
    bias = float(err.mean())
    return {
        "bias": bias,
        "ci": _cluster_ci(err, subjects, rng),
        "sd": sd,
        "loa": [bias - 1.96 * sd, bias + 1.96 * sd],
        "mae": float(np.abs(err).mean()),
    }


def proportional(y: np.ndarray, err: np.ndarray, draws: list[np.ndarray]) -> dict:
    lo_cut, hi_cut = np.quantile(y, [TAIL, 1 - TAIL])
    lo, hi = y <= lo_cut, y >= hi_cut
    boot = [slope(y[i], err[i]) for i in draws]
    return {
        "slope": slope(y, err),
        "ci": np.percentile(boot, [2.5, 97.5]).tolist() if boot else None,
        "low_cut": float(lo_cut),
        "high_cut": float(hi_cut),
        "low_bias": float(err[lo].mean()),
        "high_bias": float(err[hi].mean()),
        "low_n": int(lo.sum()),
        "high_n": int(hi.sum()),
    }


# ------------------------------------------------------------------- corrections
def fit_corrections(pred_inner: np.ndarray, y_inner: np.ndarray) -> dict:
    """Everything a correction needs, learned from training-subject predictions only."""
    b, a = np.polyfit(y_inner, pred_inner, 1)  # pred = a + b*y
    beta, alpha = np.polyfit(y_inner, pred_inner - y_inner, 1)  # err = alpha + beta*y
    return {
        "offset": float((pred_inner - y_inner).mean()),
        "a": float(a),
        "b": float(b),
        "alpha": float(alpha),
        "beta": float(beta),
    }


def correct(
    pred: np.ndarray, fit: dict, method: str, y_test: np.ndarray | None = None
) -> np.ndarray:
    if method == "offset":
        return pred - fit["offset"]
    if method == "slope":
        return (pred - fit["a"]) / fit["b"] if fit["b"] > MIN_SLOPE else pred
    if method == "uses test label":  # reference only - not available in practice
        return pred - (fit["alpha"] + fit["beta"] * y_test)
    raise KeyError(method)


METHODS = ["offset", "slope", "uses test label"]


def _fixed_model(name: str, params: dict):
    est, uses_ppg = build_model(name)
    if isinstance(est, GridSearchCV):
        est = clone(est.estimator).set_params(**params)
    return est, uses_ppg


def _inner_fit(
    t: pd.DataFrame, target: str, fold: int, name: str, params: dict
) -> dict:
    tr = t[t["fold"] != fold].reset_index(drop=True)
    est, uses_ppg = _fixed_model(name, params)
    cols = inputs(t, target, uses_ppg)
    pred = np.empty(len(tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for itr, iva in GroupKFold(n_splits=INNER_FOLDS).split(
            tr, groups=tr["patient_id"]
        ):
            m = clone(est).fit(tr.loc[itr, cols], tr.loc[itr, target])
            pred[iva] = m.predict(tr.loc[iva, cols])
    return {
        "target": target,
        "fold": fold,
        "model": name,
        **fit_corrections(pred, tr[target].to_numpy()),
    }


def debias(
    t: pd.DataFrame,
    oof: pd.DataFrame,
    folds: pd.DataFrame,
    n_jobs: int = 12,
    targets: list[str] = TARGETS,
) -> pd.DataFrame:
    params = folds.set_index(["target", "model", "fold"])["best_params"]
    jobs = [(tg, k, m) for tg in targets for k in range(N_FOLDS) for m in FIT_MODELS]
    fits = Parallel(n_jobs=n_jobs)(
        delayed(_inner_fit)(t, tg, k, m, json.loads(params[(tg, m, k)]))
        for tg, k, m in jobs
    )
    fits = pd.DataFrame(fits).set_index(["target", "fold", "model"])
    rows = []
    for (tg, k), g in oof.groupby(["target", "fold"]):
        for m in FIT_MODELS:
            fit = fits.loc[(tg, k, m)].to_dict()
            for method in METHODS:
                rows.append(
                    pd.DataFrame(
                        {
                            "target": tg,
                            "fold": k,
                            "model": m,
                            "method": method,
                            "patient_id": g["patient_id"].to_numpy(),
                            "y": g["y"].to_numpy(),
                            "original": g[m].to_numpy(),
                            "corrected": correct(
                                g[m].to_numpy(), fit, method, g["y"].to_numpy()
                            ),
                            "slope_applied": method != "slope" or fit["b"] > MIN_SLOPE,
                            "b": fit["b"],
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------- summaries
def best_ppg_model(oof: pd.DataFrame, target: str) -> str:
    d = oof[oof["target"] == target]
    return min(PPG_MODELS, key=lambda m: float(np.abs(d[m] - d["y"]).mean()))


def summarise(t: pd.DataFrame, oof: pd.DataFrame, cor: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    meta = t[["patient_id", "step", "sex", "age"]].rename(columns={"age": "age_years"})
    d = oof.merge(meta, on=["patient_id", "step"], how="left", validate="m:1")
    d["age band"] = np.where(
        d["age_years"] >= OLDER_THAN, f">= {OLDER_THAN}", f"< {OLDER_THAN}"
    )
    d["state"] = d["step"].map({"before": "rest", "after": "post-exercise"})
    out = {
        "best_ppg_model": {},
        "agreement": [],
        "proportional": [],
        "subgroups": [],
        "debias": [],
    }
    for target in TARGETS:
        g = d[d["target"] == target].reset_index(drop=True)
        y, ids = g["y"].to_numpy(), g["patient_id"].to_numpy()
        draws = _subject_draws(ids, rng)
        best = best_ppg_model(oof, target)
        out["best_ppg_model"][target] = best
        for m in BIAS_MODELS:
            err = g[m].to_numpy() - y
            out["agreement"].append(
                {"target": target, "model": m, **agreement(err, ids, rng)}
            )
            out["proportional"].append(
                {"target": target, "model": m, **proportional(y, err, draws)}
            )
        for m in ["Mean", best]:
            for factor in ["sex", "age band", "state"]:
                for level, s in g.groupby(factor):
                    err = s[m].to_numpy() - s["y"].to_numpy()
                    out["subgroups"].append(
                        {
                            "target": target,
                            "model": m,
                            "factor": factor,
                            "level": level,
                            "n_subjects": int(s["patient_id"].nunique()),
                            "bias": float(err.mean()),
                            "ci": _cluster_ci(err, s["patient_id"].to_numpy(), rng),
                        }
                    )
        c = cor[cor["target"] == target]
        for (m, method), s in c.groupby(["model", "method"], sort=False):
            s = s.reset_index(drop=True)
            ys, sid = s["y"].to_numpy(), s["patient_id"].to_numpy()
            e0, e1 = s["original"].to_numpy() - ys, s["corrected"].to_numpy() - ys
            delta = np.abs(e1) - np.abs(e0)
            prop = proportional(ys, e1, [])  # no CI needed here
            out["debias"].append(
                {
                    "target": target,
                    "model": m,
                    "method": method,
                    "valid_in_use": method != "uses test label",
                    "folds_applied": int(
                        s.groupby("fold")["slope_applied"].first().sum()
                    ),
                    "inner_b": [
                        round(float(x), 4) for x in s.groupby("fold")["b"].first()
                    ],
                    "mae_before": float(np.abs(e0).mean()),
                    "mae_after": float(np.abs(e1).mean()),
                    "mae_change": float(delta.mean()),
                    "mae_change_ci": _cluster_ci(delta, sid, rng),
                    "bias_after": float(e1.mean()),
                    "slope_before": slope(ys, e0),
                    "slope_after": prop["slope"],
                    "low_bias_after": prop["low_bias"],
                    "high_bias_after": prop["high_bias"],
                }
            )
    return out


def main() -> None:
    t = session_table()
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    folds = pd.read_csv(OUT_DIR / "folds.csv")
    cor = debias(t, oof, folds)
    out = {
        **summarise(t, oof, cor),
        "provenance": record([OUT_DIR / "oof.parquet", OUT_DIR / "folds.csv"]),
    }
    (OUT_DIR / "bias.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    for p in out["proportional"]:
        if p["model"] in ("Mean", out["best_ppg_model"][p["target"]]):
            print(
                f"{p['target']:>18} {p['model']:<11} slope {p['slope']:+.3f} "
                f"{np.round(p['ci'], 3).tolist()}  low10 {p['low_bias']:+.3f}  high10 {p['high_bias']:+.3f}"
            )
    for r in out["debias"]:
        if r["model"] == out["best_ppg_model"][r["target"]]:
            print(
                f"{r['target']:>18} {r['method']:<16} MAE {r['mae_before']:.3f} -> {r['mae_after']:.3f} "
                f"slope {r['slope_before']:+.2f} -> {r['slope_after']:+.2f}  folds {r['folds_applied']}  b {r['inner_b']}"
            )


if __name__ == "__main__":
    main()
