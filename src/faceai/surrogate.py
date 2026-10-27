"""Phase 3e (b) - other surrogate model forms, kriging uncertainty, and an accuracy/bias trade-off.

    python -m faceai.surrogate      # or: make surrogate

Applies Maaz Ahmad's "Basics of Surrogate Modeling" (MoSPSyDe group, NTU, 7 Oct 2026) to the
Phase 3 problem, on the same dev folds (the held-out test set is not read):

1. Model form (his pp. 6, 14). Kriging (a Gaussian process with one length scale per input
   plus a noise term, fitted by maximum likelihood with 3 restarts - his "multiple-start
   runs", p. 7), RBF support-vector regression, k-nearest neighbours and a small neural net
   join the Phase 3 models. Each is tuned by the same inner GroupKFold search where it has
   hyperparameters. If no form beats the constant guess on a target, the limit is the data,
   not the model.
2. Validation beyond MAE (his p. 13): median absolute error, R2, the 95th-percentile and the
   maximum absolute error.
3. Kriging uncertainty (his p. 9: the model's own variance). Each prediction gets a standard
   deviation. It is calibrated without the test fold: the GP's closed-form leave-one-out
   residuals on the training subjects (Rasmussen & Williams 2006, eqs. 5.10-5.12) set one
   scale factor so that 95 % of standardised LOO residuals fall inside +-1.96. Coverage is then
   checked on the test folds overall, in the label tails and for subjects aged >= 25.
4. A trade-off curve (Matsunami's crystallisation example: a Pareto front between two
   objectives). For the age model, a fraction lambda of the Phase 3d slope correction is
   applied; the curve shows overall MAE against the underestimation of the oldest 10 %.

Writes `artifacts/cv/surrogate_oof.parquet` and `artifacts/cv/surrogate.json`.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import linalg, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from faceai import bias as B
from faceai.cv import OUT_DIR, TARGETS, _inner_cv, inputs, session_table
from faceai.provenance import record
from faceai.robustness import _cluster_ci
from faceai.splits import N_FOLDS, OLDER_THAN, SEED

NEW_MODELS = ["Kriging (GP)", "SVR (RBF)", "kNN", "Neural net (MLP)"]
PHASE3_MODELS = ["Mean", "Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
GP_RESTARTS = 3
LAMBDAS = np.round(np.linspace(0, 1, 11), 2)


class Kriging(BaseEstimator, RegressorMixin):
    """Impute, scale, then a GP with one length scale per input and a noise term.

    After fitting, `scale_` calibrates the predictive SD from closed-form leave-one-out
    residuals on the training data, so no test data are used.
    """

    def __init__(self, restarts: int = GP_RESTARTS, random_state: int = SEED):
        self.restarts = restarts
        self.random_state = random_state

    def fit(self, X, y):
        self.prep_ = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        Z = self.prep_.fit_transform(X)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            np.ones(Z.shape[1]), (1e-2, 1e3)
        ) + WhiteKernel(1.0, (1e-5, 1e1))
        self.gp_ = GaussianProcessRegressor(
            kernel,
            normalize_y=True,
            n_restarts_optimizer=self.restarts,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp_.fit(Z, np.asarray(y, dtype=float))
        self.loo_z_ = loo_standardised_residuals(self.gp_)
        self.scale_ = float(np.quantile(np.abs(self.loo_z_), 0.95) / 1.959964)
        return self

    def predict(self, X, return_std: bool = False):
        mu, sd = self.gp_.predict(self.prep_.transform(X), return_std=True)
        return (mu, sd * self.scale_) if return_std else mu


def loo_standardised_residuals(gp: GaussianProcessRegressor) -> np.ndarray:
    """(y_i - mu_-i) / sd_-i for every training point, with the fitted kernel held fixed."""
    K = gp.kernel_(gp.X_train_) + gp.alpha * np.eye(len(gp.X_train_))
    Kinv = linalg.cho_solve(linalg.cho_factor(K, lower=True), np.eye(len(K)))
    return (Kinv @ gp.y_train_) / np.sqrt(np.diag(Kinv))


def build_surrogate(name: str):
    prep = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if name == "Kriging (GP)":
        return Kriging()
    if name == "SVR (RBF)":
        model = TransformedTargetRegressor(
            SVR(kernel="rbf"), transformer=StandardScaler()
        )
        return _inner_cv(
            Pipeline([*prep, ("model", model)]),
            {
                "model__regressor__C": [0.3, 1, 3, 10],
                "model__regressor__epsilon": [0.1, 0.5],
            },
        )
    if name == "kNN":
        return _inner_cv(
            Pipeline([*prep, ("model", KNeighborsRegressor())]),
            {"model__n_neighbors": [5, 10, 20, 40, 80]},
        )
    if name == "Neural net (MLP)":
        mlp = MLPRegressor(
            early_stopping=True, max_iter=2000, n_iter_no_change=20, random_state=SEED
        )
        model = TransformedTargetRegressor(mlp, transformer=StandardScaler())
        return _inner_cv(
            Pipeline([*prep, ("model", model)]),
            {
                "model__regressor__hidden_layer_sizes": [(16,), (32, 16)],
                "model__regressor__alpha": [1e-3, 1e-1, 1.0],
            },
        )
    raise KeyError(name)


def _fit_fold(t: pd.DataFrame, target: str, fold: int) -> pd.DataFrame:
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    cols = inputs(t, target, uses_ppg=True)
    out = pd.DataFrame(
        {
            "patient_id": va["patient_id"].to_numpy(),
            "step": va["step"].to_numpy(),
            "target": target,
            "fold": fold,
            "y": va[target].to_numpy(),
        }
    )
    for name in NEW_MODELS:
        est = build_surrogate(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if isinstance(est, Kriging):
                est.fit(tr[cols], tr[target])
                out[name], out["gp_sd"] = est.predict(va[cols], return_std=True)
                out["gp_scale"] = est.scale_
            else:
                est.fit(tr[cols], tr[target], groups=tr["patient_id"])
                out[name] = est.predict(va[cols])
    return out


def run(t: pd.DataFrame, n_jobs: int = 12) -> pd.DataFrame:
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(t, tg, k) for tg in TARGETS for k in range(N_FOLDS)
    )
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------- metrics
def errors_table(err: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    a = np.abs(err)
    return {
        "mae": float(a.mean()),
        "median_ae": float(np.median(a)),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
        "p95_ae": float(np.quantile(a, 0.95)),
        "max_ae": float(a.max()),
    }


def metrics(both: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    rows = []
    for target, d in both.groupby("target", sort=False):
        y, ids = d["y"].to_numpy(), d["patient_id"].to_numpy()
        ref = {m: np.abs(d[m].to_numpy() - y) for m in ("Mean", "Demographics")}
        ref_ci = {m: _cluster_ci(e, ids, rng) for m, e in ref.items()}
        for m in [*PHASE3_MODELS, *NEW_MODELS]:
            pred = d[m].to_numpy()
            a = np.abs(pred - y)
            ci = _cluster_ci(a, ids, rng)
            rows.append(
                {
                    "target": target,
                    "model": m,
                    **errors_table(pred - y, y, pred),
                    "ci": ci,
                    "vs_mean": float((a - ref["Mean"]).mean()),
                    "vs_mean_ci": _cluster_ci(a - ref["Mean"], ids, rng),
                    "beats_both_strict": bool(
                        ci[1] < ref_ci["Mean"][0] and ci[1] < ref_ci["Demographics"][0]
                    ),
                }
            )
    return rows


def coverage(
    new: pd.DataFrame, t: pd.DataFrame, rng: np.random.Generator
) -> list[dict]:
    d = new.merge(
        t[["patient_id", "step", "age"]].rename(columns={"age": "age_years"}),
        on=["patient_id", "step"],
        validate="m:1",
    )
    out = []
    for target, g in d.groupby("target", sort=False):
        y, mu, sd = (
            g["y"].to_numpy(),
            g["Kriging (GP)"].to_numpy(),
            g["gp_sd"].to_numpy(),
        )
        inside = np.abs(y - mu) <= 1.959964 * sd
        lo, hi = np.quantile(y, [B.TAIL, 1 - B.TAIL])
        ids = g["patient_id"].to_numpy()
        boot = [
            stats.spearmanr(sd[i], np.abs(y - mu)[i]).statistic
            for i in B._subject_draws(ids, rng)
        ]
        out.append(
            {
                "target": target,
                "coverage_all": float(inside.mean()),
                "coverage_lowest_10": float(inside[y <= lo].mean()),
                "coverage_highest_10": float(inside[y >= hi].mean()),
                f"coverage_age_ge_{OLDER_THAN}": float(
                    inside[g["age_years"].to_numpy() >= OLDER_THAN].mean()
                ),
                "mean_interval_halfwidth": float(1.959964 * sd.mean()),
                "sd_vs_abs_error_rho": float(
                    stats.spearmanr(sd, np.abs(y - mu)).statistic
                ),
                "sd_vs_abs_error_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                "loo_scale_by_fold": [
                    round(float(x), 3) for x in g.groupby("fold")["gp_scale"].first()
                ],
            }
        )
    return out


def tradeoff(
    t: pd.DataFrame, oof: pd.DataFrame, folds: pd.DataFrame, target: str = "age"
) -> dict:
    """MAE against the oldest-10 % bias as a growing share of the slope correction is applied."""
    model = B.best_ppg_model(oof, target)
    cor = B.debias(t, oof[oof["target"] == target], folds, targets=[target])
    c = cor[
        (cor["target"] == target) & (cor["model"] == model) & (cor["method"] == "slope")
    ]
    y, orig, full = (
        c["y"].to_numpy(),
        c["original"].to_numpy(),
        c["corrected"].to_numpy(),
    )
    top = y >= np.quantile(y, 1 - B.TAIL)
    curve = []
    for lam in LAMBDAS:
        p = orig + lam * (full - orig)
        curve.append(
            {
                "lambda": float(lam),
                "mae": float(np.abs(p - y).mean()),
                "top10_bias": float((p - y)[top].mean()),
            }
        )
    return {
        "target": target,
        "model": model,
        "top10_cut": float(np.quantile(y, 1 - B.TAIL)),
        "curve": curve,
    }


def main() -> None:
    t = session_table()
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    folds = pd.read_csv(OUT_DIR / "folds.csv")
    new = run(t)
    new.to_parquet(OUT_DIR / "surrogate_oof.parquet", index=False)
    both = oof.merge(
        new.drop(columns=["gp_sd", "gp_scale"]),
        on=["patient_id", "step", "target", "fold", "y"],
        validate="1:1",
    )
    assert len(both) == len(oof), "every Phase 3 prediction must be matched"
    rng = np.random.default_rng(SEED)
    out = {
        "metrics": metrics(both, rng),
        "kriging_coverage": coverage(new, t, rng),
        "tradeoff": tradeoff(t, oof, folds),
        "provenance": record([OUT_DIR / "oof.parquet", OUT_DIR / "folds.csv"]),
    }
    (OUT_DIR / "surrogate.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    for r in out["metrics"]:
        print(
            f"{r['target']:>18} {r['model']:<17} MAE {r['mae']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] "
            f"vs mean {r['vs_mean']:+.3f} {np.round(r['vs_mean_ci'], 3).tolist()} R2 {r['r2']:+.2f} max {r['max_ae']:.2f}"
            f"{'  STRICT' if r['beats_both_strict'] else ''}"
        )
    for c in out["kriging_coverage"]:
        print(c)
    print(out["tradeoff"])


if __name__ == "__main__":
    main()
