"""Phase 3 - overfitting diagnostics on the committed 5 folds.

    python -m faceai.diagnostics        # or: make diagnostics

Overfitting is visible as a GAP: error on the data a model was fitted to (train) is lower
than error on subjects it has never seen (validation). Three views of that gap:

  regularization curves   Ridge (L2) and Lasso (L1) across their penalty strength alpha.
                          Weak penalty -> flexible model, strong penalty -> the constant.
                          For Lasso, the number of features left non-zero is also recorded.
  boosting curve          LightGBM (31 leaves, LightGBM's default) scored after 1..1000
                          rounds: train error keeps falling while validation error turns up.
  learning curves         the full nested-CV procedure from `faceai.cv`, refitted on 10-100%
                          of each training fold's SUBJECTS: does more data close the gap?

These are diagnostics, not model selection: nothing here feeds back into `faceai.cv`.
Dev folds only; the held-out test set is not touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from faceai.cv import (
    LASSO_ALPHAS,
    OUT_DIR,
    RIDGE_ALPHAS,
    build_model,
    inputs,
    lgbm,
    mae,
    session_table,
)
from faceai.splits import N_FOLDS, SEED

DIAG_TARGETS = ["hba1c", "cholesterol", "age"]  # Joshua's two + the positive control
BOOST_ROUNDS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
LEARNING_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_MODELS = ["Ridge (L2)", "LightGBM"]


def _linear(model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _split(t: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return t[t["fold"] != fold], t[t["fold"] == fold]


def _alpha_fold(t: pd.DataFrame, target: str, fold: int) -> list[dict]:
    tr, va = _split(t, fold)
    cols = inputs(t, target, uses_ppg=True)
    rows = []
    for name, cls, grid in [
        ("Ridge (L2)", Ridge, RIDGE_ALPHAS),
        ("Lasso (L1)", Lasso, LASSO_ALPHAS),
    ]:
        for alpha in grid:
            kw = {"max_iter": 100_000} if cls is Lasso else {}
            pipe = _linear(cls(alpha=alpha, **kw)).fit(tr[cols], tr[target])
            coef = pipe.named_steps["model"].coef_
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "fold": fold,
                    "alpha": float(alpha),
                    "train_mae": mae(tr[target], pipe.predict(tr[cols])),
                    "val_mae": mae(va[target], pipe.predict(va[cols])),
                    "nonzero": int(np.sum(np.abs(coef) > 1e-12)),
                    "n_inputs": len(cols),
                }
            )
    return rows


def _boost_fold(t: pd.DataFrame, target: str, fold: int) -> list[dict]:
    tr, va = _split(t, fold)
    cols = inputs(t, target, uses_ppg=True)
    model = lgbm(num_leaves=31, n_estimators=max(BOOST_ROUNDS)).fit(
        tr[cols], tr[target]
    )
    return [
        {
            "target": target,
            "fold": fold,
            "rounds": k,
            "train_mae": mae(tr[target], model.predict(tr[cols], num_iteration=k)),
            "val_mae": mae(va[target], model.predict(va[cols], num_iteration=k)),
        }
        for k in BOOST_ROUNDS
    ]


def _learning_fold(t: pd.DataFrame, target: str, fold: int) -> list[dict]:
    tr_all, va = _split(t, fold)
    subjects = np.sort(tr_all["patient_id"].unique())
    rng = np.random.default_rng([SEED, fold])
    order = rng.permutation(subjects)  # nested subsets: 10% is inside 20%, and so on
    rows = []
    for frac in LEARNING_FRACTIONS:
        keep = set(order[: round(frac * len(order))])
        tr = tr_all[tr_all["patient_id"].isin(keep)]
        for name in LEARNING_MODELS:
            est, uses_ppg = build_model(name)
            cols = inputs(t, target, uses_ppg)
            if isinstance(est, GridSearchCV):
                est.fit(tr[cols], tr[target], groups=tr["patient_id"])
            else:
                est.fit(tr[cols], tr[target])
            rows.append(
                {
                    "target": target,
                    "model": name,
                    "fold": fold,
                    "fraction": frac,
                    "n_train_subjects": len(keep),
                    "train_mae": mae(tr[target], est.predict(tr[cols])),
                    "val_mae": mae(va[target], est.predict(va[cols])),
                }
            )
    return rows


def run(n_jobs: int = 12) -> dict[str, pd.DataFrame]:
    t = session_table()
    jobs = [(target, k) for target in DIAG_TARGETS for k in range(N_FOLDS)]
    out = {}
    for name, fn in [
        ("alpha_curves", _alpha_fold),
        ("boosting_curve", _boost_fold),
        ("learning_curves", _learning_fold),
    ]:
        res = Parallel(n_jobs=n_jobs)(delayed(fn)(t, target, k) for target, k in jobs)
        out[name] = pd.DataFrame([r for rows in res for r in rows])
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in run().items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows -> {OUT_DIR / f'{name}.csv'}")


if __name__ == "__main__":
    main()
