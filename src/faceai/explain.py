"""Phase 3b - which inputs drive each model, and is that the same in every fold?

    python -m faceai.explain        # or: make explain

Explains "why", not just "what". For each target, the Phase 3 models are refitted on each
training fold (same inner-CV tuning) and scored with grouped permutation importance on the
validation fold: how much validation MAE rises when one GROUP of inputs is shuffled across
that fold's sessions. A group's columns are shuffled together, so correlated features
(e.g. the four respiration estimates) are judged as one instead of splitting the credit.

An explanation is only reported as a finding if it is stable: the Spearman correlation
between every pair of folds' group importances is recorded, because model diagnostics are
often less reliable than assumed. Dev folds only; the held-out test set is not touched.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GridSearchCV

from faceai.cv import OUT_DIR, TARGETS, build_model, inputs, mae, session_table
from faceai.features import MODEL_FEATURES, features_path
from faceai.provenance import record
from faceai.splits import N_FOLDS, SEED

EXPLAIN_TARGETS = [
    "age",
    "arterial_stiffness",
    "respiratory_rate",
    "bmi",
    "hba1c",
    "cholesterol",
]
EXPLAIN_MODELS = ["Ridge (L2)", "LightGBM"]
N_REPEATS = 20
GROUPS = {
    "heart rate": ["hr_mean_bpm", "hr_spectral_bpm"],
    "rate variability": ["ibi_sdnn_ms", "ibi_rmssd_ms", "ibi_pnn50_pct"],
    "respiration": [
        "resp_rate_rifv",
        "resp_rate_riav",
        "resp_rate_riiv",
        "resp_rate_fused",
    ],
    "pulse shape": ["rise_time_s", "width50_s", "systolic_frac", "apg_b_a"],
    "beat statistics": ["amp_cv", "skewness", "kurtosis"],
    "signal quality": ["sqi_template_r"],
    "age": ["age"],
    "sex": ["sex_m"],
    "BMI": ["bmi"],
}
PPG_GROUPS = [g for g in GROUPS if g not in ("age", "sex", "BMI")]
assert sorted(c for g in PPG_GROUPS for c in GROUPS[g]) == sorted(MODEL_FEATURES)


def _fit(name: str, tr: pd.DataFrame, target: str):
    est, uses_ppg = build_model(name)
    cols = inputs(tr, target, uses_ppg)
    if isinstance(est, GridSearchCV):
        est.fit(tr[cols], tr[target], groups=tr["patient_id"])
    else:
        est.fit(tr[cols], tr[target])
    return est, cols


def grouped_importance(
    est, X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Mean rise in MAE when each group's columns are shuffled together (N_REPEATS times)."""
    base = mae(y, est.predict(X))
    out = {}
    for group, cols in GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue  # e.g. the age group when age is the target
        rises = []
        for _ in range(N_REPEATS):
            shuffled = X.copy()
            shuffled[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            rises.append(mae(y, est.predict(shuffled)) - base)
        out[group] = float(np.mean(rises))
    return out


def _fold(t: pd.DataFrame, target: str, fold: int) -> list[dict]:
    tr, va = t[t["fold"] != fold], t[t["fold"] == fold]
    rng = np.random.default_rng([SEED, fold, TARGETS.index(target)])
    rows = []
    for name in EXPLAIN_MODELS:
        est, cols = _fit(name, tr, target)
        base = mae(va[target], est.predict(va[cols]))
        for group, rise in grouped_importance(
            est, va[cols], va[target].to_numpy(), rng
        ).items():
            rows.append(
                {"target": target, "model": name, "fold": fold, "group": group,
                 "mae_rise": rise, "val_mae": base}
            )  # fmt: skip
    return rows


def stability(imp: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of group importances between every pair of folds."""
    rows = []
    for (target, model), d in imp.groupby(["target", "model"]):
        wide = d.pivot(index="group", columns="fold", values="mae_rise")
        rhos = [
            stats.spearmanr(wide[a], wide[b]).statistic
            for a, b in itertools.combinations(wide.columns, 2)
        ]
        top = wide.mean(axis=1).idxmax()
        top_every_fold = bool((wide.idxmax(axis=0) == top).all())
        rows.append(
            {"target": target, "model": model, "rho_mean": float(np.mean(rhos)),
             "rho_min": float(np.min(rhos)), "top_group": top,
             "top_group_in_every_fold": top_every_fold}
        )  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    t = session_table()
    jobs = [(target, k) for target in EXPLAIN_TARGETS for k in range(N_FOLDS)]
    res = Parallel(n_jobs=12)(delayed(_fold)(t, tg, k) for tg, k in jobs)
    imp = pd.DataFrame([r for rows in res for r in rows])
    imp.to_csv(OUT_DIR / "explain.csv", index=False)
    stab = stability(imp)
    stab.to_csv(OUT_DIR / "explain_stability.csv", index=False)
    (OUT_DIR / "explain_provenance.json").write_text(
        json.dumps(record([features_path()]), indent=2) + "\n", encoding="utf-8"
    )
    summary = imp.groupby(["target", "model", "group"])["mae_rise"].mean().round(3)
    for (target, model), s in summary.groupby(level=[0, 1]):
        top = s.droplevel([0, 1]).sort_values(ascending=False).head(3)
        st = stab[(stab["target"] == target) & (stab["model"] == model)].iloc[0]
        print(
            f"{target:>18} {model:<11} top {top.to_dict()} | fold rho {st['rho_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
