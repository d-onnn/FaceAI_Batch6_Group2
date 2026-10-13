"""Phase 3 - clinical-threshold and within-person-change views of the out-of-fold results.

    python -m faceai.clinical        # or: make clinical

Reads `artifacts/cv/oof.parquet` (dev folds only; the held-out test set is not touched).

Clinical thresholds, each applied with the conditions of its own standard:
  HbA1c             >= 5.7 % (prediabetes or diabetes) and >= 6.5 % (diabetes range),
                    American Diabetes Association diagnostic criteria.
  cholesterol       >= 5.2 mmol/L total cholesterol (~200 mg/dL, the NCEP ATP III upper
                    limit of "desirable").
  respiratory rate  share of sessions within +-3 breaths/min of the label (the brief's
                    clinical tolerance).
  SpO2              ARMS (root-mean-square error). ISO 80601-2-61:2017 asks for <= 4 % over
                    reference SaO2 spanning 70-100 %; these labels do not span that range, so
                    no number here can establish compliance - reported, not claimed.
For the binary thresholds, discrimination is the AUROC of the continuous prediction, with
subject-cluster bootstrap CIs. The constant predictors have no ranking, so their AUROC is
0.5 by construction and is not computed. Sensitivity and specificity are also given for the
threshold applied directly to the prediction - which shows how rarely predictions that
regress to the cohort mean ever cross a clinical cut-off.

Within-person change (a person's two sessions, rest then post-exercise): does a model or
feature move in the same direction as the label? Spearman rho between the per-subject
changes, with a subject bootstrap CI. Positive control: PPG heart rate against the `pulse`
label, which rises sharply after exercise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from faceai.cv import OUT_DIR, session_table
from faceai.data import audit
from faceai.features import features_path
from faceai.provenance import record
from faceai.splits import SEED

N_BOOT = 2000
RANKED_MODELS = ["Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
ALL_MODELS = ["Mean", "Median", *RANKED_MODELS]
THRESHOLDS = [
    ("hba1c", 5.7, "HbA1c >= 5.7 % (prediabetes or diabetes, ADA)"),
    ("hba1c", 6.5, "HbA1c >= 6.5 % (diabetes range, ADA)"),
    ("cholesterol", 5.2, "Total cholesterol >= 5.2 mmol/L (NCEP ATP III)"),
]
RR_TOLERANCE = 3.0  # breaths/min
SPO2_ARMS_LIMIT = 4.0  # %, ISO 80601-2-61:2017, valid only over SaO2 70-100 %
CHANGE_TARGETS = ["respiratory_rate", "spo2", "arterial_stiffness"]
CHANGE_MODELS = ["Ridge (L2)", "Lasso (L1)", "LightGBM"]


def _subject_draws(subjects: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Row indices for each bootstrap draw, resampling whole subjects."""
    codes, uniq = pd.factorize(subjects)
    rows_of = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    picks = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    return [np.concatenate([rows_of[k] for k in draw]) for draw in picks]


def auroc_ci(y: np.ndarray, score: np.ndarray, draws: list[np.ndarray]) -> dict:
    point = float(roc_auc_score(y, score))
    boot = [roc_auc_score(y[i], score[i]) for i in draws if 0 < y[i].sum() < len(i)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"auroc": point, "ci": [float(lo), float(hi)], "valid_draws": len(boot)}


def thresholds(oof: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    out = []
    for target, cut, label in THRESHOLDS:
        d = oof[oof["target"] == target].reset_index(drop=True)
        y = (d["y"].to_numpy() >= cut).astype(int)
        draws = _subject_draws(d["patient_id"].to_numpy(), rng)
        entry = {
            "target": target,
            "threshold": cut,
            "label": label,
            "sessions": len(d),
            "positive_sessions": int(y.sum()),
            "positive_subjects": int(d.loc[y == 1, "patient_id"].nunique()),
            "models": {},
        }
        for m in RANKED_MODELS:
            pred = d[m].to_numpy()
            flag = pred >= cut
            entry["models"][m] = {
                **auroc_ci(y, pred, draws),
                "predicted_positive_sessions": int(flag.sum()),
                "sensitivity_at_cut": float(flag[y == 1].mean()),
                "specificity_at_cut": float((~flag[y == 0]).mean()),
            }
        out.append(entry)
    return out


def tolerances(oof: pd.DataFrame, t: pd.DataFrame) -> dict:
    rr = oof[oof["target"] == "respiratory_rate"]
    within = {
        m: float(np.mean(np.abs(rr[m] - rr["y"]) <= RR_TOLERANCE)) for m in ALL_MODELS
    }
    by_state = {
        step: {
            m: float(np.mean(np.abs(g[m] - g["y"]) <= RR_TOLERANCE)) for m in ALL_MODELS
        }
        for step, g in rr.groupby("step")
    }
    sp = oof[oof["target"] == "spo2"]
    arms = {m: float(np.sqrt(np.mean((sp[m] - sp["y"]) ** 2))) for m in ALL_MODELS}
    return {
        "respiratory_rate": {
            "tolerance": RR_TOLERANCE,
            "share_within": within,
            "share_within_by_state": by_state,
            "label_sd": float(t["respiratory_rate"].std()),
        },
        "spo2": {
            "arms": arms,
            "iso_limit": SPO2_ARMS_LIMIT,
            "label_min": float(t["spo2"].min()),
            "label_max": float(t["spo2"].max()),
            "label_share_below_90": float(np.mean(t["spo2"] < 90)),
            "label_share_below_70": float(np.mean(t["spo2"] < 70)),
        },
    }


def _rho_ci(dx: np.ndarray, dy: np.ndarray, rng: np.random.Generator) -> dict:
    point = stats.spearmanr(dx, dy).statistic
    n = len(dx)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, size=n)
        if np.ptp(dx[i]) > 0 and np.ptp(dy[i]) > 0:
            boot.append(stats.spearmanr(dx[i], dy[i]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Direction agreement only where both moved: a prediction that did not change has no
    # direction. It mostly reflects the average post-exercise shift; rho asks whether a
    # bigger personal change is predicted as bigger.
    both = (dy != 0) & (dx != 0)
    agree = float(np.mean(np.sign(dx[both]) == np.sign(dy[both])))
    return {
        "rho": float(point),
        "ci": [float(lo), float(hi)],
        "subjects": n,
        "label_moved": int((dy != 0).sum()),
        "signal_unchanged": int((dx == 0).sum()),
        "direction_agreement": agree,
    }


def change(oof: pd.DataFrame, t: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    rows = []
    for target in CHANGE_TARGETS:
        d = oof[oof["target"] == target]
        wide = d.pivot(index="patient_id", columns="step", values=["y", *CHANGE_MODELS])
        wide = wide.dropna()
        dy = (wide[("y", "after")] - wide[("y", "before")]).to_numpy()
        for m in CHANGE_MODELS:
            dx = (wide[(m, "after")] - wide[(m, "before")]).to_numpy()
            rows.append({"target": target, "signal": m, **_rho_ci(dx, dy, rng)})
    # Feature-level: PPG respiration estimate vs RR label; positive control HR vs pulse.
    db = audit.load_db().drop_duplicates(["patient_id", "step"])
    f = t.merge(db[["patient_id", "step", "pulse"]], on=["patient_id", "step"])
    for feature, label, name in [
        ("hr_mean_bpm", "pulse", "PPG heart rate vs pulse label (positive control)"),
        ("resp_rate_fused", "respiratory_rate", "PPG respiration estimate vs RR label"),
    ]:
        w = f.pivot(index="patient_id", columns="step", values=[feature, label])
        w = w.dropna()
        dx = (w[(feature, "after")] - w[(feature, "before")]).to_numpy()
        dy = (w[(label, "after")] - w[(label, "before")]).to_numpy()
        rows.append({"target": label, "signal": name, **_rho_ci(dx, dy, rng)})
    return rows


def main() -> None:
    t = session_table()
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    rng = np.random.default_rng(SEED)
    out = {
        "thresholds": thresholds(oof, rng),
        "tolerances": tolerances(oof, t),
        "change": change(oof, t, rng),
        "provenance": record([OUT_DIR / "oof.parquet", features_path()]),
    }
    (OUT_DIR / "clinical.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    for e in out["thresholds"]:
        best = max(e["models"].items(), key=lambda kv: kv[1]["auroc"])
        print(
            f"{e['label']}: {e['positive_subjects']} positive subjects; best AUROC "
            f"{best[0]} {best[1]['auroc']:.3f} {np.round(best[1]['ci'], 3).tolist()}"
        )
    for r in out["change"]:
        print(
            f"change {r['target']:>18} {r['signal']:<48} rho {r['rho']:+.3f} "
            f"{np.round(r['ci'], 3).tolist()}"
        )


if __name__ == "__main__":
    main()
