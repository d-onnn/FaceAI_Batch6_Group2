"""Phase 3c - our errors judged by each accuracy standard's own pass rule.

    python -m faceai.standards       # or: make standards

Reads `artifacts/cv/oof.parquet` (dev folds only; the held-out test set is not touched) and
writes `artifacts/cv/standards.json`.

Accuracy standards are rarely written as an MAE, so each is applied in its own form:
  HbA1c        NGSP manufacturer certification: 36 of 40 results within +-5 % of an NGSP
               Secondary Reference Laboratory (ngsp.org/critsumm.asp; in force since 2019).
  cholesterol  NCEP goals bias <= 3 % and CV <= 3 %, which imply a total allowable error of
               8.9 % for one measurement (3 + 1.96 x 3): ~95 % of results within +-8.9 %
               (CDC Cholesterol Reference Method Laboratory Network protocol).
  SpO2         ARMS <= 4.0 % over SaO2 70-100 % (ISO 80601-2-61); the FDA 2013 510(k)
               guidance lists <= 3.0 % as typical for transmittance sensors.
  stiffness    ARTERY Society 2010 pulse-wave-velocity validation (Wilkinson et al.,
               doi:10.1016/j.artres.2010.03.001), reference minus test: excellent |mean
               difference| <= 0.5 m/s and SD <= 0.8; acceptable < 1.0 and SD <= 1.5; else poor.
  RR           within +-3 breaths/min, the brief's tolerance. No share is specified.
  temperature  within +-0.3 C, the ISO 80601-2-56 laboratory (bench) accuracy limit of the
               instrument, as tabulated by Wang et al., Sensors 2022 (doi:10.3390/s22010215).

None of these can certify anything here. Each assumes a reference our labels are not (per
the dataset paper: an EasyTouch finger-prick analyser for HbA1c and cholesterol, a Beurer
PO 60 pulse oximeter rather than arterial SaO2, an unnamed non-contact thermometer, and a
stiffness index with no stated unit from a PPG or sphygmography device), and the labels do
not span the ranges the standards test. What the exercise shows is the constant predictor
beside each model: a rule that both land on the same side of says nothing about the model.

"Implied MAE" translates a rule into the MAE it allows if errors were unbiased and normal
(MAE = sqrt(2/pi) x SD); a coverage rule "share p within +-w" gives SD = w / z((1+p)/2).
"Required R2" is the share of this cohort's label variance a model must explain to get its
error SD that low - negative means a constant already does.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from faceai.clinical import ALL_MODELS, _subject_draws
from faceai.cv import OUT_DIR
from faceai.provenance import record
from faceai.splits import SEED

MAE_PER_SD = float(np.sqrt(2 / np.pi))
ARTERY_ACCEPTABLE_SD = 1.5  # m/s
CRITERIA = [
    {
        "target": "hba1c",
        "kind": "share",
        "tolerance": 0.05,
        "relative": True,
        "required_share": 36 / 40,
        "standard": "NGSP manufacturer certification",
        "rule": "36 of 40 results within ±5 % of an NGSP reference laboratory",
        "reference_assumed": "NGSP Secondary Reference Laboratory",
        "our_reference": "EasyTouch finger-prick analyser",
    },
    {
        "target": "cholesterol",
        "kind": "share",
        "tolerance": 0.089,
        "relative": True,
        "required_share": 0.95,
        "standard": "NCEP total allowable error (CDC CRMLN)",
        "rule": "≈95 % of results within ±8.9 % (bias ≤ 3 %, CV ≤ 3 %)",
        "reference_assumed": "CDC reference method",
        "our_reference": "EasyTouch finger-prick analyser",
    },
    {
        "target": "spo2",
        "kind": "arms",
        "limit": 4.0,
        "standard": "ISO 80601-2-61 pulse oximeter accuracy",
        "rule": "ARMS ≤ 4.0 % over SaO₂ 70–100 % (FDA 2013: ≤ 3.0 % typical)",
        "reference_assumed": "arterial blood SaO₂ (CO-oximetry), 70–100 %",
        "our_reference": "Beurer PO 60 pulse oximeter, labels 86–99 %",
    },
    {
        "target": "arterial_stiffness",
        "kind": "mean_sd",
        "standard": "ARTERY Society PWV device validation",
        "rule": "acceptable: absolute mean difference < 1.0 and SD ≤ 1.5 m/s",
        "reference_assumed": "carotid–femoral PWV, m/s",
        "our_reference": "Eldar PPG or BPLab Angio index, unit not stated",
    },
    {
        "target": "respiratory_rate",
        "kind": "share",
        "tolerance": 3.0,
        "relative": False,
        "required_share": None,
        "standard": "Brief's clinical tolerance",
        "rule": "within ±3 breaths/min (no required share)",
        "reference_assumed": "counted breaths",
        "our_reference": "physician count over one minute",
    },
    {
        "target": "temperature",
        "kind": "share",
        "tolerance": 0.3,
        "relative": False,
        "required_share": None,
        "standard": "ISO 80601-2-56 laboratory accuracy",
        "rule": "within ±0.3 °C (a bench limit for the instrument, not a patient test)",
        "reference_assumed": "a laboratory reference source",
        "our_reference": "non-contact thermometer, model not stated",
    },
]
PRIMARY = {"share": "share_within", "arms": "arms", "mean_sd": "sd_difference"}


def statistic(c: dict, pred: np.ndarray, y: np.ndarray) -> dict:
    if c["kind"] == "share":
        window = c["tolerance"] * (np.abs(y) if c["relative"] else 1.0)
        return {"share_within": float(np.mean(np.abs(pred - y) <= window))}
    if c["kind"] == "arms":
        return {"arms": float(np.sqrt(np.mean((pred - y) ** 2)))}
    d = y - pred  # ARTERY convention: reference minus test device
    return {"mean_difference": float(d.mean()), "sd_difference": float(d.std(ddof=1))}


def verdict(c: dict, s: dict) -> str:
    if c["kind"] == "share":
        if c["required_share"] is None:
            return "no pass rule"
        return "meets" if s["share_within"] >= c["required_share"] else "fails"
    if c["kind"] == "arms":
        return "meets" if s["arms"] <= c["limit"] else "fails"
    m, sd = abs(s["mean_difference"]), s["sd_difference"]
    if m <= 0.5 and sd <= 0.8:
        return "excellent"
    if m < 1.0 and sd <= ARTERY_ACCEPTABLE_SD:
        return "acceptable"
    return "poor"


def implied(c: dict, y: np.ndarray) -> dict | None:
    """The MAE a rule allows, and the R2 it requires, for unbiased normal errors."""
    if c["kind"] == "share":
        if c["required_share"] is None:
            return None
        sd = c["tolerance"] / stats.norm.ppf((1 + c["required_share"]) / 2)
        if c["relative"]:  # the error SD scales with the true value
            mae, mse = MAE_PER_SD * sd * np.mean(y), sd**2 * np.mean(y**2)
        else:
            mae, mse = MAE_PER_SD * sd, sd**2
    else:
        sd = c["limit"] if c["kind"] == "arms" else ARTERY_ACCEPTABLE_SD
        mae, mse = MAE_PER_SD * sd, sd**2
    return {"mae": float(mae), "required_r2": float(1 - mse / np.var(y))}


def evaluate(oof: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    out = []
    for c in CRITERIA:
        d = oof[oof["target"] == c["target"]].reset_index(drop=True)
        y = d["y"].to_numpy()
        draws = _subject_draws(d["patient_id"].to_numpy(), rng)
        key = PRIMARY[c["kind"]]
        models = {}
        for m in ALL_MODELS:
            pred = d[m].to_numpy()
            s = statistic(c, pred, y)
            boot = [statistic(c, pred[i], y[i])[key] for i in draws]
            models[m] = {
                **s,
                "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                "verdict": verdict(c, s),
                "mae": float(np.mean(np.abs(pred - y))),
                "r2": float(1 - np.mean((pred - y) ** 2) / np.var(y)),
            }
        out.append(
            {
                **c,
                "statistic": key,
                "sessions": len(d),
                "subjects": int(d["patient_id"].nunique()),
                "label_mean": float(y.mean()),
                "label_sd": float(y.std()),
                "label_range": [float(y.min()), float(y.max())],
                "implied": implied(c, y),
                "models": models,
            }
        )
    return out


def main() -> None:
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    out = {
        "criteria": evaluate(oof, np.random.default_rng(SEED)),
        "provenance": record([OUT_DIR / "oof.parquet"]),
    }
    (OUT_DIR / "standards.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    for e in out["criteria"]:
        vals = "  ".join(
            f"{m.split(' ')[0]} {v[e['statistic']]:.3f} ({v['verdict']})"
            for m, v in e["models"].items()
        )
        print(f"{e['target']:>18} {e['statistic']:<14} {vals}")


if __name__ == "__main__":
    main()
