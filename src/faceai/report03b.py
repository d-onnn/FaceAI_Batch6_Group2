"""Phase 3b report -> reports/03b_extensions.md + reports/figures/03b_*.png

    python -m faceai.report03b        # or: make report03b

Four extensions of the Phase 3 robustness work, each adapted from an idea reviewed on the
2026 NTU Chua Thian Poh Pinnacle Prize posters (see the "Origin" line of each section):
a quality gate, a camera-timing stress test, a clinical-threshold view and a
within-person-change test. Reads only saved artifacts; dev folds only.
"""

from __future__ import annotations

import datetime as dt
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceai.cv import OUT_DIR
from faceai.data.hub import ROOT
from faceai.features import CAMERAS, MODEL_FEATURES, SNR_LEVELS_DB, load_features
from faceai.report03 import (
    BLUE,
    COLOR,
    INK,
    INK2,
    LABEL,
    MUTED,
    ORANGE,
    SHORT,
    md_table,
    save,
    style,
)
from faceai.robustness import GATE_QUANTILE, GATE_SQI

REPORT = ROOT / "reports" / "03b_extensions.md"
FOCUS = ["age", "respiratory_rate", "arterial_stiffness"]
NOISE_LEVELS = ["clean", *[f"{s:+d} dB" for s in SNR_LEVELS_DB]]
CAM_LEVELS = [f"camera {c}" for c in CAMERAS]
CAM_SHORT = {
    "clean": "contact\n100 Hz",
    "camera FullHDwebcam": "FullHD",
    "camera USBVideo": "USB",
    "camera IriunWebcam": "Iriun",
}
RAMP = [
    "#86b6ef",
    "#3987e5",
    "#1c5cab",
    "#0d366b",
]  # ordinal blue, validated (--ordinal)
MODELS6 = ["Mean", "Median", "Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]


def tick(level: str) -> str:
    return {"clean": "clean", "+0 dB": "0"}.get(level, level.replace(" dB", ""))


def title_of(t: str) -> str:
    return LABEL[t].replace(" — positive control", " (control)")


def suptitle(fig, text: str, y: float = 1.04) -> None:
    fig.suptitle(
        text, x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK, y=y
    )


def _m(g: pd.DataFrame, target: str, model: str, level: str, col: str) -> float:
    d = g[(g["target"] == target) & (g["model"] == model) & (g["level"] == level)]
    return float(d[col].mean())


# ------------------------------------------------------------------------- figures
def fig_gate(gate: pd.DataFrame) -> str:
    g = gate[(gate["model"] == "LightGBM") & gate["headline"]]
    fig, axes = plt.subplots(
        2, 3, figsize=(10.5, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.7, 1]}
    )
    x = np.arange(len(NOISE_LEVELS))
    cols = ["mae_ungated", "mae_gated", "mae_constant", "coverage"]
    for j, t in enumerate(FOCUS):
        d = g[g["target"] == t].groupby("level")[cols].mean().reindex(NOISE_LEVELS)
        ax = axes[0, j]
        ax.plot(
            x, d["mae_constant"], color=MUTED, linewidth=1.2, label="Constant guess"
        )
        ax.plot(x, d["mae_ungated"], "-o", color=ORANGE, markersize=4, label="No gate")
        ax.plot(x, d["mae_gated"], "-o", color=BLUE, markersize=4, label="Quality gate")
        ax.set_title(title_of(t), loc="left")
        axes[1, j].plot(x, 100 * d["coverage"], "-o", color=INK2, markersize=4)
        axes[1, j].set_ylim(-5, 105)
        axes[1, j].set_xticks(x)
        axes[1, j].set_xticklabels([tick(v) for v in NOISE_LEVELS], fontsize=8)
        axes[1, j].set_xlabel("In-band SNR of injected noise, dB")
    axes[0, 0].set_ylabel("Validation MAE (LightGBM)")
    axes[1, 0].set_ylabel("Recordings predicted, %")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=3)
    suptitle(
        fig, "Quality gate: predict only when the recording is clean enough", y=1.02
    )
    return save(fig, "03b_quality_gate.png")


def fig_tradeoff(gate: pd.DataFrame) -> str:
    g = gate[gate["model"] == "LightGBM"]
    levels = ["clean", "+20 dB", "+10 dB", "+5 dB"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    cols = ["coverage", "mae_gated", "mae_constant"]
    for ax, t in zip(axes, FOCUS, strict=True):
        d = g[g["target"] == t].groupby(["level", "quantile"])[cols].mean()
        for colr, lv in zip(RAMP, levels, strict=True):
            s = d.loc[lv].sort_index()
            name = lv if lv == "clean" else f"{tick(lv)} dB"
            ax.plot(100 * s["coverage"], s["mae_gated"], "-o", color=colr, markersize=4,
                    label=name)  # fmt: skip
            h = s.loc[GATE_QUANTILE]
            ax.plot(100 * h["coverage"], h["mae_gated"], "o", markersize=9,
                    markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.2)  # fmt: skip
        ax.axhline(d["mae_constant"].mean(), color=MUTED, linewidth=1.0)
        ax.set_xlabel("Recordings predicted, %")
        ax.set_title(title_of(t), loc="left")
    axes[0].set_ylabel("MAE (abstentions get the constant)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=4)
    suptitle(
        fig,
        "Error vs coverage as the gate threshold moves (ringed = 5th-percentile headline)",
        y=1.13,
    )
    return save(fig, "03b_gate_tradeoff.png")


def camera_feature_table() -> pd.DataFrame:
    clean = load_features()
    clean = clean[clean["readable"]].set_index(["patient_id", "step"])
    rows = []
    for cam in CAMERAS:
        f = load_features(camera=cam)
        f = f[f["readable"]].set_index(["patient_id", "step"]).loc[clean.index]
        row = {"camera": cam}
        for k in [
            "hr_mean_bpm",
            "rise_time_s",
            "apg_b_a",
            "ibi_rmssd_ms",
            "sqi_template_r",
        ]:
            ok = clean[k].notna() & f[k].notna()
            row[f"{k} mean shift"] = float((f.loc[ok, k] - clean.loc[ok, k]).mean())
            row[f"{k} r"] = float(np.corrcoef(f.loc[ok, k], clean.loc[ok, k])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def fig_camera(cam: pd.DataFrame, res: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    levels = ["clean", *CAM_LEVELS]
    x = np.arange(len(levels))
    for ax, t in zip(axes, FOCUS, strict=True):
        d = cam[(cam["target"] == t) & (cam["model"] == "LightGBM")]
        a = d[d["trained_on"] == "clean"].groupby("level")["val_mae"].mean()
        b = d[d["trained_on"] == "same camera"].groupby("level")["val_mae"].mean()
        a, b = a.reindex(levels), b.reindex(levels)
        ax.axhline(res[t]["models"]["Mean"]["mae"], color=MUTED, linewidth=1.0)
        ax.plot(x, a, "o", color=ORANGE, markersize=7, label="Fitted on contact PPG")
        ax.plot(x[1:], b.iloc[1:], "o", color=BLUE, markersize=7,
                label="Refitted on the same camera timing")  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels([CAM_SHORT[v] for v in levels], fontsize=8)
        ax.set_title(title_of(t), loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Validation MAE (LightGBM)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2)
    suptitle(
        fig,
        "Camera timing: the contact pulse read at each camera's real frame times",
        y=1.13,
    )
    return save(fig, "03b_camera_timing.png")


def fig_clinical(clin: dict) -> str:
    th = clin["thresholds"]
    models = ["Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    offs = np.linspace(-0.24, 0.24, len(models))
    for i, e in enumerate(th):
        y0 = len(th) - 1 - i
        for o, m in zip(offs, models, strict=True):
            v = e["models"][m]
            ax.plot(v["ci"], [y0 - o] * 2, color=COLOR[m], linewidth=1.6)
            ax.plot(v["auroc"], y0 - o, "o", color=COLOR[m], markersize=6,
                    label=m if i == 0 else None)  # fmt: skip
    ax.axvline(0.5, color=INK2, linewidth=1.0)
    ax.set_yticks(range(len(th)))
    names = [
        f"{e['label'].split(' (')[0]}\n({e['positive_subjects']} positive dev subjects)"
        for e in reversed(th)
    ]
    ax.set_yticklabels(names, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC with 95% CI (0.5 = chance)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4)
    suptitle(fig, "Clinical thresholds: who is above the cut-off?", y=1.06)
    return save(fig, "03b_clinical_thresholds.png")


def change_label(r: dict) -> str:
    target = SHORT.get(r["target"], "Pulse")
    name = r["signal"] if r["signal"] in COLOR else r["signal"].split(" vs ")[0]
    return f"{target}: {name}"


def fig_change(clin: dict) -> str:
    ch = clin["change"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, r in enumerate(ch):
        y = len(ch) - 1 - i
        colr = (
            INK if "positive control" in r["signal"] else COLOR.get(r["signal"], INK2)
        )
        ax.plot(r["ci"], [y, y], color=colr, linewidth=1.6)
        ax.plot(r["rho"], y, "o", color=colr, markersize=6)
    ax.axvline(0, color=INK2, linewidth=1.0)
    ax.set_yticks(range(len(ch)))
    ax.set_yticklabels([change_label(r) for r in reversed(ch)], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "Spearman ρ between a person's change and the label's change (95% CI)"
    )
    suptitle(
        fig,
        "Within-person change, rest → post-exercise (black = heart-rate control)",
        y=1.03,
    )
    return save(fig, "03b_within_person_change.png")


# --------------------------------------------------------------------------- report
def prov_lines(name: str, p: dict) -> list[str]:
    files = p["inputs_sha256"]
    if len(files) <= 2:
        inputs = ", ".join(f"`{k}` {v[:12]}…" for k, v in files.items())
    else:
        inputs = f"{len(files)} files, each SHA-256 recorded in the JSON"
    return [
        (
            f"- **{name}** — generated {p['generated']}; commit"
            f" `{p['git_commit'][:7]}` (uncommitted paths: {p['git_dirty_paths']});"
            f" split SHA-256 `{p['splits_assignment_sha256'][:12]}…`; db.csv blob"
            f" `{p['db_csv_git_blob'][:12]}…`; lockfile `{p['lockfile_sha256'][:12]}…`;"
            f" inputs: {inputs}."
        )
    ]


def gate_section(gate: pd.DataFrame, figs: dict) -> list[str]:
    g = gate[gate["headline"]]
    levels = NOISE_LEVELS + CAM_LEVELS
    shown = [*FOCUS, "bmi", "hba1c", "cholesterol"]
    mean = g.groupby(["target", "model", "level"])[
        ["coverage", "mae_ungated", "mae_gated"]
    ].mean()
    cov = mean["coverage"].groupby("level").mean().reindex(levels)
    heads = ["clean", *[tick(v) for v in NOISE_LEVELS[1:]], "FullHD", "USB", "Iriun"]
    cov_row = "| Recordings predicted | " + " | ".join(f"{100 * c:.1f}%" for c in cov)
    rows = []
    for t in shown:
        for m in ["Ridge (L2)", "LightGBM"]:
            cells = [
                f"{mean.loc[(t, m, lv), 'mae_ungated']:.3f} → "
                f"{mean.loc[(t, m, lv), 'mae_gated']:.3f}"
                for lv in levels
            ]
            rows.append(f"| {SHORT[t]}, {m} | " + " | ".join(cells) + " |")
    worst_clean_cost = max(
        _m(g, t, m, "clean", "mae_gated") - _m(g, t, m, "clean", "mae_ungated")
        for t in shown
        for m in ["Ridge (L2)", "LightGBM"]
    )
    age0 = (_m(g, "age", "LightGBM", "+0 dB", "mae_ungated"),
            _m(g, "age", "LightGBM", "+0 dB", "mae_gated"))  # fmt: skip
    rr0 = (_m(g, "respiratory_rate", "Ridge (L2)", "+0 dB", "mae_ungated"),
           _m(g, "respiratory_rate", "Ridge (L2)", "+0 dB", "mae_gated"))  # fmt: skip
    abstain10 = 100 - 100 * cov["+10 dB"]
    return [
        "## 1. Quality gate — verify every recording before predicting",
        "",
        "_Origin: BetterMush (inline pass/reject of every meal instead of batch sampling) and",
        "Pure-Stream (only items that pass inspection are diverted; everything else takes the",
        "normal route)._",
        "",
        f"A model predicts only when the recording's template-match quality (`{GATE_SQI}`) is",
        f"at least the {round(100 * GATE_QUANTILE)}th percentile of its own training fold; otherwise it",
        "abstains and the training-fold mean is reported instead. The threshold comes from",
        "training data only and was fixed before any gate result was seen.",
        "",
        f"![gate]({figs['gate']})",
        "",
        "Validation MAE, **no gate → gated** (noise columns are in-band SNR in dB; the last",
        "three are the camera-timed inputs of section 2). Coverage is the same for every target",
        "because the gate looks only at signal quality.",
        "",
        "| Target, model | " + " | ".join(heads) + " |",
        "|" + "---|" * (len(heads) + 1),
        cov_row + " |",
        *rows,
        "",
        "**Reading it.** The gate turns catastrophic failures into the constant guess: age at",
        f"0 dB goes from {age0[0]:.1f} to {age0[1]:.1f} years, respiratory rate (Ridge) from",
        f"{rr0[0]:.2f} to {rr0[1]:.2f}/min. On clean contact PPG it adds at most",
        f"{max(worst_clean_cost, 0):.3f} to any shown target's error. But the 5th-percentile",
        f"setting is conservative: at +10 dB it abstains on {abstain10:.0f}% of recordings,",
        "discarding predictions that were still better than the constant. The sweep below",
        "shows the trade-off to choose a deployment setting from.",
        "",
        "**Blind spot.** The gate does not fire on the camera-timed inputs (coverage stays",
        "about 95%) even though age error rises there (section 2): quality is high but the pulse",
        "shape has shifted. A quality gate catches noise, not a change of input domain.",
        "",
        f"![tradeoff]({figs['tradeoff']})",
        "",
    ]


def camera_section(cam: pd.DataFrame, figs: dict) -> list[str]:
    ft = camera_feature_table()
    show = ft[
        ["camera", "hr_mean_bpm mean shift", "hr_mean_bpm r", "rise_time_s mean shift",
         "rise_time_s r", "apg_b_a mean shift", "apg_b_a r", "sqi_template_r r"]
    ].copy()  # fmt: skip
    show.columns = [
        "Camera", "HR shift (bpm)", "HR r", "Rise-time shift (s)", "Rise-time r",
        "APG b/a shift", "APG b/a r", "Quality r",
    ]  # fmt: skip
    ct = cam.groupby(["target", "trained_on", "model", "level"])["val_mae"].mean()
    ct = ct.unstack("level")[["clean", *CAM_LEVELS]].reset_index()
    ct = ct[ct["model"] != "Mean"]
    ct["target"] = ct["target"].map(SHORT)
    ct.columns = ["Target", "Fitted on", "Model", "Contact 100 Hz", *CAMERAS]
    a = (
        cam[(cam["target"] == "age") & (cam["model"] == "LightGBM")]
        .groupby(["trained_on", "level"])["val_mae"]
        .mean()
    )
    rng = lambda col: f"{ft[col].min():.2f}–{ft[col].max():.2f}"
    shift_ms = 1000 * ft["rise_time_s mean shift"]
    return [
        "## 2. Camera timing — a controlled scenario variation",
        "",
        "_Origin: Graphyti (vary one controlled factor at a time to test a model on conditions",
        "that are hard to collect)._",
        "",
        "The contact pulse is read at every frame each camera actually captured (real frame",
        "times from `meta/`, which share the PPG clock: ~30 fps FullHD/USB, ~24 fps Iriun, with",
        "their jitter and gaps), then re-interpolated to 100 Hz exactly as a camera signal will",
        "be. This isolates the camera's **timing** only — not its optics, site or noise.",
        "",
        f"![camera]({figs['camera']})",
        "",
        "Feature change versus the 100 Hz contact value (mean shift; Pearson r):",
        "",
        *md_table(show.round(4)),
        "",
        "Validation MAE by input:",
        "",
        *md_table(ct.round(3).fillna("—")),
        "",
        (
            "**Reading it.** Heart rate is unchanged by camera timing"
            f" (r ≥ {ft['hr_mean_bpm r'].min():.4f})."
        ),
        (
            f"Pulse-shape features shift slightly and consistently: rise time +{shift_ms.min():.0f} to"
            f" +{shift_ms.max():.0f} ms,"
        ),
        (
            f"APG b/a {ft['apg_b_a mean shift'].min():+.3f} to {ft['apg_b_a mean shift'].max():+.3f}"
            f" (r {rng('rise_time_s r')} and {rng('apg_b_a r')}); the quality score"
        ),
        f"agrees less (r {rng('sqi_template_r r')}). Rate-based targets are untouched. The",
        (
            "shape-dependent age model degrades —"
            f" {a[('clean', 'clean')]:.2f} → {a[('clean', 'camera FullHDwebcam')]:.2f} years at FullHD and"
        ),
        f"{a[('clean', 'camera IriunWebcam')]:.2f} at Iriun's ~24 fps — and refitting on the same timing",
        (
            f"recovers it ({a[('same camera', 'camera FullHDwebcam')]:.2f} /"
            f" {a[('same camera', 'camera IriunWebcam')]:.2f}). The contact-to-camera gap is"
        ),
        "therefore not mainly about frame timing: Phase 4 should spend its effort on optical",
        "noise and signal quality, and train on the signal it will be deployed on.",
        "",
    ]


def clinical_section(clin: dict, figs: dict) -> list[str]:
    rows = []
    for e in clin["thresholds"]:
        for m, v in e["models"].items():
            rows.append([
                e["label"], e["positive_subjects"], m,
                f"{v['auroc']:.3f} [{v['ci'][0]:.3f}–{v['ci'][1]:.3f}]",
                v["predicted_positive_sessions"], f"{v['sensitivity_at_cut']:.3f}",
                f"{v['specificity_at_cut']:.3f}",
            ])  # fmt: skip
    cols = [
        "Threshold", "Positive dev subjects", "Model", "AUROC [95% CI]",
        "Sessions predicted above cut-off", "Sensitivity at cut-off",
        "Specificity at cut-off",
    ]  # fmt: skip
    rr = clin["tolerances"]["respiratory_rate"]
    sp = clin["tolerances"]["spo2"]
    hb65 = clin["thresholds"][1]
    lo_hi = [
        v["auroc"]
        for e in (clin["thresholds"][0], clin["thresholds"][2])
        for v in e["models"].values()
    ]
    within = " | ".join(f"{100 * rr['share_within'][m]:.1f}%" for m in MODELS6)
    arms = " | ".join(f"{sp['arms'][m]:.2f}" for m in MODELS6)
    return [
        "## 3. Clinical thresholds — judged against recognised standards",
        "",
        "_Origin: BetterMush (built around the IDDSI framework) and NatureRisk (built around",
        "TNFD, with every conclusion linked to its evidence)._",
        "",
        f"![clinical]({figs['clinical']})",
        "",
        *md_table(pd.DataFrame(rows, columns=cols)),
        "",
        "| Standard | Mean | Median | Demographics | Ridge | Lasso | LightGBM |",
        "|---|---|---|---|---|---|---|",
        f"| Respiratory rate within ±3/min (share of sessions) | {within} |",
        f"| SpO₂ ARMS, % (ISO limit 4 % over SaO₂ 70–100 %) | {arms} |",
        "",
        "**Reading it.**",
        "",
        f"- **Diabetes-range HbA1c (≥ 6.5 %)**: only {hb65['positive_subjects']} positive dev",
        (
            "  subjects. Demographics alone rank them best (AUROC"
            f" {hb65['models']['Demographics']['auroc']:.2f}): the signal is demographic (age,"
        ),
        "  sex, BMI), not PPG. No PPG model's prediction ever reaches 6.5 %, so at the clinical",
        "  cut-off its sensitivity is zero.",
        (
            "- **HbA1c ≥ 5.7 % and cholesterol ≥ 5.2 mmol/L**: AUROC"
            f" {min(lo_hi):.2f}–{max(lo_hi):.2f} across models; adding PPG features does not"
        ),
        "  improve on demographics alone.",
        (
            "- **A constant guess meets two clinical tolerances.** It lands within ±3/min for"
            f" {100 * rr['share_within']['Mean']:.1f}%"
        ),
        f"  of sessions (the label SD is only {rr['label_sd']:.1f}/min) and has an SpO₂ ARMS of",
        (
            f"  {sp['arms']['Mean']:.2f}% — under the ISO 4 % limit. But only"
            f" {100 * sp['label_share_below_90']:.1f}% of SpO₂ labels"
        ),
        "  are below 90 % and none below 70 %, so the ISO condition (a 70–100 % reference",
        "  range) is unmet: these numbers cannot establish compliance, and a tolerance that a",
        "  constant passes says nothing about the model.",
        "",
    ]


def change_section(clin: dict, figs: dict) -> list[str]:
    ch = pd.DataFrame(clin["change"])
    ch["rho [95% CI]"] = [
        f"{r:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"
        for r, c in zip(ch["rho"], ch["ci"], strict=True)
    ]
    ch["Direction agrees"] = (100 * ch["direction_agreement"]).round(1).astype(
        str
    ) + "%"
    ch["Label"] = ch["target"].map(lambda v: SHORT.get(v, "Pulse"))
    ch = ch.rename(
        columns={
            "signal": "Signal",
            "subjects": "Subjects",
            "label_moved": "Label changed",
            "signal_unchanged": "Signal unchanged",
        }
    )
    cols = [
        "Label",
        "Signal",
        "Subjects",
        "Label changed",
        "Signal unchanged",
        "rho [95% CI]",
        "Direction agrees",
    ]
    return [
        "## 4. Within-person change — does the model follow the person?",
        "",
        "_Origin: SymSkin (an “AI health memory” that tracks each user against their own",
        "history)._",
        "",
        f"![change]({figs['change']})",
        "",
        *md_table(ch[cols]),
        "",
        "_Direction agrees_ counts only subjects whose signal and label both moved. It mostly",
        "reflects the average post-exercise shift (most people's respiratory rate rises), so it",
        "can be high while ρ is near zero; ρ asks whether a bigger personal change is",
        "predicted as bigger.",
        "",
        "**Reading it.** The positive control works: a person's PPG heart-rate change tracks",
        "their pulse-label change. Arterial-stiffness predictions follow each person's change",
        "weakly but significantly; respiratory-rate and SpO₂ predictions do not, and neither",
        "does the PPG respiration estimate itself. All three labels were taken at the separate",
        "medical station, so weak tracking may reflect the label as much as the signal.",
        "Personal baselines are worth building only for signals that pass this test.",
        "",
    ]


def main() -> None:
    style()
    res_all = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    gate = pd.read_csv(OUT_DIR / "gate.csv")
    cam = pd.read_csv(OUT_DIR / "camera.csv")
    clin = json.loads((OUT_DIR / "clinical.json").read_text(encoding="utf-8"))
    rprov = json.loads(
        (OUT_DIR / "robustness_provenance.json").read_text(encoding="utf-8")
    )
    figs = {
        "gate": fig_gate(gate),
        "tradeoff": fig_tradeoff(gate),
        "camera": fig_camera(cam, res_all["targets"]),
        "clinical": fig_clinical(clin),
        "change": fig_change(clin),
    }
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Phase 3b — Quality gate, camera timing, clinical thresholds, within-person change",
        "",
        f"_Generated {now} by `python -m faceai.report03b` from saved artifacts only._",
        "",
        "Extensions of the Phase 3 contact-PPG work (`reports/03_cv_robustness.md`), on the",
        "same dev folds and models. No new data, no video, and the held-out test set is not read.",
        "",
        *gate_section(gate, figs),
        *camera_section(cam, figs),
        *clinical_section(clin, figs),
        *change_section(clin, figs),
        "## 5. Provenance",
        "",
        *prov_lines(
            "CV results (`artifacts/cv/results.json`)", res_all["meta"]["provenance"]
        ),
        *prov_lines("Robustness (`artifacts/cv/*.csv`)", rprov),
        *prov_lines("Clinical (`artifacts/cv/clinical.json`)", clin["provenance"]),
        "",
        "## 6. Reproduce",
        "",
        "```",
        "make features   # adds the three camera-timed feature tables",
        "make cv",
        "make robustness # adds camera.csv and gate.csv",
        "make clinical",
        "make report03b",
        "```",
        "",
        f"Model inputs: `{', '.join(MODEL_FEATURES)}`.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT} and {len(figs)} figures")


if __name__ == "__main__":
    main()
