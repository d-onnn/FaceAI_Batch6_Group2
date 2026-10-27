"""Phase 3e report -> reports/03e_surrogate_lessons.md + reports/figures/03e_*.png

    python -m faceai.report03e        # or: make report03e

What the MoSPSyDe group's surrogate-modelling material (Maaz Ahmad, "Basics of Surrogate
Modeling", 7 Oct 2026; K. Matsunami's population-balance lecture and group-meeting slides)
changes for FaceAI, tested on the dev folds. Reads only saved artifacts.
"""

from __future__ import annotations

import datetime as dt
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceai.cv import OUT_DIR, TARGETS, session_table
from faceai.data.hub import ROOT
from faceai.design import FRONTAL_GB, PILOT_LIST, SIZES, STRATEGIES
from faceai.graybox import load_graybox
from faceai.report03 import (
    AQUA,
    BLUE,
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
from faceai.report03b import prov_lines, suptitle
from faceai.robustness import _cluster_ci
from faceai.splits import OLDER_THAN, SEED
from faceai.surrogate import NEW_MODELS

REPORT = ROOT / "reports" / "03e_surrogate_lessons.md"
AGE_BINS = [18, 20, 22, 25, 40, 60, 90]
STRATEGY_COLOR = {
    "random": MUTED,
    "stratified": INK2,
    "space-filling": BLUE,
    "hybrid": ORANGE,
}
DESIGN_TARGETS = ["age", "arterial_stiffness", "hba1c"]
PPG3 = ["Ridge (L2)", "Lasso (L1)", "LightGBM"]


def _one(items: list[dict], **match) -> dict:
    (r,) = [r for r in items if all(r[k] == v for k, v in match.items())]
    return r


def fig_graybox(t: pd.DataFrame, g: pd.DataFrame) -> str:
    d = t.merge(g, on=["patient_id", "step"])
    d["band"] = pd.cut(d["age"], AGE_BINS, right=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    for ax, col, color, title in [
        (axes[0], "si_mps", BLUE, "PPG stiffness index (height / ΔT, m/s)"),
        (axes[1], "arterial_stiffness", ORANGE, "Dataset 'arterial stiffness' label"),
    ]:
        grp = d.groupby("band", observed=True)[col]
        med, q1, q3, n = (
            grp.median(),
            grp.quantile(0.25),
            grp.quantile(0.75),
            grp.size(),
        )
        x = np.arange(len(med))
        ax.errorbar(
            x,
            med,
            yerr=[med - q1, q3 - med],
            fmt="o",
            color=color,
            capsize=3,
            markersize=6,
        )
        ax.set_xticks(
            x,
            [
                f"{int(b.left)}–{int(b.right) - 1}\n(n={k})"
                for b, k in zip(med.index, n, strict=True)
            ],
            fontsize=7.5,
        )
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("median and IQR per age band", fontsize=8)
    fig.tight_layout()
    suptitle(
        fig,
        "The PPG stiffness index rises with age; the dataset's stiffness label falls",
        y=1.06,
    )
    return save(fig, "03e_graybox.png")


def fig_forms(metrics: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rows = list(reversed(TARGETS))
    for k, t in enumerate(rows):
        groups = {
            "Demographics": _one(metrics, target=t, model="Demographics"),
            "best Phase 3 PPG model": min(
                (_one(metrics, target=t, model=m) for m in PPG3), key=lambda r: r["mae"]
            ),
            "best new model form": min(
                (_one(metrics, target=t, model=m) for m in NEW_MODELS),
                key=lambda r: r["mae"],
            ),
        }
        for off, (name, r), color in zip(
            (-0.22, 0, 0.22), groups.items(), (INK2, BLUE, ORANGE), strict=True
        ):
            lo, hi = r["vs_mean_ci"]
            ax.plot([lo, hi], [k + off] * 2, color=color, linewidth=1.6)
            ax.plot(
                r["vs_mean"],
                k + off,
                "o",
                color=color,
                markersize=5.5,
                label=name if k == 0 else None,
            )
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(range(len(rows)), [SHORT[t] for t in rows], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "MAE minus the constant guess's MAE, with 95 % CI (left of 0 = better)",
        fontsize=8,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    suptitle(
        fig, "No model form finds signal the linear and tree models missed", y=1.08
    )
    return save(fig, "03e_model_forms.png")


def fig_coverage(cov: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = list(reversed(TARGETS))
    keys = [("coverage_all", "all sessions", INK2), ("coverage_lowest_10", "lowest 10 % of the label", BLUE),
            ("coverage_highest_10", "highest 10 % of the label", ORANGE), (f"coverage_age_ge_{OLDER_THAN}", f"aged ≥ {OLDER_THAN}", AQUA)]  # fmt: skip
    for k, t in enumerate(rows):
        c = _one(cov, target=t)
        for off, (key, name, color) in zip(
            (-0.27, -0.09, 0.09, 0.27), keys, strict=True
        ):
            ax.plot(
                c[key],
                k + off,
                "o",
                color=color,
                markersize=5.5,
                label=name if k == 0 else None,
            )
    ax.axvline(0.95, color=INK, linewidth=1, linestyle=(0, (4, 2)))
    ax.text(0.95, len(rows) - 0.4, " nominal 95 %", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)), [SHORT[t] for t in rows], fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of true values inside the kriging 95 % interval", fontsize=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=7.5)
    suptitle(
        fig,
        "Kriging error bars hold on average and fail for the people who matter",
        y=1.1,
    )
    return save(fig, "03e_kriging_coverage.png")


def fig_design(summary: list[dict]) -> str:
    fig, axes = plt.subplots(2, len(DESIGN_TARGETS), figsize=(7.6, 4.6), sharex=True)
    for col, t in enumerate(DESIGN_TARGETS):
        for row, (m, ylabel) in enumerate(
            [("mae", "MAE"), ("tail_mae", "MAE, lowest + highest 10 %")]
        ):
            ax = axes[row, col]
            for st in STRATEGIES:
                pts = sorted(
                    (r for r in summary if r["target"] == t and r["strategy"] == st),
                    key=lambda r: r["n"],
                )
                n = [r["n"] for r in pts]
                mean = np.array([r[f"{m}_mean"] for r in pts])
                sd = np.array([r[f"{m}_sd"] for r in pts])
                ax.plot(
                    n,
                    mean,
                    "o-",
                    color=STRATEGY_COLOR[st],
                    markersize=4,
                    linewidth=1.6,
                    label=st if (row, col) == (0, 0) else None,
                )
                if sd.any():
                    ax.fill_between(
                        n,
                        mean - sd,
                        mean + sd,
                        color=STRATEGY_COLOR[st],
                        alpha=0.12,
                        linewidth=0,
                    )
            if row == 0:
                ax.set_title(
                    LABEL[t].replace(" — positive control", ""), fontsize=9, loc="left"
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 1:
                ax.set_xlabel("training subjects chosen", fontsize=8)
            ax.set_xticks(SIZES)
    fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.92))
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.01, 0.97),
        ncol=4,
        fontsize=8,
    )
    suptitle(
        fig,
        "Which subjects to record first: space-filling design against random choice",
        y=1.02,
    )
    return save(fig, "03e_design.png")


def fig_tradeoff(tr: dict) -> str:
    c = tr["curve"]
    x = [p["mae"] for p in c]
    y = [p["top10_bias"] for p in c]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(x, y, "o-", color=BLUE, linewidth=1.8, markersize=5)
    for p in c:
        if p["lambda"] in (0.0, 0.5, 1.0):
            ax.annotate(
                f"λ = {p['lambda']:g}",
                (p["mae"], p["top10_bias"]),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=7.5,
                color=INK2,
            )
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("overall MAE (years)", fontsize=8)
    ax.set_ylabel(f"mean error, oldest 10 % (≥ {tr['top10_cut']:.0f} y)", fontsize=8)
    suptitle(fig, "Age model: accuracy against underestimating older people", y=1.04)
    return save(fig, "03e_tradeoff.png")


def main() -> None:
    style()
    gb = json.loads((OUT_DIR / "graybox.json").read_text(encoding="utf-8"))
    sg = json.loads((OUT_DIR / "surrogate.json").read_text(encoding="utf-8"))
    dz = json.loads((OUT_DIR / "design.json").read_text(encoding="utf-8"))
    t = session_table()
    g = load_graybox()
    figs = {
        "graybox": fig_graybox(t, g),
        "forms": fig_forms(sg["metrics"]),
        "coverage": fig_coverage(sg["kriging_coverage"]),
        "design": fig_design(dz["summary"]),
        "tradeoff": fig_tradeoff(sg["tradeoff"]),
    }
    lines = body(gb, sg, dz, figs)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT} and {len(figs)} figures")


def _paired(
    both: pd.DataFrame, target: str, m1: str, m0: str, rng: np.random.Generator
) -> tuple[float, list[float]]:
    g = both[both["target"] == target]
    d = (np.abs(g[m1] - g["y"]) - np.abs(g[m0] - g["y"])).to_numpy()
    return float(d.mean()), _cluster_ci(d, g["patient_id"].to_numpy(), rng)


def body(gb: dict, sg: dict, dz: dict, figs: dict) -> list[str]:
    met, cov, tr = sg["metrics"], sg["kriging_coverage"], sg["tradeoff"]
    res = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))["targets"]
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    new = pd.read_parquet(OUT_DIR / "surrogate_oof.parquet")
    both = oof.merge(
        new, on=["patient_id", "step", "target", "fold", "y"], validate="1:1"
    )
    rng = np.random.default_rng(SEED)
    v, lab, wp = gb["validity"], gb["label"], gb["within_person"]
    ci_pos = lambda r: r["ci"][0] > 0
    ci_neg = lambda r: r["ci"][1] < 0
    ci_zero = lambda r: r["ci"][0] < 0 < r["ci"][1]

    # ---- guard the written claims: gray box
    assert (
        ci_pos(v["dt_vs_height"]) and ci_neg(v["dt_vs_age"]) and ci_pos(v["si_vs_age"])
    )
    assert ci_neg(lab["label_vs_age"]) and lab["label_by_age_band"]["ci"][1] < 0
    assert ci_pos(lab["label_vs_heart_rate"]) and ci_zero(lab["label_vs_si"])
    assert ci_pos(wp["dlabel_vs_dhr"]) and ci_zero(wp["dlabel_vs_dsi"])
    assert all(p["change_ci"][0] < 0 < p["change_ci"][1] for p in gb["predictive"])
    # ---- model forms
    null_targets = ["hba1c", "cholesterol", "spo2", "stress_psm25"]
    for t in null_targets:
        assert not any(
            r["vs_mean_ci"][1] < 0
            for r in met
            if r["target"] == t and r["model"] != "Mean"
        ), t
        assert _one(met, target=t, model="Kriging (GP)")["vs_mean_ci"][0] > 0, t
    svr_temp = _one(met, target="temperature", model="SVR (RBF)")
    median_temp = res["temperature"]["models"]["Median"]["mae"]
    assert svr_temp["mae"] > median_temp
    age_rows = [r for r in met if r["target"] == "age"]
    knn = _one(met, target="age", model="kNN")
    assert knn["mae"] == min(r["mae"] for r in age_rows)
    knn_vs_lgbm = _paired(both, "age", "kNN", "LightGBM", rng)
    assert knn_vs_lgbm[1][1] < 0
    g = both[both["target"] == "age"]
    older = g["y"].to_numpy() >= OLDER_THAN
    mae_old = {
        m: float(np.abs(g[m] - g["y"]).to_numpy()[older].mean())
        for m in ("kNN", "LightGBM")
    }
    assert mae_old["kNN"] > mae_old["LightGBM"]
    strict_age = [r["model"] for r in age_rows if r["beats_both_strict"]]
    max_lo, max_hi = (
        min(r["max_ae"] for r in age_rows),
        max(r["max_ae"] for r in age_rows),
    )
    worst = g.iloc[int(np.argmax(np.abs(g["LightGBM"] - g["y"]).to_numpy()))]
    # ---- kriging coverage
    cv_ = {c["target"]: c for c in cov}
    assert (
        cv_["spo2"]["coverage_lowest_10"] < 0.5
        and cv_["age"][f"coverage_age_ge_{OLDER_THAN}"] < 0.5
    )
    assert cv_["age"]["sd_vs_abs_error_ci"][0] > 0
    assert all(cv_[t]["sd_vs_abs_error_ci"][0] < 0.05 for t in null_targets)
    # ---- trade-off
    half = _one(tr["curve"], **{"lambda": 0.5})
    zero, full = tr["curve"][0], tr["curve"][-1]
    assert zero["lambda"] == 0 and full["lambda"] == 1 and half["mae"] > zero["mae"]
    # ---- design
    summ = dz["summary"]
    pil = dz["pilot_first_48"]

    def dz_row(st, n, t):
        return _one(summ, strategy=st, n=n, target=t)

    f = lambda x, d=2: f"{x:+.{d}f}"
    rci = lambda r: f"{r['rho']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"

    form_rows = []
    for t in TARGETS:
        for m in ["Mean", "Demographics", *PPG3, *NEW_MODELS]:
            r = _one(met, target=t, model=m)
            form_rows.append([
                SHORT[t], m, f"{r['mae']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]",
                "—" if m == "Mean" else f"{f(r['vs_mean'], 3)} [{f(r['vs_mean_ci'][0], 3)}, {f(r['vs_mean_ci'][1], 3)}]",
                f"{r['median_ae']:.3f}", f"{r['r2']:+.2f}", f"{r['p95_ae']:.2f}", f"{r['max_ae']:.2f}",
                "yes" if r["beats_both_strict"] else "",
            ])  # fmt: skip
    cov_rows = [
        [
            SHORT[c["target"]], f"{c['coverage_all']:.2f}", f"{c['coverage_lowest_10']:.2f}", f"{c['coverage_highest_10']:.2f}",
            f"{c[f'coverage_age_ge_{OLDER_THAN}']:.2f}", f"±{c['mean_interval_halfwidth']:.2f}",
            f"{c['sd_vs_abs_error_rho']:+.2f} [{c['sd_vs_abs_error_ci'][0]:+.2f}, {c['sd_vs_abs_error_ci'][1]:+.2f}]",
        ]
        for c in cov
    ]  # fmt: skip
    design_rows = []
    for t in DESIGN_TARGETS + ["respiratory_rate", "bmi", "cholesterol"]:
        for n in SIZES:
            rr = {st: dz_row(st, n, t) for st in STRATEGIES}
            design_rows.append([
                SHORT[t], n,
                *[f"{rr[st]['mae_mean']:.3f}" + (f" ± {rr[st]['mae_sd']:.3f}" if rr[st]["draws"] > 1 else "") for st in STRATEGIES],
                *[f"{rr[st]['tail_mae_mean']:.2f}" for st in STRATEGIES],
            ])  # fmt: skip
    trade_rows = [
        [f"{p['lambda']:g}", f"{p['mae']:.2f}", f"{p['top10_bias']:+.2f}"]
        for p in tr["curve"]
        if p["lambda"] in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
    ]

    ks_age = {n: dz_row("space-filling", n, "age") for n in SIZES}
    rnd_age = {n: dz_row("random", n, "age") for n in SIZES}
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return [
        "# Phase 3e — Surrogate-modelling lessons applied to FaceAI",
        "",
        f"_Generated {now} by `python -m faceai.report03e` from saved artifacts only._",
        "",
        "Source material: the MoSPSyDe group's meeting materials (NTU School of Chemistry,",
        "Chemical Engineering and Biotechnology) — Maaz Ahmad, *Basics of Surrogate Modeling*",
        "(7 Oct 2026); K. Matsunami, *Introduction to Population Balance Modeling* (25 Jul 2025);",
        "K. Matsunami, group-meeting slides (7 Oct 2026). Every test below uses the same contact-PPG",
        "dev folds as Phase 3 (479 subjects, 956 sessions); the held-out test set is not read.",
        "",
        "## 0. What each idea became",
        "",
        "| Idea from the material | Where | What we did here | Section |",
        "|---|---|---|---|",
        "| White-, gray- and black-box models; hybrid models | Ahmad p. 2 | Built the mechanistic PPG stiffness index (height / ΔT) and tested the stiffness label against it | 1 |",
        "| Model form must be flexible enough; the list of surrogate techniques | Ahmad pp. 6, 14 | Kriging, RBF-SVR, kNN and a neural net through the same nested CV | 2 |",
        "| Validation metrics beyond MAE | Ahmad p. 13 | Median, R², 95th-percentile and maximum absolute error for every model | 2 |",
        "| Multiple-start learning | Ahmad p. 7 | Kriging likelihood fitted from 4 starting points | 2 |",
        "| Kriging's own variance; parameter uncertainty | Ahmad p. 9; Matsunami lecture slide 20 | Calibrated kriging 95 % intervals, checked overall and in the extremes | 3 |",
        "| Data must be system-informative; adaptive sampling (explore vs exploit); DoE | Ahmad pp. 5, 15; group slides (DoE) | Space-filling (Kennard–Stone) choice of which videos to download first, simulated | 4 |",
        "| Process optimisation on a Pareto front | Matsunami lecture slide 20 | Accuracy against older-people bias for the age model | 5 |",
        "| Dynamic models (ARX, RNN/LSTM/GRU, state-space, neural ODE) | Ahmad p. 14 | Not run: our inputs are per-session summaries. Candidates for the raw-waveform track (Phase 8) | 6 |",
        "| Bayesian hyperparameter search | Ahmad p. 10 | Not needed yet: our grids are small; noted for the deep models | 6 |",
        "| Population balance modelling (distributed properties, birth/death terms) | Matsunami lecture | No direct transfer — it models particle populations. Its idea of predicting a *distribution* is what section 3 tests | 6 |",
        "",
        "## 1. Gray-box: a mechanistic stiffness index exposes the stiffness label",
        "",
        f"![graybox]({figs['graybox']})",
        "",
        "The digital-volume-pulse stiffness index (Millasseau et al., *Clin Sci* 2002;103:371–377,",
        "doi:10.1042/cs1030371) is height divided by ΔT, the time from the pulse's systolic peak to",
        "its diastolic peak or inflection. Stiffer arteries return the reflected wave sooner, so SI",
        f"rises with age. We read ΔT from every session's ensemble beat ({gb['dt_kind_counts'].get('peak', 0)} with a diastolic",
        f"peak, {gb['dt_kind_counts'].get('inflection', 0)} with an inflection, {gb['dt_kind_counts'].get('none', 0) + gb['dt_kind_counts'].get('no template', 0)} without). The figure's n counts sessions (two per",
        "subject); the statistics below resample whole subjects.",
        "",
        "| Check (Spearman ρ, subject-cluster 95 % CI) | Result |",
        "|---|---|",
        f"| ΔT vs height (must be positive: a longer path) | {rci(v['dt_vs_height'])} |",
        f"| ΔT vs age (must be negative: stiffer with age) | {rci(v['dt_vs_age'])} |",
        f"| PPG stiffness index vs age | {rci(v['si_vs_age'])} |",
        f"| **Stiffness label vs age** | **{rci(lab['label_vs_age'])}** |",
        f"| **Stiffness label, aged ≥ {OLDER_THAN} minus younger** | **{f(lab['label_by_age_band']['older_minus_younger'])} [{f(lab['label_by_age_band']['ci'][0])}, {f(lab['label_by_age_band']['ci'][1])}]** |",
        f"| Stiffness label vs heart rate | {rci(lab['label_vs_heart_rate'])} |",
        f"| Stiffness label vs PPG stiffness index | {rci(lab['label_vs_si'])} |",
        f"| Rest → exercise: label change vs heart-rate change | {rci(wp['dlabel_vs_dhr'])} |",
        f"| Rest → exercise: label change vs SI change | {rci(wp['dlabel_vs_dsi'])} |",
        "",
        "- **ΔT and SI behave like physiology** — ΔT lengthens with height and shortens with age.",
        "- **The dataset's 'arterial stiffness' label does not.** It is *lower* in older subjects,",
        "  rises with heart rate, and is unrelated to the PPG stiffness index. Arterial stiffness",
        "  increases with age, so whatever the label measures, it is not behaving as arterial",
        "  stiffness in this cohort. The dataset paper says the label came from either an Eldar",
        "  photoplethysmograph or a BPLab Angio sphygmography system, with no unit; mixing two",
        "  devices, or a heart-rate-dependent index, would explain this. **Not claimed: which one.**",
        "- **Consequence:** Phase 3's 'stiffness' models mostly learn heart rate (as the Phase 3b",
        "  explanations already found). Adding ΔT and SI to them changes nothing (all four MAE",
        "  changes have CIs spanning zero). The stiffness target should be confirmed with the",
        "  dataset authors before more modelling effort goes into it.",
        "",
        "## 2. Model form: no technique finds signal the others missed",
        "",
        f"![forms]({figs['forms']})",
        "",
        "- **HbA1c, cholesterol, SpO₂, stress:** none of the 8 model forms beats the constant guess.",
        "  Kriging is *worse* than the constant on all four: with almost no signal, its flexible fit",
        "  follows noise. The null result is the data, not the model form.",
        f"- **Temperature:** SVR edges the *mean* guess ({svr_temp['mae']:.3f}), but the *median* guess is better still",
        f"  ({median_temp:.3f}): SVR's ε-insensitive loss behaves like a median fit. No real signal.",
        f"- **Age (positive control):** {len(strict_age)} forms pass the strict rule ({', '.join(strict_age)}).",
        f"  kNN has the lowest MAE ({knn['mae']:.2f} years, {f(knn_vs_lgbm[0], 3)} vs LightGBM [{f(knn_vs_lgbm[1][0], 3)}, {f(knn_vs_lgbm[1][1], 3)}]) —",
        f"  but it is *worse* for people aged ≥ {OLDER_THAN} ({mae_old['kNN']:.1f} vs {mae_old['LightGBM']:.1f} years). A lower MAE is",
        "  not a better model for the people who matter; see section 5.",
        f"- **Worst cases (Ahmad p. 13):** every age model's maximum error is {max_lo:.0f}–{max_hi:.0f} years. For the",
        f"  LightGBM model it is a {worst['y']:.0f}-year-old predicted as {worst['LightGBM']:.0f}. MAE alone hides this; report the maximum too.",
        "",
        *md_table(
            pd.DataFrame(
                form_rows,
                columns=[
                    "Target",
                    "Model",
                    "MAE [95 % CI]",
                    "vs constant [95 % CI]",
                    "Median AE",
                    "R²",
                    "95th pct AE",
                    "Max AE",
                    "Beats both (strict)",
                ],
            )
        ),
        "",
        "## 3. Kriging uncertainty: right on average, wrong where it matters",
        "",
        f"![coverage]({figs['coverage']})",
        "",
        "Each kriging prediction carries a standard deviation, calibrated on the training subjects",
        "only (closed-form leave-one-out residuals; scale factors by fold are in the JSON).",
        "",
        *md_table(
            pd.DataFrame(
                cov_rows,
                columns=[
                    "Target",
                    "Coverage, all",
                    "Lowest 10 %",
                    "Highest 10 %",
                    f"Aged ≥ {OLDER_THAN}",
                    "Mean half-width",
                    "SD vs |error| (ρ, 95 % CI)",
                ],
            )
        ),
        "",
        f"- **SpO₂:** only {cv_['spo2']['coverage_lowest_10']:.0%} of the lowest-saturation sessions fall inside their 95 % interval.",
        f"  **Age:** only {cv_['age'][f'coverage_age_ge_{OLDER_THAN}']:.0%} of people aged ≥ {OLDER_THAN}.",
        "- The interval width does not track the error for the weak targets (ρ ≈ 0): the model",
        "  cannot tell that a person is extreme from inputs that look ordinary. Only for age",
        f"  (ρ {cv_['age']['sd_vs_abs_error_rho']:+.2f}) does the uncertainty carry information.",
        "- This is the general form of the confidence-check failure on subject 4504: an input-based",
        "  confidence score cannot flag a person whose *label* is extreme. A safe app should state",
        "  the label range the model was trained on and refuse to reassure outside it.",
        "",
        "## 4. Design of experiments: which videos to download first",
        "",
        f"![design]({figs['design']})",
        "",
        "Phase 4 needs the frontal video (58.18 GB). Labels are already on disk, so the subjects can",
        "be chosen by design. Ridge models trained on n chosen subjects, scored on the untouched",
        "fold (mean ± SD over 20 draws; space-filling is deterministic). Tail = lowest + highest",
        "10 % of each label.",
        "",
        *md_table(
            pd.DataFrame(
                design_rows,
                columns=[
                    "Target",
                    "n",
                    *[f"MAE {s}" for s in STRATEGIES],
                    *[f"tail {s}" for s in STRATEGIES],
                ],
            )
        ),
        "",
        *design_text(ks_age, rnd_age, summ, pil),
        "",
        "## 5. A Pareto front for the age model",
        "",
        f"![tradeoff]({figs['tradeoff']})",
        "",
        f"Applying a fraction λ of the Phase 3d slope correction to the {tr['model']} age model:",
        "",
        *md_table(
            pd.DataFrame(
                trade_rows,
                columns=[
                    "λ",
                    "Overall MAE (years)",
                    f"Mean error, oldest 10 % (≥ {tr['top10_cut']:.0f} y)",
                ],
            )
        ),
        "",
        f"- Halving the underestimation of the oldest 10 % (λ = 0.5: {f(zero['top10_bias'], 1)} → {f(half['top10_bias'], 1)} years)",
        f"  costs {half['mae'] - zero['mae']:.2f} years of overall MAE ({zero['mae']:.2f} → {half['mae']:.2f}).",
        "- The choice of λ is a policy decision for the team — accuracy for the young majority",
        "  against fairness to older users — not something the data can settle.",
        "",
        "## 6. Not applied, and why",
        "",
        "- **Dynamic models** (ARX, LSTM/GRU, state-space, neural ODE): our models see one summary",
        "  per session. They fit the raw-waveform track planned for Phase 8, where the pulse itself",
        "  is the time series.",
        "- **Bayesian hyperparameter search:** our grids have 16–20 points; the deep models will",
        "  need it.",
        "- **Population balance modelling:** no direct transfer. Its useful idea — model the whole",
        "  distribution, not one value — is what the kriging intervals in section 3 provide.",
        "- **Group-meeting practices** (GitHub collaborators, Zotero, data rules): already in place",
        "  here, except a shared reference library (Zotero) for the team's citations.",
        "",
        "## 7. Provenance",
        "",
        *prov_lines("Gray box (`artifacts/cv/graybox.json`)", gb["provenance"]),
        *prov_lines(
            "Surrogate forms (`artifacts/cv/surrogate.json`)", sg["provenance"]
        ),
        *prov_lines("Design (`artifacts/cv/design.json`)", dz["provenance"]),
        "",
        "## 8. Reproduce",
        "",
        "```",
        "make cv         # oof.parquet, folds.csv",
        "make graybox    # ~1 min",
        "make surrogate  # ~15 min (kriging)",
        "make design     # ~10 min",
        "make report03e",
        "```",
        "",
    ]


def design_text(ks_age: dict, rnd_age: dict, summ: list[dict], pil: dict) -> list[str]:
    def row(st, n, t):
        return _one(summ, strategy=st, n=n, target=t)

    # Guard the written claims.
    for t in ("age", "bmi"):
        assert all(
            row("space-filling", n, t)["mae_mean"] > row("random", n, t)["mae_mean"]
            for n in SIZES
        ), t
    for t in ("hba1c", "arterial_stiffness"):
        assert all(
            row("random", n, t)["space_filling_beats_share_tail"] == 1.0 for n in SIZES
        ), t
        assert all(
            row("space-filling", n, t)["mae_mean"] > row("random", n, t)["mae_mean"]
            for n in SIZES
        ), t
    assert pil["older"] > pil["random_48_older_p95"] and pil["hba1c_ge_6_5"] >= 10
    hb = {
        n: (row("space-filling", n, "hba1c"), row("random", n, "hba1c")) for n in SIZES
    }
    st = {
        n: (
            row("space-filling", n, "arterial_stiffness"),
            row("random", n, "arterial_stiffness"),
        )
        for n in SIZES
    }
    return [
        "- **A space-filling design does not train better models.** For the targets with real",
        "  signal it is worse than random choice at every size — age with 40 subjects: MAE",
        f"  {ks_age[40]['mae_mean']:.2f} against {rnd_age[40]['mae_mean']:.2f} ± {rnd_age[40]['mae_sd']:.2f}; it beats none of the 20 random draws. The",
        "  test folds are dominated by typical young subjects, and a design that over-represents",
        "  the extremes fits them worse. Exploration has a price (Ahmad p. 15).",
        "- **It does reach the extremes.** For HbA1c and arterial stiffness its error in the lowest",
        "  and highest 10 % is lower than every random draw at every size (HbA1c with 80 subjects:",
        f"  {hb[80][0]['tail_mae_mean']:.3f} against {hb[80][1]['tail_mae_mean']:.3f}; stiffness: {st[80][0]['tail_mae_mean']:.2f} against {st[80][1]['tail_mae_mean']:.2f}), while its overall MAE is",
        "  slightly worse. The hybrid sits in between.",
        "- **Proposed first download for Phase 4: the first 48 subjects of the Kennard–Stone order**",
        f"  (`{PILOT_LIST.name}`), {pil['gb']:.2f} GB of the {FRONTAL_GB} GB frontal video. It holds {pil['older']} of the",
        f"  {pil['dev_older']} dev subjects aged ≥ {OLDER_THAN} (a random 48 holds {pil['random_48_older_mean']:.1f} on average, at most {pil['random_48_older_p95']:.0f} in 95 % of",
        f"  draws) and {pil['hba1c_ge_6_5']} of the {pil['dev_hba1c_ge_6_5']} with HbA1c ≥ 6.5 %. Phase 4's first question — does camera rPPG",
        "  reproduce the contact pulse, including for the people our models fail on — needs that",
        "  range. **For training models, download at random (or everything); not by this order.**",
        "- The download still needs the team's approval (gate B); this only makes the first step",
        "  smaller and more informative. Dev subjects only — the test set stays sealed.",
    ]


if __name__ == "__main__":
    main()
