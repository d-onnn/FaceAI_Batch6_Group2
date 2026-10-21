"""Phase 3c report -> reports/03c_standards.md + reports/figures/03c_standards.png

    python -m faceai.report03c        # or: make report03c

Our out-of-fold errors set against recognised accuracy standards, each applied in its own
form (see `faceai.standards`), plus the dataset paper's Table 5 for targets that have no
device standard. Reads only saved artifacts; dev folds only.
"""

from __future__ import annotations

import datetime as dt
import json

import matplotlib.pyplot as plt

from faceai.cv import OUT_DIR, PAPER_TABLE5
from faceai.data.hub import ROOT
from faceai.report03 import COLOR, INK, INK2, LABEL, MUTED, SHORT, save, style
from faceai.report03b import prov_lines, suptitle

REPORT = ROOT / "reports" / "03c_standards.md"
PPG_MODELS = ["Ridge (L2)", "Lasso (L1)", "LightGBM"]
FIGURE_TARGETS = ["hba1c", "cholesterol", "arterial_stiffness", "spo2"]
PANEL = {  # title, MAE unit
    "hba1c": ("HbA1c — NGSP ±5 %", "HbA1c percentage points"),
    "cholesterol": ("Cholesterol — NCEP ±8.9 %", "mmol/L"),
    "arterial_stiffness": ("Arterial stiffness — ARTERY Society", "index units"),
    "spo2": ("SpO₂ — ISO 80601-2-61 ARMS", "SpO₂ percentage points"),
}
STAT_NAME = {
    "share_within": "share within tolerance",
    "arms": "ARMS",
    "sd_difference": "SD of differences",
}


def best(e: dict) -> str:
    """The PPG model with the lowest MAE."""
    return min(PPG_MODELS, key=lambda m: e["models"][m]["mae"])


def fmt_stat(e: dict, m: str) -> str:
    v = e["models"][m]
    if e["statistic"] == "share_within":
        lo, hi = (100 * x for x in v["ci"])
        return f"{100 * v['share_within']:.1f}% [{lo:.1f}–{hi:.1f}]"
    if e["statistic"] == "arms":
        return f"{v['arms']:.2f} [{v['ci'][0]:.2f}–{v['ci'][1]:.2f}]"
    return (
        f"mean {v['mean_difference']:+.2f}, SD {v['sd_difference']:.2f}"
        f" [{v['ci'][0]:.2f}–{v['ci'][1]:.2f}]"
    )


def fig_standards(crit: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.4))
    for ax, t in zip(axes.flat, FIGURE_TARGETS, strict=True):
        e = crit[t]
        m = best(e)
        bars = [("Constant guess", e["models"]["Mean"]["mae"], MUTED)]
        bars.append((f"Best PPG model:\n{m}", e["models"][m]["mae"], COLOR[m]))
        allowed = e["implied"]["mae"]
        xmax = 1.3 * max(allowed, *(b[1] for b in bars))
        for i, (_, v, c) in enumerate(bars):
            ax.barh(1 - i, v, height=0.56, color=c)
            ax.text(v, 1 - i, f"  {v:.3g}", va="center", fontsize=8.5, color=INK)
        ax.axvline(allowed, color=INK, linewidth=1.4, linestyle=(0, (4, 2)))
        right = allowed < 0.6 * xmax  # keep the label clear of the line and the edge
        ax.text(allowed + (0.02 if right else -0.02) * xmax, 1.62,
                f"standard allows ≈ {allowed:.2g}", ha="left" if right else "right",
                fontsize=8, color=INK)  # fmt: skip
        ax.set_yticks([1, 0], [b[0] for b in bars], fontsize=8)
        ax.set_ylim(-0.5, 1.9)
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)
        ax.set_title(PANEL[t][0], fontsize=9, loc="left")
        ax.set_xlabel(f"MAE, {PANEL[t][1]}", fontsize=8)
    fig.tight_layout(h_pad=1.6)
    suptitle(fig, "Our MAE against the MAE each standard's pass rule allows", y=1.05)
    fig.text(
        0.01, -0.03,
        "Allowed MAE: the MAE of unbiased, normally distributed errors that just meet the"
        " rule. Contact PPG, dev folds.",
        fontsize=7.5, color=INK2,
    )  # fmt: skip
    return save(fig, "03c_standards.png")


def standards_table(crit: dict) -> list[str]:
    out = [
        (
            "| Target | Standard: pass rule | Statistic | Constant guess [95% CI]"
            " | Best PPG model [95% CI] | Verdict (constant / model) | MAE: allowed vs ours"
            " | R²: required vs achieved |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in crit.values():
        m = best(e)
        mean, mod = e["models"]["Mean"], e["models"][m]
        imp = e["implied"]
        allowed = f"{imp['mae']:.3g} vs {mod['mae']:.3g}" if imp else "—"
        r2 = f"{imp['required_r2']:.2f} vs {mod['r2']:.2f}" if imp else "—"
        out.append(
            f"| {LABEL[e['target']]} | {e['standard']}: {e['rule']}"
            f" | {STAT_NAME[e['statistic']]} | {fmt_stat(e, 'Mean')}"
            f" | {m}: {fmt_stat(e, m)} | {mean['verdict']} / {mod['verdict']}"
            f" | {allowed} | {r2} |"
        )
    return out


def reference_table(crit: dict) -> list[str]:
    out = [
        (
            "| Target | The standard assumes a reference of | Our label comes from"
            " | Label range (dev sessions) |"
        ),
        "|---|---|---|---|",
    ]
    for e in crit.values():
        lo, hi = e["label_range"]
        out.append(
            f"| {SHORT[e['target']]} | {e['reference_assumed']} | {e['our_reference']}"
            f" | {lo:g}–{hi:g} (SD {e['label_sd']:.2f}) |"
        )
    return out


def benchmark_table(res: dict) -> list[str]:
    out = [
        (
            "| Target | Paper: constant | Paper: video model | Ours: constant"
            " | Ours: best PPG model |"
        ),
        "|---|---|---|---|---|",
    ]
    for t, (p_const, p_model) in PAPER_TABLE5.items():
        r = res[t]["models"]
        m = min(PPG_MODELS, key=lambda k: r[k]["mae"])
        out.append(
            f"| {LABEL[t]} | {p_const:.2f} | {p_model:.2f} | {r['Mean']['mae']:.3f}"
            f" | {m}: {r[m]['mae']:.3f} |"
        )
    return out


def main() -> None:
    style()
    std = json.loads((OUT_DIR / "standards.json").read_text(encoding="utf-8"))
    res = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    crit = {e["target"]: e for e in std["criteria"]}

    # Guard the written claims.
    for e in crit.values():
        assert e["models"]["Mean"]["verdict"] == e["models"][best(e)]["verdict"]
    for t in ("hba1c", "cholesterol", "arterial_stiffness"):
        assert crit[t]["implied"]["mae"] < crit[t]["models"][best(crit[t])]["mae"]
        assert crit[t]["models"]["Mean"]["verdict"] in ("fails", "poor")
    assert crit["spo2"]["models"]["Mean"]["verdict"] == "meets"
    assert crit["spo2"]["implied"]["required_r2"] < 0

    figs = {"standards": fig_standards(crit)}
    hb, ch, st, sp = (crit[t] for t in FIGURE_TARGETS)
    ratio = {t: crit[t]["models"][best(crit[t])]["mae"] / crit[t]["implied"]["mae"]
             for t in FIGURE_TARGETS}  # fmt: skip
    r2_max = max(e["models"][m]["r2"] for e in (hb, ch) for m in PPG_MODELS)
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Phase 3c — Our errors against industry and clinical accuracy standards",
        "",
        f"_Generated {now} by `python -m faceai.report03c` from saved artifacts only._",
        "",
        "Requested by the FaceAI staff: compare our MAE with industry standards. Accuracy",
        "standards are rarely written as an MAE, so each one is applied in its own form (a",
        "share within a tolerance, an ARMS, or a mean and SD of differences) on the same",
        "out-of-fold predictions as `reports/03_cv_robustness.md`: **contact PPG, 5 dev folds,",
        "479 subjects, 956 sessions. Not camera rPPG, and not the sealed test set.** The rule is",
        "then translated into the MAE it allows, so it can sit next to ours.",
        "",
        "## 1. Headline",
        "",
        f"![standards]({figs['standards']})",
        "",
        (
            f"- **HbA1c**: the NGSP rule allows an MAE of ≈ {hb['implied']['mae']:.2f}"
            f" percentage points; our best is {hb['models'][best(hb)]['mae']:.3f}"
            f" ({ratio['hba1c']:.1f}× too large), and the constant guess"
        ),
        (
            f"  {hb['models']['Mean']['mae']:.3f}. Only"
            f" {100 * hb['models'][best(hb)]['share_within']:.0f}% of predictions land"
            " within ±5 % of the label, against 90 % required."
        ),
        (
            f"- **Cholesterol**: allowed ≈ {ch['implied']['mae']:.2f} mmol/L; ours"
            f" {ch['models'][best(ch)]['mae']:.3f} ({ratio['cholesterol']:.1f}×);"
            f" {100 * ch['models'][best(ch)]['share_within']:.0f}% within ±8.9 %"
            " against ≈95 % required."
        ),
        (
            "- **To meet either lab standard on this cohort, a model would have to explain"
            f" ≈{100 * min(hb['implied']['required_r2'], ch['implied']['required_r2']):.0f}%"
            " of the label variance. Ours explain at most"
        ),
        (
            f"  {100 * r2_max:.0f}%.** The gap is not a tuning problem: the PPG features"
            " carry almost no information about either target (Phase 3)."
        ),
        (
            f"- **Arterial stiffness**: SD of differences {st['models'][best(st)]['sd_difference']:.2f}"
            f' against the ARTERY Society\'s 1.5 for "acceptable" (allowed MAE ≈'
            f" {st['implied']['mae']:.2f}; ours {st['models'][best(st)]['mae']:.3f},"
        ),
        (
            f'  {ratio["arterial_stiffness"]:.1f}×). Graded "poor" — but so is the constant,'
            " and the label is not the PWV the guideline is written for."
        ),
        (
            f"- **SpO₂**: ARMS {sp['models'][best(sp)]['arms']:.2f}, under ISO's 4.0 % and"
            " FDA's typical 3.0 %. The constant guess"
            f" ({sp['models']['Mean']['arms']:.2f}) is under it too: a constant's ARMS is"
            " roughly the label SD,"
        ),
        f"  which is only {sp['label_sd']:.2f} on this cohort. This is not a pass.",
        "- **On every standard, the best model and the constant guess get the same verdict.**",
        "  So none of these comparisons is evidence that our models work, and none could",
        "  certify anything: section 3 shows that the reference each standard assumes is not",
        "  what our labels are.",
        "",
        "## 2. Each standard applied in its own form",
        "",
        *standards_table(crit),
        "",
        "- *Verdict* applies the standard's pass rule to the number as if the test conditions",
        "  held (they do not; section 3). RR and temperature have a tolerance but no required",
        "  share, so they get no verdict.",
        "- *MAE allowed*: the MAE of unbiased, normally distributed errors that just meet the",
        '  rule (MAE = √(2/π) × SD; a rule "share p within ±w" gives SD = w / z₍₁₊ₚ₎/₂).',
        "- *R² required*: the share of this cohort's label variance a model must explain to",
        "  get its error SD that low. Negative means a constant already meets the rule.",
        "- CIs resample whole subjects (2000 draws), as in Phase 3.",
        "",
        "## 3. Why none of this certifies anything — the reference conditions",
        "",
        *reference_table(crit),
        "",
        "Every standard here is written for a measurement device checked against a better",
        "reference, across a range that includes abnormal values. Our labels come from",
        "consumer or point-of-care devices (named in the dataset paper, §2.1), and the cohort is",
        "mostly young and healthy. A model trained on these labels can at best match the label",
        "device, whose own error against the reference is unknown here. **Claims of meeting a",
        "standard would need a new study against the reference method.** Tier C targets",
        "(HbA1c, cholesterol, temperature) remain not clinically usable.",
        "",
        "## 4. Targets without a device accuracy standard — the field benchmark",
        "",
        "Age, BMI and stress (PSM-25) have no device accuracy standard. The benchmark is the",
        "dataset paper's Table 5 (Egorov et al., ACM MM 2025). The paper's models use facial",
        "video in 20-s windows on a different split, so this comparison is indicative only.",
        "",
        *benchmark_table(res["targets"]),
        "",
        "## 5. Heart rate — the standard Phase 4 will be judged on",
        "",
        "Heart rate is the field's headline rPPG metric. Not yet measured by us on video",
        "(Phase 4 is blocked on the frontal-video download). The dataset paper reports an",
        "rPPG heart-rate MAE of 3.80 bpm (POS) and 2.82 bpm (RhythmFormer) on the frontal",
        "camera, and 16.40 and 7.33 bpm on the side camera (Table 4). Two standards are",
        "commonly applied to heart rate, but **we have not checked either against its own",
        "(paywalled) text**:",
        "",
        "- ANSI/AAMI EC13 (cardiac monitors): readout error within ±10 % or ±5 bpm, whichever",
        "  is greater — quoted from the 1987 edition in secondary sources.",
        "- ANSI/CTA-2065 (consumer heart-rate monitors): published validation studies apply it",
        "  as a mean absolute percentage error below 10 %.",
        "",
        "## 6. Sources",
        "",
        "- NGSP, Summary of Certification Criteria — https://ngsp.org/critsumm.asp (read",
        "  2026-10-21). CAP's HbA1c grading limit is ±6 %: Rohlfing et al., *Clin Chim Acta*",
        "  2014;433:259–263, doi:10.1016/j.cca.2014.03.034.",
        "- CDC Cholesterol Reference Method Laboratory Network, total-cholesterol certification",
        "  protocol (as quoted; the PDF has moved from its old cdc.gov address): NCEP bias",
        "  ≤ 3.0 % and CV ≤ 3.0 %, implying 8.9 % total allowable error (3 + 1.96 × 3).",
        "- FDA, *Pulse Oximeters — Premarket Notification Submissions [510(k)s]*, final",
        "  guidance, 4 Mar 2013, Table 3 (ARMS ≤ 3.0 % typical, transmittance sensors, SpO₂",
        "  70–100 %); it refers to ISO 80601-2-61 clause 201.12.1.101, whose 4.0 % limit is",
        "  taken from study protocols that quote it (the ISO text is paywalled).",
        "- Wilkinson et al., ARTERY Society guidelines Part 1: arterial pulse wave velocity,",
        "  *Artery Research* 2010;4(2):34–40, doi:10.1016/j.artres.2010.03.001, Table 4.",
        "- Wang et al., Infrared thermography for measuring elevated body temperature,",
        "  *Sensors* 2022;22(1):215, doi:10.3390/s22010215 (table of ISO 80601-2-56 limits).",
        "- Egorov et al., Gaze into the Heart, ACM MM 2025, doi:10.1145/3746027.3758255,",
        "  arXiv:2508.17924 — label devices (§2.1), Tables 3–5.",
        "",
        "## 7. Provenance",
        "",
        *prov_lines("Standards (`artifacts/cv/standards.json`)", std["provenance"]),
        *prov_lines(
            "CV results (`artifacts/cv/results.json`)", res["meta"]["provenance"]
        ),
        "",
        "## 8. Reproduce",
        "",
        "```",
        "make cv          # writes artifacts/cv/oof.parquet",
        "make standards",
        "make report03c",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT} and {len(figs)} figure")
    print({t: round(r, 2) for t, r in ratio.items()})


if __name__ == "__main__":
    main()
