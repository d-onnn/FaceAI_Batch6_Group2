"""Phase 3d report -> reports/03d_bias.md + reports/figures/03d_bias.png

    python -m faceai.report03d        # or: make report03d

Bias in the out-of-fold errors, and what de-biasing can and cannot do (see `faceai.bias`).
Reads only saved artifacts; dev folds only.
"""

from __future__ import annotations

import datetime as dt
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceai.bias import MIN_SLOPE, TAIL
from faceai.cv import OUT_DIR, TARGETS
from faceai.data.hub import ROOT
from faceai.report03 import COLOR, INK, INK2, LABEL, MUTED, SHORT, md_table, save, style
from faceai.report03b import prov_lines, suptitle
from faceai.splits import OLDER_THAN

REPORT = ROOT / "reports" / "03d_bias.md"
FIGURE_TARGETS = ["hba1c", "cholesterol", "arterial_stiffness", "age"]


def _rows(items: list[dict], **match) -> list[dict]:
    return [r for r in items if all(r[k] == v for k, v in match.items())]


def _one(items: list[dict], **match) -> dict:
    (r,) = _rows(items, **match)
    return r


def fig_bias(oof: pd.DataFrame, best: dict) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    for ax, t in zip(axes.flat, FIGURE_TARGETS, strict=True):
        d = oof[oof["target"] == t]
        m = best[t]
        y = d["y"].to_numpy()
        err = d[m].to_numpy() - y
        ax.scatter(y, err, s=6, color=COLOR[m], alpha=0.35, linewidths=0, label=m)
        xs = np.linspace(y.min(), y.max(), 50)
        k, c = np.polyfit(y, err, 1)
        ax.plot(
            xs, c + k * xs, color=INK, linewidth=1.6, label=f"trend (slope {k:+.2f})"
        )
        ax.plot(xs, y.mean() - xs, color=MUTED, linewidth=1.2, linestyle=(0, (4, 2)),
                label="constant guess (slope −1)")  # fmt: skip
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(
            LABEL[t].replace(" — positive control", ""), fontsize=9, loc="left"
        )
        ax.set_xlabel("true value", fontsize=8)
        ax.set_ylabel("error (prediction − true)", fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(h_pad=1.4)
    suptitle(
        fig,
        "Errors fall as the true value rises: predictions are pulled toward the average",
        y=1.04,
    )
    return save(fig, "03d_bias.png")


def main() -> None:
    style()
    b = json.loads((OUT_DIR / "bias.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    best = b["best_ppg_model"]
    ag, pr, sg, db = b["agreement"], b["proportional"], b["subgroups"], b["debias"]

    # ---- guard the written claims
    for t in TARGETS:
        a = _one(ag, target=t, model=best[t])
        assert a["ci"][0] < 0 < a["ci"][1] and abs(a["bias"]) < 0.3, (
            t
        )  # no systematic bias
        assert abs(_one(pr, target=t, model="Mean")["slope"] + 1) < 0.01, t
        assert (
            abs(_one(db, target=t, model=best[t], method="offset")["mae_change"]) < 0.02
        ), t
        s = _one(db, target=t, model=best[t], method="slope")
        assert s["folds_applied"] == 0 or s["mae_after"] > s["mae_before"], (
            t
        )  # never helps MAE
        leak = _one(db, target=t, model=best[t], method="uses test label")
        assert leak["mae_after"] < leak["mae_before"]
    weak = [t for t in TARGETS if t != "age"]
    assert all(_one(pr, target=t, model=best[t])["slope"] < -0.8 for t in weak)
    age_p = _one(pr, target="age", model=best["age"])
    assert -0.5 < age_p["slope"] < -0.2
    for t in ("respiratory_rate", "arterial_stiffness", "bmi", "age"):
        s = _one(db, target=t, model=best[t], method="slope")
        assert (
            s["folds_applied"] == 5 and s["mae_change_ci"][0] > 0
        )  # worse, and surely so
    sp_old = _one(
        sg,
        target="spo2",
        model=best["spo2"],
        factor="age band",
        level=f">= {OLDER_THAN}",
    )
    assert sp_old["ci"][0] > 0
    rr = {
        lv: _one(
            sg,
            target="respiratory_rate",
            model=best["respiratory_rate"],
            factor="state",
            level=lv,
        )
        for lv in ("rest", "post-exercise")
    }
    assert rr["rest"]["ci"][0] > 0 and rr["post-exercise"]["ci"][1] < 0

    figs = {"bias": fig_bias(oof, best)}
    f = lambda v, d=3: f"{v:+.{d}f}"

    agree_rows = []
    for t in TARGETS:
        a = _one(ag, target=t, model=best[t])
        agree_rows.append([
            LABEL[t].replace(" — positive control", ""), best[t], f"{f(a['bias'])} [{f(a['ci'][0])}, {f(a['ci'][1])}]",
            f"{f(a['loa'][0], 2)} to {f(a['loa'][1], 2)}", f"{a['mae']:.3f}",
        ])  # fmt: skip
    prop_rows = []
    for t in TARGETS:
        for m in ("Mean", "Demographics", best[t]):
            p = _one(pr, target=t, model=m)
            prop_rows.append([
                SHORT[t], m, f"{p['slope']:+.2f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]",
                f"≤ {p['low_cut']:.4g}: {f(p['low_bias'], 2)}", f"≥ {p['high_cut']:.4g}: {f(p['high_bias'], 2)}",
            ])  # fmt: skip
    sub_rows = []
    for r in sg:
        if r["model"] == best[r["target"]] and (r["ci"][0] > 0 or r["ci"][1] < 0):
            sub_rows.append([
                SHORT[r["target"]], r["model"], r["factor"], r["level"], r["n_subjects"],
                f"{f(r['bias'])} [{f(r['ci'][0])}, {f(r['ci'][1])}]",
            ])  # fmt: skip
    deb_rows = []
    for t in TARGETS:
        m = best[t]
        o, s, lk = (
            _one(db, target=t, model=m, method=k)
            for k in ("offset", "slope", "uses test label")
        )
        slope_txt = (
            f"{s['mae_after']:.3f} (slope {s['slope_after']:+.2f}; top 10 % {f(s['high_bias_after'], 2)})"
            if s["folds_applied"] == 5
            else f"{s['mae_after']:.3f} (applied in {s['folds_applied']}/5 folds)"
            if s["folds_applied"]
            else "not applied: no signal to stretch"
        )
        deb_rows.append([
            SHORT[t], m, (f"{o['mae_before']:.3f} (slope {o['slope_before']:+.2f}; top 10 % "
            f"{f(_one(pr, target=t, model=m)['high_bias'], 2)})"), f"{o['mae_after']:.3f}", slope_txt,
            f"{lk['mae_after']:.3f}",
        ])  # fmt: skip

    hb = _one(pr, target="hba1c", model=best["hba1c"])
    ch = _one(pr, target="cholesterol", model=best["cholesterol"])
    age_s = _one(db, target="age", model=best["age"], method="slope")
    hb_leak = _one(db, target="hba1c", model=best["hba1c"], method="uses test label")
    age_old = _one(
        sg, target="age", model=best["age"], factor="age band", level=f">= {OLDER_THAN}"
    )
    hb_sex = {
        lv: _one(sg, target="hba1c", model=best["hba1c"], factor="sex", level=lv)
        for lv in ("F", "M")
    }
    follow_lo, follow_hi = sorted(100 * (1 + p["slope"]) for p in (hb, ch))
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Phase 3d — Bias in our errors, and what de-biasing can and cannot do",
        "",
        f"_Generated {now} by `python -m faceai.report03d` from saved artifacts only._",
        "",
        "Follow-up to the staff feedback on industry standards (`reports/03c_standards.md`). MAE",
        "hides the direction of errors; standards that judge agreement (Bland–Altman, ISO",
        "81060-2, the ARTERY Society guidelines) report bias and its spread. Same out-of-fold",
        "predictions as Phase 3: **contact PPG, 5 dev folds, 479 subjects, 956 sessions; not camera",
        "rPPG, not the sealed test set.** Error = prediction − true value (positive = too high).",
        "",
        "## 1. Headline",
        "",
        f"![bias]({figs['bias']})",
        "",
        "- **No systematic bias.** For every target the best model's mean error is within ±0.3 of",
        "  zero and its 95 % CI contains zero. Least-squares models are unbiased on average by",
        "  construction, so MAE is not hiding an offset.",
        "- **Strong proportional bias (regression to the mean).** Predictions stay near the cohort",
        "  average: low true values are overestimated and high ones underestimated. The slope of",
        "  error on the true value is −1 for a constant guess and 0 for a perfect model; our best",
        f"  models score {hb['slope']:+.2f} for HbA1c and {ch['slope']:+.2f} for cholesterol — they follow",
        f"  only {follow_lo:.0f}–{follow_hi:.0f} % of real differences. In the top 10 % of HbA1c (≥ {hb['high_cut']:.2f} %)",
        f"  predictions are {abs(hb['high_bias']):.2f} points too low on average, which is why no prediction ever",
        "  reaches the 6.5 % diabetes cut-off (Phase 3b). Age is the only target whose predictions",
        f"  follow most of the real differences (slope {age_p['slope']:+.2f}), yet it still underestimates the oldest",
        f"  10 % (≥ {age_p['high_cut']:.0f} years in this young cohort) by {abs(age_p['high_bias']):.1f} years.",
        "- **Subgroup bias follows from it.** Groups whose average differs from the cohort's",
        f"  inherit the pull to the mean: the HbA1c error averages {f(hb_sex['F']['bias'], 2)} for women and {f(hb_sex['M']['bias'], 2)} for men;",
        f"  SpO₂ is overestimated by {sp_old['bias']:.2f} points in people aged ≥ {OLDER_THAN} — the risky direction for",
        (
            "  oximetry, because it hides low oxygen; respiratory rate is "
            f"{f(rr['rest']['bias'], 1)}/min at rest and {f(rr['post-exercise']['bias'], 1)}/min after exercise."
        ),
        "- **De-biasing cannot add information.** Subtracting an average offset changes nothing.",
        "  Stretching predictions to undo the slope (fitted on training subjects only) removes the",
        "  proportional bias but multiplies the noise: MAE rises for every target where it can be",
        (
            f"  applied — age {age_s['mae_before']:.2f} → {age_s['mae_after']:.2f} years, while the oldest 10 % go from "
            f"{f(age_p['high_bias'], 1)} to {f(age_s['high_bias_after'], 1)} years."
        ),
        "  It is a fairness trade-off for a model with signal, and pointless for one without.",
        "- **Beware 'bias-corrected' results that use the true value.** The correction some",
        "  brain-age papers report puts the test label back into the prediction. Here it would cut",
        f"  HbA1c MAE from {hb_leak['mae_before']:.3f} to {hb_leak['mae_after']:.3f} — a number no deployed model could reach.",
        "  When comparing with a published benchmark, check which correction it used.",
        "",
        "## 2. Systematic bias and limits of agreement (best PPG model per target)",
        "",
        *md_table(
            pd.DataFrame(
                agree_rows,
                columns=[
                    "Target",
                    "Model",
                    "Mean error [95 % CI]",
                    "95 % limits of agreement",
                    "MAE",
                ],
            )
        ),
        "",
        "Limits of agreement = mean ± 1.96 SD of the errors: 95 % of single predictions fall inside",
        "them. They are what a Bland–Altman comparison with a reference method reports.",
        "",
        "## 3. Proportional bias (regression to the mean)",
        "",
        *md_table(
            pd.DataFrame(
                prop_rows,
                columns=[
                    "Target",
                    "Model",
                    "Slope of error on true value [95 % CI]",
                    f"Mean error, lowest {TAIL:.0%}",
                    f"Mean error, highest {TAIL:.0%}",
                ],
            )
        ),
        "",
        "A constant guess has slope −1 by construction (its error is the average minus the truth).",
        "",
        "## 4. Subgroup bias (best PPG model; groups whose 95 % CI excludes zero)",
        "",
        *md_table(
            pd.DataFrame(
                sub_rows,
                columns=[
                    "Target",
                    "Model",
                    "Factor",
                    "Group",
                    "Subjects",
                    "Mean error [95 % CI]",
                ],
            )
        ),
        "",
        f"- The age bands are defined by age itself, so the age row ({f(age_old['bias'], 1)} years for ≥ {OLDER_THAN}) is",
        "  the proportional bias of section 3 seen again, not a separate effect.",
        f"- Only 44 dev subjects are aged ≥ {OLDER_THAN}; their intervals are wide.",
        "",
        "## 5. De-biasing, done without the test labels",
        "",
        *md_table(
            pd.DataFrame(
                deb_rows,
                columns=[
                    "Target",
                    "Model",
                    "MAE as fitted",
                    "Offset removed",
                    "Slope undone",
                    "Uses test label (invalid)",
                ],
            )
        ),
        "",
        "- **How it was fitted.** In each outer fold the model was refitted, with that fold's tuned",
        "  hyperparameters, on 5 inner folds of the training subjects; the inner out-of-fold",
        "  predictions set the correction, applied to the untouched outer fold.",
        f"- **Slope undone** inverts prediction = a + b·truth. Where b ≤ {MIN_SLOPE} (the model follows less",
        "  than 5 % of real differences) the inverse would multiply the noise more than twentyfold,",
        "  so the fold is left as fitted.",
        "- **Uses test label** subtracts the expected error at the person's *true* value. It needs the",
        "  answer, so it cannot be used — it is shown only to calibrate claims made elsewhere.",
        "",
        "## 6. What we take from this",
        "",
        "1. Report bias and limits of agreement next to every MAE; it is the form device standards use.",
        "2. Our weak targets (HbA1c, cholesterol, SpO₂, stress, temperature) do not have a bias problem",
        "   that a correction can fix — they have no signal. Undoing their slope is either impossible",
        "   (nothing to stretch) or makes them worse (HbA1c, cholesterol).",
        "3. For age, a slope correction would trade overall accuracy for less underestimation of",
        "   older people. If the team cares about older users, report both numbers and decide openly.",
        "4. Any benchmark quoted as 'bias-corrected' must say whether the correction used the true",
        "   value. If it did, it is not comparable with ours.",
        "",
        "## 7. Provenance",
        "",
        *prov_lines("Bias (`artifacts/cv/bias.json`)", b["provenance"]),
        "",
        "## 8. Reproduce",
        "",
        "```",
        "make cv     # writes artifacts/cv/oof.parquet and folds.csv",
        "make bias",
        "make report03d",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT} and {len(figs)} figure")


if __name__ == "__main__":
    main()
