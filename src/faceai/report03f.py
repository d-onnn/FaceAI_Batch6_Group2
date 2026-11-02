"""Phase 3f report -> reports/03f_hba1c.md + reports/figures/03f_hba1c.png

    python -m faceai.report03f        # or: make report03f

HbA1c predicted against true, and the standard deviation of the labels, the predictions
and the errors. Reads only saved artifacts; dev folds only. HbA1c is Tier C: nothing here
is clinically usable.
"""

from __future__ import annotations

import datetime as dt
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceai.cv import OUT_DIR
from faceai.data.hub import ROOT
from faceai.report03 import COLOR, INK, INK2, MUTED, md_table, save, style
from faceai.report03b import prov_lines, suptitle
from faceai.robustness import N_BOOT
from faceai.splits import SEED

REPORT = ROOT / "reports" / "03f_hba1c.md"
MODELS = ["Mean", "Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
PPG_MODELS = ["Ridge (L2)", "Lasso (L1)", "LightGBM"]
DIABETES = 6.5  # ADA diagnostic cut-off, % HbA1c


def cluster_sd(x: np.ndarray, subjects: np.ndarray, seed: int = SEED) -> dict:
    """SD (ddof 1) over sessions, with a 95 % CI from resampling whole subjects."""
    x = x - x.mean()  # SD is shift-invariant; centring keeps the sums well conditioned
    codes, uniq = pd.factorize(subjects)
    s1 = np.bincount(codes, weights=x)
    s2 = np.bincount(codes, weights=x * x)
    n = np.bincount(codes).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    bn, b1, b2 = n[draws].sum(axis=1), s1[draws].sum(axis=1), s2[draws].sum(axis=1)
    boot = np.sqrt((b2 - b1**2 / bn) / (bn - 1))
    return {
        "sd": float(x.std(ddof=1)),
        "ci": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
    }


def fig_hba1c(d: pd.DataFrame, best: str, label: dict, spread: dict) -> str:
    fig = plt.figure(figsize=(12.4, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1, 1], wspace=0.45)

    ax = fig.add_subplot(gs[0])
    y, p = d["y"].to_numpy(), d[best].to_numpy()
    lim = np.array([3.0, 13.5])
    ax.scatter(y, p, s=9, color=COLOR[best], alpha=0.4, linewidths=0,
               label=f"{best}, one dot per session")  # fmt: skip
    ax.plot(lim, lim, color=MUTED, linewidth=1.2, linestyle=(0, (4, 2)),
            label="perfect prediction (slope 1)")  # fmt: skip
    k, c = np.polyfit(y, p, 1)
    ax.plot(lim, c + k * lim, color=INK, linewidth=1.6, label=f"trend (slope {k:+.2f})")
    ax.axvline(DIABETES, color=INK2, linewidth=0.8, linestyle=":")
    ax.axhline(DIABETES, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(DIABETES + 0.15, 3.25, "6.5 % diabetes cut-off", fontsize=7.5, color=INK2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("true HbA1c (%)")
    ax.set_ylabel("predicted HbA1c (%, out of fold)")
    ax.set_title("A  Predicted against true", loc="left")
    ax.legend(fontsize=7.5, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=1)  # fmt: skip

    rows = MODELS[::-1]  # barh draws bottom-up; this lists Mean first, top
    for i, (key, title) in enumerate(
        [("pred", "B  Spread of predictions"), ("err", "C  Spread of errors")]
    ):
        ax = fig.add_subplot(gs[i + 1])
        v = np.array([spread[m][key]["sd"] for m in rows])
        lo = np.array([spread[m][key]["ci"][0] for m in rows])
        hi = np.array([spread[m][key]["ci"][1] for m in rows])
        pos = np.arange(len(rows))
        ax.barh(pos, v, height=0.6, color=[COLOR[m] for m in rows])
        ax.errorbar(v, pos, xerr=[v - lo, hi - v], fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2)  # fmt: skip
        for j in pos:
            ax.text(hi[j] + 0.02, j, f"{v[j]:.2f}", va="center", fontsize=8, color=INK2)
        ax.axvline(label["sd"], color=INK, linewidth=1.2, linestyle=(0, (4, 2)))
        ax.text(label["sd"] - 0.02, len(rows) - 0.45, f"true HbA1c SD {label['sd']:.2f}",
                ha="right", va="bottom", fontsize=7.5, color=INK)  # fmt: skip
        ax.set_yticks(pos, rows)
        ax.set_ylim(-0.6, len(rows) + 0.1)
        ax.set_xlim(0, 0.95)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("standard deviation (% HbA1c)")
        ax.set_title(title, loc="left")

    fig.text(0.01, -0.06, "Finger contact PPG, 5-fold subject-level CV, 479 dev subjects / "
             "956 sessions; error = prediction − true. Bars: SD over sessions with a 95 % CI "
             "from resampling subjects. HbA1c is Tier C: not clinically usable.",
             fontsize=7.5, color=INK2)  # fmt: skip
    suptitle(
        fig,
        "HbA1c: predictions stay near the average, so the errors are almost as spread as the truth",
        y=1.0,
    )
    return save(fig, "03f_hba1c.png")


def main() -> None:
    style()
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    d = oof[oof["target"] == "hba1c"]
    res = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    std = json.loads((OUT_DIR / "standards.json").read_text(encoding="utf-8"))
    b = json.loads((OUT_DIR / "bias.json").read_text(encoding="utf-8"))
    best = b["best_ppg_model"]["hba1c"]
    mae = res["targets"]["hba1c"]["models"]
    (crit,) = [c for c in std["criteria"] if c["target"] == "hba1c"]

    y, subj = d["y"].to_numpy(), d["patient_id"].to_numpy()
    label = cluster_sd(y, subj)
    spread = {
        m: {
            "pred": cluster_sd(d[m].to_numpy(), subj),
            "err": cluster_sd(d[m].to_numpy() - y, subj),
        }
        for m in MODELS
    }
    r2 = {m: 1 - np.mean((d[m].to_numpy() - y) ** 2) / np.var(y) for m in MODELS}
    ratio = {m: spread[m]["err"]["sd"] / label["sd"] for m in MODELS}
    share = {m: spread[m]["pred"]["sd"] / label["sd"] for m in MODELS}
    k = float(np.polyfit(y, d[best].to_numpy(), 1)[0])
    top = d[PPG_MODELS].to_numpy().max()
    n_diab = d.loc[d["y"] >= DIABETES, "patient_id"].nunique()
    (agree,) = [
        a for a in b["agreement"] if a["target"] == "hba1c" and a["model"] == best
    ]
    (prop,) = [
        p for p in b["proportional"] if p["target"] == "hba1c" and p["model"] == best
    ]

    # ---- guard the written claims
    assert len(d) == 956 and d["patient_id"].nunique() == 479
    assert best in PPG_MODELS and min(mae, key=lambda m: mae[m]["mae"]) == best
    for m in MODELS:  # same errors as Phases 3c and 3d
        (a,) = [r for r in b["agreement"] if r["target"] == "hba1c" and r["model"] == m]
        assert abs(spread[m]["err"]["sd"] - a["sd"]) < 1e-9, m
        assert abs(r2[m] - crit["models"][m]["r2"]) < 1e-9, m
        # bias ≈ 0, so R² = 1 − ratio²
        assert abs(r2[m] - (1 - ratio[m] ** 2)) < 1e-4, m
    # trend of the prediction = 1 + slope of the error (Phase 3d)
    assert abs(k - (1 + prop["slope"])) < 1e-9
    assert top < DIABETES and n_diab == 14
    assert all(share[m] < 0.3 for m in PPG_MODELS)
    assert all(0.95 < ratio[m] < 1 for m in PPG_MODELS)
    a_mean, a_best = mae["Mean"], mae[best]
    assert a_best["ci"][0] < a_mean["mae"] < a_best["ci"][1]  # not distinguishable
    assert all(abs(a["bias"]) < 0.01 for a in b["agreement"] if a["target"] == "hba1c")
    assert max(MODELS, key=share.get) == max(MODELS, key=r2.get) == "Demographics"
    assert (
        d["Demographics"].max() > DIABETES
        and mae["Demographics"]["mae"] > a_mean["mae"]
    )

    figs = {"hba1c": fig_hba1c(d, best, label, spread)}

    def pct(v: float, d: int = 0, sign: str = "") -> str:
        return f"{100 * v:{sign}.{d}f} %"

    def sd_txt(s: dict) -> str:
        return f"{s['sd']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"

    rows = [["True HbA1c (labels)", sd_txt(label), "—", "—", "—", "—"]]
    for m in MODELS:
        rows.append([
            m, sd_txt(spread[m]["pred"]), sd_txt(spread[m]["err"]), pct(ratio[m], 1),
            f"{mae[m]['mae']:.3f} [{mae[m]['ci'][0]:.3f}, {mae[m]['ci'][1]:.3f}]", pct(r2[m], 1, "+"),
        ])  # fmt: skip
    ppg_share = sorted(share[m] for m in PPG_MODELS)
    ppg_ratio = sorted(ratio[m] for m in PPG_MODELS)
    sb, se = spread[best]["pred"]["sd"], spread[best]["err"]["sd"]
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Phase 3f — HbA1c: predicted against true, and the standard deviations",
        "",
        f"_Generated {now} by `python -m faceai.report03f` from saved artifacts only._",
        "",
        "**HbA1c is Tier C: nothing here is clinically usable.** Same out-of-fold predictions as",
        "Phase 3: **finger contact PPG, 5 dev folds, 479 subjects, 956 sessions, split by subject;",
        "not camera rPPG, not the sealed test set.** No model was refitted for this report.",
        "",
        "## 1. Headline",
        "",
        f"![HbA1c scatter and standard deviations]({figs['hba1c']})",
        "",
        f"- **The truth is spread out.** True HbA1c averages {y.mean():.2f} % with an SD of {label['sd']:.2f} %",
        f"  (range {y.min():.2f}–{y.max():.2f} %); {n_diab} dev subjects are at or above the {DIABETES} % diabetes cut-off.",
        f"- **The predictions are not.** The best PPG model, {best}, has a prediction SD of {sb:.2f} %,",
        f"  {pct(share[best])} of the true spread; across the PPG models it is {pct(ppg_share[0])[:-2]}–{pct(ppg_share[-1])}. Its trend",
        f"  has slope {k:+.2f}: a 1-point rise in true HbA1c moves the prediction by {k:.2f} points.",
        f"- **No PPG prediction reaches {DIABETES} %** (highest {top:.2f} %), so the top-right of panel A is empty",
        "  even though diabetes-range people are in the data.",
        f"- **So the errors are almost as spread as the truth.** Error SD {se:.2f} % against {label['sd']:.2f} % for the",
        f"  labels: the PPG models shrink the spread by only {pct(1 - ppg_ratio[-1])[:-2]}–{pct(1 - ppg_ratio[0])}. Variance explained is",
        f"  1 − ({se:.2f}/{label['sd']:.2f})² ≈ {pct(r2[best], 1)} for {best}, at most {pct(max(r2[m] for m in PPG_MODELS), 1)} for any PPG model.",
        f"  95 % limits of agreement: {agree['loa'][0]:+.2f} to {agree['loa'][1]:+.2f} points around the true value.",
        f"- **Same as a constant guess.** {best} MAE {a_best['mae']:.3f} [{a_best['ci'][0]:.3f}, {a_best['ci'][1]:.3f}] against",
        f"  {a_mean['mae']:.3f} for predicting the training mean; the constant's error SD is {spread['Mean']['err']['sd']:.2f} %.",
        "",
        "## 2. Standard deviations (all models, % HbA1c)",
        "",
        *md_table(
            pd.DataFrame(
                rows,
                columns=[
                    "Model",
                    "SD of predictions [95 % CI]",
                    "SD of errors [95 % CI]",
                    "Error SD ÷ label SD",
                    "MAE [95 % CI]",
                    "Variance explained (R²)",
                ],
            )
        ),
        "",
        "- SDs are over the 956 sessions (ddof 1). CIs resample the 479 subjects with both their",
        f"  sessions together ({N_BOOT} draws, seed {SEED}), as everywhere else in Phase 3.",
        "- Error = prediction − true. Mean error is ≈ 0 for every model (`reports/03d_bias.md`), so",
        "  the error SD is the spread of the misses and R² = 1 − (error SD ÷ label SD)².",
        "- The mean baseline is refitted on each fold's training subjects, so its prediction SD is",
        "  small but not 0.",
        "- Demographics (age, sex, BMI) has the largest prediction spread and R², and its",
        f"  predictions do pass {DIABETES} %, yet its MAE is worse than the mean's: MAE counts every miss",
        "  equally, while R² squares them and so rewards moving towards the rare high values.",
        "",
        "## 3. Provenance",
        "",
        *prov_lines(
            "Out-of-fold predictions (`artifacts/cv/results.json`)",
            res["meta"]["provenance"],
        ),
        *prov_lines("Bias (`artifacts/cv/bias.json`)", b["provenance"]),
        *prov_lines("Standards (`artifacts/cv/standards.json`)", std["provenance"]),
        "",
        "## 4. Reproduce",
        "",
        "```",
        "make cv standards bias   # writes artifacts/cv/oof.parquet, standards.json, bias.json",
        "make report03f",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT} and {len(figs)} figure")


if __name__ == "__main__":
    main()
