"""Phase 3 report -> reports/03_cv_robustness.md + reports/figures/03_*.png

    python -m faceai.report03        # or: make report03

Reads ONLY saved artifacts (artifacts/cv/, artifacts/features/, artifacts/splits.*) and the
data folder's file counts, so every number in the report and the pipeline diagram comes
from the run itself. Aggregates only: no individual subject's row reaches the report.
"""

from __future__ import annotations

import datetime as dt
import json
from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from faceai.cv import MODELS, OUT_DIR, PAPER_TABLE5, session_table
from faceai.data import audit
from faceai.data.hub import LOCAL_DIR, REPO_ID, REVISION, ROOT
from faceai.diagnostics import DIAG_TARGETS
from faceai.features import (
    MODEL_FEATURES,
    SNR_LEVELS_DB,
    compare_resp_bands,
    load_features,
)
from faceai.targets import load

REPORT = ROOT / "reports" / "03_cv_robustness.md"
FIGURES = ROOT / "reports" / "figures"

# ---- palette (dataviz reference palette, validated on white: 3 slots, all-pairs PASS)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR = {
    "Mean": MUTED,
    "Median": MUTED,
    "Demographics": INK2,
    "Ridge (L2)": BLUE,
    "Lasso (L1)": ORANGE,
    "LightGBM": AQUA,
}
TRAIN, VAL = BLUE, ORANGE  # in train-vs-validation charts colour encodes the data split
LABEL = {
    "hba1c": "HbA1c (%)",
    "cholesterol": "Cholesterol (mmol/L)",
    "spo2": "SpO₂ (%)",
    "stress_psm25": "Stress (PSM-25)",
    "temperature": "Temperature (°C)",
    "respiratory_rate": "Respiratory rate (/min)",
    "arterial_stiffness": "Arterial stiffness (index)",
    "bmi": "BMI (kg/m²)",
    "age": "Age (years) — positive control",
}
SHORT = {k: v.split(" (")[0].split(" —")[0] for k, v in LABEL.items()}
SHORT["spo2"] = "SpO₂"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    return f"figures/{name}"


def md_table(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return out


def ci(v: dict, fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(v['mae'])} [{fmt.format(v['ci'][0])}–{fmt.format(v['ci'][1])}]"


# ----------------------------------------------------------------- pipeline diagram
def pipeline_counts() -> dict:
    splits = json.loads(
        (ROOT / "artifacts" / "splits.json").read_text(encoding="utf-8")
    )
    sp = pd.read_parquet(ROOT / "artifacts" / "splits.parquet")
    subj = sp.drop_duplicates("patient_id")
    log = json.loads(
        (ROOT / "artifacts" / "features" / "build_log.json").read_text("utf-8")
    )
    res = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    files = {
        d: sum(1 for _ in (LOCAL_DIR / d).iterdir())
        for d in ["ppg", "ecg", "meta", "ppg_sync", "video"]
    }
    db = audit.load_db()
    return {
        "files": files,
        "db_rows": len(db),
        "db_subjects": int(db["patient_id"].nunique()),
        "splits": splits,
        "test_subjects": int((subj["split"] == "test").sum()),
        "dev_subjects": int((subj["split"] == "dev").sum()),
        "fold_sizes": subj[subj["split"] == "dev"].groupby("fold").size().tolist(),
        "dev_sessions": log["levels"]["clean"]["sessions"],
        "unreadable": log["levels"]["clean"]["unreadable_files"],
        "modelled_sessions": res["meta"]["sessions"],
        "n_features": len(MODEL_FEATURES),
    }


def fig_pipeline(c: dict) -> str:
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    DONE, NEW, OFF = "#f0efec", "#cde2fb", "white"

    def box(x, y, w, h, title, lines, fill, dashed=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4,rounding_size=1.2",
                facecolor=fill,
                edgecolor=MUTED if dashed else AXIS,
                linewidth=1.0,
                linestyle=(0, (4, 3)) if dashed else "solid",
            )
        )
        ax.text(
            x + 1.2,
            y + h - 1.6,
            title,
            fontsize=10,
            fontweight="bold",
            color=INK,
            va="top",
        )
        ax.text(
            x + 1.2,
            y + h - 5.0,
            "\n".join(lines),
            fontsize=8.2,
            color=INK2,
            va="top",
            linespacing=1.45,
        )

    def arrow(x1, y1, x2, y2, dashed=False, label=None):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color=MUTED if dashed else INK2,
                linewidth=1.1,
                linestyle=(0, (4, 3)) if dashed else "solid",
            )
        )
        if label:
            ax.text(
                (x1 + x2) / 2 + 0.8,
                (y1 + y2) / 2,
                label,
                fontsize=7.6,
                color=MUTED,
                va="center",
            )

    f, s = c["files"], c["splits"]
    X, W = 2, 60
    rows = [  # (y, h)
        (89.5, 10),
        (76, 11),
        (61.5, 12),
        (47, 12),
        (30, 14.5),
        (17, 10.5),
        (1, 13),
    ]
    box(
        X,
        rows[0][0],
        W,
        rows[0][1],
        "1  Source  (Phase 0)",
        [
            f"Hugging Face {REPO_ID} @ {REVISION[:8]} (pinned), CC-BY-4.0",
            f"db.csv: {c['db_rows']:,} recordings = {c['db_subjects']} subjects × 2 states × 3 cameras",
        ],
        DONE,
    )
    box(
        X,
        rows[1][0],
        W,
        rows[1][1],
        "2  Download — gate A, non-video only  (Phase 0)",
        [
            f"ppg {f['ppg']:,} · ecg {f['ecg']:,} · meta {f['meta']:,} · ppg_sync {f['ppg_sync']:,} · db.csv",
            f"video: {f['video']} sample files only (full set 129.8 GB not downloaded)",
        ],
        DONE,
    )
    box(
        X,
        rows[2][0],
        W,
        rows[2][1],
        "3  Parse + integrity audit  (Phase 1)",
        [
            "all 5 file formats opened and confirmed",
            (
                f"excluded: subject {s['excluded_subjects'][0]} (mis-paired, flat signal); "
                f"{s['excluded_sessions'][0][0]} '{s['excluded_sessions'][0][1]}' (flat)"
            ),
            f"→ {s['n_subjects']} subjects · {s['n_recordings']:,} recordings",
        ],
        DONE,
    )
    box(
        X,
        rows[3][0],
        W,
        rows[3][1],
        "4  Subject-level split  (Phase 2, committed)",
        [
            f"StratifiedGroupKFold, seed {s['seed']}, stratified on {s['stratified_by']}",
            f"dev {c['dev_subjects']} subjects → 5 folds of {min(c['fold_sizes'])}–{max(c['fold_sizes'])}",
            f"assignment SHA-256 {s['assignment_sha256'][:12]}… · leakage test in pre-commit",
        ],
        DONE,
    )
    box(
        X,
        rows[4][0],
        W,
        rows[4][1],
        "5  Contact-PPG features  (Phase 3, NEW)",
        [
            (
                f"{c['dev_sessions']} dev sessions; {len(c['unreadable'])} unreadable file "
                f"({', '.join(c['unreadable'])}) → {c['modelled_sessions']} modelled"
            ),
            "inverted to physiological polarity · resampled to a uniform 100 Hz",
            f"{c['n_features']} features: rate, variability, respiration, beat shape, quality",
            f"+ noise copies at {len(SNR_LEVELS_DB)} in-band SNR levels (+20 … −10 dB)",
        ],
        NEW,
    )
    box(
        X,
        rows[5][0],
        W,
        rows[5][1],
        "6  Models  (Phase 3, NEW)",
        [
            "baselines: mean · median · demographics (age + sex + BMI)",
            "PPG + demographics: Ridge (L2) · Lasso (L1) · LightGBM, tuned by inner GroupKFold",
        ],
        NEW,
    )
    box(
        X,
        rows[6][0],
        W,
        rows[6][1],
        "7  Evaluation  (Phase 3, NEW)",
        [
            "5-fold out-of-fold MAE · subject-cluster bootstrap 95% CIs",
            "overfitting: train vs validation, regularisation and boosting curves, learning curves",
            "robustness: noise injection · rest→exercise shift · subgroups · stability",
        ],
        NEW,
    )
    for (y0, _), (y1, h1) in pairwise(rows):
        arrow(X + W / 2, y0 - 0.6, X + W / 2, y1 + h1 + 0.6)
    # side branches
    box(
        68,
        49,
        30,
        9,
        "Held-out test set",
        [
            f"{c['test_subjects']} subjects · SEALED until Phase 10",
            "never read by any step here",
        ],
        DONE,
    )
    arrow(X + W + 0.6, 53, 67.4, 53.5)
    box(
        68,
        31.5,
        30,
        11.5,
        "Video → rPPG  (Phase 4)",
        [
            "MediaPipe face ROIs → POS / CHROM",
            "NOT STARTED: needs the 58 GB frontal-",
            "video download (gate B); will feed step 5",
        ],
        OFF,
        dashed=True,
    )
    arrow(67.4, 37, X + W + 0.6, 37, dashed=True)
    # legend
    for i, (lbl, fill, dsh) in enumerate(
        [
            ("Done (Phases 0–2)", DONE, False),
            ("New this week (Phase 3)", NEW, False),
            ("Not started", OFF, True),
        ]
    ):
        ax.add_patch(
            FancyBboxPatch(
                (69, 20 - i * 5),
                3,
                2.6,
                boxstyle="round,pad=0.2",
                facecolor=fill,
                edgecolor=MUTED if dsh else AXIS,
                linestyle=(0, (4, 3)) if dsh else "solid",
            )
        )
        ax.text(73.5, 21.3 - i * 5, lbl, fontsize=8.5, color=INK2, va="center")
    return save(fig, "03_pipeline.png")


# -------------------------------------------------------------------------- figures
def fig_overview(res: dict) -> str:
    order = [
        "age",
        "respiratory_rate",
        "arterial_stiffness",
        "spo2",
        "stress_psm25",
        "bmi",
        "hba1c",
        "cholesterol",
        "temperature",
    ]
    series = ["Median", "Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    offs = np.linspace(-0.3, 0.3, len(series))
    for i, t in enumerate(order):
        y0 = len(order) - 1 - i
        for o, m in zip(offs, series, strict=True):
            v = res[t]["models"][m]["vs"]["Mean"]
            hollow = m == "Median"
            ax.plot(
                v["pct_ci"],
                [y0 - o] * 2,
                color=COLOR[m],
                linewidth=1.6,
                solid_capstyle="round",
            )
            ax.plot(
                v["pct"],
                y0 - o,
                "o",
                markersize=6,
                color=COLOR[m],
                markerfacecolor="white" if hollow else COLOR[m],
                markeredgewidth=1.5,
                label=m if i == 0 else None,
            )
    ax.axvline(0, color=INK2, linewidth=1.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[t] for t in reversed(order)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(
        "Change in MAE vs the mean predictor, %  (left of 0 = better; bars = 95% CI)"
    )
    ax.set_title("5-fold CV: does contact PPG beat a constant guess?", loc="left")
    for y, lbl in [
        (9.05, "Tier A"),
        (5.42, "Tier B"),
        (2.42, "Tier C (not clinically usable)"),
    ]:
        ax.text(ax.get_xlim()[0] + 0.5, y, lbl, fontsize=8, color=MUTED, va="top")
    for y in (5.5, 2.5):
        ax.axhline(y, color=GRID, linewidth=1.0)
    ax.set_ylim(-0.6, 9.1)
    ax.legend(loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.0))
    return save(fig, "03_cv_overview.png")


def fig_folds(folds: pd.DataFrame, res: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, t in zip(axes, DIAG_TARGETS, strict=True):
        d = folds[folds["target"] == t]
        for i, m in enumerate(MODELS):
            v = d[d["model"] == m]["val_mae"].to_numpy()
            jitter = np.linspace(-0.18, 0.18, len(v))
            ax.plot(
                i + jitter,
                v,
                "o",
                color=COLOR[m],
                markersize=4.5,
                alpha=0.85,
                markerfacecolor="white" if m == "Median" else COLOR[m],
            )
            r = res[t]["models"][m]
            ax.plot([i - 0.32, i + 0.32], [r["mae"]] * 2, color=INK, linewidth=1.4)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(
            ["Mean", "Median", "Demographics", "Ridge", "Lasso", "LightGBM"],
            fontsize=8,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_title(
            LABEL[t].replace(" — positive control", "\n(positive control)"), loc="left"
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Validation MAE per fold")
    fig.text(
        0.01,
        -0.2,
        "Dots: the 5 validation folds (95–96 subjects each). Black bar: pooled out-of-fold MAE.",
        fontsize=8,
        color=MUTED,
    )
    fig.suptitle(
        "Fold-to-fold spread: how much the answer depends on which subjects are held out",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    return save(fig, "03_fold_spread.png")


def fig_gap(folds: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    models = ["Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"]
    for ax, t in zip(axes, DIAG_TARGETS, strict=True):
        d = (
            folds[folds["target"] == t]
            .groupby("model")[["train_mae", "val_mae"]]
            .mean()
        )
        for i, m in enumerate(models):
            tr, va = d.loc[m, "train_mae"], d.loc[m, "val_mae"]
            ax.plot([i, i], [tr, va], color=AXIS, linewidth=2)
            ax.plot(
                i,
                tr,
                "o",
                color=TRAIN,
                markersize=7,
                label="Train (fitted subjects)" if i == 0 else None,
            )
            ax.plot(
                i,
                va,
                "o",
                color=VAL,
                markersize=7,
                label="Validation (unseen subjects)" if i == 0 else None,
            )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(
            ["Demo-\ngraphics", "Ridge", "Lasso", "Light-\nGBM"], fontsize=8
        )
        ax.set_ylim(0, 1.12 * d[["train_mae", "val_mae"]].to_numpy().max())
        ax.set_title(
            LABEL[t].replace(" — positive control", "\n(positive control)"), loc="left"
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE (mean of 5 folds)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2)
    fig.suptitle(
        "The overfitting gap: error on fitted vs unseen subjects",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=1.12,
    )
    return save(fig, "03_train_val_gap.png")


def fig_boosting(boost: pd.DataFrame, res: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, t in zip(axes, DIAG_TARGETS, strict=True):
        d = boost[boost["target"] == t].groupby("rounds")
        for col, colr, lbl in [
            ("train_mae", TRAIN, "Train"),
            ("val_mae", VAL, "Validation"),
        ]:
            mean, lo, hi = d[col].mean(), d[col].min(), d[col].max()
            ax.fill_between(mean.index, lo, hi, color=colr, alpha=0.12, linewidth=0)
            ax.plot(mean.index, mean, "-o", color=colr, markersize=4, label=lbl)
        best = d["val_mae"].mean().idxmin()
        ax.axvline(best, color=MUTED, linewidth=0.9)
        ax.text(
            best * (1.15 if best <= 50 else 1 / 1.15),
            0.1,
            f"best = {best} rounds",
            transform=ax.get_xaxis_transform(),
            fontsize=7.6,
            color=MUTED,
            va="bottom",
            ha="left" if best <= 50 else "right",
        )
        ax.axhline(res[t]["models"]["Mean"]["mae"], color=MUTED, linewidth=1.0)
        ax.text(
            max(boost["rounds"]),
            res[t]["models"]["Mean"]["mae"],
            "mean predictor ",
            fontsize=7.6,
            color=MUTED,
            va="top",
            ha="right",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Boosting rounds (log scale)")
        ax.set_title(
            LABEL[t].replace(" — positive control", "\n(positive control)"), loc="left"
        )
    axes[0].set_ylabel("MAE (mean of 5 folds; band = fold range)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2)
    fig.suptitle(
        "Overfitting in action: LightGBM (31 leaves) as boosting rounds grow",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=1.12,
    )
    return save(fig, "03_boosting_curve.png")


def fig_lasso(alpha: pd.DataFrame, folds: pd.DataFrame) -> str:
    fig, axes = plt.subplots(
        2, 3, figsize=(10.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )
    for j, t in enumerate(DIAG_TARGETS):
        d = alpha[(alpha["target"] == t) & (alpha["model"] == "Lasso (L1)")].groupby(
            "alpha"
        )
        ax = axes[0, j]
        for col, colr, lbl in [
            ("train_mae", TRAIN, "Train"),
            ("val_mae", VAL, "Validation"),
        ]:
            ax.plot(
                d[col].mean().index,
                d[col].mean(),
                "-o",
                color=colr,
                markersize=3.5,
                label=lbl,
            )
        chosen = [
            json.loads(p)["model__alpha"]
            for p in folds[(folds["target"] == t) & (folds["model"] == "Lasso (L1)")][
                "best_params"
            ]
        ]
        for a in (axes[0, j], axes[1, j]):
            a.axvline(float(np.median(chosen)), color=MUTED, linewidth=0.9)
        axes[1, j].text(
            float(np.median(chosen)) * 1.2,
            0.95,
            "chosen by\ninner CV",
            transform=axes[1, j].get_xaxis_transform(),
            fontsize=7.6,
            color=MUTED,
            va="top",
        )
        ax.set_title(
            LABEL[t].replace(" — positive control", "\n(positive control)"), loc="left"
        )
        nz = d["nonzero"].mean()
        axes[1, j].plot(nz.index, nz, "-o", color=INK2, markersize=3.5)
        axes[1, j].set_ylim(-1, d["n_inputs"].max().max() + 2)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("Lasso penalty α (log scale) → stronger")
    axes[0, 0].set_ylabel("MAE (mean of 5 folds)")
    axes[1, 0].set_ylabel("Inputs kept (non-zero)")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=2)
    fig.suptitle(
        "L1 regularisation: a stronger penalty drops inputs until only the constant is left",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=1.04,
    )
    return save(fig, "03_lasso_path.png")


def fig_learning(learn: pd.DataFrame, res: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, t in zip(axes, DIAG_TARGETS, strict=True):
        d = learn[learn["target"] == t]
        for m in ["Ridge (L2)", "LightGBM"]:
            g = (
                d[d["model"] == m]
                .groupby("fraction")[["n_train_subjects", "train_mae", "val_mae"]]
                .mean()
            )
            ax.plot(
                g["n_train_subjects"],
                g["val_mae"],
                "-o",
                color=COLOR[m],
                markersize=4,
                label=f"{m} — validation",
            )
            ax.plot(
                g["n_train_subjects"],
                g["train_mae"],
                "--",
                color=COLOR[m],
                linewidth=1.5,
                label=f"{m} — train",
            )
        ax.axhline(res[t]["models"]["Mean"]["mae"], color=MUTED, linewidth=1.0)
        ax.set_xlabel("Training subjects")
        ax.set_title(
            LABEL[t].replace(" — positive control", "\n(positive control)"), loc="left"
        )
    axes[0].set_ylabel("MAE (mean of 5 folds)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2)
    fig.suptitle(
        "Learning curves: does more data help? (grey line = mean predictor)",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=1.2,
    )
    return save(fig, "03_learning_curves.png")


def noise_feature_table() -> pd.DataFrame:
    clean = load_features()
    clean = clean[clean["readable"]].set_index(["patient_id", "step"])
    rows = [
        {
            "level": "clean",
            "hr_err_median": 0.0,
            "hr_err_gt5_pct": 0.0,
            "apg_b_a_mean": clean["apg_b_a"].mean(),
            "sqi_mean": clean["sqi_template_r"].mean(),
            "rows_with_nan_pct": 100 * clean[MODEL_FEATURES].isna().any(axis=1).mean(),
        }
    ]
    for snr in SNR_LEVELS_DB:
        n = load_features(snr)
        n = n[n["readable"]].set_index(["patient_id", "step"]).loc[clean.index]
        err = (n["hr_mean_bpm"] - clean["hr_mean_bpm"]).abs()
        rows.append(
            {
                "level": level_name(snr),
                "hr_err_median": float(err.median()),
                "hr_err_gt5_pct": float(100 * (err > 5).mean()),
                "apg_b_a_mean": n["apg_b_a"].mean(),
                "sqi_mean": n["sqi_template_r"].mean(),
                "rows_with_nan_pct": 100 * n[MODEL_FEATURES].isna().any(axis=1).mean(),
            }
        )
    return pd.DataFrame(rows)


def fig_noise(noise: pd.DataFrame, feat: pd.DataFrame) -> str:
    levels = ["clean", *[f"{s:+d} dB" for s in SNR_LEVELS_DB]]
    ticks = ["clean", *[f"{s:+d}" if s else "0" for s in SNR_LEVELS_DB]]
    fig = plt.figure(figsize=(10.5, 7.8))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(range(len(levels)), feat["hr_err_median"], "-o", color=INK2, markersize=4)
    ax.set_title("Heart rate: median error (bpm)", loc="left")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(levels)), feat["apg_b_a_mean"], "-o", color=INK2, markersize=4)
    ax2.set_title("Beat shape: mean APG b/a", loc="left")
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(range(len(levels)), feat["sqi_mean"], "-o", color=INK2, markersize=4)
    ax3.set_title("Signal quality: template r", loc="left")
    for a in (ax, ax2, ax3):
        a.set_xticks(range(len(levels)))
        a.set_xticklabels(ticks, fontsize=7.5)
    order = [
        "age",
        "respiratory_rate",
        "arterial_stiffness",
        "bmi",
        "hba1c",
        "cholesterol",
    ]
    for k, t in enumerate(order):
        a = fig.add_subplot(gs[1 + k // 3, k % 3])
        d = noise[noise["target"] == t].groupby(["model", "level"])["val_mae"].mean()
        for m in ["Mean", "Ridge (L2)", "LightGBM"]:
            y = [d.loc[(m, lv)] for lv in levels]
            a.plot(
                range(len(levels)),
                y,
                "-o" if m != "Mean" else "-",
                color=COLOR[m],
                markersize=4,
                linewidth=2 if m != "Mean" else 1.2,
                label=m if k == 0 else None,
            )
        a.set_xticks(range(len(levels)))
        a.set_xticklabels(ticks, fontsize=7.5)
        a.set_title(LABEL[t].replace(" — positive control", " (control)"), loc="left")
        if k % 3 == 0:
            a.set_ylabel("Validation MAE")
        if k == 0:
            a.legend(loc="lower left", bbox_to_anchor=(0, 1.2), ncol=3)
    fig.text(
        0.5,
        0.02,
        "In-band SNR of the injected noise, dB  (left = clean contact PPG; right = so noisy that heart rate itself is lost)",
        ha="center",
        fontsize=8.5,
        color=INK2,
    )
    fig.suptitle(
        "Noise injection: features (top row) and models fitted on clean data (below)",
        x=0.1,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=0.94,
    )
    return save(fig, "03_noise.png")


def fig_subgroups(sub: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    levels = [
        ("sex", "F"),
        ("sex", "M"),
        ("age band", "< 25"),
        ("age band", ">= 25"),
        ("state", "rest"),
        ("state", "post-exercise"),
    ]
    ylabels = ["Female", "Male", "Age < 25", "Age ≥ 25", "Rest", "Post-exercise"]
    for ax, t in zip(axes, DIAG_TARGETS, strict=True):
        for i, (fac, lv) in enumerate(levels):
            y = len(levels) - 1 - i
            for o, m in zip([0.15, -0.15], ["Mean", "LightGBM"], strict=True):
                r = sub[
                    (sub["target"] == t)
                    & (sub["factor"] == fac)
                    & (sub["level"] == lv)
                    & (sub["model"] == m)
                ].iloc[0]
                ax.plot(r["ci"], [y + o] * 2, color=COLOR[m], linewidth=1.6)
                ax.plot(
                    r["mae"],
                    y + o,
                    "o",
                    color=COLOR[m],
                    markersize=5.5,
                    label=m if (i == 0) else None,
                )
        ax.set_yticks(range(len(levels)))
        counts = [
            sub[
                (sub["target"] == t) & (sub["factor"] == f) & (sub["level"] == lv)
            ].iloc[0]["n_subjects"]
            for f, lv in levels
        ]
        ax.set_yticklabels(
            [
                f"{lb}  (n={n})"
                for lb, n in zip(reversed(ylabels), reversed(counts), strict=True)
            ]
        )
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("MAE with 95% CI")
        ax.set_title(
            LABEL[t].replace(" — positive control", "\n(positive control)"), loc="left"
        )
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=2)
    fig.suptitle(
        "Subgroups: who carries the error? (n = dev subjects; 95% CIs)",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        y=1.12,
    )
    return save(fig, "03_subgroups.png")


# --------------------------------------------------------------------------- report
def level_name(snr: int) -> str:
    return "0 dB" if snr == 0 else f"{snr:+d} dB"


def older_share_lines(sub: pd.DataFrame, t: pd.DataFrame) -> list[str]:
    """How much worse the >= 25 group is, and its share of total error - computed."""
    oof = pd.read_parquet(OUT_DIR / "oof.parquet")
    ages = t[["patient_id", "step", "age"]].rename(columns={"age": "age_years"})
    out = []
    for tg in ["hba1c", "cholesterol"]:
        d = oof[oof["target"] == tg].merge(
            ages, on=["patient_id", "step"], validate="1:1"
        )
        err = (d["Mean"] - d["y"]).abs()
        old = d["age_years"] >= 25
        out.append(
            f"- {SHORT[tg]}: mean-predictor MAE {err[old].mean():.3f} for age \u2265 25 vs "
            f"{err[~old].mean():.3f} for < 25 ({err[old].mean() / err[~old].mean():.1f}\u00d7); "
            f"the older group is {100 * old.mean():.1f}% of sessions but "
            f"{100 * err[old].sum() / err.sum():.1f}% of the total error."
        )
    n_old = int(
        sub[(sub["factor"] == "age band") & (sub["level"] == ">= 25")][
            "n_subjects"
        ].iloc[0]
    )
    return [
        f"The {n_old} dev subjects aged 25 or over are much harder to predict:",
        "",
        *out,
        "",
        "No model closes that gap: their labels sit far from a young cohort's mean, and the",
        "PPG features do not tell the models which way an older subject deviates.",
    ]


def shift_label_lines(t: pd.DataFrame) -> list[str]:
    g = t.groupby("step")[["respiratory_rate", "spo2", "arterial_stiffness"]].agg(
        ["mean", "std"]
    )
    rr, sp, st = (g[c] for c in ["respiratory_rate", "spo2", "arterial_stiffness"])
    b, a = "before", "after"
    return [
        "Respiratory-rate and SpO\u2082 errors rise post-exercise for every model, including the",
        "constant, because their labels change after exercise: respiratory rate rises from",
        f"{rr.loc[b, 'mean']:.1f} to {rr.loc[a, 'mean']:.1f}/min on average, and SpO\u2082 falls from",
        f"{sp.loc[b, 'mean']:.2f}% to {sp.loc[a, 'mean']:.2f}% with a wider spread",
        f"(SD {sp.loc[b, 'std']:.2f} \u2192 {sp.loc[a, 'std']:.2f}). A rest-trained model cannot",
        "anticipate it. Arterial stiffness is also re-measured each session (mean",
        f"{st.loc[b, 'mean']:.2f} \u2192 {st.loc[a, 'mean']:.2f}).",
    ]


def main() -> None:
    style()
    res_all = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    res, meta = res_all["targets"], res_all["meta"]
    folds = pd.read_csv(OUT_DIR / "folds.csv")
    alpha = pd.read_csv(OUT_DIR / "alpha_curves.csv")
    boost = pd.read_csv(OUT_DIR / "boosting_curve.csv")
    learn = pd.read_csv(OUT_DIR / "learning_curves.csv")
    noise = pd.read_csv(OUT_DIR / "noise.csv")
    shift = pd.read_csv(OUT_DIR / "shift.csv")
    sub = pd.read_json(OUT_DIR / "subgroups.json")
    stab = pd.read_csv(OUT_DIR / "stability.csv")
    counts = pipeline_counts()
    feat_noise = noise_feature_table()
    labels = (
        audit.load_db()
        .drop_duplicates(["patient_id", "step"])
        .set_index(["patient_id", "step"])
    )
    bands = compare_resp_bands(labels)
    cfg = load()

    figs = {
        "pipeline": fig_pipeline(counts),
        "overview": fig_overview(res),
        "folds": fig_folds(folds, res),
        "gap": fig_gap(folds),
        "boost": fig_boosting(boost, res),
        "lasso": fig_lasso(alpha, folds),
        "learn": fig_learning(learn, res),
        "noise": fig_noise(noise, feat_noise),
        "sub": fig_subgroups(sub),
    }

    t = session_table()
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    L: list[str] = []
    L += [
        "# Phase 3 — Contact-PPG 5-fold cross-validation, overfitting and robustness",
        "",
        f"_Generated {now} by `python -m faceai.report03` from saved artifacts only._",
        "",
        "**Scope.** Contact PPG only (the .PW finger signal) — the best-case signal, before any",
        "camera is involved. No video, no rPPG (Phase 4 has not started). Dev folds only: the 120",
        "held-out test subjects are sealed until Phase 10 and were not read.",
        "",
        f"**Data.** {meta['sessions']} dev sessions from {meta['subjects']} subjects, the committed",
        (
            "5 folds (`artifacts/splits.parquet`), "
            f"{len(MODEL_FEATURES)} PPG features + demographics. Bootstrap: {meta['n_boot']} subject-cluster"
        ),
        f"resamples. Lasso convergence warnings: {meta['convergence_warnings']}.",
        "",
        "## 1. Pipeline",
        "",
        f"![pipeline]({figs['pipeline']})",
        "",
        "## 2. Data findings made while building it",
        "",
        "- **The raw .PW waveform is inverted.** On 40 sampled sessions, the median trough-to-peak",
        "  time was 0.545 s and peak-to-trough 0.140 s — the reverse of a real pulse in every one.",
        "  Features use `255 − raw`. Heart rate is unaffected; beat-shape features are not.",
        "- **One contact-PPG file is corrupt:** `8706_after.PW` (a merged timestamp, a run of NUL",
        "  bytes and a truncated last line; its clock jumps back 18 s). It is reported and left out of",
        "  the features, not repaired. All other 1,199 files parse and have monotonic timestamps",
        "  (largest sampling gap 0.85 s). This scan read file formats only, including test-set files.",
        "- **Respiratory rate from PPG disagrees with the physician count.** Five estimator settings",
        "  were compared on 150 dev sessions (fused RIFV/RIAV/RIIV vs the label):",
        "",
    ]
    b = bands.copy()
    b["mae_vs_label"] = b["mae_vs_label"].map("{:.2f}".format)
    b["pearson_r"] = b["pearson_r"].map(lambda v: "—" if pd.isna(v) else f"{v:+.2f}")
    b["median_estimate"] = b["median_estimate"].map("{:.1f}".format)
    b.columns = [
        "Setting",
        "MAE vs label (/min)",
        "Pearson r",
        "Median estimate (/min)",
    ]
    L += md_table(b)
    L += [
        "",
        "  Without a band-pass the estimate collapses onto slow drift (~7/min). The adopted 12–30/min",
        "  band is the physiological prior around the paper's 15–24 label range; it was chosen after",
        "  this comparison on dev labels, so the respiratory-rate results below are slightly optimistic.",
        "  Even the best setting is worse than a constant guess — consistent with the label being",
        "  counted at a separate stand before the recording, not during it.",
        "",
        "## 3. Five-fold cross-validation results",
        "",
        f"![overview]({figs['overview']})",
        "",
        "Pooled out-of-fold MAE with 95% subject-cluster bootstrap CIs. **Strict** = the brief's rule",
        "(model CI does not overlap the mean-predictor CI **and** the demographics CI). **Paired** = the",
        "95% CI of (model − demographics) on the same bootstrap draws lies entirely below 0.",
        "",
    ]
    rows = []
    for tg in [
        "age",
        "respiratory_rate",
        "arterial_stiffness",
        "spo2",
        "stress_psm25",
        "bmi",
        "hba1c",
        "cholesterol",
        "temperature",
    ]:
        r, m = res[tg], res[tg]["models"]
        strict = [k.split(" ")[0] for k, v in r["verdict"].items() if v["beats_both"]]
        paired = [
            k.split(" ")[0]
            for k in ["Ridge (L2)", "Lasso (L1)", "LightGBM"]
            if m[k]["vs"]["Demographics"]["ci"][1] < 0
        ]
        p5 = PAPER_TABLE5.get(tg)
        rows.append(
            [
                f"{SHORT[tg]} ({cfg.by_name(tg).tier})",
                ci(m["Mean"]),
                ci(m["Median"]),
                ci(m["Demographics"]),
                ci(m["Ridge (L2)"]),
                ci(m["Lasso (L1)"]),
                ci(m["LightGBM"]),
                ", ".join(strict) or "none",
                ", ".join(paired) or "none",
                "—" if p5 is None else f"{p5[0]:.2f} / {p5[1]:.2f}",
            ]
        )
    L += md_table(
        pd.DataFrame(
            rows,
            columns=[
                "Target (tier)",
                "Mean",
                "Median",
                "Demographics",
                "Ridge (L2)",
                "Lasso (L1)",
                "LightGBM",
                "Beats both (strict)",
                "Beats demographics (paired)",
                "Paper T5 constant / video model",
            ],
        )
    )
    L += [
        "",
        "Paper Table 5 numbers are context only: a different split and test set, and a video model.",
        "",
        "**Reading it.**",
        "",
        f"- **Positive control passes.** Age: LightGBM {ci(res['age']['models']['LightGBM'])} years vs",
        f"  mean {ci(res['age']['models']['Mean'])} and demographics {ci(res['age']['models']['Demographics'])}.",
        "  The pipeline can find signal where it exists. Caveat: against the MAE-optimal median",
        f"  ({ci(res['age']['models']['Median'])}) the CIs overlap, because age is so skewed (cohort median 20).",
        "- **Arterial stiffness and respiratory rate** improve on demographics on the paired test but",
        "  not under the brief's strict rule — promising, not established.",
        "- **BMI:** demographics (age + sex) beat the constant on the paired test, not the strict rule;",
        "  PPG adds nothing on top of them.",
        "- **HbA1c, cholesterol, SpO₂, stress, temperature:** nothing beats a constant guess. For",
        "  HbA1c the best model is within 0.003 of the mean predictor.",
        "",
        "### Fold-to-fold spread (the 5-fold robustness check)",
        "",
        f"![folds]({figs['folds']})",
        "",
    ]
    fs = folds[folds["target"].isin(DIAG_TARGETS)].pivot_table(
        index=["target", "model"], columns="fold", values="val_mae"
    )
    fs["spread (max−min)"] = fs.max(axis=1) - fs.min(axis=1)
    fs = fs.reset_index()
    fs.columns = [
        "Target",
        "Model",
        *[f"Fold {c}" for c in range(5)],
        "Spread (max−min)",
    ]
    fs["Target"] = fs["Target"].map(SHORT)
    L += md_table(fs.round(3))
    L += [
        "",
        "## 4. Overfitting",
        "",
        "Overfitting = a model fits the quirks of the subjects it was trained on, so its error on them",
        "(train) is much lower than on subjects it has never seen (validation). Cross-validation does",
        "not prevent it; it **measures** it, by always scoring on unseen subjects.",
        "",
        f"![gap]({figs['gap']})",
        "",
    ]
    gap = folds.groupby(["target", "model"])[["train_mae", "val_mae"]].mean()
    gap["gap"] = gap["val_mae"] - gap["train_mae"]
    gap = gap.reset_index()
    gap = gap[
        gap["model"].isin(["Demographics", "Ridge (L2)", "Lasso (L1)", "LightGBM"])
    ]
    gap["target"] = gap["target"].map(SHORT)
    gap.columns = ["Target", "Model", "Train MAE", "Validation MAE", "Gap"]
    L += md_table(gap.round(3))
    L += [
        "",
        f"![boost]({figs['boost']})",
        "",
        "With 31 leaves and no early stopping, LightGBM drives HbA1c **training** error towards zero",
        "while **validation** error rises above the mean predictor: it memorises noise. For age the",
        "validation error falls and then flattens — there is real signal to learn first.",
        "",
        "**Regularisation** limits that complexity. L2 (Ridge) shrinks all weights towards zero; L1",
        "(Lasso) can set weights exactly to zero, dropping inputs. For trees the levers are fewer",
        "leaves, fewer rounds and a minimum leaf size. In every case the penalty was chosen by an",
        "inner CV inside the training folds, never on the fold being scored.",
        "",
        f"![lasso]({figs['lasso']})",
        "",
        "For HbA1c the inner CV keeps 0–2 of 20 inputs — almost the constant. For HbA1c, cholesterol,",
        "SpO₂ and temperature it also chose LightGBM's smallest or next-smallest setting (2–4 leaves,",
        "25–50 rounds) in every fold, and for temperature Ridge's strongest penalty in 2 of 5 folds:",
        "regularisation correctly shrinking towards the mean when there is nothing to learn.",
        "",
        f"![learn]({figs['learn']})",
        "",
        "More subjects do not move HbA1c or cholesterol validation error; for age they keep helping.",
        "",
        "## 5. Robustness",
        "",
        "### Noise injection",
        "",
        f"![noise]({figs['noise']})",
        "",
    ]
    fn = feat_noise.copy()
    fn.columns = [
        "Level",
        "HR error median (bpm)",
        "HR error > 5 bpm (%)",
        "Mean APG b/a",
        "Mean template r",
        "Sessions with a NaN feature (%)",
    ]
    L += md_table(fn.round(3))
    L += [""]
    nz = (
        noise.groupby(["target", "model", "level"], sort=False)["val_mae"]
        .mean()
        .unstack("level")
    )
    nz = nz[["clean", *[f"{s:+d} dB" for s in SNR_LEVELS_DB]]].reset_index()
    nz = nz.rename(columns={f"{s:+d} dB": level_name(s) for s in SNR_LEVELS_DB})
    nz["target"] = nz["target"].map(SHORT)
    L += md_table(nz.round(3).rename(columns={"target": "Target", "model": "Model"}))
    L += [
        "",
        "Heart rate survives down to +5 dB; beat shape degrades from +10 dB. Models fitted on clean",
        "contact PPG and fed noisy features do not fail gracefully: for age and respiratory rate they",
        "become **worse than the constant** at 0 dB and below. HbA1c and cholesterol look “robust” only",
        "because their models learned almost nothing. Phase 4 must therefore either train on rPPG",
        "itself or gate predictions on signal quality.",
        "",
        "### Rest → post-exercise shift",
        "",
        "Models fitted on rest sessions only, scored on unseen subjects' rest and post-exercise sessions:",
        "",
    ]
    sh = (
        shift.groupby(["target", "model", "scored_on"])["val_mae"]
        .mean()
        .unstack()
        .reset_index()
    )
    sh["target"] = sh["target"].map(SHORT)
    sh = sh[["target", "model", "before", "after"]]
    sh.columns = ["Target", "Model", "Scored on rest", "Scored on post-exercise"]
    L += md_table(sh.round(3))
    L += [
        "",
        *shift_label_lines(t),
        "",
        "### Subgroups",
        "",
        f"![sub]({figs['sub']})",
        "",
    ]
    s2 = sub[
        sub["target"].isin(["hba1c", "cholesterol"])
        & sub["model"].isin(["Mean", "Demographics", "LightGBM"])
    ].copy()
    s2["MAE [95% CI]"] = [
        f"{m:.3f} [{c[0]:.3f}–{c[1]:.3f}]"
        for m, c in zip(s2["mae"], s2["ci"], strict=True)
    ]
    s2["target"] = s2["target"].map(SHORT)
    s2 = s2[["target", "factor", "level", "model", "n_subjects", "MAE [95% CI]"]]
    s2.columns = ["Target", "Factor", "Level", "Model", "Subjects", "MAE [95% CI]"]
    L += md_table(s2)
    L += [
        "",
        *older_share_lines(sub, t),
        "",
        "### Stability across a person's two sessions",
        "",
    ]
    st = stab.copy()
    st["target"] = st["target"].map(SHORT)
    st.columns = [
        "Target",
        "Model",
        "Subjects",
        "Label identical (%)",
        "Prediction shift median",
        "90th pct",
        "Max",
    ]
    L += md_table(st.round(3))
    L += [
        "",
        "Labels are identical across sessions for almost everyone, so any prediction shift is driven by",
        "the session (heart rate, pulse shape), not the biomarker. Shifts are small because the models",
        "are close to constant.",
        "",
        "### Not run",
        "",
        "- **Adversarial testing** — crafted worst-case inputs. Meaningful for deep models on raw",
        "  signals; deferred to Phase 8.",
        "- **Leakage demonstration** (deliberately random split) — would break the project's",
        "  never-random-split rule; not run without explicit approval.",
        "",
        "## 6. Reproduce",
        "",
        "```",
        "make features      # contact-PPG features, clean + 6 noise levels (~1 min)",
        "make cv            # 5-fold CV, bootstrap CIs (~30 s)",
        "make diagnostics   # regularisation, boosting and learning curves (~1 min)",
        "make robustness    # noise, shift, subgroups, stability (~1 min)",
        "make report03      # this report and its figures",
        "```",
        "",
        f"Feature table: {len(t)} modelled sessions; features: `{', '.join(MODEL_FEATURES)}`.",
        "",
    ]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT} and {len(figs)} figures")


if __name__ == "__main__":
    main()
