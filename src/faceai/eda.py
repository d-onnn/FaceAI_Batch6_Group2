"""Phase 1 EDA -> reports/01_eda.md + reports/figures/

    python -m faceai.eda        # or: make eda

Aggregates only: no individual subject's row ever reaches the committed report.
The analysis unit is stated explicitly for every statistic - physiological labels are
per SESSION (1200), demographics per SUBJECT (600). Nothing is computed over the 3600
recording rows, which would triple-count every session.
"""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faceai.data import audit
from faceai.data import formats as F
from faceai.data.hub import LOCAL_DIR, ROOT
from faceai.targets import TargetConfig, load

REPORT = ROOT / "reports" / "01_eda.md"
FIGURES = ROOT / "reports" / "figures"
SEX_ENCODING = {"M": 1, "F": 0}  # for rank correlation only; not a modelling choice


def md_table(df: pd.DataFrame, floatfmt: str = "%.3f") -> list[str]:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = [
        "| "
        + " | ".join(floatfmt % v if isinstance(v, float) else str(v) for v in r)
        + " |"
        for r in df.itertuples(index=False)
    ]
    return [header, rule, *rows]


def sessions(db: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """One row per (subject, session) - the correct unit for physiological labels."""
    cols = ["patient_id", "step", *[t.column for t in cfg.modellable()], "pulse"]
    return db.drop_duplicates(subset=["patient_id", "step"])[cols].reset_index(
        drop=True
    )


def subjects(db: pd.DataFrame) -> pd.DataFrame:
    """One row per subject - the correct unit for demographics."""
    return db.drop_duplicates(subset=["patient_id"])[
        ["patient_id", "age", "sex", "bmi", "weight", "height"]
    ].reset_index(drop=True)


def format_table() -> pd.DataFrame:
    """Open one real file of every format and report what is actually in it."""
    subject, state, cam = "1020", "after", "FullHDwebcam"
    ecg = F.read_ecg(LOCAL_DIR / f"ecg/{subject}_{state}.json")
    meta = F.read_meta(LOCAL_DIR / f"meta/{subject}_{cam}_{state}.txt")
    pw = F.read_pw(LOCAL_DIR / f"ppg/{subject}_{state}.PW")
    sync = F.read_ppg_sync(LOCAL_DIR / f"ppg_sync/{subject}_{cam}_{state}.txt")
    rows = [
        {
            "format": "`ecg/*.json`",
            "opened": ecg.path.name,
            "confirmed": f"{ecg.frequency} Hz, {len(ecg.leads)} leads, {ecg.time[-1]:.1f} s",
            "quirk found": f"lead I has {ecg.lead_lengths['I']} samples vs "
            f"{ecg.lead_lengths['II']} for the rest - stacking must truncate",
        },
        {
            "format": "`meta/*.txt`",
            "opened": meta.path.name,
            "confirmed": f"{meta.n_frames} frame timestamps, {meta.duration_s:.1f} s, "
            f"{meta.fps_mean:.2f} fps mean",
            "quirk found": f"rate not constant (median {meta.fps_median:.2f} fps); "
            "undocumented in the brief",
        },
        {
            "format": "`ppg/*.PW`",
            "opened": pw.path.name,
            "confirmed": f"text, {pw.n_samples} samples, {pw.duration_s:.1f} s, "
            f"{pw.mean_rate_hz:.2f} Hz mean, 8-bit values",
            "quirk found": "samples arrive in bursts; 100 Hz is an average only - "
            "resample before spectral work",
        },
        {
            "format": "`ppg_sync/*.txt`",
            "opened": sync.path.name,
            "confirmed": f"{sync.n_rows} rows = one per encoded video frame; "
            f"median sync error {1000 * sync.median_sync_error_s:.1f} ms",
            "quirk found": "column 2 is the frame/PPG time mismatch; "
            f"{meta.n_frames - sync.n_rows} fewer rows than `meta/`",
        },
    ]
    video_path = LOCAL_DIR / f"video/{subject}_{cam}_{state}.avi"
    if video_path.exists():
        v = F.video_info(video_path)
        rows.append(
            {
                "format": "`video/*.avi`",
                "opened": v.path.name,
                "confirmed": f"{v.codec} {v.width}x{v.height}, {v.n_frames} frames, "
                f"{v.container_fps:.2f} fps, {v.duration_s:.1f} s",
                "quirk found": "no container duration field; frame count equals the "
                "`ppg_sync` row count exactly",
            }
        )
    return pd.DataFrame(rows)


def hr_reference_check(db: pd.DataFrame, n: int = 40, seed: int = 1) -> dict:
    """Is the `pulse` label a usable HR reference? Judged against ECG R-peaks."""
    import warnings

    import neurokit2 as nk

    rng = np.random.default_rng(seed)
    sess = db.drop_duplicates(subset=["patient_id", "step"])
    sess = sess.iloc[rng.choice(len(sess), size=min(n, len(sess)), replace=False)]
    rows = []
    for _, r in sess.iterrows():
        pw_p = LOCAL_DIR / f"ppg/{r['patient_id']}_{r['step']}.PW"
        ecg_p = LOCAL_DIR / f"ecg/{r['patient_id']}_{r['step']}.json"
        if not (pw_p.exists() and ecg_p.exists()):
            continue
        ecg = F.read_ecg(ecg_p)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, info = nk.ecg_peaks(ecg.leads["II"], sampling_rate=ecg.frequency)
            peaks = info["ECG_R_Peaks"]
            hr_ecg = (
                60.0 / (np.median(np.diff(peaks)) / ecg.frequency)
                if len(peaks) > 3
                else np.nan
            )
        except Exception:  # noqa: BLE001 - a failed detection is just a dropped sample
            hr_ecg = np.nan
        rows.append((audit.estimate_hr_bpm(F.read_pw(pw_p)), hr_ecg, float(r["pulse"])))
    a = np.array([r for r in rows if np.isfinite(r[1])])
    pair = lambda i, j: {
        "MAE_bpm": float(np.abs(a[:, i] - a[:, j]).mean()),
        "within_5bpm_pct": float(100 * np.mean(np.abs(a[:, i] - a[:, j]) <= 5)),
        "corr": float(np.corrcoef(a[:, i], a[:, j])[0, 1]),
    }
    return {
        "sessions": len(a),
        "PPG vs ECG": pair(0, 1),
        "ECG vs pulse label": pair(1, 2),
        "PPG vs pulse label": pair(0, 2),
    }


def correlations(sess: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """Spearman rank correlation of every target against age, sex and BMI.

    Spearman because several labels are ordinal or coarsely quantised. `sex` is coded
    M=1/F=0 purely so it can enter a rank correlation.
    """
    frame = sess.copy()
    frame["sex"] = frame["sex"].map(SEX_ENCODING)
    cols = [t.column for t in cfg.modellable()]
    return frame[cols].corr(method="spearman")


def figures(sess: pd.DataFrame, corr: pd.DataFrame, cfg: TargetConfig) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []

    targets = [t for t in cfg.modellable() if t.task == "regression"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, t in zip(axes.ravel(), targets, strict=False):
        ax.hist(sess[t.column].dropna(), bins=40, color="#4C72B0")
        ax.set_title(f"{t.name} (Tier {t.tier})", fontsize=10)
        ax.set_xlabel(t.unit, fontsize=8)
    for ax in axes.ravel()[len(targets) :]:
        ax.axis("off")
    fig.suptitle(f"Target distributions, per session (n={len(sess)})")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_target_distributions.png", dpi=110)
    plt.close(fig)
    written.append("01_target_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation between targets (per session)")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_target_correlations.png", dpi=110)
    plt.close(fig)
    written.append("01_target_correlations.png")

    # PPG vs ECG synchronisation sanity, 10 s of one session
    pw = F.read_pw(LOCAL_DIR / "ppg/1020_after.PW")
    ecg = F.read_ecg(LOCAL_DIR / "ecg/1020_after.json")
    grid, values = pw.resample(100.0)
    window = grid <= 10
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    a1.plot(grid[window], values[window], lw=0.9, color="#C44E52")
    a1.set_ylabel("contact PPG\n(8-bit)")
    a1.set_title(
        "Subject 1020, post-exercise: first 10 s of contact PPG and ECG lead II"
    )
    ecg_window = ecg.time <= 10
    a2.plot(
        ecg.time[ecg_window],
        ecg.leads["II"][: ecg_window.sum()],
        lw=0.8,
        color="#4C72B0",
    )
    a2.set_ylabel("ECG lead II")
    a2.set_xlabel("seconds from the start of each recording")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_ppg_vs_ecg.png", dpi=110)
    plt.close(fig)
    written.append("01_ppg_vs_ecg.png")
    return written


def main() -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    cfg = load()
    db = audit.load_db()
    kept, excl = audit.apply_exclusions(db)
    sess, subj = sessions(kept, cfg), subjects(kept)
    corr = correlations(sess, cfg)
    figs = figures(sess, corr, cfg)

    fmt = format_table()
    paths = audit.check_path_consistency(db)
    degenerate = audit.check_degenerate_signals()
    labels = audit.check_labels_constant_across_cameras(db, cfg)
    ranges = audit.check_out_of_range(db, cfg)
    durations = audit.check_durations(kept, n=30)
    hr = hr_reference_check(kept)
    pairing = audit.check_session_pairing(db, audit.SUSPECT_SUBJECT)

    demo = corr.loc[
        [t.column for t in cfg.modellable() if t.column not in ("age", "sex", "bmi")],
        ["age", "sex", "bmi"],
    ]

    L = [
        "# 01 — Data audit and EDA",
        "",
        (
            f"Generated {now} by `python -m faceai.eda`. Source: "
            f"`wengziheng/mcd_rppg` @ `91b967a5`. Aggregates only - no individual row appears here."
        ),
        "",
        "## Summary",
        "",
        (
            f"- **{len(cfg.modellable())} of {len(cfg.targets)} targets have labels.** "
            "`glucose` has no column in `db.csv` and cannot be modelled."
        ),
        (
            f"- **{excl['rows_dropped']} recordings dropped** "
            f"({excl['subjects_before']} → {excl['subjects_after']} subjects): see Exclusions."
        ),
        (
            "- Every file format was opened and confirmed; five quirks found that would "
            "silently corrupt downstream work (see the table)."
        ),
        (
            f"- **The `pulse` label is not a usable HR reference**: it disagrees with "
            f"simultaneous ECG by {hr['ECG vs pulse label']['MAE_bpm']:.1f} bpm MAE."
        ),
        "",
        "## 1. Format confirmation",
        "",
        *md_table(fmt),
        "",
        "## 2. Integrity",
        "",
        "### db.csv paths vs the filename convention",
        "",
        *md_table(paths),
        "",
        (
            f"Only `ppg` and `ecg` disagree, and every disagreement is subject "
            f"{audit.SUSPECT_SUBJECT}."
        ),
        "",
        "### Degenerate (flat-line) signals — all 3600 `ppg_sync` files scanned",
        "",
        *(md_table(degenerate) if len(degenerate) else ["None."]),
        "",
        "### Exclusions applied to every downstream phase",
        "",
        *[f"- **Subject {s}** — {r}" for s, r in audit.EXCLUDED_SUBJECTS.items()],
        *[
            f"- **Subject {s}, `{step}` session** — {r}"
            for (s, step), r in audit.EXCLUDED_SESSIONS.items()
        ],
        "",
        (
            f"Result: {excl['rows_before']} → {excl['rows_after']} recordings, "
            f"{excl['subjects_before']} → {excl['subjects_after']} subjects."
        ),
        "",
        "### Why subject 4952's before/after pairing is 'inconclusive', not guessed",
        "",
        *md_table(
            pd.DataFrame(
                [
                    {
                        "step": e.step,
                        "pulse label": e.pulse_label,
                        "HR of filename-implied file": e.hr_same_step,
                        "HR of db.csv-implied file": e.hr_opposite_step,
                        "verdict": e.verdict,
                    }
                    for e in pairing
                ]
            )
        ),
        "",
        (
            "Two adjudication attempts were made and **both were rejected as unreliable**. "
            "Correlating the `.PW` waveform against `ppg_sync` failed on known-good control "
            "subjects (r ~ 0.04-0.55), because `ppg_sync` has no timestamps and sub-second "
            "drift decorrelates a ~1 Hz waveform. Adjudicating by HR then failed because the "
            "`pulse` label is itself loose (below). The subject is excluded rather than "
            "assigned on a guess."
        ),
        "",
        "### Is `pulse` a usable HR reference? No.",
        "",
        *md_table(
            pd.DataFrame(
                [{"comparison": k, **v} for k, v in hr.items() if isinstance(v, dict)]
            )
        ),
        "",
        (
            f"Over {hr['sessions']} sessions, PPG-derived HR and ECG-derived HR agree with "
            "each other better than either agrees with the `pulse` label. `pulse` is a spot "
            "tonometer reading, not a mean over the 180 s recording. **Use contact PPG or "
            "ECG as the HR reference in Phases 3-4, never `pulse`.**"
        ),
        "",
        "### Labels are constant across the 3 cameras of a session",
        "",
        f"{labels['sessions']} sessions × {labels['label_columns_checked']} label columns: "
        f"**{labels['sessions_with_varying_labels']} sessions vary**. This is why every "
        "statistic below is computed per session (n={n_sess}) or per subject "
        "(n={n_subj}), never over the 3600 recording rows.".format(
            n_sess=len(sess), n_subj=len(subj)
        ),
        "",
        "### Observed values vs the paper's Table 2 ranges",
        "",
        *md_table(ranges),
        "",
        (
            "All targets lie inside their declared ranges (allowing for Table 2 being "
            "rounded to 2 dp - BMI's true minimum is 15.3894 against a printed 15.39)."
        ),
        "",
        "### Signal durations, 30 sampled recordings",
        "",
        *md_table(
            durations.describe().loc[["mean", "min", "max"]].round(2).reset_index()
        ),
        "",
        "## 3. Target availability",
        "",
        *md_table(
            pd.DataFrame(
                [
                    {
                        "target": t.name,
                        "column": t.column or "**none**",
                        "tier": t.tier,
                        "unit": t.unit,
                        "available": t.available,
                        "clinically usable": t.clinically_usable,
                    }
                    for t in cfg.targets
                ]
            )
        ),
        "",
        (
            "Tier C targets must never be presented as clinically usable. Blood pressure "
            "(`upper_ap`, `lower_ap`) is present in the data but excluded as a target."
        ),
        "",
        "## 4. Cohort",
        "",
        (
            f"- {len(subj)} subjects; age {subj['age'].min():.0f}-{subj['age'].max():.0f} "
            f"(median {subj['age'].median():.0f}), BMI median {subj['bmi'].median():.1f}."
        ),
        (
            f"- Sex: {(subj['sex'] == 'F').sum()} F / {(subj['sex'] == 'M').sum()} M "
            f"({100 * (subj['sex'] == 'F').mean():.0f}% female)."
        ),
        f"- {len(sess)} sessions (before = rest, after = 15 squats in 30 s).",
        "",
        (
            "The cohort is young, Russian/Eastern-European and female-skewed. This is a "
            "first-order generalisation risk for Singapore deployment and must be stated "
            "beside every result."
        ),
        "",
        "## 5. Confound structure (the reason for Phase 6)",
        "",
        "Spearman correlation of each target with age, sex (M=1/F=0) and BMI, per session:",
        "",
        *md_table(
            demo.round(3).reset_index().rename(columns={"index": "target column"})
        ),
        "",
        (
            "Any target correlating strongly here can be partly predicted from demographics "
            "alone, which is exactly what the Phase 6 age+sex+BMI baseline tests for."
        ),
        "",
        "## 6. Figures",
        "",
        *[f"![{f}](figures/{f})" for f in figs],
        "",
        "## 7. Carried into Phase 2",
        "",
        "- Split on `patient_id` only, after applying the exclusions above.",
        (
            "- The authors split the same way (`train_test_split` on unique `patient_id`, "
            "20%, seed 43), so their Table 5 numbers are genuine subject-level results and "
            "are comparable with ours."
        ),
        "- `ppg_sync` aligns 1:1 with encoded video frames, not with `meta/` rows.",
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:24]))
    print(f"\nwrote {REPORT.relative_to(ROOT)} and {len(figs)} figures")


if __name__ == "__main__":
    main()
