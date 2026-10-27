"""Phase 3e (c) - which videos to download first: a space-filling design against random choice.

    python -m faceai.design         # or: make design

Phase 4 (camera rPPG) is blocked on downloading 58.18 GB of frontal video (gate B). Maaz
Ahmad's surrogate-modelling talk (MoSPSyDe group, NTU, 7 Oct 2026) makes two points that bear
on it: training data must be "system-informative", covering the input space (his p. 5), and
when data are expensive, choose them by design - adaptive sampling that balances
exploration and exploitation (his p. 15). Here the expensive item is the video; every
subject's labels are already on disk.

Simulation on the dev folds (the held-out test set is not read). For each outer fold, a
subset of n training subjects is chosen by:
  - random:          uniformly at random (20 draws);
  - stratified:      random within sex x age band, as the splits are (20 draws);
  - space-filling:   Kennard-Stone maximin design (Technometrics 1969;11:137-148,
                     doi:10.1080/00401706.1969.10490666) over the subjects' standardised
                     labels and sex - each pick is the subject farthest from those chosen;
  - hybrid:          half space-filling, half random from the rest (20 draws).
A Ridge model per target is trained on the chosen subjects' contact-PPG sessions (inner
GroupKFold tuning as in Phase 3) and scored on the untouched fold: MAE, MAE in the lowest and
highest 10 % of each label, and the share of test sessions whose label lies inside the
chosen subjects' range.

Also writes a proposed first download: the Kennard-Stone order over all dev subjects. The
committed list holds only rank, patient_id, fold and cumulative GB - no health labels.
Outputs: `artifacts/cv/design.json` and `reports/03e_pilot_download_order.csv`.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from faceai.bias import TAIL
from faceai.cv import OUT_DIR, TARGETS, build_model, inputs, session_table
from faceai.data.hub import ROOT
from faceai.provenance import record
from faceai.splits import N_FOLDS, OLDER_THAN, SEED

STRATEGIES = ["random", "stratified", "space-filling", "hybrid"]
SIZES = [40, 80, 160]
REPEATS = 20
FRONTAL_GB = (
    58.18  # FullHDwebcam total for 600 subjects x 2 sessions (CLAUDE.md, Phase 0)
)
FRONTAL_VIDEOS = 1200
PILOT_LIST = ROOT / "reports" / "03e_pilot_download_order.csv"


def subject_table(t: pd.DataFrame) -> pd.DataFrame:
    """One row per dev subject: fold, sex and each label averaged over its sessions."""
    s = t.groupby("patient_id").agg(
        fold=("fold", "first"),
        sex_m=("sex_m", "first"),
        **{c: (c, "mean") for c in TARGETS},
    )
    return s.reset_index()


def design_space(s: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    cols = [*TARGETS, "sex_m"]
    return ((s[cols] - ref[cols].mean()) / ref[cols].std(ddof=0)).to_numpy()


def kennard_stone(Z: np.ndarray, n: int) -> list[int]:
    """Kennard & Stone (1969): the two most distant points, then repeatedly the point whose
    distance to its nearest chosen point is largest."""
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    chosen = [int(i), int(j)]
    nearest = np.minimum(d[i], d[j])
    nearest[chosen] = -1
    while len(chosen) < min(n, len(Z)):
        k = int(np.argmax(nearest))
        chosen.append(k)
        nearest = np.minimum(nearest, d[k])
        nearest[chosen] = -1
    return chosen[:n]


def select(
    strategy: str, pool: pd.DataFrame, n: int, rng: np.random.Generator
) -> np.ndarray:
    ids = pool["patient_id"].to_numpy()
    if strategy == "random":
        return rng.choice(ids, n, replace=False)
    if strategy == "stratified":
        stratum = pool["sex_m"].astype(int).astype(str) + (
            pool["age"] >= OLDER_THAN
        ).astype(int).astype(str)
        picks = []
        for _, g in pool.groupby(stratum):
            k = round(n * len(g) / len(pool))
            picks += list(
                rng.choice(g["patient_id"].to_numpy(), min(k, len(g)), replace=False)
            )
        rest = np.setdiff1d(ids, picks)
        while len(picks) < n:  # rounding shortfall
            picks.append(rng.choice(np.setdiff1d(rest, picks)))
        return np.array(picks[:n])
    Z = design_space(pool, pool)
    if strategy == "space-filling":
        return ids[kennard_stone(Z, n)]
    if strategy == "hybrid":
        ks = ids[kennard_stone(Z, n // 2)]
        return np.concatenate(
            [ks, rng.choice(np.setdiff1d(ids, ks), n - len(ks), replace=False)]
        )
    raise KeyError(strategy)


def score(t: pd.DataFrame, chosen: np.ndarray, fold: int, cuts: dict) -> dict:
    tr = t[t["patient_id"].isin(chosen)]
    va = t[t["fold"] == fold]
    out = {}
    for target in TARGETS:
        est, uses_ppg = build_model("Ridge (L2)")
        cols = inputs(t, target, uses_ppg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est.fit(tr[cols], tr[target], groups=tr["patient_id"])
        y = va[target].to_numpy()
        e = np.abs(est.predict(va[cols]) - y)
        lo, hi = cuts[target]
        out[target] = {
            "mae": float(e.mean()),
            "tail_mae": float(e[(y <= lo) | (y >= hi)].mean()),
            "range_coverage": float(
                np.mean((y >= tr[target].min()) & (y <= tr[target].max()))
            ),
        }
    return out


def _run_one(t, s, strategy, n, fold, rep, cuts):
    rng = np.random.default_rng([SEED, fold, n, rep, STRATEGIES.index(strategy)])
    pool = s[s["fold"] != fold]
    chosen = select(strategy, pool, n, rng)
    older = int((pool.set_index("patient_id").loc[chosen, "age"] >= OLDER_THAN).sum())
    return {
        "strategy": strategy,
        "n": n,
        "fold": fold,
        "rep": rep,
        "older_chosen": older,
        "scores": score(t, chosen, fold, cuts),
    }


def simulate(t: pd.DataFrame, s: pd.DataFrame, n_jobs: int = 12) -> list[dict]:
    cuts = {c: tuple(np.quantile(t[c], [TAIL, 1 - TAIL])) for c in TARGETS}
    jobs = [
        (st, n, k, r)
        for st in STRATEGIES
        for n in SIZES
        for k in range(N_FOLDS)
        for r in range(1 if st == "space-filling" else REPEATS)
    ]
    return Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(t, s, st, n, k, r, cuts) for st, n, k, r in jobs
    )


def summarise(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        for target, v in r["scores"].items():
            rows.append(
                {
                    "strategy": r["strategy"],
                    "n": r["n"],
                    "fold": r["fold"],
                    "rep": r["rep"],
                    "older_chosen": r["older_chosen"],
                    "target": target,
                    **v,
                }
            )
    df = pd.DataFrame(rows)
    # Fold-average each draw first, then describe the spread over draws.
    per_draw = df.groupby(["strategy", "n", "rep", "target"], as_index=False)[
        ["mae", "tail_mae", "range_coverage", "older_chosen"]
    ].mean()
    out = []
    for (st, n, target), g in per_draw.groupby(["strategy", "n", "target"]):
        ks = per_draw[
            (per_draw["strategy"] == "space-filling")
            & (per_draw["n"] == n)
            & (per_draw["target"] == target)
        ].iloc[0]
        out.append(
            {
                "strategy": st,
                "n": int(n),
                "target": target,
                "draws": len(g),
                **{
                    f"{m}_mean": float(g[m].mean())
                    for m in ("mae", "tail_mae", "range_coverage", "older_chosen")
                },
                **{
                    f"{m}_sd": float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
                    for m in ("mae", "tail_mae")
                },
                # share of this strategy's draws that the space-filling design beats
                "space_filling_beats_share_mae": float(np.mean(ks["mae"] < g["mae"])),
                "space_filling_beats_share_tail": float(
                    np.mean(ks["tail_mae"] < g["tail_mae"])
                ),
            }
        )
    return out


def pilot(s: pd.DataFrame) -> pd.DataFrame:
    """Every dev subject in Kennard-Stone order, with the frontal-video GB downloaded so far."""
    p = s.iloc[kennard_stone(design_space(s, s), len(s))].reset_index(drop=True)
    p.insert(0, "rank", np.arange(1, len(p) + 1))
    p["cumulative_gb"] = np.round(p["rank"] * 2 * FRONTAL_GB / FRONTAL_VIDEOS, 2)
    return p[["rank", "patient_id", "fold", "sex_m", *TARGETS, "cumulative_gb"]]


def main() -> None:
    t = session_table()
    s = subject_table(t)
    runs = simulate(t, s)
    p = pilot(s)
    PILOT_LIST.parent.mkdir(parents=True, exist_ok=True)
    # Committed file: ids and order only - no health labels (CLAUDE.md: no patient data in git).
    p[["rank", "patient_id", "fold", "cumulative_gb"]].to_csv(PILOT_LIST, index=False)
    first = p.head(48)
    rng = np.random.default_rng(SEED)
    rand_older = [
        int((s.loc[rng.choice(len(s), 48, replace=False), "age"] >= OLDER_THAN).sum())
        for _ in range(1000)
    ]
    out = {
        "summary": summarise(runs),
        "pilot_first_48": {
            "gb": float(first["cumulative_gb"].iloc[-1]),
            "older": int((first["age"] >= OLDER_THAN).sum()),
            "hba1c_ge_6_5": int((first["hba1c"] >= 6.5).sum()),
            "random_48_older_mean": float(np.mean(rand_older)),
            "random_48_older_p95": float(np.percentile(rand_older, 95)),
            "dev_subjects": len(s),
            "dev_older": int((s["age"] >= OLDER_THAN).sum()),
            "dev_hba1c_ge_6_5": int((s["hba1c"] >= 6.5).sum()),
        },
        "provenance": record([OUT_DIR.parent / "splits.parquet"]),
    }
    (OUT_DIR / "design.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    for r in out["summary"]:
        if r["target"] in ("age", "hba1c", "arterial_stiffness"):
            print(
                f"{r['target']:>18} n {r['n']:>3} {r['strategy']:<13} MAE {r['mae_mean']:.3f}±{r['mae_sd']:.3f} "
                f"tail {r['tail_mae_mean']:.3f}±{r['tail_mae_sd']:.3f} cover {r['range_coverage_mean']:.3f} "
                f"older {r['older_chosen_mean']:.1f}  KS beats {r['space_filling_beats_share_mae']:.2f}/{r['space_filling_beats_share_tail']:.2f}"
            )
    print(out["pilot_first_48"])


if __name__ == "__main__":
    main()
