"""Phase 0 dataset-access probe  ->  reports/00_access.md

    python -m faceai.data.probe

Downloads ONLY README.md, readme.txt and db.csv (~0.94 MB) at the pinned revision.
Everything else comes from file metadata: no video, PPG or ECG is fetched. The report
holds aggregates only (schema, missingness, ranges), never an individual's row.
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import re
import shutil
import sys

import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError

from faceai.data import hub

REPORT = hub.ROOT / "reports" / "00_access.md"
SMALL_FILES = ["README.md", "readme.txt", "db.csv"]
CAM = r"(FullHDwebcam|USBVideo|IriunWebcam)"
PATTERNS = {  # folder -> (filename regex, files expected per subject)
    "video": (re.compile(rf"^(\d+)_{CAM}_(before|after)\.avi$"), 6),
    "meta": (re.compile(rf"^(\d+)_{CAM}_(before|after)\.txt$"), 6),
    "ppg_sync": (re.compile(rf"^(\d+)_{CAM}_(before|after)\.txt$"), 6),
    "ppg": (re.compile(r"^(\d+)_(before|after)\.PW$"), 2),
    "ecg": (re.compile(r"^(\d+)_(before|after)\.json$"), 2),
}
# Group-2 targets -> name fragments to look for. This only SURFACES candidate columns;
# the real column->target mapping is decided by a human in Phase 1 (configs/targets.yaml).
TARGETS = {
    "SpO2": ["saturation", "spo2"],
    "temperature": ["temperature", "temp"],
    "glucose": ["glucose", "sugar"],
    "HbA1c": ["glyc", "hba1c", "hemoglobin"],
    "cholesterol": ["cholesterol"],
    "respiratory rate": ["respiratory", "resp"],
    "arterial stiffness": ["rigidity", "stiffness"],
    "stress (PSM-25)": ["stress", "psm"],
    "age": ["age"],
    "sex": ["gender", "sex"],
    "BMI": ["bmi"],
}
BLOOD_PRESSURE = ["upper_ap", "lower_ap", "systolic", "diastolic", "pressure"]


def matches(column: str, fragments: list[str]) -> bool:
    c = column.lower()
    parts = re.split(r"[^a-z0-9]+", c)
    # short fragments ("age", "bmi", "sex", "psm") must be a whole word: "age" is in "image"
    return any(f in parts if len(f) <= 3 else f in c for f in fragments)


def md_escape(x: object) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def table(header: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(md_escape(c) for c in r) + " |" for r in rows]
    return out


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    api = HfApi()

    # ---- 1. provenance -----------------------------------------------------------
    try:
        api.dataset_info(hub.CANONICAL_REPO_ID)
        canonical = "accessible"
    except Exception as e:  # noqa: BLE001 - the failure itself is the finding
        canonical = (
            f"NOT accessible - {type(e).__name__}: {str(e).splitlines()[0][:160]}"
        )
    pinned = hub.info()
    main_sha = hub.info(revision=None).sha
    mirror = hub.info(hub.MIRROR_REPO_ID, revision=None)
    ours = {s.rfilename: (s.size, hub.file_hash(s)) for s in pinned.siblings or []}
    theirs = {s.rfilename: (s.size, hub.file_hash(s)) for s in mirror.siblings or []}
    mismatched = sorted(
        f for f in set(ours) | set(theirs) if ours.get(f) != theirs.get(f)
    )
    unhashed = [f for f, (_, h) in ours.items() if not h]
    card = pinned.card_data.to_dict() if pinned.card_data else {}
    licence = card.get("license")

    # ---- 2. contents from metadata ---------------------------------------------------
    folders = collections.OrderedDict()
    for f, (size, _) in sorted(ours.items()):
        key = f.split("/")[0] + "/" if "/" in f else f
        n, b = folders.get(key, (0, 0))
        folders[key] = (n + 1, b + (size or 0))
    total = sum(b for _, b in folders.values())

    naming, per_subject = [], collections.defaultdict(collections.Counter)
    video_by = collections.defaultdict(int)
    for folder, (rx, _) in PATTERNS.items():
        names = [f.split("/", 1)[1] for f in ours if f.startswith(folder + "/")]
        bad = [n for n in names if not rx.match(n)]
        for n in names:
            if m := rx.match(n):
                per_subject[m.group(1)][folder] += 1
                if folder == "video":
                    video_by[m.group(2)] += ours[f"video/{n}"][0] or 0
        naming.append(
            [f"`{folder}/`", len(names), len(names) - len(bad), len(bad), bad[:3]]
        )
    subjects = sorted(per_subject, key=int)
    incomplete = {
        s: {
            f: per_subject[s][f]
            for f, (_, want) in PATTERNS.items()
            if per_subject[s][f] != want
        }
        for s in subjects
    }
    incomplete = {s: v for s, v in incomplete.items() if v}

    # ---- 3. the only downloads ---------------------------------------------------------
    try:
        paths = dict(zip(SMALL_FILES, hub.fetch(SMALL_FILES), strict=True))
    except GatedRepoError:
        sys.exit(
            f"\nACCESS DENIED to {hub.REPO_ID}. Open https://huggingface.co/datasets/"
            f"{hub.REPO_ID}, click 'Agree and access', then in YOUR terminal run "
            "`hf auth login` with a read token. No report was written (the public "
            "dataset card may already have downloaded - it is served even when gated)."
        )
    integrity = []
    for name, path in paths.items():
        expected = ours[name][1]
        got = hub.git_blob_id(path)  # these three are small git-stored (non-LFS) files
        integrity.append(
            [f"`{name}`", path.stat().st_size, f"`{got[:12]}`", got == expected]
        )
    if not all(r[-1] for r in integrity):
        sys.exit(
            f"INTEGRITY FAILURE - downloaded bytes do not match the repo: {integrity}"
        )
    readme_txt = paths["readme.txt"].read_text(encoding="utf-8").strip()

    # ---- 4. db.csv: encoding + delimiter are detected and reported, never assumed --------
    raw = paths["db.csv"].read_bytes()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as e:
        sys.exit(
            f"db.csv is not {encoding}: {e}. Inspect it manually - not guessing a codec."
        )
    dialect = csv.Sniffer().sniff("\n".join(text.splitlines()[:5]), delimiters=",;\t")
    df = pd.read_csv(paths["db.csv"], sep=dialect.delimiter, encoding=encoding)

    ids = set(subjects)
    schema, id_columns = [], []
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            rng = f"{s.min():.4g} .. {s.max():.4g}" if s.notna().any() else "-"
            detail = f"{s.nunique()} unique"
        else:
            rng = "-"
            vc = s.astype(str).value_counts()
            detail = (
                f"{len(vc)} unique: "
                + ", ".join(f"{k[:24]} ({v})" for k, v in vc.head(3).items())
                if len(vc) <= 20
                else f"{len(vc)} unique, e.g. `{str(s.dropna().iloc[0])[:60]}`"
            )
        overlap = (
            s.dropna().astype(str).str.replace(r"\.0$", "", regex=True).isin(ids).mean()
        )
        if overlap >= 0.9:
            id_columns.append(c)
        schema.append(
            [f"`{c}`", str(s.dtype), miss, f"{100 * miss / len(df):.1f}%", rng, detail]
        )

    # Does each camera token always correspond to one viewing angle? The dataset card
    # describes the cameras by device, the filenames by token: this settles the mapping.
    camera_view = (
        pd.crosstab(df["camera"], df["view"])
        if {"camera", "view"} <= set(df.columns)
        else pd.DataFrame()
    )

    target_rows = []
    for target, frags in TARGETS.items():
        cols = [c for c in df.columns if matches(c, frags)]
        filled = [f"`{c}`: {int(df[c].notna().sum())}/{len(df)}" for c in cols]
        target_rows.append([target, ", ".join(filled) if cols else "**NONE FOUND**"])
    bp_cols = [c for c in df.columns if matches(c, BLOOD_PRESSURE)]

    # ---- 5. download options -------------------------------------------------------------
    nonvideo = total - folders.get("video/", (0, 0))[1]
    ecg = folders.get("ecg/", (0, 0))[1]
    video_all = folders.get("video/", (0, 0))[1]
    per_subj = video_all / len(subjects)
    free = shutil.disk_usage(hub.ROOT).free
    options = [
        [
            "A",
            "all non-video data, every subject",
            gb(nonvideo),
            f"{gb(nonvideo - ecg)} without ECG",
        ],
        [
            "B",
            "A + frontal `FullHDwebcam` video, every subject",
            gb(nonvideo + video_by["FullHDwebcam"]),
            "",
        ],
        [
            "C",
            "A + all 3 cameras for N=50 subjects",
            gb(nonvideo + per_subj * 50),
            "N=100: " + gb(nonvideo + per_subj * 100),
        ],
        ["D", "everything", gb(total), f"leaves {gb(free - total)} free"],
    ]

    # ---- report ------------------------------------------------------------------------
    L = [
        "# 00 — Dataset access probe",
        "",
        (
            f"Generated {now} by `python -m faceai.data.probe`. Downloaded only "
            f"{', '.join(f'`{f}`' for f in SMALL_FILES)}; everything else below comes "
            "from file metadata."
        ),
        "",
        "## Summary",
        "",
        f"- **Licence:** `{licence}` (from the dataset card).",
        f"- **Source:** `{hub.REPO_ID}` pinned at `{hub.REVISION}` (gated: `{pinned.gated}`).",
        f"- **Full download:** **{gb(total)}** in {len(ours)} files; video alone is {gb(video_all)}.",
        (
            f"- **Subjects:** {len(subjects)} (IDs {subjects[0]}–{subjects[-1]}, not contiguous); "
            f"{len(incomplete)} with an incomplete file set."
        ),
        (
            f"- **db.csv:** {df.shape[0]} rows × {df.shape[1]} columns; encoding `{encoding}`, "
            f"delimiter `{dialect.delimiter!r}`."
        ),
        "- **Targets with no candidate column:** "
        + (", ".join(r[0] for r in target_rows if "NONE" in r[1]) or "none"),
        "",
        "## 1. Provenance",
        "",
        f"- Canonical repo named by the card, `{hub.CANONICAL_REPO_ID}`: **{canonical}**.",
        (
            f"- Pinned `{hub.REPO_ID}` revision `{hub.REVISION}`; its `main` is now `{main_sha}` "
            + (
                "(unchanged)."
                if main_sha == hub.REVISION
                else "(**MOVED since pinning** - review before re-pinning)."
            )
        ),
        (
            f"- Mirror identity vs `{hub.MIRROR_REPO_ID}` @ `{mirror.sha}`: "
            f"{len(ours)} vs {len(theirs)} files, **{len(mismatched)} differing** by size "
            f"or hash, {len(unhashed)} without a hash. "
            + (
                "Byte-identical."
                if not mismatched and not unhashed
                else f"Differences: {mismatched[:5]}"
            )
        ),
        (
            "- Cite the paper, not the mirror: Egorov et al., *Gaze into the Heart*, "
            "ACM MM 2025, DOI 10.1145/3746027.3758255."
        ),
        "",
        "## 2. Contents (metadata)",
        "",
        *table(
            ["folder", "files", "size"],
            [[f"`{k}`", n, gb(b)] for k, (n, b) in folders.items()],
        ),
        "",
        "Video by camera: "
        + ", ".join(f"`{k}` {gb(v)}" for k, v in sorted(video_by.items()))
        + ".",
        "",
        "### Filename patterns (checked per folder)",
        "",
        *table(
            ["folder", "files", "match", "no match", "examples of no match"], naming
        ),
        "",
        (
            "Expected per subject: 6 each of `video/`, `meta/` and `ppg_sync/` "
            "(camera × state); 2 each of `ppg/` and `ecg/` (one per session, **not** per camera)."
        ),
        "",
        (
            f"Incomplete subjects ({len(incomplete)}): {dict(list(incomplete.items())[:10])}"
            if incomplete
            else "Every subject has the complete file set."
        ),
        "",
        "## 3. Downloaded files (integrity-checked against the repo)",
        "",
        *table(["file", "bytes", "git blob id", "matches repo"], integrity),
        "",
        f"`readme.txt` says: `{readme_txt}`",
        "",
        "## 4. `db.csv` schema",
        "",
        (
            f"{df.shape[0]} rows × {df.shape[1]} columns. Subject-ID candidate column(s) "
            "(≥90% of values are subject IDs from the filenames): "
            f"{', '.join(f'`{c}`' for c in id_columns) or '**none found**'}."
        ),
        "",
        *table(["column", "dtype", "missing", "missing %", "range", "values"], schema),
        "",
        "### Camera ↔ viewing angle",
        "",
        *(
            table(
                ["camera \\ view", *camera_view.columns],
                [
                    [i, *r]
                    for i, r in zip(
                        camera_view.index, camera_view.to_numpy(), strict=True
                    )
                ],
            )
            + [
                "",
                (
                    "Each camera maps to exactly one angle."
                    if (camera_view.to_numpy() > 0).sum() == len(camera_view)
                    else "**A camera appears at more than one angle** - do not treat the "
                    "camera token as the view."
                ),
            ]
            if not camera_view.empty
            else ["(no `camera`/`view` columns in db.csv)"]
        ),
        "",
        "## 5. Group-2 target availability (candidate columns only)",
        "",
        (
            "Matching is by name fragment, to surface candidates. **The real mapping is set "
            "by a person in Phase 1** (`configs/targets.yaml`), with units checked against "
            "paper Table 2."
        ),
        "",
        *table(["target", "candidate column(s): non-missing / rows"], target_rows),
        "",
        (
            "Blood-pressure columns present (**not** a Group-2 target; 2023 ESH): "
            f"{', '.join(f'`{c}`' for c in bp_cols) or 'none'}."
        ),
        "",
        "## 6. Download options (for the go / no-go)",
        "",
        f"Free disk now: {gb(free)}. Nothing beyond this report's three small files has been downloaded.",
        "",
        *table(["option", "contents", "size", "note"], options),
        "",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")

    print(f"licence            : {licence}")
    print(f"source             : {hub.REPO_ID}@{hub.REVISION}  (gated={pinned.gated})")
    print(f"canonical          : {canonical}")
    print(
        f"mirror identical   : {not mismatched and not unhashed}  ({len(mismatched)} differing)"
    )
    print(
        f"full download size : {gb(total)}  ({len(ours)} files; video {gb(video_all)})"
    )
    print(f"subjects           : {len(subjects)}  (incomplete: {len(incomplete)})")
    print(
        f"db.csv             : {df.shape[0]} rows x {df.shape[1]} cols, {encoding}, sep={dialect.delimiter!r}"
    )
    print(f"subject-ID columns : {id_columns}")
    print("\n--- db.csv header / dtype / missing / range ---")
    for r in schema:
        print(f"  {r[0]:<34} {r[1]:<9} missing {r[2]:>5} ({r[3]:>6})  range {r[4]}")
    if not camera_view.empty:
        print("\n--- camera x view ---")
        print(camera_view.to_string().replace("\n", "\n  "))
    print("\n--- target candidates ---")
    for t, cols in target_rows:
        print(f"  {t:<20} {cols}")
    print(f"  {'(blood pressure)':<20} {bp_cols}  <- excluded as a target")
    print(f"\nwrote {REPORT.relative_to(hub.ROOT)}")


if __name__ == "__main__":
    main()
