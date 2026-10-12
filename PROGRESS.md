# PROGRESS — FaceAI Group 2

Running checklist (brief rule 5). Each phase ends with a Definition-of-Done gate: run the
checks, show the output, and move on only once they pass **and** the lead approves.

## Phase 0 — Environment + hardware + dataset access (no large download)
- [x] `.gitignore`: `env/` records are tracked; `artifacts/`, `mlruns/`, `*.db`, `data/` are ignored
- [x] Group 2 packages installed with the verified core frozen — 92 new, **0 changed**; `pip check` clean
- [x] `make` 4.4.1 (conda-forge); telemetry off (conda env vars + `faceai/__init__.py`)
- [x] `env/lockfile.txt`; `requirements.txt` / `environment.yml` / `pyproject.toml` regenerated from the live env
- [x] `faceai.hardware`, `faceai.data.hub`, `faceai.data.probe`, `faceai.lockfile`; `Makefile`; `CLAUDE.md`
- [x] HF gate accepted + `hf auth login` — **Joshua** (a token is never handled by the agent)
- [x] `env/hardware.json` recorded **on AC power** — `cuda` / RTX 5050 / sm_120, no warnings
- [x] `reports/00_access.md` — cc-by-4.0, 135.18 GB, `db.csv` 3600×25, **glucose absent**
- [x] DoD run + local commit (no remote)
- [x] **GATE — download plan approved by Joshua on 2026-09-23: option A, non-video only.**
      9,603 files / 5.33 GB (`ppg/`, `ppg_sync/`, `ecg/`, `meta/`, `db.csv`) for all 600 subjects.
      **No video downloaded.** Option B (frontal video, +58.2 GB) is deferred until Phase 3 shows
      whether any target carries signal even from clean contact PPG.

## Phase 1 — Data audit + EDA ✅
DoD: `reports/01_eda.md` with the format-confirmation table, `configs/targets.yaml`, and the
target × demographics correlation matrix. States which of the 11 targets actually have labels.

- [x] Option A downloaded and verified: 9,603 files, 0 missing, 0 unexpected, no video
- [x] All 5 formats opened and confirmed (`src/faceai/data/formats.py`) + 3 sample videos
- [x] `configs/targets.yaml` + `faceai.targets` — **10 of 11 targets have labels; glucose has none**
- [x] Integrity audit (`faceai.data.audit`): paths, ranges, label-constancy, flat-line scan
- [x] **Exclusions**: subject 4952 entirely; subject 7412's `after` session → 599 subjects
- [x] `reports/01_eda.md` + 3 figures; 23 tests green; `make test | audit | eda`
- [ ] Phase 2 gate: plan approval before any splitting code

## Phase 2 — Subject-level splits + leakage tests (BLOCKING for all modelling) ✅
DoD: `pytest tests/test_no_leakage.py` green; `artifacts/splits.parquet` written; a table of
#subjects / #recordings per split and per camera/state.

- [x] `src/faceai/splits.py` — seed 42, 20% held-out test, 5-fold `StratifiedGroupKFold`
      grouped on `patient_id`, stratified on sex × (age≥25)
- [x] **test 120 subjects / 720 recordings; dev 479 / 2871**; folds 95–96 subjects,
      63–64 F / 31–33 M, 8–9 subjects aged 25+ in every fold
- [x] `tests/test_no_leakage.py` — **12 tests green**, including two that confirm the guard
      REJECTS deliberately leaked splits, and determinism vs the committed file
- [x] `artifacts/splits.parquet` + `splits.json` committed (gitignore exception); provenance
      records seed, sklearn version, exclusions, db.csv blob id and an assignment SHA-256
- [x] `.pre-commit-config.yaml` — local hooks (ruff + the leakage test), installed
- [x] `reports/02_splits.md`
- [x] Phase 3 gate: plan approved by Joshua on 2026-10-12 ("Resume"). The optional leakage
      demonstration (a deliberately random split) was NOT approved and was not run.

**The held-out test set is SEALED until Phase 10.** No phase before it may read those rows.

## Phase 3 — Contact-PPG ceiling models ✅
DoD: `reports/03_ceiling.md` (target × [mean | demo-only | ours | paper] with CIs) plus an honest
paragraph on which targets carry real signal even from clean contact PPG.

Delivered as `reports/03_cv_robustness.md` (the DoD table is its section 3), extended for the
2026-10-12 meeting with overfitting diagnostics, robustness tests and a pipeline diagram.

- [x] `faceai.features` — 17 contact-PPG features per dev session (957; 956 readable), plus
      copies at 6 in-band SNR levels (+20 … −10 dB); `make features`
- [x] `faceai.cv` — 5-fold CV on the committed folds: mean / median / demographics baselines,
      Ridge, Lasso, LightGBM tuned by inner GroupKFold; subject-cluster bootstrap CIs; `make cv`
- [x] `faceai.diagnostics` — regularisation paths, boosting curve, learning curves
- [x] `faceai.robustness` — noise injection, rest→exercise shift, subgroups, stability
- [x] `faceai.report03` — report + 9 figures, every number read from saved artifacts
- [x] `tests/test_features.py`, `tests/test_cv.py` — 56 tests green in total
- [x] Verdict (strict brief rule): only the age positive control beats both baselines.
      Arterial stiffness and respiratory rate pass the paired test only. HbA1c, cholesterol,
      SpO₂, stress and temperature do not beat a constant.
- Not done: adversarial testing (deferred to Phase 8 deep models); the APG c/d/e waves and
  aging index (unvalidated at 100 Hz / 8-bit; Phase 5).

## Phase 4 — rPPG extraction + signal quality (cached, resumable)
DoD: `reports/04_rppg.md` with rPPG-vs-contact HR MAE by camera/state and SQI distributions; a
resumable cache under `artifacts/rppg/` that re-runs without recomputing finished files.

## Phase 5 — Feature bank
DoD: `artifacts/features.parquet` + data dictionary in `reports/05_features.md`; unit tests that
features are finite and in sane ranges; no subject leakage in any aggregation.

## Phase 6 — Baselines + confound analysis
DoD: `reports/06_confounds.md` verdict table; explicit statement of which targets pass.

## Phase 7 — Tabular models
DoD: `reports/07_tabular.md` leaderboard vs mean / demographics / paper baselines, with CIs.

## Phase 8 — Deep / foundation-model track
DoD: `reports/08_deep.md` comparing deep vs tabular vs baselines; whether the foundation
embedding helps any target beyond demographics.

## Phase 9 — Multi-task model
DoD: `reports/09_multitask.md` comparison table (multi-task vs best single-task per target).

## Phase 10 — Evaluation suite
DoD: `reports/10_eval/` with all plots and tables, and a machine-readable `results.json`.

## Phase 11 — Results tiering + honest synthesis
DoD: `reports/11_summary.md` decision table.

## Phase 12 — Privacy-by-design demo (Gradio)
DoD: app runs locally; a test asserts no video file persists after a request; screenshots saved.

## Phase 13 — Model cards, dataset card, report scaffold, team README
DoD: `MODEL_CARDS/`, `DATASET_CARD.md`, `report/`, `README.md`, `CONTRIBUTING.md` all present.
