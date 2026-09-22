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
- [ ] **GATE — download plan go / no-go (Joshua).** No video before this.

## Phase 1 — Data audit + EDA
DoD: `reports/01_eda.md` with the format-confirmation table, `configs/targets.yaml`, and the
target × demographics correlation matrix. States which of the 11 targets actually have labels.

## Phase 2 — Subject-level splits + leakage tests (BLOCKING for all modelling)
DoD: `pytest tests/test_no_leakage.py` green; `artifacts/splits.parquet` written; a table of
#subjects / #recordings per split and per camera/state.

## Phase 3 — Contact-PPG ceiling models
DoD: `reports/03_ceiling.md` (target × [mean | demo-only | ours | paper] with CIs) plus an honest
paragraph on which targets carry real signal even from clean contact PPG.

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
