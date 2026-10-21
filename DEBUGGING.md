# Debugging and verification log — FaceAI Group 2

Every defect found so far: how it was caught, the root cause, the fix, and what now stops it
from coming back. Then the standing checks that catch the next one. Facts are from
`CLAUDE.md`, `PROGRESS.md` and the reports; numbers are on the dev folds only.

## 1. Standing checks — how bugs get caught

| Check | What it catches | Where |
|---|---|---|
| Subject-level leakage guard, plus a test that a deliberately leaky split is **rejected** (the guard is tested too) | the same person in training and validation, which inflates every score | `faceai.splits.assert_no_leakage`, `tests/test_no_leakage.py`, pre-commit hook |
| Positive control: age must be predictable from pulse shape | a pipeline that silently learns nothing (LightGBM 2.995 years vs 5.477 for the constant) | `reports/03_cv_robustness.md` |
| Two baselines for every target: a constant guess and age + sex + BMI | "results" that are really the cohort average or demographics | `faceai.cv` |
| Heart rate vs the `pulse` label in the rest → exercise change test | a change test that cannot detect a change the body certainly made (ρ = 0.38) | `faceai.clinical` |
| Synthetic-signal unit tests with known answers | wrong maths: a known heart rate is recovered, flipping a pulse flips its shape features, too-short input gives NaN rather than a number, noise hits the stated SNR exactly, importance finds the only informative input, the implied-MAE formula matches simulated errors | `tests/` (75 tests) |
| Report generators assert their own written claims | a sentence in a report that stops being true after a rerun: the report refuses to build | `faceai.report03b`, `faceai.report03c` |
| Run provenance in every results file | numbers from uncommitted code, a different split, a different dataset file or environment | `faceai.provenance` |
| Full rerun reproduces the outputs | hidden state or randomness: `make phase3` reproduced the Phase 3 report and figures byte for byte, apart from the timestamp (2026-10-13) | `Makefile` |
| ruff lint + format | dead code, unsafe string joins, import errors | pre-commit hook |

## 2. Defects found and fixed

**Data defects** (in the released dataset; none is repaired, each is excluded or handled
in code and the reason recorded):

| # | Symptom | How it was caught | Root cause | Fix | Guard now |
|---|---|---|---|---|---|
| 1 | Subject 4952's signals did not match its session | Phase 1 audit: path-consistency check across `db.csv` columns | `db.csv` points its `ppg`/`ecg` at the opposite session, and all 6 `ppg_sync` files are flat lines | subject excluded before splitting | `tests/test_audit.py` — exclusions are exact, each carries a written reason |
| 2 | Subject 7412's post-exercise session had no signal | Phase 1 audit: scan of all 3600 `ppg_sync` files for degenerate signals | 3 flat `ppg_sync` files | that session excluded; its rest session kept | same |
| 3 | Beat-shape features were implausible | Phase 3: upstroke vs downstroke timing on 40 dev sessions — median rise 0.545 s vs fall 0.140 s, the reverse of a real pulse in every one | the raw `.PW` waveform is **inverted** | features computed on `255 − raw` | `test_physiological_polarity_and_shape`, `test_real_pulses_have_physiological_polarity` |
| 4 | One file failed to parse | Phase 3 feature build | `ppg/8706_after.PW` is write-corrupted: merged timestamp, NUL bytes, truncated last line, clock jumps back 18 s | left out as unreadable, never repaired | `test_only_the_known_corrupt_file_is_unreadable`; 956 sessions asserted |
| 5 | Timestamp parsing crashed partway through some files | format tests | timestamps mix precision (most have microseconds, some fall exactly on the second); pandas infers the format from row 1 | parse with `format="ISO8601"` | `test_timestamps_parse_with_and_without_microseconds` |
| 6 | The `pulse` label disagreed with the ECG | Phase 1 cross-check of three heart-rate sources | `pulse` is off from simultaneous ECG by ~9 bpm MAE, while PPG and ECG agree to ~4.6 | `pulse` is not used as a heart-rate reference | `test_hr_estimator_tracks_ecg_better_than_the_pulse_label` |

**Method and code defects** (ours):

| # | Symptom | How it was caught | Root cause | Fix | Guard now |
|---|---|---|---|---|---|
| 7 | Respiration estimates near 7/min for everyone | checking estimates against the physician count | without a band-pass the spectral peak sits on slow drift | 0.2–0.6 Hz band-pass, 12–30/min search (chosen on dev labels, disclosed). Still worse than the constant: 2.43 vs 1.38 | documented in `CLAUDE.md` |
| 8 | The noise robustness test barely moved heart rate (< 0.4 bpm) | sanity-checking the test itself | white noise puts only ~15 % of its power in the 0.5–8 Hz pulse band | noise confined to the pulse band, at an exact in-band SNR | `test_noise_hits_the_stated_in_band_snr`, `test_noise_is_reproducible` |
| 9 | Tuned hyperparameters sat on the edge of the search grid (Ridge α = 10⁴ for SpO₂ in every fold) | reading the chosen values per fold | the grid was too narrow, so the "optimum" was not one | grids widened | comment in `faceai.cv`; alpha and boosting curves in `faceai.diagnostics` |
| 10 | A code comment said respiration needs ~10 s; the code required 40 s | reviewing the 30-s recording test | stale comment (behaviour was always 40 s) | named constant `RESP_MIN_S` | `test_shortened_recordings_cannot_give_respiration_under_40_s` |
| 11 | Direction agreement in the change test looked worse than it was | reviewing the metric definition | a prediction that did not move was counted as a wrong direction | counted only where both the label and the prediction moved | `test_change_rho_and_direction_ignore_signals_that_did_not_move` |
| 12 | The report generator could not find a column in `gate.csv` | report generation | a stale `gate.csv` left from before the column was added | robustness step rerun | provenance records the commit and inputs behind every results file |
| 13 | Real warnings drowned in thousands of imputer warnings | reading run logs | features that are entirely missing in the camera-timed and short-recording refits | warning filtered only around those refits | `faceai.robustness._variant_fold` |
| 14 | The USB camera appeared as "US" in a figure | reading the rendered figure | a typo in a label map | fixed | figures are read before release |
| 15 | A table in the Phase 3c report rendered broken | proofreading the generated Markdown | an absolute-value bar pair inside a table cell; Markdown reads each bar as a column break | reworded as "absolute mean difference" | proofreading |
| 16 | Crashes printing Unicode on Windows; `make help` failing under `sh` | running on both shells | cp1252 console encoding; parentheses in `echo` | UTF-8 output; parentheses removed | — |

**Claims corrected before they reached a report**: "30-s recordings fail only below
~62 bpm" (failures go up to 99 bpm); a respiratory-rate "state proxy" explanation (tested
and refuted, so not claimed).

## 3. Found, not yet fixed

- `reports/03b_extensions.md`, recording-length table: the header `median |diff| (bpm)`
  has unescaped pipes, so GitHub does not render that table. Fix: rename the column in
  `faceai.report03b` and regenerate.

## 4. Known gaps — checks we do not have yet

- **Label-permutation control**: retrain on shuffled labels; every model should fall to the
  constant. It would catch leakage the split guard cannot (for example via features).
- Adversarial and perturbation testing beyond pulse-band noise (deferred to Phase 8).
- Nothing in the camera rPPG pipeline is tested yet: Phase 4 is blocked on the
  frontal-video download.
