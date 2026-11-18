# rPPG health models: sex, stress and temperature from face video

Deonne · Oct–Nov 2026 · Google Colab + Google Drive

This folder holds the notebooks behind the project report (`OVERALL REPORT`). Pipeline (face video → skin regions → POS/CHROM/green pulse signals → features and waveforms → models) was tested on three targets.

| Target | Outcome | Headline |
|---|---|---|
| **Sex** | Works; packaged for deployment | rPPG-only fusion AUC 0.824 [0.784, 0.858], accuracy 78.1% (majority 66.6%); age + BMI alone 0.698 |
| **Stress** | Negative on MCD-rPPG | MCD (PSM-25): no model beats a constant. UBFC-Phys: stressful vs relaxed task at chance for every input, contact sensors included |
| **Temperature** | Negative on MCD-rPPG | MCD label SD is 0.13 °C; real labels behave like shuffled labels. Only a thermal camera works (+40% vs constant) |

---

## Notebooks

### A. Sex prediction (working result)

| Notebook | What it does | Main output |
|---|---|---|
| `Week9_Sex_Prediction_Improved.ipynb` | 12-channel CNN, random convolutions, pulse-shape features, fusion; recording-day folds; controls | Fusion AUC 0.824; CNN 0.816; age + BMI 0.698 |
| `sex_stack.ipynb` | Tests whether the signal is pulse or face appearance: pulse-removed, phase-scrambled, skin-colour groups; stacked model | Holds within skin-colour groups (0.804); phase scrambling lowers AUC; stacked 0.859 (exploratory) |
| `FINAL_EXTRACTION.ipynb` | Writes the shared library `sexpred_lib.py`; extracts missing ROI traces; builds inputs for full, 120 s, 60 s and 30 s clips; checks them against Week 3 and Week 9 | `sex_deploy/inputs/` |
| `sex_rPPG_only.ipynb` | **Package A**: Week 9 fusion (CNN + random convolutions + gradient boosting), quality gate, calibration | `sex_deploy/packages/<A>/` |
| `sex_rPPG_plus_appearance.ipynb` | **Package B**: Package A + skin-colour model. Reuses A's cross-validation cache | `sex_deploy/packages/<B>/` |

Each package folder contains the models (5 CNN seeds), calibrators, `config.json`, `predict.py` (command line), `MODEL_CARD.md` and `deployment_tests.csv`.

```python
from sexpred_lib import SexPredictor
p = SexPredictor("sex_deploy/packages/<package>")
p.predict(before_video, after_video, before_meta, after_meta)
# -> probability_male, prediction; None if no session passes the quality gate
```

One video (before or after) and clips of 30–120 s are supported. The threshold is `balanced` (default) or `accuracy`.

### B. Stress: evidence that MCD-rPPG cannot support it

| Notebook | What it shows |
|---|---|
| `Week2_Baselines_Temperature_Stress.ipynb` | Baseline ladder. PSM-25 is a trait (0.7% of patients change score between sessions); demographics and even measured vitals do not beat the constant (MAE 1.117) |
| `Week3_Stress_rPPG_600_FrontCam_v2.ipynb` | rPPG extraction for all 600 patients (**also the source of the ROI traces used by the sex packages**). Sections 12–14: within-person features and 1D-CNN; all at or below the constant; contact-PPG ceiling fails the same way |
| `Week5_Stress_rPPG_Audio_Transformer.ipynb` | MCD videos have no audio. Transformer on rPPG and contact PPG collapses to a constant. Only recording date/time helps (+4.93% over demographics), which points to batch structure, not physiology |
| `Week7_rPPG_QualityGate_Validation.ipynb` + Week 7 simulation notebook | The pipeline finds planted signals and finds nothing in shuffled labels, so the method works. Real label correlations (0.022) are far below the detectable level (≈ 0.45–0.6). Quality gate fixes HR error but not stress |

### C. Stress on a within-person dataset

| Notebook | What it shows |
|---|---|
| `Week5b_UBFC-Phys_Stress_v2.ipynb` | 56 subjects. Part 1 (sections 0–7): first extraction and models. Part 2 (sections 8–12): improved rPPG (motion regressed out, window gate) and nested leave-subjects-out tests. Rest vs task is detected mainly through face motion (0.995); cleaned rPPG physiology 0.823 ≈ contact BVP 0.887. Stressful vs relaxed: chance for all inputs |

### D. Temperature: evidence that MCD-rPPG cannot support it

| Notebook | What it shows |
|---|---|
| `Week2_Baselines_Temperature_Stress.ipynb` | Constant MAE 0.093 °C; nothing beats it |
| `Week4_PhysioNet_MCD_Temperature.ipynb` | MCD label SD 0.133 °C (none above 37.5 °C). Temperature can explain at most ~0.8% of HR variance. Thermal ceiling on PhysioNet: +40%, fever AUC 0.986 |
| `Week5_Temperature_rPPG_Audio_Transformer.ipynb` | Transformer predicts HR well (+68%, positive control) but temperature like shuffled labels (−0.8% vs −1.7%); contact-PPG ceiling also fails |
| `Week7_rPPG_QualityGate_Validation.ipynb` | Quality-gated rPPG still −2.8%. Simulation: label SD must reach ~0.8 °C before an effect is visible |
| `Week6_v3_Toadstool_rPPG_SkinTemperature.ipynb` (optional) | Outside MCD: on Toadstool and PhysioNet wearable, pulse does not predict wrist temperature beyond a time trend |

---

## How to run

1. **Environment:** Google Colab with a GPU (CNN and transformer cells). Mount Drive in the first cell.
2. **Data on Drive:**
   - `MyDrive/MCD_rPPG/`: MCD-rPPG `db.csv`, `raw/video`, meta files, contact PPG.
   - `MyDrive/UBFC-Phys/`: videos, BVP and EDA for s1–s56.
   - PhysioNet face-oral-temp and wearable-device (downloaded inside Week 4 and Week 6).
3. **Order for the sex packages:** `Week3_Stress_rPPG_600_FrontCam_v2` (ROI traces) → `Week9` (signal cache) → `FINAL_EXTRACTION` → `sex_rPPG_only` → `sex_rPPG_plus_appearance` → `sex_stack`.
4. Every other notebook runs on its own once the Week 3 features (`MCD_rPPG/week3_stress_rppg600/features/`) exist.

| Notebook | Output folder on Drive |
|---|---|
| Week 3 | `MCD_rPPG/week3_stress_rppg600/` |
| Week 4 | `MCD_rPPG/week4_outputs/` |
| Week 5 stress | `MCD_rPPG/week5_stress_multimodal/results/` |
| Week 5 temperature | `MCD_rPPG/week5_outputs/` |
| Week 7 simulation | `MCD_rPPG/SIMULATED_populated_data_NOT_REAL_MCD/` (simulated, not real data) |
| Week 5b UBFC-Phys | `UBFC-Phys/processed/results/` |
| Sex packages | `sex_deploy/` |

## Evaluation rules used everywhere

- Folds are grouped by person (repeated 5-fold CV; fixed test split `random_state=43` on MCD; leave-subjects-out on UBFC-Phys).
- Every result is compared with a constant (or chance) on the same rows, with 95% bootstrap intervals by person.
- Contact PPG is the ceiling for rPPG; a heart-rate positive control and shuffled labels are run alongside.

## Known limitations

- **Sex:** The Week 10 stacked model used a contact-HR mask, which a camera cannot provide.
- **Licences:** Toadstool is "research and educational use"; PhysioNet sets are CC0 / ODC-By.
