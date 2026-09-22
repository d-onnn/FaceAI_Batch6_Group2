# 00 — Dataset access probe

Generated 2026-09-23T01:40:51+08:00 by `python -m faceai.data.probe`. Downloaded only `README.md`, `readme.txt`, `db.csv`; everything else below comes from file metadata.

## Summary

- **Licence:** `cc-by-4.0` (from the dataset card).
- **Source:** `wengziheng/mcd_rppg` pinned at `91b967a5f2bab735d35f81722ff95e629358753b` (gated: `auto`).
- **Full download:** **135.18 GB** in 13204 files; video alone is 129.84 GB.
- **Subjects:** 600 (IDs 1020–9998, not contiguous); 0 with an incomplete file set.
- **db.csv:** 3600 rows × 25 columns; encoding `utf-8`, delimiter `','`.
- **Targets with no candidate column:** glucose

## 1. Provenance

- Canonical repo named by the card, `kyegorov/mcd_rppg`: **NOT accessible - RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-6ab2bda3-37c037cf08c8cd4e585c16c9;3fa6b106-04b2-41bf-8105-caa2e68350fc)**.
- Pinned `wengziheng/mcd_rppg` revision `91b967a5f2bab735d35f81722ff95e629358753b`; its `main` is now `91b967a5f2bab735d35f81722ff95e629358753b` (unchanged).
- Mirror identity vs `Shubh782004/mcd_rppg` @ `fff4831302ece779bc45c9833b09da9750a46a72`: 13204 vs 13204 files, **0 differing** by size or hash, 0 without a hash. Byte-identical.
- Cite the paper, not the mirror: Egorov et al., *Gaze into the Heart*, ACM MM 2025, DOI 10.1145/3746027.3758255.

## 2. Contents (metadata)

| folder | files | size |
|---|---|---|
| `.gitattributes` | 1 | 0.00 GB |
| `README.md` | 1 | 0.00 GB |
| `db.csv` | 1 | 0.00 GB |
| `ecg/` | 1200 | 3.77 GB |
| `meta/` | 3600 | 0.61 GB |
| `ppg/` | 1200 | 0.73 GB |
| `ppg_sync/` | 3600 | 0.22 GB |
| `readme.txt` | 1 | 0.00 GB |
| `video/` | 3600 | 129.84 GB |

Video by camera: `FullHDwebcam` 58.18 GB, `IriunWebcam` 23.36 GB, `USBVideo` 48.31 GB.

### Filename patterns (checked per folder)

| folder | files | match | no match | examples of no match |
|---|---|---|---|---|
| `video/` | 3600 | 3600 | 0 | [] |
| `meta/` | 3600 | 3600 | 0 | [] |
| `ppg_sync/` | 3600 | 3600 | 0 | [] |
| `ppg/` | 1200 | 1200 | 0 | [] |
| `ecg/` | 1200 | 1200 | 0 | [] |

Expected per subject: 6 each of `video/`, `meta/` and `ppg_sync/` (camera × state); 2 each of `ppg/` and `ecg/` (one per session, **not** per camera).

Every subject has the complete file set.

## 3. Downloaded files (integrity-checked against the repo)

| file | bytes | git blob id | matches repo |
|---|---|---|---|
| `README.md` | 5840 | `b30c8da51a1c` | True |
| `readme.txt` | 32 | `9b6a80266a1a` | True |
| `db.csv` | 931811 | `b6db00bc074a` | True |

`readme.txt` says: `This is repository for  dataset`

## 4. `db.csv` schema

3600 rows × 25 columns. Subject-ID candidate column(s) (≥90% of values are subject IDs from the filenames): `patient_id`.

| column | dtype | missing | missing % | range | values |
|---|---|---|---|---|---|
| `patient_id` | int64 | 0 | 0.0% | 1020 .. 9998 | 600 unique |
| `weight` | float64 | 0 | 0.0% | 43 .. 168 | 71 unique |
| `height` | float64 | 0 | 0.0% | 147 .. 201 | 46 unique |
| `bmi` | float64 | 0 | 0.0% | 15.39 .. 47.03 | 436 unique |
| `age` | float64 | 0 | 0.0% | 18 .. 83 | 40 unique |
| `sex` | object | 0 | 0.0% | - | 2 unique: F (2394), M (1206) |
| `upper_ap` | float64 | 0 | 0.0% | 80 .. 202 | 97 unique |
| `lower_ap` | float64 | 0 | 0.0% | 50 .. 108 | 53 unique |
| `saturation` | float64 | 0 | 0.0% | 86 .. 99 | 11 unique |
| `temperature` | float64 | 0 | 0.0% | 36 .. 37.5 | 12 unique |
| `hemoglobin` | float64 | 0 | 0.0% | 8.1 .. 17.3 | 77 unique |
| `glycated_hemoglobin` | float64 | 0 | 0.0% | 3.4 .. 13.02 | 189 unique |
| `cholesterol` | float64 | 0 | 0.0% | 0.9 .. 8 | 133 unique |
| `respiratory` | float64 | 0 | 0.0% | 15 .. 24 | 10 unique |
| `rigidity` | float64 | 0 | 0.0% | 1.75 .. 34.02 | 670 unique |
| `pulse` | float64 | 0 | 0.0% | 49 .. 153 | 95 unique |
| `stress` | float64 | 0 | 0.0% | 1 .. 7.52 | 107 unique |
| `step` | object | 0 | 0.0% | - | 2 unique: after (1800), before (1800) |
| `camera` | object | 0 | 0.0% | - | 3 unique: FullHDwebcam (1200), USBVideo (1200), IriunWebcam (1200) |
| `view` | object | 0 | 0.0% | - | 3 unique: front (1200), left (1200), right (1200) |
| `ecg` | object | 0 | 0.0% | - | 1200 unique, e.g. `ecg/1020_after.json` |
| `ppg` | object | 0 | 0.0% | - | 1200 unique, e.g. `ppg/1020_after.PW` |
| `video` | object | 0 | 0.0% | - | 3600 unique, e.g. `video/1020_FullHDwebcam_after.avi` |
| `meta` | object | 0 | 0.0% | - | 3600 unique, e.g. `meta/1020_FullHDwebcam_after.txt` |
| `ppg_sync` | object | 0 | 0.0% | - | 3600 unique, e.g. `ppg_sync/1020_FullHDwebcam_after.txt` |

### Camera ↔ viewing angle

| camera \ view | front | left | right |
|---|---|---|---|
| FullHDwebcam | 1200 | 0 | 0 |
| IriunWebcam | 0 | 0 | 1200 |
| USBVideo | 0 | 1200 | 0 |

Each camera maps to exactly one angle.

## 5. Group-2 target availability (candidate columns only)

Matching is by name fragment, to surface candidates. **The real mapping is set by a person in Phase 1** (`configs/targets.yaml`), with units checked against paper Table 2.

| target | candidate column(s): non-missing / rows |
|---|---|
| SpO2 | `saturation`: 3600/3600 |
| temperature | `temperature`: 3600/3600 |
| glucose | **NONE FOUND** |
| HbA1c | `hemoglobin`: 3600/3600, `glycated_hemoglobin`: 3600/3600 |
| cholesterol | `cholesterol`: 3600/3600 |
| respiratory rate | `respiratory`: 3600/3600 |
| arterial stiffness | `rigidity`: 3600/3600 |
| stress (PSM-25) | `stress`: 3600/3600 |
| age | `age`: 3600/3600 |
| sex | `sex`: 3600/3600 |
| BMI | `bmi`: 3600/3600 |

Blood-pressure columns present (**not** a Group-2 target; 2023 ESH): `upper_ap`, `lower_ap`.

## 6. Download options (for the go / no-go)

Free disk now: 246.83 GB. Nothing beyond this report's three small files has been downloaded.

| option | contents | size | note |
|---|---|---|---|
| A | all non-video data, every subject | 5.33 GB | 1.56 GB without ECG |
| B | A + frontal `FullHDwebcam` video, every subject | 63.51 GB |  |
| C | A + all 3 cameras for N=50 subjects | 16.15 GB | N=100: 26.97 GB |
| D | everything | 135.18 GB | leaves 111.65 GB free |
