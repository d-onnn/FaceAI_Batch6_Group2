# 02 — Subject-level splits

Generated 2026-09-23T02:30:42+08:00 by `python -m faceai.splits`.

**Splitting is by `patient_id` only.** All recordings of a subject sit in the same partition. The held-out test set is sealed until Phase 10 — no phase before it may read those rows.

## Provenance

- `generated`: 2026-09-23T02:30:42+08:00
- `seed`: 42
- `n_folds`: 5
- `test_fraction`: 0.2
- `stratified_by`: sex x (age >= 25)
- `sklearn`: 1.9.0
- `excluded_subjects`: [4952]
- `excluded_sessions`: [[7412, 'after']]
- `db_csv_git_blob_id`: b6db00bc074a4a8e717319315b6253dca9f287a6
- `n_subjects`: 599
- `n_recordings`: 3591
- `assignment_sha256`: 08f6fb63072e0fc76febbd6a333661a74e5bf757a96fe62a23a76bcf3d9fceed

## Subjects and recordings per split

| split | subjects | recordings |
|---|---|---|
| dev | 479 | 2871 |
| test | 120 | 720 |

### Per camera

| split | FullHDwebcam | IriunWebcam | USBVideo |
|---|---|---|---|
| dev | 957 | 957 | 957 |
| test | 240 | 240 | 240 |

### Per state

| split | after | before |
|---|---|---|
| dev | 1434 | 1437 |
| test | 360 | 360 |

## Cross-validation folds (dev only)

| fold | subjects | female | male | age_ge_25 | age_median | age_max | recordings |
|---|---|---|---|---|---|---|---|
| 0 | 96 | 64 | 32 | 9 | 20.000 | 74.000 | 576 |
| 1 | 96 | 64 | 32 | 9 | 20.000 | 73.000 | 576 |
| 2 | 96 | 63 | 33 | 9 | 20.000 | 65.000 | 576 |
| 3 | 96 | 64 | 32 | 8 | 20.000 | 77.000 | 573 |
| 4 | 95 | 64 | 31 | 9 | 20.000 | 78.000 | 570 |

Folds are `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped on `patient_id` and stratified on sex x (age >= 25).

Stratifying on age keeps older subjects present in every fold — only 55 of 599 subjects are 25 or older, so an unstratified draw could leave a fold with almost none, making `age` and `arterial_stiffness` unstable. **The trade-off: CV is slightly more optimistic than a purely random split**, because every fold is guaranteed a representative age and sex mix.

## Comparability with the paper

The authors split the same way — `train_test_split` on unique `patient_id`, `test_size=0.2`, `random_state=43` — so their Table 5 numbers are genuine subject-level results. We match their 20% test fraction, but their exact split cannot be reproduced: they had 600 subjects, we have 599 after the Phase 1 exclusions.
