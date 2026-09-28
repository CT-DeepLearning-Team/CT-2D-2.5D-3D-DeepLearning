# Dataset Documentation

## 1. Dataset Selection

The project uses:

**LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)**

LIDC-IDRI is a publicly available thoracic CT dataset containing:

- CT scan images
- Pulmonary nodule annotations
- Radiologist assessments
- Malignancy ratings


## 2. Why LIDC-IDRI?

The dataset is suitable for this project because:

- It contains real CT volumetric data.
- It provides nodule-level annotations.
- It allows comparison between 2D, 2.5D and 3D deep learning approaches.
- It is widely used in medical image analysis research.


## 3. Learning Task

The project focuses on:

**Pulmonary nodule malignancy-risk classification**

The goal is to classify nodules according to their malignancy risk based on CT image information.

This project does not perform clinical cancer diagnosis.


## 4. Data Organization

The dataset will be organized as:


data/

├── raw/
│ ├── original CT scans
│ └── annotations
│
└── processed/
├── extracted nodules
├── normalized images
└── train/validation/test splits



## 5. Data Preprocessing Pipeline

The common preprocessing pipeline will include:

1. Loading CT scans.
2. Extracting pulmonary nodules using annotations.
3. Normalizing CT intensity values.
4. Preparing inputs according to each model requirement.


## 6. Data Split Strategy

To ensure a fair comparison:

All approaches will use:

- The same patients.
- The same training set.
- The same validation set.
- The same test set.

The split must be performed at patient level to prevent data leakage.


## 7. Shared Data Policy

All team members will use the same processed dataset.

Only the input representation changes:

| Approach | Input Representation | Index into the (104, 72, 80) crop |
|---|---|---|
| 2D | Single central slice | `vol[52]` |
| 2.5D | 5 neighbouring slices | `vol[50:55]` |
| 3D | Full volume | `vol` |

The nodule is already at the geometric centre of every crop (measured mask
centre-of-mass: z 51.1–52.1, y 35.2–35.7, x 39.3–39.9), so slice 52 is the
defined central slice and no per-nodule recentring is required.


## 8. Dataset Preparation Status

**Complete.** The built dataset is `final_team_dataset_v2_3class`
(see `docs/preprocessing_contract.md` for how to consume it).

| | |
|---|---|
| Consensus nodules | 7385 (2672 rated, 4713 unrated) |
| Technical QC exclusions | 27 (23 suspicious cluster, 4 incomplete crop coverage) |
| Supervised cohort | 2654 — 873 benign / 1226 indeterminate / 555 malignant |
| Train / val / test (nodules) | 1876 / 382 / 396 |
| Train / val / test (patients) | 602 / 134 / 134 |
| SSL pretraining pool | 5133 nodules from 692 train-split patients |
| Arrays | `(104, 72, 80)` float32, 1 mm isotropic, raw HU |

Verified independently: 7385 samples and 7385 masks with matching IDs, constant
array shape, zero patient overlap and zero nodule overlap between splits, and no
validation or test patient in the SSL pool.

The task is **three-class**, not binary — see the label policy in the README.
A binary subset is available for sensitivity analysis: `binary_eligible` marks
370 nodules (273 benign / 97 malignant), and `strict_3_reader_eligible` marks 220.
