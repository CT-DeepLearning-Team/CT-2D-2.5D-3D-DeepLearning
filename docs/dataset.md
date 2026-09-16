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

| Approach | Input Representation |
|---|---|
| 2D | Single slice |
| 2.5D | Multiple slices |
| 3D | Full volume |


## 8. Dataset Preparation Status

Current status:

🚧 Dataset preparation not started.
