# CT-2D-2.5D-3D-DeepLearning
# From Slices to Volumes: Comparing 2D, 2.5D and 3D Deep Learning for CT Pulmonary Nodule Malignancy-Risk Classification

## Project Overview

Computed Tomography (CT) is a three-dimensional imaging modality composed of consecutive slices. However, many deep learning approaches simplify CT data by using only individual slices or limited contextual information.

This project investigates how the amount of spatial context provided to deep learning models affects pulmonary nodule malignancy-risk classification.

We compare three approaches:

- **2D Deep Learning**
  - Uses a single CT slice.
  - Implemented by Bahodir.

- **2.5D Multi-Slice Deep Learning**
  - Uses multiple neighboring CT slices.
  - Implemented by Fatima.

- **3D Volumetric Deep Learning**
  - Uses complete 3D CT volumes.
  - Implemented by Zaineb.


## Research Question

**How does the amount of spatial context available to a deep learning model affect pulmonary nodule malignancy-risk classification from CT scans?**

The project studies whether additional three-dimensional information improves classification performance and whether the improvement justifies the additional computational cost.


## Dataset

The project uses the:

**LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)**

The dataset contains thoracic CT scans with pulmonary nodule annotations and radiologist malignancy assessments.

The task is defined as:

**Pulmonary nodule malignancy-risk classification**

not clinical cancer diagnosis.


## Experimental Design

All three models use:

- The same CT nodules
- The same patient-level train/validation/test split
- The same evaluation protocol
- The same classification objective

This ensures a fair comparison between different levels of spatial information.


## Project Structure
CT-2D-2.5D-3D-DeepLearning/

├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── common/
│ ├── 2d/
│ ├── 2p5d/
│ └── 3d/
│
├── models/
│
├── results/
│
├── figures/
│
├── papers/
│
└── README.md



## Team Members

| Member | Responsibility |
|---|---|
| Bahodir | 2D CNN baseline |
| Fatima | 2.5D multi-slice model |
| Zaineb | 3D volumetric model, preprocessing pipeline and integration |


## Evaluation Metrics

Models will be compared using:

- ROC-AUC
- Accuracy
- Balanced Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score

Additionally, computational requirements will be analyzed:

- Number of parameters
- Training time
- Inference time
- GPU memory usage


## Goal

The objective is not only to find the highest-performing model, but to understand:

- How much spatial context is needed?
- When does 2.5D provide advantages over 2D?
- Does full 3D processing justify its computational cost?


## Status

🚧 Project under development
