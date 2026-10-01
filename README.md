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

## Final 3D Track — Zaineb

The final 3D track evaluates full CT volumes under the same fixed patient-level
V2 splits used by the project. The final implementation, training/evaluation
entry points, saved small result artifacts, and reproducible figures are kept
separate from the existing 2D and 2.5D work.

### Approach

- Input: `[B, 1, 104, 72, 80]` CT volumes
- Architecture: lightweight 3D ResNet-18 with blocks `[2,2,2,2]` and channels `16 → 32 → 64 → 128`
- Preprocessing: clip HU values to `[-1000,400]`, then normalize with `(x + 1000) / 1400`
- Dropout: `0.20`; seed: `42`
- 3-class parameter count: `2,073,747`; binary parameter count: `2,073,618`
- Checkpoint selection: validation Macro-F1

### Experiments and current results

- **Model A:** supervised 3-class baseline
- **Model B:** MoCo pretraining followed by supervised 3-class fine-tuning
- **Model C:** supervised 3-class model with mild training-only augmentation
- **Binary:** benign-versus-malignant sensitivity analysis with indeterminate nodules excluded

Model C is the strongest final three-class 3D model in the saved results:
test accuracy `0.545455`, balanced accuracy `0.546072`, Macro-F1 `0.540307`,
and ROC-AUC `0.716477`. The binary experiment reaches test accuracy
`0.760766`, balanced accuracy `0.753809`, Macro-F1 `0.755384`, and ROC-AUC
`0.816797`; these binary metrics are not directly equivalent to the 3-class
metrics.

See [3D_METHOD.md](docs/3D_METHOD.md) for the protocol and [3D_RESULTS.md](docs/3D_RESULTS.md)
for the complete results and interpretation. Reproducible figures are in
[results/3d/figures](results/3d/figures/), and the machine-readable table is
[results/3d/final_3d_metrics.csv](results/3d/final_3d_metrics.csv).
