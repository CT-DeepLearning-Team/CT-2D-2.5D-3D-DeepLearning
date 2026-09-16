# Project Proposal

## Title

From Slices to Volumes: Comparing 2D, 2.5D and 3D Deep Learning Approaches for CT Pulmonary Nodule Malignancy-Risk Classification


## 1. Background and Motivation

Computed Tomography (CT) provides three-dimensional medical images composed of multiple consecutive slices.

Deep learning methods have achieved strong performance in medical image analysis, but different approaches use different levels of spatial information:

- 2D models analyze individual CT slices.
- 2.5D models combine several neighboring slices.
- 3D models process complete volumetric information.

The main research question is whether increasing spatial context improves classification performance and whether the additional computational cost is justified.


## 2. Research Question

How does the amount of spatial context available to a deep learning model affect pulmonary nodule malignancy-risk classification from CT scans?


## 3. Objectives

The project aims to:

1. Implement and compare three deep learning approaches for CT image classification.
2. Study the impact of spatial information on model performance.
3. Analyze the trade-off between accuracy and computational cost.
4. Provide a fair comparison using the same dataset and evaluation protocol.


## 4. Dataset

The project uses the:

LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)

The dataset contains thoracic CT scans with pulmonary nodule annotations and malignancy assessments.

The task is defined as:

Pulmonary nodule malignancy-risk classification.

It is not intended as clinical cancer diagnosis.


## 5. Methodology

All models will follow the same experimental framework:

- Same dataset
- Same preprocessing pipeline
- Same patient-level train/validation/test split
- Same evaluation metrics


### Approach 1: 2D Deep Learning

Responsible member: Bahodir

The model will use individual CT slices as input.

Main goal:
Establish a baseline using traditional 2D convolutional neural networks.


### Approach 2: 2.5D Deep Learning

Responsible member: Fatima

The model will use multiple neighboring CT slices as input.

Main goal:
Evaluate whether limited 3D context improves performance compared with 2D.


### Approach 3: 3D Deep Learning

Responsible member: Zaineb

The model will process complete CT volumes.

Responsibilities:

- Develop 3D deep learning pipeline
- Manage common preprocessing
- Integrate results from all approaches
- Lead final comparison


## 6. Evaluation

Models will be compared using:

- Accuracy
- Balanced Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score
- Specificity

Computational analysis:

- Number of parameters
- Training time
- Inference time
- Memory requirements


## 7. Expected Outcome

The project will provide an experimental comparison showing:

- The effect of spatial information on CT classification.
- The advantages and limitations of 2D, 2.5D and 3D approaches.
- The relationship between model performance and computational requirements.


## 8. Team Workflow

Phase 1:
Literature review and dataset preparation.

Phase 2:
Implementation of individual deep learning approaches.

Phase 3:
Evaluation and comparison.

Phase 4:
Research paper writing and presentation preparation.
