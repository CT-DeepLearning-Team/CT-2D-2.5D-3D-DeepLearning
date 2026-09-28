# CT-2D-2.5D-3D-DeepLearning
# From Slices to Volumes: Comparing 2D, 2.5D and 3D Deep Learning for CT Pulmonary Nodule Malignancy-Risk Classification

## Project Overview

Computed Tomography (CT) is a three-dimensional imaging modality composed of consecutive slices. However, many deep learning approaches simplify CT data by using only individual slices or limited contextual information.

This project investigates how the amount of spatial context provided to deep learning models affects pulmonary nodule malignancy-risk classification.

We compare three approaches, all on identical data, labels, splits and metrics:

| Approach | Input | Encoder | Member |
|---|---|---|---|
| **2D** | single central slice `(1, 72, 80)` | ResNet-18 | Bahodir |
| **2.5D** | 5 neighbouring slices `(5, 72, 80)` | ResNet-18 | Fatima |
| **3D** | full crop `(104, 72, 80)` | 3D ResNet-10 | Zaineb |

The encoders differ **only** in input channels / dimensionality, so a performance
gap is attributable to spatial context rather than architecture.

## Research Question

**How does the amount of spatial context available to a deep learning model affect pulmonary nodule malignancy-risk classification from CT scans?**

And, second: **does self-supervised pretraining improve each approach?**

## Task: three classes

The task is **pulmonary nodule malignancy-risk classification**, not clinical
cancer diagnosis. LIDC-IDRI radiologist assessments include genuinely uncertain
cases, so we do not force a binary split. The project label rule
(`docs/label_policy_v2.md`) is:

| Median of reader scores | Class | `class_index_v2` |
|---|---|---|
| 1.0 – 2.0 | benign | 0 |
| 2.5 – 3.0 | indeterminate | 1 |
| 3.5 – 5.0 | malignant | 2 |

Original LIDC ratings are integers 1–5; medians of 2.5 and 3.5 arise from reader
disagreement and are **not** original radiologist categories. Original reader
scores, medians, reader counts and disagreement flags are preserved in the
metadata for sensitivity analyses (`midpoint_median_v2` flags 449 such nodules).

## Dataset

`final_team_dataset_v2_3class`, derived from **LIDC-IDRI**.

| | |
|---|---|
| Consensus nodules | 7385 (2672 rated, 4713 unrated) |
| Supervised cohort | **2654** — 873 benign / 1226 indeterminate / 555 malignant |
| Train / val / test (nodules) | 1876 / 382 / 396 |
| Train / val / test (patients) | 602 / 134 / 134 |
| SSL pretraining pool | 5133 nodules, train patients only |
| Arrays | `(104, 72, 80)` float32, 1 mm isotropic, **raw HU** |

Patient-level splits, verified to have zero patient and zero nodule overlap, and
an SSL pool containing no validation or test patients.

Set the dataset location once:

```bash
export CT_DATA_ROOT=~/Documents/final_team_dataset_v2_3class
```

## Two experiments per approach

- **Model A** — random initialisation → supervised 3-class training.
- **Model B** — **MoCo v2** self-supervised pretraining on `ssl_pretrain_train.csv`
  (no labels, train patients only) → fine-tuning on the same labelled cohort as A.

Model B uses an identical fine-tuning recipe to Model A, so the A→B difference
measures pretraining alone. The SSL method is frozen team-wide as MoCo v2.

## Project Structure

```
CT-2D-2.5D-3D-DeepLearning/
├── data/cache/                 # generated central-slice cache (gitignored)
├── docs/
│   ├── preprocessing_contract.md   # <- shared rules: READ THIS FIRST
│   ├── project_proposal.md
│   ├── dataset.md
│   └── experimental_protocol.md
├── scripts/
│   ├── build_cache.py
│   ├── sweep_2d.sh
│   └── aggregate_2d.py
├── src/
│   ├── common/     # config, data, transforms, metrics, plots, seeding
│   ├── 2d/         # model, moco_pretrain, train_supervised
│   ├── 2p5d/
│   └── 3d/
└── results/2d/REPORT.md
```

## Evaluation Metrics

Identical for all three approaches (`src/common/metrics.py`):

- Accuracy, Balanced Accuracy
- Macro Precision / Recall / **F1** (macro-F1 is the primary selection metric)
- Per-class Precision / Recall / F1
- Confusion matrix
- Multiclass ROC-AUC (one-vs-rest, macro and weighted)
- Training vs validation curves, as overfitting evidence

Plus computational cost: parameters, training time, inference time, memory.

Because the test split is only 396 nodules, headline numbers are reported as
**mean ± std over 3 seeds** with **bootstrap 95% CIs**. Differences smaller than
those intervals should not be interpreted as real.

**Reference baseline:** always predicting *indeterminate* gives test macro-F1
≈ 0.21 and accuracy ≈ 0.47. Accuracy near 0.50 is therefore only marginally
above the majority rate, while macro-F1 near 0.50 is a genuine result.

## Reproducing the 2D track

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export CT_DATA_ROOT=~/Documents/final_team_dataset_v2_3class
.venv/bin/python scripts/build_cache.py                  # ~170 MB slice cache

# Model A (3 seeds)
for s in 0 1 2; do .venv/bin/python src/2d/train_supervised.py \
    --init random --seed $s --eval-test; done

# Model B: MoCo v2 pretraining, then fine-tune (3 seeds)
.venv/bin/python src/2d/moco_pretrain.py
for s in 0 1 2; do .venv/bin/python src/2d/train_supervised.py \
    --init moco --moco-ckpt results/2d/moco/moco_encoder.pt \
    --seed $s --eval-test; done

.venv/bin/python scripts/aggregate_2d.py                 # -> results/2d/REPORT.md
```

## Team rule

Do not independently change labels, patient splits, class definitions,
evaluation metrics, or the general training strategy. Shared preprocessing lives
in `src/common/` and is documented in `docs/preprocessing_contract.md` — import
it rather than reimplementing, and discuss changes before making them.

## Status

- [x] Dataset built, validated, splits verified leakage-free
- [x] Shared preprocessing / metrics / plotting (`src/common/`)
- [x] 2D — Model A and Model B (Bahodir)
- [ ] 2.5D — Model A and Model B (Fatima)
- [x] 3D — Model A done, Model B in progress (Zaineb)
- [ ] Final comparison and paper
