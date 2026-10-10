# From Slices to Volumes

## Comparing 2D, 2.5D and 3D Deep Learning for Pulmonary Nodule Malignancy-Risk Classification

This repository contains the completed team study of how CT spatial context,
self-supervised pretraining, and training-time augmentation affect pulmonary
nodule malignancy-risk classification. It includes the LIDC-IDRI preparation
pipeline, model implementations, experiment reports, saved metrics, and figures
for all three representations.

**Main finding:** the augmented 2.5D model achieved the strongest held-out
three-class performance in this study. Augmentation improved test Macro-F1 in
all three tracks, while the tested MoCo configurations did not improve it over
the supervised baselines. More spatial context alone did not guarantee better
generalization.

> Labels are aggregated radiologist malignancy-risk assessments, not
> pathology-confirmed cancer diagnoses. These experiments do not establish
> clinical diagnostic performance.

## Project status

**Completed.** All three model tracks, the final 2.5D report, and the final
2D vs 2.5D vs 3D comparison are complete. This repository contains the final
source code, reports, metrics, and figures for the project.

## Study overview

| Track | Team member | Input representation | Architecture |
|---|---|---|---|
| 2D | Bahodir | One central CT slice | Single-channel ResNet-18 |
| 2.5D | Fatima | Five neighboring CT slices | Five-channel ResNet-18 |
| 3D | Zaineb | Full local CT volume | Lightweight 3D ResNet-18 |

The study addresses four questions:

1. How does spatial context affect classification across 2D, 2.5D, and 3D?
2. Does MoCo self-supervised pretraining improve downstream classification?
3. Does training-only augmentation improve generalization?
4. How does performance change when indeterminate-risk nodules are excluded?

The tracks share the final V2 cohort, label policy, fixed patient assignments,
and HU window. They differ in architecture capacity, optimization settings,
and representation-specific augmentation. The comparison therefore reflects
these experimental implementations rather than isolating dimensionality alone.

## Dataset and evaluation

The dataset is derived from **LIDC-IDRI** CT scans and radiologist annotations.
The preparation workflow covers DICOM inventory, CT-series selection, XML
parsing, SOP/UID matching, canonical CT reconstruction, reader masks, consensus
nodules, isotropic resampling, crop generation, technical quality control,
risk aggregation, and patient-level splitting.

| Dataset stage | Nodules | Patients |
|---|---:|---:|
| Consensus nodules | 7,385 | 990 |
| Rated consensus nodules | 2,672 | — |
| Final supervised cohort | 2,654 | — |
| SSL training pool | 5,133 | 692 |

The pipeline processed 1,010 patients, inventoried 244,527 DICOM records,
parsed 1,318 XML files, and identified 1,018 valid CT series.

### Labels

| Class | Aggregated median malignancy score | Nodules |
|---|---|---:|
| Benign | 1.0–2.0 | 873 |
| Indeterminate | 2.5–3.0 | 1,226 |
| Malignant | 3.5–5.0 | 555 |

### Fixed splits

| Split | Nodules | Patients |
|---|---:|---:|
| Train | 1,876 | 602 |
| Validation | 382 | 134 |
| Test | 396 | 134 |

Patient overlap between supervised splits is zero. The SSL pool contains only
training patients and excludes validation and test patients. Checkpoints are
selected by **validation Macro-F1**; the held-out test set is used for final
evaluation.

The supplementary binary task excludes indeterminate nodules while retaining
patient assignments: 1,000 training, 219 validation, and 209 test nodules.

### Input preprocessing

Final crops have shape `(104, 72, 80)` in `(Z, Y, X)` order at 1 mm isotropic
spacing. All tracks apply the same base intensity transformation:

```python
x = np.clip(x, -1000, 400)
x = (x + 1000) / 1400  # [0, 1]
```

The 2D track additionally uses training-split mean/std standardization. The
2.5D and final 3D tracks use the normalized intensities without that additional
step. Masks support preparation, centering, and audit; they are not input
channels for the classifiers. See the individual reports for slice selection
and other implementation details.

## Experimental design

| Experiment | Training recipe |
|---|---|
| Model A | Random initialization; supervised three-class baseline without augmentation |
| Model B | MoCo pretraining, then supervised three-class fine-tuning without supervised augmentation |
| Model C | Random initialization; supervised three-class training with mild training-only augmentation |
| Binary sensitivity | Benign versus malignant classification using the Model C recipe |

MoCo pretraining ignores malignancy labels. Validation and test inputs are
never augmented.

## Final results

### Three-class test Macro-F1

| Representation | Model A | Model B: MoCo | Model C: augmentation |
|---|---:|---:|---:|
| 2D | 0.6059 ± 0.024 | 0.5957 ± 0.009 | 0.6134 ± 0.007 |
| 2.5D | 0.6075 | 0.5125 | **0.6537** |
| 3D | 0.4928 | 0.4730 | 0.5403 |

The 2D values are means ± standard deviations across three seeds. The 2.5D
results are single selected runs; the 3D results are frozen seed-42 runs.
Small differences do not establish statistical significance.

![Three-class Macro-F1 across representations and experiments](results/team_comparison/figures/abc_macro_f1_grouped.png)

### Model C held-out performance

| Representation | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 2D | 0.6128 | 0.6110 | 0.6134 | 0.7595 |
| 2.5D | **0.6869** | **0.6413** | **0.6537** | **0.7894** |
| 3D | 0.5455 | 0.5461 | 0.5403 | 0.7165 |

### Binary sensitivity analysis

| Representation | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 2D | 0.8293 | 0.8182 | 0.8227 | 0.8644 |
| 2.5D | 0.8708 | 0.8614 | 0.8664 | 0.9215 |
| 3D | 0.7608 | 0.7538 | 0.7554 | 0.8168 |

Binary classification produced higher scores in every track. This is consistent
with indeterminate-risk ambiguity contributing to the difficulty of the primary
task. Binary and three-class scores are not directly equivalent: the binary
task has fewer classes and a different cohort.

The full validation/test tables and interpretation are in the
[final team comparison](docs/TEAM_COMPARISON.md). Exact saved values are in
[final_team_metrics.csv](results/team_comparison/final_team_metrics.csv).

## Findings and limitations

- **2.5D Model C performed best in the final three-class comparison.** Five
  neighboring slices provided a useful context/optimization compromise in this
  study; this does not establish that 2.5D is universally preferable.
- **Augmentation improved test Macro-F1 in all tracks.** The gains over Model A
  were +0.0075 for 2D, +0.0462 for 2.5D, and +0.0475 for 3D. Improvement did not
  eliminate overfitting, particularly in the 2D track.
- **The tested MoCo recipes did not improve downstream test Macro-F1.** This
  finding applies to the configurations evaluated here.
- **The indeterminate class remained a major source of difficulty.** Removing
  it simplified the task and increased performance across representations.

The study uses a limited, imbalanced cohort and one fixed patient-level split.
Model capacity, training settings, and augmentation differ across tracks, and
multi-seed reporting is available only for 2D. Labels reflect radiologist risk
ratings. These limits constrain causal, statistical, and clinical conclusions.

## Reproducing the workflow

### Install dependencies

The pinned packages are listed in [requirements.txt](requirements.txt). The
reported experiments used Python 3.13 on macOS with PyTorch MPS.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CPU-only Linux development, install the matching CPU builds first:

```bash
python -m pip install torch==2.14.0 torchvision==0.29.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The source data-preparation pipeline has additional dependencies:

```bash
python -m pip install -r data_pipeline/requirements.txt
```

### Supply external data

Medical data and model checkpoints are **not committed**. Obtain LIDC-IDRI
separately and follow [data/README.md](data/README.md) for the processed V2
layout. Keep datasets and local path configurations outside the source tree.

The shared 2D configuration accepts `CT_DATA_ROOT` and `CT_BINARY_ROOT`.
The final 3D configuration currently contains a machine-specific dataset path
in [config.py](src/3d/final_common_hu/config.py); configure that path for your
machine before running data-dependent commands. Review the 2.5D scripts for
their dataset paths as well.

For raw-data preparation, copy
[data_pipeline/config/default.example.json](data_pipeline/config/default.example.json)
to a local configuration, set the input/output paths, and follow the
[data-pipeline instructions](data_pipeline/README.md).

### Run experiments and inspect results

Use the track reports for the exact final settings rather than assuming that
script defaults reproduce the reported experiments:

- [2D report](results/2d/REPORT.md) and [analysis](results/2d/ANALYSIS.md)
- [2.5D report](docs/2P5D_REPORT.md)
- [3D method](docs/3D_METHOD.md), [results](docs/3D_RESULTS.md), and [report](docs/3D_REPORT.md)
- [Final cross-representation comparison](docs/TEAM_COMPARISON.md)

Training and evaluation entry points are in `src/2d/`, `scripts/`, and
`scripts/3d/`. The existing `scripts/3d/smoke_tests.py` requires the external
dataset. Figure-generation scripts read committed result artifacts; they
write into `results/`, and the team-comparison plotter assumes a macOS font
path. Review output locations and platform assumptions before running them.

## Repository guide

```text
├── data/                  # External-data policy
├── data_pipeline/         # Cleaning source and example configuration
├── docs/                  # Dataset, method, and final comparison reports
├── src/
│   ├── 2d/               # 2D model and training implementation
│   ├── 2p5d/             # 2.5D track directory
│   ├── 3d/               # Final 3D implementation
│   └── common/           # Shared data, preprocessing, and evaluation utilities
├── scripts/               # Training, evaluation, plotting, and utilities
├── binary_experiment/     # Binary metadata and report
├── results/
│   ├── 2d/               # Three-seed results and analysis
│   ├── 2.5d/             # Final 2.5D results
│   ├── 3d/               # Final 3D results
│   └── team_comparison/  # Combined metrics and figures
└── requirements.txt       # Pinned experiment dependencies
```

## Dataset documentation

- [Dataset preparation](docs/DATASET_PREPARATION.md)
- [Cleaning pipeline](docs/CLEANING_PIPELINE.md)
- [Label policy](docs/LABEL_POLICY.md)
- [Fixed data splits](docs/DATA_SPLITS.md)
- [Data validation](docs/DATA_VALIDATION.md)
- [Preprocessing contract](docs/preprocessing_contract.md)
- [Experimental protocol](docs/experimental_protocol.md)

The repository provides the completed experimental record for the team study.
Raw scans, processed arrays, masks, and checkpoints must be supplied separately
to rerun data-dependent experiments.
