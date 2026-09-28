# Experimental Protocol (frozen)

## Research questions

1. How does spatial context affect pulmonary nodule malignancy-risk
   classification: 2D vs 2.5D vs 3D?
2. Does MoCo v2 self-supervised pretraining improve each approach?

## Shared and frozen

Same samples, same `class_index_v2` labels, same patient-level splits, same
eligibility rules, same HU window, same metrics. See
`docs/preprocessing_contract.md`.

| | train | validation | test |
|---|---|---|---|
| nodules | 1876 | 382 | 396 |
| patients | 602 | 134 | 134 |

Roles: **train** — the model learns here, and SSL pretraining uses these
patients only. **validation** — checkpoint selection, overfitting monitoring,
hyperparameter search. **test** — final evaluation only, once, after the
configuration is frozen.

## Encoders (the only intended difference)

| approach | input | encoder |
|---|---|---|
| 2D | `(1, 72, 80)` | ResNet-18 |
| 2.5D | `(5, 72, 80)` | ResNet-18 |
| 3D | `(104, 72, 80)` | 3D ResNet-10 |

## Two experiments

- **Model A**: random initialisation → supervised training.
- **Model B**: MoCo v2 pretraining on `metadata/ssl_pretrain_train.csv`
  (5133 nodules, train patients only, no labels) → fine-tuning with an
  identical recipe to Model A.

## Overfitting controls

The labelled cohort is small (1876 training nodules) relative to encoder
capacity (ResNet-18 is 11.7M parameters), and it overfits within a handful of
epochs. All of the following are used:

- patient-level splits, re-asserted programmatically on every run
- augmentation on the training split only
- dropout before the classifier, and weight decay
- early stopping on validation macro-F1
- saved training vs validation curves
- test split untouched until the configuration is frozen

## Model selection

Primary metric: **validation macro-F1** (equal weight to benign, indeterminate
and malignant, which matters because *indeterminate* is 47% of the data).

## Reporting

Every headline number is **mean ± std over 3 seeds** with a **bootstrap 95% CI**
on the test macro-F1. With 396 test nodules the sampling error is a few points,
so differences smaller than the intervals must not be read as real. A
majority-class baseline is reported alongside (test macro-F1 ≈ 0.21,
accuracy ≈ 0.47).

Reported metrics: accuracy, balanced accuracy, macro precision/recall/F1,
per-class precision/recall/F1, confusion matrix, multiclass one-vs-rest ROC-AUC,
plus parameters and training time.
