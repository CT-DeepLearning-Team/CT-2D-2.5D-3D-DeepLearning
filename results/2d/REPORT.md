# 2D Results — Bahodir

ResNet-18 on the defined central CT slice. Final numbers for the team comparison.

## Setup

| | |
|---|---|
| Representation | **2D**, single central slice |
| Input | slice 52 of the (104, 72, 80) 1 mm crop → 72×80, 1 channel |
| Architecture | torchvision **ResNet-18**, stock, `conv1` changed to 1 input channel, dropout 0.5 before the classifier |
| Parameters | **11,171,779** (11.17 M) |
| HU window | clip [-1000, 400] → [0,1], then standardised with 3-class train-split mean/std (0.3651 / 0.3145) |
| Optimiser | AdamW, lr 1e-3, weight decay 0.05, cosine schedule |
| Regularisation | dropout 0.5, strong train-only augmentation, label smoothing 0.10, early stopping (patience 20) |
| Class imbalance | inverse-frequency weights in the loss |
| Selection | best **validation macro-F1**; test evaluated only after the configuration was frozen |
| Seeds | 0, 1, 2 — all numbers are mean ± std |
| Device | Apple M2, PyTorch MPS |

Inference: 0.7548 ms/nodule (batch 128), 5.934 ms for a single nodule; peak GPU memory 1226.9 MB.

**MoCo v2 pretraining (for Model B):** 200 epochs on 5133 nodules from 692 training patients (labels unused), K=4096, T=0.2, m=0.999, SGD lr 0.015, batch 128. InfoNCE loss 7.482 → 4.138; instance-discrimination accuracy → 0.863. Took 57 min.

## Reference baselines (always predict the majority class)

| task | macro F1 | accuracy | balanced accuracy |
|---|---|---|---|
| 3-class (predict *indeterminate*) | 0.2138 | 0.4722 | 0.3333 |
| binary (predict *benign*) | 0.3589 | 0.5598 | 0.5000 |

These matter for reading the numbers: 3-class accuracy near 0.50 is only marginally above the majority rate, whereas macro-F1 near 0.50 is a genuine result.

## 3-class supervised baseline (Model A)

Seeds: 0, 1, 2 (n = 3), reported as mean ± std.

| metric | validation | test |
|---|---|---|
| **macro F1** | 0.5823 ± 0.0144 | **0.5908 ± 0.0241** |
| accuracy | 0.5768 ± 0.0174 | **0.5850 ± 0.0320** |
| balanced accuracy | 0.5887 ± 0.0134 | **0.5920 ± 0.0158** |
| macro precision | 0.5842 ± 0.0177 | 0.5974 ± 0.0323 |
| macro recall | 0.5887 ± 0.0134 | 0.5920 ± 0.0158 |
| ROC-AUC (OvR macro) | 0.7364 ± 0.0090 | **0.7511 ± 0.0067** |

**Best validation macro-F1 (selection metric):** 0.5823 ± 0.0144  
**Best epoch:** [14, 11, 19] (of [34, 31, 39] run)

Per-class test metrics:

| class | precision | recall | F1 |
|---|---|---|---|
| benign | 0.4913 ± 0.0512 | 0.5499 ± 0.0282 | **0.5158 ± 0.0149** |
| indeterminate | 0.6099 ± 0.0167 | 0.5740 ± 0.0996 | **0.5877 ± 0.0561** |
| malignant | 0.6909 ± 0.0377 | 0.6522 ± 0.0407 | **0.6689 ± 0.0111** |

Test macro-F1 bootstrap 95% CI per seed: [0.543, 0.642]; [0.510, 0.608]; [0.567, 0.666]

Test confusion matrix, pooled over 3 seeds (rows = true, columns = predicted; order benign, indeterminate, malignant):

```
                    benign indetermi malignant
  true benign           193       140        18
  true indetermin       175       322        64
  true malignant         31        65       180
```

Single-seed (seed 0) confusion matrix, for a per-run view:

```
  true benign            65        44         8
  true indetermin        61       103        23
  true malignant         11        17        64
```

Overfitting: train − validation macro-F1 was +0.386 at the selected epoch and +0.456 by the last epoch run.  
Validation → test drift in macro-F1: +0.0085.

Training settings: lr 0.001, weight decay 0.05, dropout 0.5, augmentation `strong`, label smoothing 0.1, batch size 64, AdamW + cosine schedule, inverse-frequency class weights, early stopping patience 20 on validation macro-F1, max 80 epochs.  
Training time: 3.5 min/seed | parameters: 11,171,779

## 3-class MoCo v2 + fine-tuning (Model B)

Seeds: 0, 1, 2 (n = 3), reported as mean ± std.

| metric | validation | test |
|---|---|---|
| **macro F1** | 0.5767 ± 0.0056 | **0.5803 ± 0.0261** |
| accuracy | 0.5654 ± 0.0057 | **0.5732 ± 0.0292** |
| balanced accuracy | 0.5796 ± 0.0042 | **0.5774 ± 0.0264** |
| macro precision | 0.5771 ± 0.0070 | 0.5850 ± 0.0244 |
| macro recall | 0.5796 ± 0.0042 | 0.5774 ± 0.0264 |
| ROC-AUC (OvR macro) | 0.7326 ± 0.0060 | **0.7515 ± 0.0174** |

**Best validation macro-F1 (selection metric):** 0.5767 ± 0.0056  
**Best epoch:** [11, 41, 10] (of [31, 61, 30] run)

Per-class test metrics:

| class | precision | recall | F1 |
|---|---|---|---|
| benign | 0.4791 ± 0.0445 | 0.5043 ± 0.0458 | **0.4905 ± 0.0407** |
| indeterminate | 0.5798 ± 0.0245 | 0.5793 ± 0.0431 | **0.5793 ± 0.0329** |
| malignant | 0.6960 ± 0.0164 | 0.6486 ± 0.0359 | **0.6713 ± 0.0266** |

Test macro-F1 bootstrap 95% CI per seed: [0.496, 0.591]; [0.542, 0.639]; [0.554, 0.653]

Test confusion matrix, pooled over 3 seeds (rows = true, columns = predicted; order benign, indeterminate, malignant):

```
                    benign indetermi malignant
  true benign           177       160        14
  true indetermin       172       325        64
  true malignant         22        75       179
```

Single-seed (seed 0) confusion matrix, for a per-run view:

```
  true benign            58        53         6
  true indetermin        70        97        20
  true malignant          9        27        56
```

Overfitting: train − validation macro-F1 was +0.420 at the selected epoch and +0.438 by the last epoch run.  
Validation → test drift in macro-F1: +0.0037.

Training settings: lr 0.001, weight decay 0.05, dropout 0.5, augmentation `strong`, label smoothing 0.1, batch size 64, AdamW + cosine schedule, inverse-frequency class weights, early stopping patience 20 on validation macro-F1, max 80 epochs.  
Training time: 2.2 min/seed | parameters: 11,171,779

## Binary supervised (sensitivity analysis)

Seeds: 0, 1, 2 (n = 3), reported as mean ± std.

| metric | validation | test |
|---|---|---|
| **macro F1** | 0.8020 ± 0.0243 | **0.8323 ± 0.0218** |
| accuracy | 0.8113 ± 0.0248 | **0.8357 ± 0.0215** |
| balanced accuracy | 0.8028 ± 0.0219 | **0.8304 ± 0.0213** |
| macro precision | 0.8019 ± 0.0260 | 0.8353 ± 0.0225 |
| macro recall | 0.8028 ± 0.0219 | 0.8304 ± 0.0213 |
| ROC-AUC | 0.8515 ± 0.0170 | **0.8790 ± 0.0171** |

**Best validation macro-F1 (selection metric):** 0.8020 ± 0.0243  
**Best epoch:** [21, 34, 12] (of [41, 54, 32] run)

Per-class test metrics:

| class | precision | recall | F1 |
|---|---|---|---|
| benign | 0.8388 ± 0.0167 | 0.8746 ± 0.0245 | **0.8563 ± 0.0194** |
| malignant | 0.8319 ± 0.0300 | 0.7862 ± 0.0223 | **0.8083 ± 0.0243** |

Test macro-F1 bootstrap 95% CI per seed: [0.816, 0.908]; [0.758, 0.866]; [0.764, 0.871]

Test confusion matrix, pooled over 3 seeds (rows = true, columns = predicted; order benign, malignant):

```
                    benign malignant
  true benign           307        44
  true malignant         59       217
```

Single-seed (seed 0) confusion matrix, for a per-run view:

```
  true benign           106        11
  true malignant         17        75
```

Overfitting: train − validation macro-F1 was +0.192 at the selected epoch and +0.204 by the last epoch run.  
Validation → test drift in macro-F1: +0.0303.

Training settings: lr 0.001, weight decay 0.05, dropout 0.5, augmentation `strong`, label smoothing 0.1, batch size 64, AdamW + cosine schedule, inverse-frequency class weights, early stopping patience 20 on validation macro-F1, max 80 epochs.  
Training time: 2.5 min/seed | parameters: 11,171,266

## Comparison 1 — supervised vs MoCo on the 3-class task

| metric | Model A (supervised) | Model B (MoCo) | Δ (B − A) |
|---|---|---|---|
| test macro F1 | 0.5908 | 0.5803 | -0.0105 |
| test accuracy | 0.5850 | 0.5732 | -0.0118 |
| test balanced accuracy | 0.5920 | 0.5774 | -0.0146 |
| test ROC-AUC (OvR macro) | 0.7511 | 0.7515 | +0.0004 |
| test F1 benign | 0.5158 | 0.4905 | -0.0254 |
| test F1 indeterminate | 0.5877 | 0.5793 | -0.0084 |
| test F1 malignant | 0.6689 | 0.6713 | +0.0024 |
| best validation macro F1 | 0.5823 | 0.5767 | -0.0056 |

## Comparison 2 — 3-class vs binary (sensitivity analysis)

Both supervised, same architecture, same settings, same seeds. The binary cohort is the same images with *indeterminate* removed (1000 / 219 / 209 nodules).

| metric | 3-class | binary | Δ |
|---|---|---|---|
| test macro F1 | 0.5908 | 0.8323 | +0.2415 |
| test accuracy | 0.5850 | 0.8357 | +0.2507 |
| test balanced accuracy | 0.5920 | 0.8304 | +0.2384 |
| test ROC-AUC | 0.7511 (OvR macro) | 0.8790 (binary) | +0.1279 |
| test F1 benign | 0.5158 | 0.8563 | +0.3404 |
| test F1 malignant | 0.6689 | 0.8083 | +0.1394 |

Note the two tasks are not directly comparable on macro-F1 alone — a 2-class macro-F1 has a higher chance level (0.33 vs 0.21 for the majority baseline). The per-class F1 for benign and malignant, and the ROC-AUC, are the fairer comparison.

## Analysis and conclusion

See **`results/2d/ANALYSIS.md`** for the written analysis requested in `README_TEAM_EXPERIMENTS.txt`: supervised vs MoCo, whether SSL helped, easiest/hardest class, what the confusion matrix shows, overfitting, validation-to-test agreement, 3-class vs binary, and limitations.

## Files

- `results/2d/REPORT.md` — this report
- `results/2d/ANALYSIS.md` — written analysis and conclusion
- `results/2d/summary.json` — machine-readable numbers
- `results/2d/<tag>/seed<N>/result.json` — per-seed metrics and history
- `results/2d/<tag>/seed<N>/curves.png` — train vs validation curves
- `results/2d/<tag>/confusion_test_pooled.png` — pooled confusion matrix
- `results/2d/computational_cost.json` — parameters, latency, memory
