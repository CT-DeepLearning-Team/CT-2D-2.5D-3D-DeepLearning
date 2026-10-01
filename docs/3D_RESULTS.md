# Final 3D Results

The values below are copied from the official saved histories and final test
result JSON files. The train accuracy column is the deterministic post-hoc
accuracy of the frozen best checkpoint on its training split; it is not a new
training run. Test inference was not rerun for this integration.

| Experiment | Best epoch | Train accuracy | Validation accuracy | Validation Macro-F1 | Test accuracy | Test balanced accuracy | Test Macro-F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A | 9 | 0.960554 | 0.510471 | 0.513904 | 0.494949 | 0.479044 | 0.492761 | 0.656366 |
| Model B — MoCo | 12 | 0.651386 | 0.502618 | 0.512672 | 0.459596 | 0.468331 | 0.473008 | 0.669224 |
| Model C — augmentation | 16 | 0.858209 | 0.531414 | 0.535629 | 0.545455 | 0.546072 | 0.540307 | 0.716477 |
| Binary | 16 | 0.984000 | 0.803653 | 0.795394 | 0.760766 | 0.753809 | 0.755384 | 0.816797 |

The machine-readable table is `results/3d/final_3d_metrics.csv`.

## Interpretation

### Model A

Model A reaches high deterministic training accuracy but has a severe
train-validation accuracy gap, indicating a clear overfitting and
generalization limitation. Its test Macro-F1 is `0.492761`.

### Model B

MoCo pretraining produces a smaller train-validation accuracy gap than Model A,
but it does not improve overall downstream classification Macro-F1 in this
experiment. Its test ROC-AUC is slightly higher than Model A's, so SSL should
not be described as a failure; it simply did not improve the overall selected
downstream classification result here.

### Model C

Model C is the strongest final three-class 3D model in this set. Mild
training-only augmentation improves test accuracy, balanced accuracy, Macro-F1,
and ROC-AUC relative to Models A and B. Its validation and test Macro-F1 are
close (`0.535629` versus `0.540307`).

### Binary sensitivity experiment

The binary experiment is substantially easier, with test Macro-F1 `0.755384`
and ROC-AUC `0.816797`. This supports the interpretation that the indeterminate
malignancy-risk group contributes strongly to the difficulty of the 3-class
task, although it is not evidence that the indeterminate group is the only
cause. Binary and 3-class metrics are not directly equivalent.

## Figures

The reproducible generator is `scripts/plot_3d_results.py`. It produces the
figures under `results/3d/figures/` from the committed histories, confusion
matrices, and final metrics table.
