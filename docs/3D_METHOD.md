# Final 3D Method

## Task and data

The 3D track classifies pulmonary-nodule malignancy risk using the fixed
patient-level V2 train, validation, and test splits. The 3D volumes have shape
`[104, 72, 80]` and enter the model as `[B, 1, 104, 72, 80]`. Masks are kept
for audit/visualization and are not model inputs.

The committed code expects the built dataset to remain outside the repository;
the dataset itself and `.npy` samples are not committed.

## Frozen preprocessing

For every split and experiment:

```text
clip to [-1000, 400]
normalize with (x + 1000) / 1400
```

This produces a float volume in `[0, 1]`. Validation and test inference are
deterministic and use no augmentation.

## Architecture

The final classifier is a lightweight 3D ResNet-18:

- residual block layout: `[2, 2, 2, 2]`
- channels: `16 → 32 → 64 → 128`
- one input channel
- adaptive global-average pooling
- dropout: `0.20`
- standard linear classification head
- seed: `42`
- parameters: `2,073,747` for the 3-class head and `2,073,618` for the binary head

The classifier returns logits. Softmax is used only for probability-based
evaluation.

## Experiments

- **Model A:** randomly initialized supervised 3-class baseline, without augmentation.
- **Model B:** MoCo-pretrained encoder followed by supervised 3-class fine-tuning.
- **Model C:** randomly initialized supervised 3-class model with mild training-only augmentation.
- **Binary:** the Model C recipe for benign-versus-malignant sensitivity analysis; indeterminate nodules are excluded.

All supervised experiments use weighted cross-entropy, AdamW, batch size 4,
learning rate `1e-4`, weight decay `1e-4`, validation Macro-F1 checkpoint
selection, and early stopping. Model B transfers only the pretrained query
backbone; its classifier is newly initialized.

## MoCo pretraining

MoCo uses 5,133 training-patient nodules with labels ignored during SSL:

- queue size: `4096`
- momentum: `0.999`
- temperature: `0.07`
- projection MLP: `128 → 128 → 128`
- batch size: `4`
- AdamW, learning rate `3e-4`, weight decay `1e-4`
- 50 epochs
- cosine learning-rate schedule

The MoCo history is retained under `results/3d/moco_pretraining/`.

## Reproduction layout

- implementation: `src/3d/final_common_hu/`
- training/evaluation entry points: `scripts/3d/`
- binary metadata: `binary_experiment/`
- saved small result artifacts and figures: `results/3d/`

Weights remain local and are intentionally excluded from GitHub.
