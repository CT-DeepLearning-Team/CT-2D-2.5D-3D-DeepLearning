#!/bin/bash
# Full 2D track: Model A (3 seeds) -> MoCo v2 pretraining -> Model B (3 seeds)
# -> aggregated report. Hyperparameters come from the validation-only sweep
# (scripts/sweep_2d.sh) and are fixed here so the run is reproducible.
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python

LR=${LR:-3e-4}
WD=${WD:-5e-2}
DROPOUT=${DROPOUT:-0.5}
AUG=${AUG:-strong}
LS=${LS:-0.10}
EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-20}
MOCO_EPOCHS=${MOCO_EPOCHS:-200}

COMMON="--epochs $EPOCHS --patience $PATIENCE --lr $LR --wd $WD \
        --dropout $DROPOUT --aug $AUG --label-smoothing $LS --eval-test"

echo "=================== MODEL A (random init) ==================="
for s in 0 1 2; do
  echo "--- Model A seed $s ---"
  $PY src/2d/train_supervised.py --init random --seed "$s" $COMMON
done

echo "=================== MoCo v2 PRETRAINING ==================="
$PY src/2d/moco_pretrain.py --epochs "$MOCO_EPOCHS"

echo "=================== MODEL B (MoCo -> fine-tune) ==================="
for s in 0 1 2; do
  echo "--- Model B seed $s ---"
  $PY src/2d/train_supervised.py --init moco \
      --moco-ckpt results/2d/moco/moco_encoder.pt --seed "$s" $COMMON
done

echo "=================== AGGREGATE ==================="
$PY scripts/aggregate_2d.py
