#!/bin/bash
# Team-requested final structure. Every model below uses the IDENTICAL recipe
# (ResNet-18, lr 1e-3, AdamW wd 0.05, cosine, dropout 0.5, label smoothing 0.10,
# class weights, early stopping on val macro-F1) and differs ONLY in the
# augmentation setting and the task, so the A-vs-C contrast isolates
# augmentation. Validation and test are never augmented.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON="--epochs 80 --patience 20 --lr 1e-3 --wd 5e-2 --dropout 0.5 \
        --label-smoothing 0.10 --eval-test"

for s in 0 1 2; do
  echo "########## A_noaug seed $s ##########"
  $PY src/2d/train_supervised.py --task 3class --init random --aug none \
      --tag model_a_noaug --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
for s in 0 1 2; do
  echo "########## B_noaug seed $s ##########"
  $PY src/2d/train_supervised.py --task 3class --init moco \
      --moco-ckpt results/2d/moco/moco_encoder.pt --aug none \
      --tag model_b_noaug --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
for s in 0 1 2; do
  echo "########## C_mild seed $s ##########"
  $PY src/2d/train_supervised.py --task 3class --init random --aug mild \
      --tag model_c_mild --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
for s in 0 1 2; do
  echo "########## BINARY_mild seed $s ##########"
  $PY src/2d/train_supervised.py --task binary --init random --aug mild \
      --tag binary_mild --seed "$s" $COMMON 2>&1 | grep -vE "^ep +[0-9]+ \|"
done
echo "########## MODEL C SET DONE ##########"
