#!/bin/bash
# Validation-only hyperparameter search for the 2D model.
# The test split is NOT evaluated here (no --eval-test): selection uses
# validation macro-F1 only, per the team protocol.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run () {  # name lr wd dropout aug ls
  echo "########## $1 ##########"
  $PY src/2d/train_supervised.py --tag "sweep/$1" --seed 0 \
      --epochs 45 --patience 12 \
      --lr "$2" --wd "$3" --dropout "$4" --aug "$5" --label-smoothing "$6" \
      2>&1 | grep -E "^ep |best val|early stopping|macro F1 " | tail -6
}
run c1_base      3e-4 1e-4 0.3 mild   0.05
run c2_wd        3e-4 5e-2 0.5 mild   0.05
run c3_strongaug 3e-4 5e-2 0.5 strong 0.05
run c4_lowlr     1e-4 5e-2 0.5 strong 0.10
run c5_heavy     1e-4 1e-1 0.5 strong 0.10
run c6_higherlr  1e-3 5e-2 0.5 strong 0.10
echo "########## sweep done ##########"
