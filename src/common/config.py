"""Shared, frozen configuration for the 2D/2.5D/3D comparison.

Every value here is part of the team protocol. Changing any of it breaks
comparability between the three approaches, so it must be agreed jointly
(see the team rule in docs/project_proposal.md).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- dataset root
# Override with:  export CT_DATA_ROOT=/path/to/final_team_dataset_v2_3class
DATA_ROOT = Path(
    os.environ.get(
        "CT_DATA_ROOT",
        Path.home() / "Documents" / "final_team_dataset_v2_3class",
    )
).expanduser()

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results"

METADATA = DATA_ROOT / "metadata"
SPLIT_CSV = {
    "train": METADATA / "supervised_train.csv",
    "val": METADATA / "supervised_validation.csv",
    "test": METADATA / "supervised_test.csv",
}
SSL_CSV = METADATA / "ssl_pretrain_train.csv"

# ------------------------------------------------------------------- geometry
# Every stored crop is (104, 72, 80) at 1 mm isotropic spacing, and the nodule
# is centred: mask centre-of-mass measured over 25 random samples was
# z 51.1-52.1, y 35.2-35.7, x 39.3-39.9 -- i.e. the geometric centre.
SAMPLE_SHAPE_ZYX = (104, 72, 80)
CENTRAL_SLICE = SAMPLE_SHAPE_ZYX[0] // 2   # 52 -> the "defined central slice"
SLICE_HW = SAMPLE_SHAPE_ZYX[1:]            # (72, 80)

# ------------------------------------------------------------- HU windowing
# Arrays are stored as RAW Hounsfield units (not normalised), so the window is
# our choice. Observed range across 120 random samples: minima to -3024
# (out-of-FOV padding, 16/120 samples below -1100) and maxima to +3080.
# Unclipped, those outliers dominate any normalisation, so we clip first.
# This is a standard lung window covering air, lung parenchyma and soft tissue.
HU_MIN, HU_MAX = -1000.0, 400.0

# ---------------------------------------------------------------- label space
NUM_CLASSES = 3
CLASS_NAMES = ("benign", "indeterminate", "malignant")  # class_index_v2 0,1,2
LABEL_COL = "class_index_v2"
ID_COL = "consensus_nodule_id"
PATIENT_COL = "patient_id"
PATH_COL = "sample_path"

# --------------------------------------------------------------- model select
PRIMARY_METRIC = "macro_f1"   # validation macro-F1 selects the checkpoint
SEEDS = (0, 1, 2)             # every result is reported as mean +/- std
