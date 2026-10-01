"""Frozen paths and protocol constants for the final 3D comparison."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path("/Volumes/Zainaba-1/final_team_dataset_v2_3class")
BINARY_METADATA_DIR = PROJECT_ROOT / "binary_experiment" / "metadata"
FINAL_RUN_ROOT = PROJECT_ROOT / "runs_final_common_hu"
EXPECTED_VOLUME_SHAPE = (104, 72, 80)
CLASS_NAMES_3 = ("benign", "indeterminate", "malignant")
CLASS_NAMES_BINARY = ("benign", "malignant")
WIDTHS = (16, 32, 64, 128)
BLOCK_COUNTS = (2, 2, 2, 2)
SEED = 42
CLIP_MIN = -1000.0
CLIP_MAX = 400.0
NORMALIZATION_DENOMINATOR = 1400.0
ROTATION_DEGREES = 5.0
TRANSLATION_VOXELS = 3.0
INTENSITY_SCALE_RANGE = (0.95, 1.05)
INTENSITY_SHIFT_RANGE = (-0.03, 0.03)
GAUSSIAN_NOISE_STD = 0.01
