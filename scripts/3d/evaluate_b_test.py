"""One-time frozen test evaluator for final common-HU Model B.

This script is intentionally separate from the train/validation loaders. It
opens supervised_test.csv only when explicitly executed by the user and never
applies augmentation during inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "3d"))

from final_common_hu.config import (  # noqa: E402
    CLASS_NAMES_3,
    CLIP_MAX,
    CLIP_MIN,
    DATASET_ROOT,
    FINAL_RUN_ROOT,
    NORMALIZATION_DENOMINATOR,
)
from final_common_hu.metrics import classification_metrics, roc_auc_ovr  # noqa: E402
from final_common_hu.model import CommonHUResNet3D18  # noqa: E402
from final_common_hu.preprocessing import preprocess_volume  # noqa: E402


RUN_DIR = FINAL_RUN_ROOT / "3class_B_moco_resnet18"
CHECKPOINT_PATH = RUN_DIR / "best_model.pt"
TEST_CSV_PATH = DATASET_ROOT / "metadata" / "supervised_test.csv"
OUTPUT_DIR = RUN_DIR / "final_test"


def _resolve_dataset_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part.startswith("._") for part in path.parts):
        raise ValueError(f"Invalid dataset-relative path: {value}")
    resolved = (DATASET_ROOT / path).resolve()
    resolved.relative_to(DATASET_ROOT.resolve())
    return resolved


def _load_test_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with TEST_CSV_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if Path(row["sample_path"]).name.startswith("._"):
                continue
            rows.append(row)
    return rows


class FrozenTestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        sample_id = row.get("consensus_nodule_id") or Path(row["sample_path"]).stem
        volume = np.load(_resolve_dataset_path(row["sample_path"]), allow_pickle=False)
        return {
            "image": torch.from_numpy(preprocess_volume(volume)),
            "label": torch.tensor(int(row["class_index_v2"]), dtype=torch.long),
            "sample_id": sample_id,
            "patient_id": row["patient_id"],
        }


def _device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label\\predicted_label", *CLASS_NAMES_3])
        for name, row in zip(CLASS_NAMES_3, matrix):
            writer.writerow([name, *row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Model B once on supervised_test.csv")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    args = parser.parse_args()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("best_epoch") != checkpoint.get("completed_epoch"):
        raise RuntimeError("best_model.pt is not the frozen best-epoch checkpoint")
    if checkpoint.get("best_epoch") != checkpoint.get("epoch"):
        raise RuntimeError("best_model.pt epoch fields disagree")
    if tuple(checkpoint.get("class_names", ())) != CLASS_NAMES_3:
        raise RuntimeError("Checkpoint class definitions do not match Model B")

    model = CommonHUResNet3D18(num_classes=3, dropout=0.20)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    device = _device(args.device)
    model.to(device)

    rows = _load_test_rows()
    dataset = FrozenTestDataset(rows)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    class_weights = torch.as_tensor(checkpoint["class_weights"], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    sample_ids: list[str] = []
    patient_ids: list[str] = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            count = int(labels.shape[0])
            total_loss += float(loss.item()) * count
            total_samples += count
            targets.extend(labels.cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            sample_ids.extend(batch["sample_id"])
            patient_ids.extend(batch["patient_id"])

    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = classification_metrics(targets, predictions, CLASS_NAMES_3)
    metrics["loss"] = total_loss / total_samples
    metrics["roc_auc_ovr"] = roc_auc_ovr(targets, probability_array, CLASS_NAMES_3)
    metrics["checkpoint_path"] = str(CHECKPOINT_PATH)
    metrics["checkpoint_epoch"] = int(checkpoint["best_epoch"])
    metrics["num_samples"] = total_samples
    metrics["preprocessing"] = {
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "normalization": "(x + 1000) / 1400",
        "augmentation": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test_results.json").write_text(json.dumps(metrics, indent=2))
    _write_confusion_matrix(metrics["confusion_matrix"], OUTPUT_DIR / "test_confusion_matrix.csv")
    with (OUTPUT_DIR / "test_probabilities.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "patient_id", "true_label", *[f"{name}_probability" for name in CLASS_NAMES_3]])
        for sample_id, patient_id, target, probs in zip(sample_ids, patient_ids, targets, probability_array):
            writer.writerow([sample_id, patient_id, target, *[float(value) for value in probs]])

    print(json.dumps(metrics, indent=2))
    print(f"saved_results={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
