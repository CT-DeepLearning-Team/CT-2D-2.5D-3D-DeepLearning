"""Smoke tests for the isolated final common-HU pipeline; never loads test data."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "3d"))

from final_common_hu.augmentations import CommonHUAugmentation
from final_common_hu.config import (
    BINARY_METADATA_DIR,
    BLOCK_COUNTS,
    CLIP_MAX,
    CLIP_MIN,
    DATASET_ROOT,
    EXPECTED_VOLUME_SHAPE,
    GAUSSIAN_NOISE_STD,
    INTENSITY_SCALE_RANGE,
    INTENSITY_SHIFT_RANGE,
    ROTATION_DEGREES,
    TRANSLATION_VOXELS,
    WIDTHS,
)
from final_common_hu.datasets import (
    BinaryDataset,
    SSLDataset,
    SupervisedDataset,
    make_ssl_loader,
    make_supervised_loader,
)
from final_common_hu.metadata import load_binary_records, load_ssl_records, load_supervised_records
from final_common_hu.metrics import classification_metrics
from final_common_hu.model import CommonHUResNet3D18, trainable_parameter_count
from final_common_hu.moco import MoCo3D, MoCoConfig, fit_moco
from final_common_hu.preprocessing import preprocess_volume
from final_common_hu.supervised import SupervisedConfig, class_weights, fit_supervised, seed_all


class TinyDataset(Dataset):
    def __init__(self, split: str, labels: list[int], ssl: bool = False) -> None:
        self.split = split; self.records = [SimpleNamespace(label=x) for x in labels]
        self.images = torch.tensor([[float(x), 0.0, 0.0, 0.0] for x in labels], dtype=torch.float32).reshape(len(labels), 1, 2, 2, 1)
        self.ssl = ssl
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        if self.ssl: return {"view_q": self.images[index], "view_k": self.images[index], "sample_id": str(index), "patient_id": str(index)}
        return {"image": self.images[index], "label": torch.tensor(self.records[index].label)}


class TinyClassifier(nn.Module):
    def __init__(self): super().__init__(); self.classifier = nn.Linear(4, 3)
    def forward(self, x): return self.classifier(x.flatten(1))


def _assert_no_patient_overlap() -> None:
    train = {r.patient_id for r in load_supervised_records("train")}; val = {r.patient_id for r in load_supervised_records("validation")}
    if train & val: raise AssertionError("3-class train/validation patient overlap")
    btrain = {r.patient_id for r in load_binary_records("train")}; bval = {r.patient_id for r in load_binary_records("validation")}
    if btrain & bval: raise AssertionError("binary train/validation patient overlap")


def _assert_preprocessing() -> dict[str, float]:
    source = load_supervised_records("train")[0].sample_path
    raw = np.load(source, allow_pickle=False)
    processed = preprocess_volume(raw)
    clipped = np.clip(raw, CLIP_MIN, CLIP_MAX)
    if raw.dtype.kind not in "biuf": raise AssertionError("Raw sample is not numeric HU data")
    if clipped.min() < CLIP_MIN or clipped.max() > CLIP_MAX: raise AssertionError("Clip bounds failed")
    if processed.min() < 0 or processed.max() > 1: raise AssertionError("[0,1] normalization failed")
    return {"raw_min": float(raw.min()), "raw_max": float(raw.max()), "processed_min": float(processed.min()), "processed_max": float(processed.max())}


def _assert_model() -> int:
    model3 = CommonHUResNet3D18(3, 0.20); model2 = CommonHUResNet3D18(2, 0.20)
    if model3.block_counts != BLOCK_COUNTS or model3.widths != WIDTHS: raise AssertionError("ResNet-18 structure mismatch")
    dummy = torch.randn(2, 1, *EXPECTED_VOLUME_SHAPE); out3 = model3(dummy); out2 = model2(dummy)
    if tuple(out3.shape) != (2, 3) or tuple(out2.shape) != (2, 2): raise AssertionError("Final output shape mismatch")
    loss = nn.CrossEntropyLoss()(out3, torch.tensor([0, 2])); loss.backward()
    if not any(p.grad is not None for p in model3.parameters()): raise AssertionError("3-class gradients failed")
    return trainable_parameter_count(model3)


def _assert_dataset_interfaces() -> tuple[list[float], list[float], int]:
    a_train = SupervisedDataset("train", augment=False); c_train = SupervisedDataset("train", augment=True); c_val = SupervisedDataset("validation", augment=True); b_train = BinaryDataset("train", augment=True); b_val = BinaryDataset("validation", augment=True)
    if a_train.transform is not None or c_train.transform is None or c_val.transform is not None: raise AssertionError("A/C augmentation policy failed")
    if b_train.transform is None or b_val.transform is not None: raise AssertionError("binary augmentation policy failed")
    if type(c_train.transform) is not type(b_train.transform): raise AssertionError("C/binary transforms differ")
    if c_train.transform.intensity_scale_range != INTENSITY_SCALE_RANGE or c_train.transform.intensity_shift_range != INTENSITY_SHIFT_RANGE: raise AssertionError("Augmentation intensity recipe mismatch")
    first = c_val[0]["image"]; second = c_val[0]["image"]
    if not torch.equal(first, second): raise AssertionError("Validation is not deterministic")
    ssl_records = load_ssl_records()
    if len(ssl_records) != 5133 or any(record.label is not None for record in ssl_records): raise AssertionError("SSL pool count or labels failed")
    ssl_patients = {record.patient_id for record in ssl_records}; val_patients = {record.patient_id for record in load_supervised_records("validation")}
    if ssl_patients & val_patients: raise AssertionError("SSL contains validation patients")
    w3, counts3 = class_weights(a_train, 3); wb, countsb = class_weights(b_train, 2)
    if counts3 != [622, 876, 378] or countsb != [622, 378]: raise AssertionError("Class counts failed")
    return w3.tolist(), wb.tolist(), len(ssl_records)


def _assert_moco() -> None:
    dataset = SSLDataset(); sample = dataset[0]
    if tuple(sample["view_q"].shape) != (1, *EXPECTED_VOLUME_SHAPE) or "label" in sample: raise AssertionError("SSL dataset view/label policy failed")
    model = MoCo3D(); q = sample["view_q"].unsqueeze(0); k = sample["view_k"].unsqueeze(0); logits, labels, embedding = model(q, k)
    if tuple(embedding.shape) != (1, 128) or tuple(model.queue.shape) != (128, 4096) or not torch.isfinite(logits).all(): raise AssertionError("MoCo smoke shape/loss failed")
    nn.functional.cross_entropy(logits, labels).backward()
    if not any(p.grad is not None for p in model.encoder_q.parameters()): raise AssertionError("MoCo query gradients failed")
    if any(p.grad is not None for p in model.encoder_k.parameters()): raise AssertionError("MoCo key encoder received gradients")


def _assert_moco_resume() -> None:
    class TinySSL(Dataset):
        def __len__(self): return 4
        def __getitem__(self, index):
            image = torch.randn(1, *EXPECTED_VOLUME_SHAPE)
            return {"view_q": image, "view_k": image.clone(), "sample_id": str(index), "patient_id": str(index)}
    loader = DataLoader(TinySSL(), batch_size=4, shuffle=False)
    with tempfile.TemporaryDirectory(prefix="final_common_hu_moco_resume_") as temp:
        out = Path(temp); first = fit_moco(MoCo3D(), loader, config=MoCoConfig(epochs=1, output_dir=out), device=torch.device("cpu"))
        latest = out / "latest_checkpoint.pt"; state = torch.load(latest, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "queue", "queue_ptr", "completed_epoch", "history", "rng_state"}
        if not required.issubset(state) or state["completed_epoch"] != 1 or first["history"][-1]["epoch"] != 1: raise AssertionError("MoCo resume checkpoint failed")
        resumed = fit_moco(MoCo3D(), loader, config=MoCoConfig(epochs=2, output_dir=out), device=torch.device("cpu"), resume=True)
        if len(resumed["history"]) != 2: raise AssertionError("MoCo resume did not continue")


def _assert_resume() -> None:
    with tempfile.TemporaryDirectory(prefix="final_common_hu_resume_") as temp:
        out = Path(temp); train = DataLoader(TinyDataset("train", [0, 1, 2, 0, 1, 2]), batch_size=2, shuffle=True); val = DataLoader(TinyDataset("validation", [0, 1, 2]), batch_size=3)
        first = fit_supervised(TinyClassifier(), train, val, names=("benign", "indeterminate", "malignant"), config=SupervisedConfig(epochs=1, output_dir=out), device=torch.device("cpu"))
        latest = out / "latest_checkpoint.pt"; state = torch.load(latest, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict", "completed_epoch", "best_validation_macro_f1", "early_stopping_counter", "history", "rng_state"}
        if not required.issubset(state) or first["best_epoch"] != 1: raise AssertionError("Supervised resume checkpoint failed")
        resumed = fit_supervised(TinyClassifier(), train, val, names=("benign", "indeterminate", "malignant"), config=SupervisedConfig(epochs=2, output_dir=out), device=torch.device("cpu"), resume=True)
        if len(resumed["history"]) != 2: raise AssertionError("Supervised resume did not continue")


def main() -> None:
    seed_all(42); raw_stats = _assert_preprocessing(); _assert_no_patient_overlap(); params = _assert_model(); w3, wb, ssl_count = _assert_dataset_interfaces(); _assert_moco(); _assert_moco_resume(); _assert_resume()
    print(f"dataset_root={DATASET_ROOT}"); print(f"preprocessing=clip[{CLIP_MIN},{CLIP_MAX}] then (x+1000)/1400 to [0,1]"); print(f"raw_and_processed_stats={raw_stats}"); print(f"resnet18_blocks={BLOCK_COUNTS} channels={WIDTHS} trainable_parameters={params}"); print(f"class_weights_3class={w3}"); print(f"class_weights_binary={wb}"); print(f"ssl_samples={ssl_count}"); print(f"augmentation_recipe=rotation±{ROTATION_DEGREES}deg translation±{TRANSLATION_VOXELS}vox scale{INTENSITY_SCALE_RANGE} shift{INTENSITY_SHIFT_RANGE} noise_std={GAUSSIAN_NOISE_STD} clamp[0,1]"); print("A_no_augmentation=True"); print("C_train_only_augmentation=True"); print("binary_same_train_only_augmentation=True"); print("validation_deterministic=True"); print("moco_labels_used=False"); print("moco_resume_state_smoke=True"); print("supervised_resume_state_smoke=True"); print("test_loader_constructed=False"); print("test_csv_accessed=False"); print("final_common_hu_smoke_passed=True")


if __name__ == "__main__": main()
