"""Final common-HU datasets and train/validation-only loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import CommonHUAugmentation
from .config import DATASET_ROOT
from .metadata import Record, load_binary_records, load_ssl_records, load_supervised_records
from .preprocessing import preprocess_mask, preprocess_volume


class SupervisedDataset(Dataset):
    def __init__(self, split: str, *, augment: bool) -> None:
        self.split = split
        self.records = load_supervised_records(split)
        self.transform = CommonHUAugmentation() if augment and split == "train" else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = torch.from_numpy(preprocess_volume(np.load(record.sample_path, allow_pickle=False)))
        mask = torch.from_numpy(preprocess_mask(np.load(record.mask_path, allow_pickle=False)))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {"image": image, "mask": mask, "label": torch.tensor(record.label, dtype=torch.long), "sample_id": record.sample_id, "patient_id": record.patient_id}


class BinaryDataset(Dataset):
    def __init__(self, split: str, *, augment: bool) -> None:
        self.split = split
        self.records = load_binary_records(split)
        self.transform = CommonHUAugmentation() if augment and split == "train" else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = torch.from_numpy(preprocess_volume(np.load(record.sample_path, allow_pickle=False)))
        mask = torch.from_numpy(preprocess_mask(np.load(record.mask_path, allow_pickle=False)))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return {"image": image, "mask": mask, "label": torch.tensor(record.label, dtype=torch.long), "sample_id": record.sample_id, "patient_id": record.patient_id}


class SSLDataset(Dataset):
    def __init__(self) -> None:
        self.split = "ssl_train"
        self.records = load_ssl_records()
        self.transform = CommonHUAugmentation()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = torch.from_numpy(preprocess_volume(np.load(record.sample_path, allow_pickle=False)))
        view_q, _ = self.transform(image, None)
        view_k, _ = self.transform(image, None)
        return {"view_q": view_q, "view_k": view_k, "sample_id": record.sample_id, "patient_id": record.patient_id}


def make_supervised_loader(split: str, *, batch_size: int = 4, num_workers: int = 0, augment: bool = False) -> DataLoader:
    if split not in ("train", "validation"):
        raise ValueError("Final supervised loaders only permit train and validation")
    return DataLoader(SupervisedDataset(split, augment=augment), batch_size=batch_size, shuffle=split == "train", num_workers=num_workers)


def make_binary_loader(split: str, *, batch_size: int = 4, num_workers: int = 0, augment: bool = False) -> DataLoader:
    if split not in ("train", "validation"):
        raise ValueError("Final binary loaders only permit train and validation")
    return DataLoader(BinaryDataset(split, augment=augment), batch_size=batch_size, shuffle=split == "train", num_workers=num_workers)


def make_ssl_loader(*, batch_size: int = 4, num_workers: int = 0) -> DataLoader:
    return DataLoader(SSLDataset(), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
