"""Read-only metadata access for final train/validation/SSL pools."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import BINARY_METADATA_DIR, DATASET_ROOT, PROJECT_ROOT


@dataclass(frozen=True)
class Record:
    sample_id: str
    patient_id: str
    sample_path: Path
    mask_path: Path
    label: int | None
    metadata: Mapping[str, str]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part.startswith("._") for part in path.parts):
        raise ValueError(f"Invalid dataset-relative path: {value}")
    resolved = (root / path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_supervised_records(split: str) -> list[Record]:
    if split not in ("train", "validation"):
        raise ValueError("Final supervised pipeline only permits train and validation")
    rows = _read(DATASET_ROOT / "metadata" / f"supervised_{split}.csv")
    records = []
    for row in rows:
        if Path(row["sample_path"]).name.startswith("._"):
            continue
        records.append(Record(
            row.get("consensus_nodule_id") or Path(row["sample_path"]).stem,
            row["patient_id"], _resolve(DATASET_ROOT, row["sample_path"]),
            _resolve(DATASET_ROOT, row["mask_path"]), int(row["class_index_v2"]), dict(row)
        ))
    return records


def load_binary_records(split: str) -> list[Record]:
    if split not in ("train", "validation"):
        raise ValueError("Final binary pipeline only permits train and validation")
    rows = _read(BINARY_METADATA_DIR / f"binary_{split}.csv")
    records = []
    for row in rows:
        label = int(row["binary_class_index"])
        records.append(Record(
            row.get("consensus_nodule_id") or Path(row["sample_path"]).stem,
            row["patient_id"], _resolve(DATASET_ROOT, row["sample_path"]),
            _resolve(DATASET_ROOT, row["mask_path"]), label, dict(row)
        ))
    return records


def load_ssl_records() -> list[Record]:
    rows = _read(DATASET_ROOT / "metadata" / "ssl_pretrain_train.csv")
    records = []
    for row in rows:
        records.append(Record(
            row.get("consensus_nodule_id") or Path(row["sample_path"]).stem,
            row["patient_id"], _resolve(DATASET_ROOT, row["sample_path"]),
            _resolve(DATASET_ROOT, row["mask_path"]), None, dict(row)
        ))
    return records
