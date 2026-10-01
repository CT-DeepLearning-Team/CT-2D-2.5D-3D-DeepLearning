"""Finalize a completed Phase 2 worker staging tree without duplicating data."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.canonical import write_csv, write_validation_report
from scripts.common import write_json, write_jsonl
from scripts.run_phase2_parallel import REPO_ROOT, rebase_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.parent if path.parent.name == "config" else Path.cwd()
    canonical_root = Path(config.get("canonical_output_root", "processed/canonical")).expanduser()
    if not canonical_root.is_absolute():
        canonical_root = (base / canonical_root).resolve()
    config["canonical_output_root"] = str(canonical_root)
    return config


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_worker_paths(worker_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata_rows = read_csv_rows(worker_root / "metadata.csv")
    label_rows = read_csv_rows(worker_root / "labels.csv")
    for row in metadata_rows:
        for key in ("volume_path", "metadata_path", "nodule_jsonl_path"):
            row[key] = rebase_path(row.get(key), worker_root)
    for row in label_rows:
        row["mask_path"] = rebase_path(row.get("mask_path"), worker_root)

    with (worker_root / "failures.jsonl").open(encoding="utf-8") as handle:
        failures = [json.loads(line) for line in handle if line.strip()]
    stats = json.loads((worker_root / "statistics.json").read_text(encoding="utf-8"))

    # Rewrite paths inside nodule records and volume sidecars while the worker
    # tree still exists; the files are moved afterward without copying bytes.
    for source_path in (worker_root / "nodules").rglob("*.jsonl"):
        rows = []
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["mask_path"] = rebase_path(row.get("mask_path"), worker_root)
                    rows.append(row)
        write_jsonl(source_path, rows)
    for source_path in (worker_root / "volumes").rglob("*.json"):
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
        metadata["volume_path"] = rebase_path(metadata.get("volume_path"), worker_root)
        write_json(source_path, metadata)
    return metadata_rows, label_rows, failures, stats


def move_worker_artifacts(worker_root: Path, final_root: Path) -> None:
    for directory in ("volumes", "masks", "nodules"):
        source = worker_root / directory
        target = final_root / directory
        for child in sorted(source.iterdir()):
            shutil.move(str(child), str(target / child.name))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    final_root = Path(config["canonical_output_root"])
    staging_root = final_root / ".phase2_workers"
    workers = sorted(staging_root.glob("worker_[0-9][0-9]"))
    if len(workers) != 8:
        raise RuntimeError(f"Expected 8 completed worker directories, found {len(workers)}")
    if not all((worker / "statistics.json").exists() for worker in workers):
        raise RuntimeError("At least one worker is missing statistics.json")
    for extra in staging_root.iterdir():
        if extra not in workers:
            if any(path.is_file() for path in extra.rglob("*")):
                raise RuntimeError(f"Unexpected non-empty staging entry: {extra}")
            shutil.rmtree(extra)

    # The existing final tree contains only partial generated output from the
    # failed copy attempt. Remove those exact generated directories so the
    # completed staged tree can be moved into place without extra disk usage.
    for directory in ("volumes", "masks", "nodules"):
        target = final_root / directory
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

    metadata_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_patients: set[str] = set()
    missing_details: list[dict[str, Any]] = []
    malignancy_counts: Counter[str] = Counter()
    selected_series = volumes_created = masks_created = missing_series = unmatched = 0

    for worker_root in workers:
        worker_metadata, worker_labels, worker_failures, stats = prepare_worker_paths(worker_root)
        metadata_rows.extend(worker_metadata)
        label_rows.extend(worker_labels)
        failures.extend(worker_failures)
        selected_patients.update(stats.get("selected_patient_ids", []))
        selected_series += int(stats.get("selected_series", 0))
        volumes_created += int(stats.get("volumes_created", 0))
        masks_created += int(stats.get("masks_created", 0))
        missing_series += int(stats.get("missing_annotation_series", 0))
        missing_details.extend(stats.get("missing_annotation_details", []))
        malignancy_counts.update(stats.get("malignancy_original_counts", {}))
        unmatched += sum(int(row.get("unmatched_roi_count") or 0) for row in worker_labels)
        move_worker_artifacts(worker_root, final_root)
        shutil.rmtree(worker_root)

    metadata_rows.sort(key=lambda row: str(row.get("volume_id")))
    label_rows.sort(key=lambda row: str(row.get("annotation_key")))
    write_csv(final_root / "metadata.csv", metadata_rows, [
        "volume_id", "patient_id", "study_instance_uid", "series_instance_uid", "volume_path", "metadata_path",
        "nodule_jsonl_path", "source_instance_count", "shape_zyx", "spacing_zyx_mm", "affine_array_zyx_to_patient",
        "xml_annotation_file_count", "nodule_annotation_count", "mask_count", "status", "error",
    ])
    write_csv(final_root / "labels.csv", label_rows, [
        "annotation_key", "volume_id", "patient_id", "study_instance_uid", "series_instance_uid", "xml_source_path",
        "reader_index", "reader_id", "annotation_version", "nodule_id", "malignancy", "malignancy_original", "roi_count",
        "matched_roi_count", "unmatched_roi_count", "out_of_bounds_polygon_count", "mask_voxel_count", "centroid_voxel_zyx",
        "centroid_patient_xyz", "mask_path", "annotation_status",
    ])
    write_jsonl(final_root / "failures.jsonl", failures)
    shutil.rmtree(staging_root)

    volume_bytes = sum(path.stat().st_size for path in (final_root / "volumes").rglob("*") if path.is_file())
    storage_bytes = sum(path.stat().st_size for path in final_root.rglob("*") if path.is_file())
    stats = {
        "selected_patients": len(selected_patients),
        "selected_patient_ids": sorted(selected_patients),
        "selected_series": selected_series,
        "volumes_created": volumes_created,
        "nodule_annotations_extracted": len(label_rows),
        "masks_created": masks_created,
        "missing_annotation_series": missing_series,
        "unmatched_roi_references": unmatched,
        "failed_series": len(failures),
        "malignancy_original_counts": dict(malignancy_counts),
        "canonical_storage_bytes": storage_bytes,
        "canonical_storage_gib": storage_bytes / (1024**3),
        "full_dataset_volume_storage_estimate_gib": volume_bytes / (1024**3),
        "missing_annotation_details": missing_details,
    }
    result = {"stats": stats, "metadata_rows": metadata_rows, "label_rows": label_rows, "failures": failures, "missing_annotation_series": missing_details}
    write_json(final_root / "statistics.json", stats)
    report = write_validation_report(final_root, result)
    write_json(final_root / "run_config.json", {**config, "workers": 8, "full_dataset": True, "aggregation": "same_filesystem_move"})
    print(json.dumps({"status": report["status"], "counts": report["counts"], "storage": report["storage"]}, indent=2, sort_keys=True))
    return 0 if report["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
