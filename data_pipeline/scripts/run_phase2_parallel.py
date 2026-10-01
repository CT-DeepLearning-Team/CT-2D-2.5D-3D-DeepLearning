"""Run Phase 2 in patient-partitioned workers and deterministically aggregate outputs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scripts.canonical import (
    load_phase1_manifests,
    process_canonical_dataset,
    select_series,
    write_csv,
    write_validation_report,
)
from scripts.common import configure_logging, write_json, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("lidc_phase2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.parent if path.parent.name == "config" else Path.cwd()
    for key in ("raw_dicom_root", "xml_root", "output_root"):
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = (base / value).resolve()
        config[key] = str(value)
    canonical_root = Path(config.get("canonical_output_root", "processed/canonical")).expanduser()
    if not canonical_root.is_absolute():
        canonical_root = (base / canonical_root).resolve()
    config["canonical_output_root"] = str(canonical_root)
    return config


def chunks(values: list[str], count: int) -> list[list[str]]:
    result: list[list[str]] = [[] for _ in range(count)]
    for index, value in enumerate(values):
        result[index % count].append(value)
    return [chunk for chunk in result if chunk]


def rebase_path(value: Any, worker_root: Path) -> Any:
    """Convert worker-relative output paths to repository-relative paths."""
    if not value:
        return value
    path = Path(str(value))
    if not path.is_absolute():
        path = worker_root.parent.parent / path
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(value)


def worker_task(
    worker_index: int,
    patient_ids: list[str],
    manifests: dict[str, list[dict[str, Any]]],
    raw_root: Path,
    xml_root: Path,
    worker_root: Path,
) -> tuple[int, dict[str, Any]]:
    LOGGER.info("Worker %02d starting: %d patients", worker_index, len(patient_ids))
    result = process_canonical_dataset(
        manifests,
        raw_root,
        xml_root,
        worker_root,
        patient_ids=patient_ids,
    )
    LOGGER.info(
        "Worker %02d complete: volumes=%d nodules=%d masks=%d failures=%d",
        worker_index,
        result["stats"]["volumes_created"],
        result["stats"]["nodule_annotations_extracted"],
        result["stats"]["masks_created"],
        result["stats"]["failed_series"],
    )
    return worker_index, result


def copy_worker_artifacts(worker_root: Path, final_root: Path) -> None:
    for directory in ("volumes", "masks"):
        source = worker_root / directory
        if source.exists():
            shutil.copytree(source, final_root / directory, dirs_exist_ok=True)

    source_nodules = worker_root / "nodules"
    target_nodules = final_root / "nodules"
    for source_path in source_nodules.rglob("*.jsonl"):
        relative = source_path.relative_to(source_nodules)
        target_path = target_nodules / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["mask_path"] = rebase_path(row.get("mask_path"), worker_root)
                    rows.append(row)
        write_jsonl(target_path, rows)

    # Sidecars contain a repository-relative volume_path. Rewrite it after
    # moving the worker artifact into the final canonical tree.
    for source_path in (worker_root / "volumes").rglob("*.json"):
        relative = source_path.relative_to(worker_root / "volumes")
        target_path = final_root / "volumes" / relative
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
        metadata["volume_path"] = rebase_path(metadata.get("volume_path"), worker_root)
        write_json(target_path, metadata)


def aggregate_results(
    results: list[tuple[int, dict[str, Any]]],
    final_root: Path,
    selected_patients: list[str],
    selected_series_count: int,
) -> dict[str, Any]:
    metadata_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    malignancy_counts: Counter[str] = Counter()
    volumes_created = masks_created = unmatched = 0

    for worker_index, result in sorted(results):
        worker_root = final_root / ".phase2_workers" / f"worker_{worker_index:02d}"
        copy_worker_artifacts(worker_root, final_root)
        for row in result["metadata_rows"]:
            for key in ("volume_path", "metadata_path", "nodule_jsonl_path"):
                row[key] = rebase_path(row.get(key), worker_root)
            metadata_rows.append(row)
        for row in result["label_rows"]:
            row["mask_path"] = rebase_path(row.get("mask_path"), worker_root)
            label_rows.append(row)
        failures.extend(result["failures"])
        missing.extend(result["missing_annotation_series"])
        stats = result["stats"]
        volumes_created += stats["volumes_created"]
        masks_created += stats["masks_created"]
        unmatched += stats["unmatched_roi_references"]
        malignancy_counts.update(stats["malignancy_original_counts"])

    metadata_fields = [
        "volume_id", "patient_id", "study_instance_uid", "series_instance_uid", "volume_path", "metadata_path",
        "nodule_jsonl_path", "source_instance_count", "shape_zyx", "spacing_zyx_mm", "affine_array_zyx_to_patient",
        "xml_annotation_file_count", "nodule_annotation_count", "mask_count", "status", "error",
    ]
    label_fields = [
        "annotation_key", "volume_id", "patient_id", "study_instance_uid", "series_instance_uid", "xml_source_path",
        "reader_index", "reader_id", "annotation_version", "nodule_id", "malignancy", "malignancy_original", "roi_count",
        "matched_roi_count", "unmatched_roi_count", "out_of_bounds_polygon_count", "mask_voxel_count", "centroid_voxel_zyx",
        "centroid_patient_xyz", "mask_path", "annotation_status",
    ]
    metadata_rows.sort(key=lambda row: str(row["volume_id"]))
    label_rows.sort(key=lambda row: str(row["annotation_key"]))
    write_csv(final_root / "metadata.csv", metadata_rows, metadata_fields)
    write_csv(final_root / "labels.csv", label_rows, label_fields)
    write_jsonl(final_root / "failures.jsonl", failures)

    # Remove only the generated worker staging directory after aggregation.
    shutil.rmtree(final_root / ".phase2_workers")
    storage_bytes = sum(path.stat().st_size for path in final_root.rglob("*") if path.is_file())
    stats = {
        "selected_patients": len(selected_patients),
        "selected_patient_ids": selected_patients,
        "selected_series": selected_series_count,
        "volumes_created": volumes_created,
        "nodule_annotations_extracted": len(label_rows),
        "masks_created": masks_created,
        "missing_annotation_series": len(missing),
        "unmatched_roi_references": unmatched,
        "failed_series": len(failures),
        "malignancy_original_counts": dict(malignancy_counts),
        "canonical_storage_bytes": storage_bytes,
        "canonical_storage_gib": storage_bytes / (1024**3),
        "full_dataset_volume_storage_estimate_gib": None,
        "missing_annotation_details": missing,
    }
    write_json(final_root / "statistics.json", stats)
    return {
        "stats": stats,
        "metadata_rows": metadata_rows,
        "label_rows": label_rows,
        "failures": failures,
        "missing_annotation_series": missing,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    config = load_config(args.config)
    log_path = REPO_ROOT / "logs" / "phase2.log"
    configure_logging(log_path, verbose=args.verbose, logger_name="lidc_phase2")

    manifests = load_phase1_manifests(REPO_ROOT / "processed/manifests")
    selected_series, selected_patients = select_series(manifests["series"])
    final_root = Path(config["canonical_output_root"])
    final_root.mkdir(parents=True, exist_ok=True)
    for directory in ("volumes", "masks", "nodules"):
        (final_root / directory).mkdir(parents=True, exist_ok=True)
    staging_root = final_root / ".phase2_workers"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    patient_chunks = chunks(selected_patients, min(args.workers, len(selected_patients)))
    LOGGER.info("Starting full Phase 2: %d patients, %d CT series, %d workers", len(selected_patients), len(selected_series), len(patient_chunks))
    worker_results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=len(patient_chunks), thread_name_prefix="phase2") as executor:
        futures = {
            executor.submit(
                worker_task,
                worker_index,
                patient_ids,
                manifests,
                Path(config["raw_dicom_root"]),
                Path(config["xml_root"]),
                staging_root / f"worker_{worker_index:02d}",
            ): worker_index
            for worker_index, patient_ids in enumerate(patient_chunks)
        }
        for future in as_completed(futures):
            worker_index = futures[future]
            try:
                worker_results.append(future.result())
            except Exception:  # noqa: BLE001 - retain failure visibility and finish other workers
                LOGGER.exception("Worker %02d failed at worker level", worker_index)
                worker_results.append((worker_index, {
                    "stats": {"volumes_created": 0, "nodule_annotations_extracted": 0, "masks_created": 0, "unmatched_roi_references": 0, "failed_series": len([row for row in selected_series if str(row.get("patient_id")) in set(patient_chunks[worker_index])]), "malignancy_original_counts": {}},
                    "metadata_rows": [], "label_rows": [], "failures": [{"stage": "worker", "worker_index": worker_index, "error": "worker-level exception; see phase2.log"}], "missing_annotation_series": [],
                }))

    result = aggregate_results(worker_results, final_root, selected_patients, len(selected_series))
    report = write_validation_report(final_root, result)
    write_json(final_root / "run_config.json", {**config, "workers": args.workers, "full_dataset": True})
    LOGGER.info("Full Phase 2 complete: %s", json.dumps(report["counts"], sort_keys=True))
    print(json.dumps({"status": report["status"], "counts": report["counts"], "storage": report["storage"]}, indent=2, sort_keys=True))
    return 0 if report["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
