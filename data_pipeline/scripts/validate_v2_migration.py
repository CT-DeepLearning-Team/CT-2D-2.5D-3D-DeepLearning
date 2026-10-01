"""Read-only migration integrity validation for the final V2 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TOTAL_NODULES = 7385
EXPECTED_SUPERVISED = {
    "train": {"total": 1876, "benign": 622, "indeterminate": 876, "malignant": 378},
    "validation": {"total": 382, "benign": 134, "indeterminate": 163, "malignant": 85},
    "test": {"total": 396, "benign": 117, "indeterminate": 187, "malignant": 92},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("path/to/final_team_dataset"))
    parser.add_argument("--destination", type=Path, default=Path("path/to/final_team_dataset_v2_3class"))
    parser.add_argument("--random-count", type=int, default=25)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def npy_files(path: Path) -> dict[str, Path]:
    return {item.name: item for item in path.glob("*.npy") if not item.name.startswith("._")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_array_tree(source_dir: Path, destination_dir: Path, random_count: int) -> dict[str, object]:
    source = npy_files(source_dir)
    destination = npy_files(destination_dir)
    missing = sorted(set(source) - set(destination))
    extra = sorted(set(destination) - set(source))
    common = sorted(set(source) & set(destination))
    size_mismatches = [name for name in common if source[name].stat().st_size != destination[name].stat().st_size]
    hash_mismatches: list[str] = []
    hashes_checked = 0
    for index, name in enumerate(common, start=1):
        if sha256(source[name]) != sha256(destination[name]):
            hash_mismatches.append(name)
        hashes_checked += 1
        if index % 1000 == 0:
            print(f"hashed {source_dir.name}: {index}/{len(common)}")
    rng = random.Random(20250923 + len(source_dir.name))
    random_names = rng.sample(common, min(random_count, len(common)))
    random_mismatches: list[str] = []
    try:
        import numpy as np

        for name in random_names:
            left = np.load(source[name], allow_pickle=False)
            right = np.load(destination[name], allow_pickle=False)
            if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
                random_mismatches.append(name)
    except Exception as exc:  # pragma: no cover - environment-specific failure
        random_mismatches.append(f"RANDOM_CHECK_ERROR: {exc}")
    return {
        "source_count": len(source),
        "destination_count": len(destination),
        "missing": missing,
        "extra": extra,
        "common_count": len(common),
        "size_mismatches": size_mismatches,
        "sha256_files_checked": hashes_checked,
        "sha256_mismatches": hash_mismatches,
        "random_files_checked": len(random_names),
        "random_mismatches": random_mismatches,
    }


def check_required_structure(source: Path, destination: Path) -> dict[str, object]:
    required_dirs = ["samples", "masks", "metadata", "docs"]
    required_files = ["statistics_v2.json", "manifest_v2.json"]
    missing_dirs = [name for name in required_dirs if not (destination / name).is_dir()]
    missing_files = [name for name in required_files if not (destination / name).is_file()]
    source_top = sorted(item.name for item in source.iterdir())
    destination_top = sorted(item.name for item in destination.iterdir())
    intentional = {"samples", "masks", "metadata", "docs", "source_metadata", "statistics_v2.json", "manifest_v2.json", "RESUME_STATUS.md"}
    apple_double = sorted(name for name in destination_top if name.startswith("._"))
    unexpected_destination = sorted(name for name in destination_top if name not in intentional and not name.startswith("._"))
    return {
        "required_directories": required_dirs,
        "required_files": required_files,
        "missing_directories": missing_dirs,
        "missing_files": missing_files,
        "source_top_level": source_top,
        "destination_top_level": destination_top,
        "intentional_v2_additions": ["metadata/", "docs/", "source_metadata/", "statistics_v2.json", "manifest_v2.json"],
        "ignored_apple_double_entries": apple_double,
        "unexpected_destination_top_level": unexpected_destination,
    }


def check_metadata(source: Path, destination: Path) -> dict[str, object]:
    rows = read_csv(destination / "metadata" / "all_nodules_v2.csv")
    sample_names = set(npy_files(destination / "samples"))
    mask_names = set(npy_files(destination / "masks"))
    missing_samples = sorted({Path(row["sample_path"]).name for row in rows if Path(row["sample_path"]).name not in sample_names})
    missing_masks = sorted({Path(row["mask_path"]).name for row in rows if Path(row["mask_path"]).name not in mask_names})
    ids = [row["consensus_nodule_id"] for row in rows]
    original_metadata_match = sha256(source / "metadata.csv") == sha256(destination / "source_metadata" / "original_metadata.csv")
    return {
        "row_count": len(rows),
        "unique_consensus_nodule_ids": len(set(ids)),
        "missing_sample_references": missing_samples,
        "missing_mask_references": missing_masks,
        "original_metadata_copy_sha256_match": original_metadata_match,
    }


def check_supervised(destination: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for split, expected in EXPECTED_SUPERVISED.items():
        rows = read_csv(destination / "metadata" / f"supervised_{split}.csv")
        classes = Counter(row["class_v2"] for row in rows)
        actual = {"total": len(rows), **{label: classes.get(label, 0) for label in ("benign", "indeterminate", "malignant")}}
        result[split] = {"actual": actual, "expected": expected, "match": actual == expected}
    return result


def check_leakage(destination: Path) -> dict[str, object]:
    rows = read_csv(destination / "metadata" / "all_nodules_v2.csv")
    by_split = {split: {row["patient_id"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")}
    overlaps = {
        f"{left}_vs_{right}": sorted(by_split[left] & by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    ssl = read_csv(destination / "metadata" / "ssl_pretrain_train.csv")
    ssl_patients = {row["patient_id"] for row in ssl}
    return {
        "split_patient_counts": {split: len(patients) for split, patients in by_split.items()},
        "split_overlaps": overlaps,
        "leakage_count": sum(len(value) for value in overlaps.values()),
        "ssl_rows": len(ssl),
        "ssl_non_train_rows": sum(row["split"] != "train" for row in ssl),
        "ssl_validation_patient_overlap": len(ssl_patients & by_split["validation"]),
        "ssl_test_patient_overlap": len(ssl_patients & by_split["test"]),
    }


def render_report(result: dict[str, object], source: Path, destination: Path) -> str:
    structure = result["structure"]
    samples = result["samples"]
    masks = result["masks"]
    metadata = result["metadata"]
    supervised = result["supervised"]
    ssl = result["leakage"]
    failures = result["failures"]
    status = "PASS" if not failures else "FAIL"
    lines = [
        "# V2 Migration Validation Report",
        "",
        f"- Migration status: **{status}**",
        f"- Source: `{source}`",
        f"- Destination: `{destination}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Folder structure",
        "",
        f"- Missing required directories: `{structure['missing_directories']}`",
        f"- Missing required files: `{structure['missing_files']}`",
        f"- Unexpected destination top-level entries: `{structure['unexpected_destination_top_level']}`",
        f"- Ignored macOS AppleDouble filesystem entries: `{structure['ignored_apple_double_entries']}`",
        "- V2 layout additions (`metadata/`, `docs/`, `source_metadata/`, V2 reports) are intentional.",
        "",
        "## Array migration",
        "",
        "| Tree | Source | Destination | Missing | Extra | SHA-256 checked | Hash mismatches | Random checked | Random mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, check in (("samples", samples), ("masks", masks)):
        lines.append(f"| {label} | {check['source_count']} | {check['destination_count']} | {len(check['missing'])} | {len(check['extra'])} | {check['sha256_files_checked']} | {len(check['sha256_mismatches'])} | {check['random_files_checked']} | {len(check['random_mismatches'])} |")
    lines += [
        "",
        "All common array filenames were compared by file size and SHA-256. Random files were additionally loaded with NumPy and compared for shape, dtype, dimensions, and element equality.",
        "",
        "## Metadata",
        "",
        f"- `all_nodules_v2.csv` rows: **{metadata['row_count']}** (expected {TOTAL_NODULES})",
        f"- Unique consensus IDs: **{metadata['unique_consensus_nodule_ids']}**",
        f"- Missing sample references: **{len(metadata['missing_sample_references'])}**",
        f"- Missing mask references: **{len(metadata['missing_mask_references'])}**",
        f"- Preserved original metadata SHA-256 match: **{metadata['original_metadata_copy_sha256_match']}**",
        "",
        "## Supervised cohorts",
        "",
        "| Split | Actual | Expected | Match |",
        "|---|---|---|---|",
    ]
    for split, check in supervised.items():
        lines.append(f"| {split} | `{check['actual']}` | `{check['expected']}` | {check['match']} |")
    lines += [
        "",
        f"- SSL rows: **{ssl['ssl_rows']}** (expected 5133)",
        f"- SSL non-train rows: **{ssl['ssl_non_train_rows']}**",
        f"- SSL/validation patient overlap: **{ssl['ssl_validation_patient_overlap']}**",
        f"- SSL/test patient overlap: **{ssl['ssl_test_patient_overlap']}**",
        "",
        "## Patient leakage",
        "",
        f"- Train/validation/test patient leakage count: **{ssl['leakage_count']}**",
        f"- Split patient counts: `{ssl['split_patient_counts']}`",
        "",
        "## Conclusion",
        "",
    ]
    if failures:
        lines.append("Dataset migration validation failed. See the failed checks above and the machine-readable mismatch details below.")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(failures, indent=2))
        lines.append("```")
    else:
        lines.append("Dataset is safe for team sharing")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    structure = check_required_structure(source, destination)
    samples = compare_array_tree(source / "samples", destination / "samples", args.random_count)
    masks = compare_array_tree(source / "masks", destination / "masks", args.random_count)
    metadata = check_metadata(source, destination)
    supervised = check_supervised(destination)
    leakage = check_leakage(destination)
    failures: dict[str, object] = {}
    if structure["missing_directories"] or structure["missing_files"] or structure["unexpected_destination_top_level"]:
        failures["structure"] = structure
    for name, check in (("samples", samples), ("masks", masks)):
        if check["missing"] or check["extra"] or check["size_mismatches"] or check["sha256_mismatches"] or check["random_mismatches"]:
            failures[name] = check
    if metadata["row_count"] != TOTAL_NODULES or metadata["unique_consensus_nodule_ids"] != TOTAL_NODULES or metadata["missing_sample_references"] or metadata["missing_mask_references"] or not metadata["original_metadata_copy_sha256_match"]:
        failures["metadata"] = metadata
    if any(not check["match"] for check in supervised.values()):
        failures["supervised"] = supervised
    if leakage["leakage_count"] or leakage["ssl_non_train_rows"] or leakage["ssl_validation_patient_overlap"] or leakage["ssl_test_patient_overlap"] or leakage["ssl_rows"] != 5133:
        failures["leakage_or_ssl"] = leakage
    result = {"structure": structure, "samples": samples, "masks": masks, "metadata": metadata, "supervised": supervised, "leakage": leakage, "failures": failures}
    report = render_report(result, source, destination)
    if args.write_report:
        report_path = destination / "docs" / "V2_MIGRATION_VALIDATION_REPORT.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"Wrote {report_path}")
    print(report)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
