"""Build the V2 three-class dataset from finalized Phase 3 outputs.

The builder copies existing sample/mask arrays and derives manifests. It never
regenerates, resamples, or edits the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger("lidc_phase3_v2")

EXPECTED = {
    "total_nodules": 7385,
    "rated_nodules": 2672,
    "technical_exclusions": 27,
    "supervised_total": 2654,
    "supervised_classes": {"benign": 873, "indeterminate": 1226, "malignant": 555},
    "supervised_split_counts": {
        "train": {"total": 1876, "benign": 622, "indeterminate": 876, "malignant": 378},
        "validation": {"total": 382, "benign": 134, "indeterminate": 163, "malignant": 85},
        "test": {"total": 396, "benign": 117, "indeterminate": 187, "malignant": 92},
    },
    "ssl_train": 5133,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("path/to/final_team_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("path/to/final_team_dataset_v2_3class"))
    parser.add_argument("--resume", action="store_true", help="Resume an existing partial V2 destination without replacing existing array files.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_true(value: str | bool | None) -> bool:
    return value is True or str(value).strip().lower() == "true"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def median_value(row: dict[str, str]) -> float | None:
    value = row.get("malignancy_median", "").strip()
    return float(value) if value else None


def class_v2(row: dict[str, str]) -> str:
    median = median_value(row)
    if median is None:
        return ""
    if 1.0 <= median <= 2.0:
        return "benign"
    if 2.5 <= median <= 3.0:
        return "indeterminate"
    if 3.5 <= median <= 5.0:
        return "malignant"
    raise ValueError(f"Unsupported malignancy median {median} for {row['consensus_nodule_id']}")


def technical_reasons(row: dict[str, str], source_root: Path) -> list[str]:
    reasons: list[str] = []
    if is_true(row.get("suspicious_cluster")):
        reasons.append("suspicious_cluster")
    if not is_true(row.get("crop_coverage_ok")):
        reasons.append("incomplete_crop_coverage")
    for column, reason in (("sample_path", "missing_sample_file"), ("mask_path", "missing_mask_file")):
        value = row.get(column, "")
        if not value or not (source_root / value).is_file():
            reasons.append(reason)
    return reasons


def enrich_rows(source_root: Path, source_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_row in source_rows:
        row = dict(source_row)
        nodule_id = row["consensus_nodule_id"]
        if nodule_id in seen:
            raise ValueError(f"Duplicate consensus_nodule_id: {nodule_id}")
        seen.add(nodule_id)
        reasons = technical_reasons(row, source_root)
        label = class_v2(row)
        rated = bool(row.get("malignancy_median", "").strip())
        qc_ok = not reasons
        row.update(
            {
                "technical_qc_ok": str(qc_ok),
                "technical_exclusion_reason": "|".join(reasons),
                "malignancy_rating_exists_v2": str(rated),
                "class_v2": label,
                "class_index_v2": {"benign": "0", "indeterminate": "1", "malignant": "2", "": ""}[label],
                "midpoint_median_v2": str(median_value(row) in (2.5, 3.5)),
                "supervised_eligible_v2": str(qc_ok and bool(label)),
                "ssl_eligible_v2": str(qc_ok and row.get("split") == "train"),
            }
        )
        rows.append(row)

    technical = [row for row in rows if not is_true(row["technical_qc_ok"])]
    supervised = [row for row in rows if is_true(row["supervised_eligible_v2"])]
    ssl = [row for row in rows if is_true(row["ssl_eligible_v2"])]
    counts: dict[str, object] = {
        "total_nodules": len(rows),
        "rated_nodules": sum(is_true(row["malignancy_rating_exists_v2"]) for row in rows),
        "unrated_nodules": sum(not is_true(row["malignancy_rating_exists_v2"]) for row in rows),
        "technical_exclusions": len(technical),
        "technical_exclusions_by_reason": Counter(
            reason for row in technical for reason in row["technical_exclusion_reason"].split("|") if reason
        ),
        "technical_exclusions_by_split": Counter(row["split"] for row in technical),
        "supervised_total": len(supervised),
        "supervised_class_counts": Counter(row["class_v2"] for row in supervised),
        "supervised_split_counts": {
            split: Counter(row["class_v2"] for row in supervised if row["split"] == split)
            for split in ("train", "validation", "test")
        },
        "ssl_train": len(ssl),
        "supervised_patient_counts": {
            split: len({row["patient_id"] for row in supervised if row["split"] == split})
            for split in ("train", "validation", "test")
        },
        "ssl_patient_count": len({row["patient_id"] for row in ssl}),
        "patients_total": len({row["patient_id"] for row in rows}),
    }
    return rows, counts


def json_ready(value: object) -> object:
    if isinstance(value, Counter):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def assert_expected(counts: dict[str, object]) -> None:
    checks = [
        ("total_nodules", counts["total_nodules"], EXPECTED["total_nodules"]),
        ("rated_nodules", counts["rated_nodules"], EXPECTED["rated_nodules"]),
        ("technical_exclusions", counts["technical_exclusions"], EXPECTED["technical_exclusions"]),
        ("supervised_total", counts["supervised_total"], EXPECTED["supervised_total"]),
        ("supervised_class_counts", json_ready(counts["supervised_class_counts"]), EXPECTED["supervised_classes"]),
        ("ssl_train", counts["ssl_train"], EXPECTED["ssl_train"]),
    ]
    split_counts = counts["supervised_split_counts"]
    for split, expected in EXPECTED["supervised_split_counts"].items():
        counter = split_counts[split]
        actual = {
            "total": sum(counter.values()),
            "benign": counter.get("benign", 0),
            "indeterminate": counter.get("indeterminate", 0),
            "malignant": counter.get("malignant", 0),
        }
        checks.append((f"supervised_split_counts.{split}", actual, expected))
    failures = [(name, actual, expected) for name, actual, expected in checks if actual != expected]
    if failures:
        details = "\n".join(f"{name}: actual={actual!r}, expected={expected!r}" for name, actual, expected in failures)
        raise ValueError("V2 expected-count validation failed before output creation:\n" + details)


def check_space(source_root: Path, output_root: Path, resume: bool = False) -> dict[str, int]:
    source_paths = [
        path
        for subdir in ("samples", "masks")
        for path in (source_root / subdir).glob("*.npy")
        if not path.name.startswith("._")
    ]
    if resume:
        array_bytes = sum(
            source_path.stat().st_size
            for source_path in source_paths
            if not (output_root / source_path.parent.name / source_path.name).is_file()
            or (output_root / source_path.parent.name / source_path.name).stat().st_size != source_path.stat().st_size
        )
    else:
        array_bytes = sum(path.stat().st_size for path in source_paths)
    usage = shutil.disk_usage(output_root.parent)
    required = int(array_bytes * 1.05) + 100 * 1024 * 1024
    if usage.free < required:
        raise OSError(f"Insufficient space: required about {required} bytes, available {usage.free} bytes")
    return {"array_bytes": array_bytes, "required_bytes_with_margin": required, "available_bytes": usage.free}


def copy_arrays(source_root: Path, output_root: Path) -> dict[str, int]:
    for subdir in ("samples", "masks"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    # A single tar stream avoids thousands of slow exFAT metadata transactions.
    archive = subprocess.Popen(
        ["tar", "-C", str(source_root), "--exclude", "._*", "-cf", "-", "samples", "masks"],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    extract = subprocess.Popen(["tar", "-C", str(output_root), "-xf", "-"], stdin=archive.stdout)
    archive.stdout.close()
    extract_code = extract.wait()
    archive_code = archive.wait()
    if extract_code != 0 or archive_code != 0:
        raise RuntimeError(f"tar copy failed: archive={archive_code}, extract={extract_code}")
    copied = {}
    for subdir in ("samples", "masks"):
        copied[subdir] = len([path for path in (output_root / subdir).glob("*.npy") if not path.name.startswith("._")])
        LOGGER.info("Copied %s: %d files", subdir, copied[subdir])
    return copied


def resume_arrays(source_root: Path, output_root: Path) -> dict[str, int]:
    """Copy missing or size-incomplete array paths into an existing destination."""
    copied: dict[str, int] = {}
    for subdir in ("samples", "masks"):
        source_dir = source_root / subdir
        destination_dir = output_root / subdir
        destination_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--size-only", "--exclude", "._*", f"{source_dir}/", f"{destination_dir}/"],
            check=True,
        )
        copied[subdir] = len([path for path in destination_dir.glob("*.npy") if not path.name.startswith("._")])
        LOGGER.info("Resume state for %s: %d files present", subdir, copied[subdir])
    return copied


def copy_source_metadata(source_root: Path, output_root: Path) -> None:
    destination = output_root / "source_metadata"
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {
        "metadata.csv": "original_metadata.csv",
        "labels.csv": "original_labels.csv",
        "splits.csv": "original_splits.csv",
        "excluded_nodules.csv": "original_excluded_nodules.csv",
        "phase3_config.json": "original_phase3_config.json",
        "statistics.json": "original_statistics.json",
        "validation_report.json": "original_validation_report.json",
        "validation_report.md": "original_validation_report.md",
        "README.md": "original_README.md",
        "dataset_card.md": "original_dataset_card.md",
    }
    for source_name, destination_name in mapping.items():
        source_path = source_root / source_name
        if source_path.is_file():
            shutil.copy2(source_path, destination / destination_name)


def validate_leakage(rows: list[dict[str, str]]) -> dict[str, object]:
    patients = {split: {row["patient_id"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")}
    overlaps = {
        f"{left}_vs_{right}": sorted(patients[left] & patients[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    supervised = [row for row in rows if is_true(row["supervised_eligible_v2"])]
    ssl = [row for row in rows if is_true(row["ssl_eligible_v2"])]
    supervised_cohorts = {
        split: [row for row in supervised if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    violations = {
        "all_nodules_split_overlap": overlaps,
        "supervised_train_non_train": sorted({row["patient_id"] for row in supervised_cohorts["train"] if row["split"] != "train"}),
        "supervised_validation_non_validation": sorted({row["patient_id"] for row in supervised_cohorts["validation"] if row["split"] != "validation"}),
        "supervised_test_non_test": sorted({row["patient_id"] for row in supervised_cohorts["test"] if row["split"] != "test"}),
        "ssl_non_train": sorted({row["patient_id"] for row in ssl if row["split"] != "train"}),
    }
    leakage_count = sum(len(value) for value in overlaps.values()) + sum(
        len(value) for key, value in violations.items() if key != "all_nodules_split_overlap"
    )
    return {"leakage_count": leakage_count, "violations": violations}


def validate_arrays(source_root: Path, output_root: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    result = {"missing_sample_count": 0, "missing_mask_count": 0, "array_size_mismatch_count": 0, "array_id_mismatch_count": 0}
    for row in rows:
        for column, subdir, missing_key in (("sample_path", "samples", "missing_sample_count"), ("mask_path", "masks", "missing_mask_count")):
            source_path = source_root / row[column]
            destination_path = output_root / subdir / source_path.name
            if not destination_path.is_file():
                result[missing_key] += 1
                continue
            if source_path.stat().st_size != destination_path.stat().st_size:
                result["array_size_mismatch_count"] += 1
            if destination_path.stem != row["consensus_nodule_id"]:
                result["array_id_mismatch_count"] += 1
    if any(result.values()):
        raise ValueError(f"Copied-array validation failed: {result}")
    return result


def write_docs(output_root: Path, counts: dict[str, object], validation: dict[str, object]) -> None:
    docs = output_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    classes = counts["supervised_class_counts"]
    split_counts = counts["supervised_split_counts"]
    (docs / "README.md").write_text(
        f"""# Final V2 Team Dataset — Three-Class LIDC-IDRI

This self-contained dataset reuses finalized Phase 3 samples and masks without modifying, regenerating, or resampling them.

Use `metadata/supervised_train.csv`, `metadata/supervised_validation.csv`, and `metadata/supervised_test.csv` for supervised learning. Use `metadata/ssl_pretrain_train.csv` only for self-supervised pretraining. All architectures use the same `samples/`, `masks/`, labels, and patient-level splits.

Totals: {counts['total_nodules']} consensus nodules; {counts['rated_nodules']} rated; {counts['unrated_nodules']} unrated; {counts['technical_exclusions']} technical exclusions; {counts['supervised_total']} supervised ({classes['benign']} benign, {classes['indeterminate']} indeterminate, {classes['malignant']} malignant); {counts['ssl_train']} SSL train-patient nodules.

See `dataset_card_v2.md`, `label_policy_v2.md`, `experimental_protocol_v2.md`, and `validation_report_v2.md`.
""",
        encoding="utf-8",
    )
    (docs / "dataset_card_v2.md").write_text(
        f"""# Dataset Card — Final V2 Three-Class Team Dataset

## Scope

This release is derived from LIDC-IDRI consensus nodules. It contains {counts['total_nodules']} consensus nodules and reuses the existing standardized 3D samples and masks exactly as stored in the source dataset.

## Labels

The project-defined operational rule uses `malignancy_median`: 1.0–2.0 benign, 2.5–3.0 indeterminate, and 3.5–5.0 malignant. These are radiologist assessments, not histopathological ground truth.

## Quality control and cohorts

Technical QC excludes suspicious clusters, incomplete crop coverage, and missing array files. Technical exclusions: {counts['technical_exclusions']}. Rated nodules: {counts['rated_nodules']}. Supervised cohort: {counts['supervised_total']}. SSL train-only pool: {counts['ssl_train']}.

## Limitations

Malignancy scores are subjective reader assessments. Midpoint medians are derived from multiple integer ratings and are not original LIDC rating categories. Patient-level splits are inherited from the source release.
""",
        encoding="utf-8",
    )
    (docs / "label_policy_v2.md").write_text(
        """# Label Policy V2

This is a project-defined operational three-class rule, not an original LIDC label taxonomy.

- Median 1.0–2.0 → `benign` (`class_index_v2 = 0`)
- Median 2.5–3.0 → `indeterminate` (`class_index_v2 = 1`)
- Median 3.5–5.0 → `malignant` (`class_index_v2 = 2`)

Original LIDC ratings are integer values 1–5. Values 2.5 and 3.5 arise from medians of multiple reader scores; they are not original radiologist rating categories. `midpoint_median_v2` identifies these cases. Original ratings, reader/session identities, counts, and disagreement information are preserved for sensitivity analyses.
""",
        encoding="utf-8",
    )
    (docs / "experimental_protocol_v2.md").write_text(
        """# Experimental Protocol V2

## Research question

How does increasing spatial CT context from 2D to 2.5D to 3D affect pulmonary nodule malignancy-risk classification?

## Shared protocol

All architectures use the same master samples, labels, patient-level splits, eligibility rules, and evaluation metrics. The 2D model is ResNet-18 on the defined central slice; the 2.5D model is ResNet-18 with fixed neighboring slices as input channels; the 3D model is a lightweight 3D ResNet-10 on the full standardized crop. Outputs are benign, indeterminate, and malignant.

## Two experiments

Experiment A trains from normal/random initialization on `metadata/supervised_train.csv`. Experiment B pretrains self-supervised on `metadata/ssl_pretrain_train.csv`, then fine-tunes on the supervised training cohort. The SSL pretext task is not frozen by this release. Validation data are used for model selection; test data are used only after methodology is frozen.

Use training-only augmentation, dropout where appropriate, weight decay, early stopping, learning curves, and validation-based checkpoint selection. Primary model-selection metric: macro F1. Report accuracy, balanced accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, and one-vs-rest multiclass ROC-AUC.
""",
        encoding="utf-8",
    )
    report_lines = [
        "# Validation Report V2",
        "",
        f"- Total consensus nodules: {counts['total_nodules']}",
        f"- Rated / unrated: {counts['rated_nodules']} / {counts['unrated_nodules']}",
        f"- Technical exclusions: {counts['technical_exclusions']}",
        f"- Supervised total: {counts['supervised_total']}",
        f"- Benign / indeterminate / malignant: {classes['benign']} / {classes['indeterminate']} / {classes['malignant']}",
        f"- SSL train: {counts['ssl_train']}",
        "",
        "| Split | Total | Benign | Indeterminate | Malignant | Patients |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        counter = split_counts[split]
        report_lines.append(f"| {split} | {sum(counter.values())} | {counter.get('benign', 0)} | {counter.get('indeterminate', 0)} | {counter.get('malignant', 0)} | {counts['supervised_patient_counts'][split]} |")
    report_lines += [
        "",
        f"- Patient leakage count: {validation['leakage_count']}",
        "- Missing sample files: 0",
        "- Missing mask files: 0",
        "- Array size mismatches: 0",
        "- Array ID mismatches: 0",
        "",
        "Unrated nodules remain in `metadata/all_nodules_v2.csv` and may be used for SSL only when they pass technical QC and belong to train patients.",
    ]
    (docs / "validation_report_v2.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def build(source_root: Path, output_root: Path, resume: bool = False) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    for name in ("metadata.csv", "labels.csv", "splits.csv"):
        if not (source_root / name).is_file():
            raise FileNotFoundError(f"Missing required source file: {source_root / name}")
    if output_root.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite or reuse destination: {output_root}")
    if resume and not output_root.is_dir():
        raise FileNotFoundError(f"Cannot resume; destination directory does not exist: {output_root}")

    _, source_rows = read_csv(source_root / "metadata.csv")
    rows, counts = enrich_rows(source_root, source_rows)
    assert_expected(counts)
    space = check_space(source_root, output_root, resume=resume)
    LOGGER.info("Preflight passed: %d total, %d supervised, %d SSL, %d technical exclusions", counts["total_nodules"], counts["supervised_total"], counts["ssl_train"], counts["technical_exclusions"])

    output_root.mkdir(parents=True, exist_ok=True)
    copied = resume_arrays(source_root, output_root) if resume else copy_arrays(source_root, output_root)
    copy_source_metadata(source_root, output_root)
    fields = list(rows[0].keys())
    write_csv(output_root / "metadata" / "all_nodules_v2.csv", fields, rows)
    write_csv(output_root / "metadata" / "qc_exclusions.csv", fields, [row for row in rows if not is_true(row["technical_qc_ok"])])
    supervised = [row for row in rows if is_true(row["supervised_eligible_v2"])]
    for split in ("train", "validation", "test"):
        write_csv(output_root / "metadata" / f"supervised_{split}.csv", fields, [row for row in supervised if row["split"] == split])
    write_csv(output_root / "metadata" / "ssl_pretrain_train.csv", fields, [row for row in rows if is_true(row["ssl_eligible_v2"])])

    leakage = validate_leakage(rows)
    array_validation = validate_arrays(source_root, output_root, rows)
    if leakage["leakage_count"] != 0:
        raise ValueError(f"Patient leakage detected: {leakage}")
    counts.update(array_validation)
    counts.update({"source_array_copy_counts": copied, "patient_leakage_count": leakage["leakage_count"], "space_preflight": space})
    (output_root / "statistics_v2.json").write_text(json.dumps(json_ready(counts), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "dataset_version": "final_team_dataset_v2_3class",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_path": str(source_root),
        "label_policy": {"benign": "1.0-2.0", "indeterminate": "2.5-3.0", "malignant": "3.5-5.0", "class_index": {"benign": 0, "indeterminate": 1, "malignant": 2}},
        "technical_qc_policy": ["suspicious_cluster == False", "crop_coverage_ok == True", "sample_path exists", "mask_path exists"],
        "checks_performed": {"expected_count_validation": True, "patient_leakage_count": leakage["leakage_count"], **array_validation},
        "file_structure": {"arrays": ["samples/*.npy", "masks/*.npy"], "metadata": ["metadata/all_nodules_v2.csv", "metadata/qc_exclusions.csv", "metadata/supervised_train.csv", "metadata/supervised_validation.csv", "metadata/supervised_test.csv", "metadata/ssl_pretrain_train.csv"], "docs": ["docs/README.md", "docs/dataset_card_v2.md", "docs/label_policy_v2.md", "docs/experimental_protocol_v2.md", "docs/validation_report_v2.md"]},
    }
    (output_root / "manifest_v2.json").write_text(json.dumps(json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_docs(output_root, counts, leakage)
    return counts


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = build(args.source_root, args.output_root, resume=args.resume)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
