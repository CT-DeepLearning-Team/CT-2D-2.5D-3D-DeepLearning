"""Validation metrics and human-readable Phase 1 report generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json


def build_validation_report(
    dicom_rows: list[dict[str, Any]],
    xml_rows: list[dict[str, Any]],
    series_rows: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    config: dict[str, Any],
    patient_rows: list[dict[str, Any]] | None = None,
    patient_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    patient_rows = patient_rows or []
    patient_failures = patient_failures or []
    match_counts = Counter(row["match_status"] for row in matches)
    dicom_status = Counter(row.get("status") for row in dicom_rows)
    xml_status = Counter(row.get("status") for row in xml_rows)
    included = [row for row in series_rows if row.get("included")]
    issues: list[dict[str, str]] = []

    def issue(severity: str, code: str, message: str) -> None:
        issues.append({"severity": severity, "code": code, "message": message})

    if not dicom_rows:
        issue("error", "NO_DICOM_FILES", "No DICOM files were inventoried.")
    if dicom_status["ok"] == 0:
        issue("error", "NO_READABLE_DICOM", "No readable DICOM headers were found.")
    if not series_rows:
        issue("error", "NO_SERIES", "No DICOM series were detected.")
    if not included:
        issue("error", "NO_INCLUDED_CT_SERIES", "No series passed the initial CT inclusion rules.")
    if patient_failures:
        issue("error", "PATIENT_PROCESSING_FAILURES", f"{len(patient_failures)} patients failed during patient-level processing.")
    if xml_status["unreadable"]:
        issue("warning", "UNREADABLE_XML", f"{xml_status['unreadable']} XML files could not be parsed.")
    if config.get("patient_ids") and not xml_rows:
        issue("warning", "NO_XML_FOR_FILTERED_PATIENT", "No XML annotation file matched the filtered DICOM patient study/series UIDs.")
    if match_counts["unmatched"]:
        issue("warning", "UNMATCHED_XML_SOPS", f"{match_counts['unmatched']} XML SOP references were not found in the DICOM inventory.")
    if match_counts["matched_wrong_series"] or match_counts["matched_wrong_study"]:
        issue("error", "UID_CONTEXT_MISMATCH", "At least one SOP UID matched a DICOM object in the wrong study or series.")
    duplicate_sops = sum(row.get("duplicate_sop_instance_uid_count", 0) for row in series_rows)
    if duplicate_sops:
        issue("error", "DUPLICATE_SOP_UID", f"{duplicate_sops} duplicate SOP UID occurrences were detected within series.")

    status = "fail" if any(item["severity"] == "error" for item in issues) else ("warn" if issues else "pass")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_phase": "phase1_indexing",
        "status": status,
        "config": config,
        "counts": {
            "dicom_inventory_records": len(dicom_rows),
            "readable_dicom_records": dicom_status["ok"],
            "unreadable_dicom_records": dicom_status["unreadable"],
            "xml_inventory_records": len(xml_rows),
            "readable_xml_records": xml_status["ok"],
            "unreadable_xml_records": xml_status["unreadable"],
            "detected_series": len(series_rows),
            "ct_series_found": sum("CT" in row.get("modalities", []) for row in series_rows),
            "non_ct_series_detected": sum("CT" not in row.get("modalities", []) for row in series_rows),
            "included_ct_series": len(included),
            "excluded_series": len(series_rows) - len(included),
            "sop_references": len(matches),
            "matched_sop_references": match_counts["matched"],
            "unmatched_sop_references": match_counts["unmatched"],
            "ambiguous_sop_references": match_counts["ambiguous_duplicate_dicom_sop_uid"],
            "wrong_context_sop_references": match_counts["matched_wrong_series"] + match_counts["matched_wrong_study"],
            "total_patients": len(patient_rows),
            "successful_patients": sum(row.get("status") == "success" for row in patient_rows),
            "patients_without_valid_ct_series": sum(row.get("status") == "no_valid_ct_series" for row in patient_rows),
            "failed_patients": len(patient_failures),
            "xml_without_dicom_patients": sum(row.get("status") == "xml_without_dicom_patient" for row in patient_rows),
            "ct_xml_annotation_files": sum(row.get("annotation_type") == "lidc_ct" and row.get("status") == "ok" for row in xml_rows),
        },
        "breakdowns": {
            "dicom_status": dict(dicom_status),
            "xml_status": dict(xml_status),
            "sop_match_status": dict(match_counts),
            "series_decisions": dict(Counter(row.get("decision") for row in series_rows)),
            "series_exclusion_reasons": dict(
                Counter(reason for row in series_rows for reason in row.get("exclusion_reasons", []))
            ),
        },
        "issues": issues,
        "patient_failures": patient_failures,
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    counts = report["counts"]
    lines = [
        "# LIDC-IDRI Phase 1 Validation Report",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Generated: `{report['generated_at_utc']}`",
        "",
        "## Counts",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for item in report["issues"]:
            lines.append(f"- **{item['severity'].upper()}** `{item['code']}`: {item['message']}")
    else:
        lines.append("No validation issues were recorded.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    write_json(output_dir / "validation_report.json", report)
    write_markdown_report(output_dir / "validation_report.md", report)
