"""Run Phase 1 of the LIDC-IDRI preprocessing pipeline.

This phase indexes source data only. It does not write patches, masks, crops,
or deep-learning-ready datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow both `python -m scripts.run_phase1` and `python scripts/run_phase1.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import configure_logging, write_json, write_jsonl  # noqa: E402
from scripts.dicom_inventory import inventory_dicom  # noqa: E402
from scripts.series_detection import detect_ct_series  # noqa: E402
from scripts.sop_matching import match_sop_uids  # noqa: E402
from scripts.validation import build_validation_report, write_reports  # noqa: E402
from scripts.xml_inventory import inventory_xml  # noqa: E402

LOGGER = logging.getLogger("lidc_phase1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--patient-id", action="append", dest="patient_ids", help="Restrict indexing to one or more patient IDs.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.parent if path.parent.name == "config" else Path.cwd()
    for key in ("raw_dicom_root", "xml_root", "output_root"):
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = (base / value).resolve()
        config[key] = str(value)
    return config


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    patient_ids = args.patient_ids or config.get("patient_ids", [])

    output_root = Path(config["output_root"])
    manifests = output_root / "manifests"
    logs = REPO_ROOT / "logs"
    manifests.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    configure_logging(logs / "phase1.log", verbose=args.verbose)

    # Record the resolved configuration before doing work.
    resolved_config = {**config, "patient_ids": patient_ids}
    write_json(manifests / "run_config.json", resolved_config)

    dicom_rows = inventory_dicom(
        Path(config["raw_dicom_root"]),
        patient_ids=patient_ids,
        compute_hashes=bool(config.get("compute_file_hashes", False)),
        workers=int(config.get("dicom_workers", 4)),
    )
    write_jsonl(manifests / "dicom_instances.jsonl", dicom_rows)

    # XML filenames/directories are not guaranteed to encode the DICOM patient
    # ID. Inventory all XML when necessary, then filter by the authoritative
    # DICOM Study/SeriesInstanceUID pair for a patient-scoped run.
    xml_rows = inventory_xml(Path(config["xml_root"]), patient_ids=patient_ids)
    if patient_ids:
        target_series = {
            (row.get("StudyInstanceUID"), row.get("SeriesInstanceUID"))
            for row in dicom_rows
            if row.get("status") == "ok"
        }
        before = len(xml_rows)
        xml_rows = [
            row
            for row in xml_rows
            if row.get("annotation_type") == "lidc_ct"
            and (
                (row.get("study_instance_uid"), row.get("series_instance_uid")) in target_series
                or any(
                    str(row.get("patient_hint") or "").lower() == str(patient_id).lower()
                    for patient_id in patient_ids
                )
            )
        ]
        logging.getLogger("lidc_phase1").info("DICOM UID filter retained %d/%d XML inventory records", len(xml_rows), before)

    # Preserve the complete XML inventory, but only CT LIDC XML participates in
    # CT series linkage and SOP matching. CXR/XML records can reference a CT
    # series in their header while their SOP UIDs belong to DX images.
    ct_xml_rows = [row for row in xml_rows if row.get("status") == "ok" and row.get("annotation_type") == "lidc_ct"]
    write_jsonl(manifests / "xml_files.jsonl", xml_rows)

    # Process series patient-by-patient so a local failure is captured and the
    # remaining patients continue. Inventory itself remains a complete global
    # table and is never discarded because of one patient's error.
    dicom_by_patient: dict[str, list[dict]] = {}
    for row in dicom_rows:
        if row.get("status") != "ok":
            continue
        key = str(row.get("PatientID") or "UNKNOWN")
        dicom_by_patient.setdefault(key, []).append(row)
    series_by_pair = {
        (row.get("StudyInstanceUID"), row.get("SeriesInstanceUID")): str(row.get("PatientID") or "UNKNOWN")
        for row in dicom_rows
        if row.get("status") == "ok"
    }
    xml_by_patient: dict[str, list[dict]] = {}
    for row in ct_xml_rows:
        patient_id = series_by_pair.get((row.get("study_instance_uid"), row.get("series_instance_uid")), "UNMATCHED_XML")
        row["patient_id"] = patient_id
        xml_by_patient.setdefault(patient_id, []).append(row)

    patient_rows: list[dict] = []
    patient_failures: list[dict] = []
    series_rows: list[dict] = []
    for patient_id in sorted(dicom_by_patient):
        try:
            patient_dicom = dicom_by_patient[patient_id]
            patient_xml = xml_by_patient.get(patient_id, [])
            patient_series = detect_ct_series(patient_dicom, patient_xml)
            series_rows.extend(patient_series)
            selected_count = sum(row.get("included", False) for row in patient_series)
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "status": "success" if selected_count else "no_valid_ct_series",
                    "dicom_instance_count": len(patient_dicom),
                    "xml_annotation_file_count": len(patient_xml),
                    "detected_series_count": len(patient_series),
                    "included_ct_series_count": selected_count,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate one patient and continue
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Patient %s failed during CT-series processing", patient_id)
            failure = {
                "patient_id": patient_id,
                "stage": "ct_series_detection",
                "error": error,
            }
            patient_failures.append(failure)
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "status": "failed",
                    "dicom_instance_count": len(dicom_by_patient.get(patient_id, [])),
                    "xml_annotation_file_count": len(xml_by_patient.get(patient_id, [])),
                    "detected_series_count": 0,
                    "included_ct_series_count": 0,
                    "error": error,
                }
            )

    # XML files whose study/series pair was not present in the DICOM inventory
    # are still matched globally and remain visible in the matching table.
    unmatched_xml_patients = sorted(set(xml_by_patient) - set(dicom_by_patient))
    for patient_id in unmatched_xml_patients:
        patient_rows.append(
            {
                "patient_id": patient_id,
                "status": "xml_without_dicom_patient",
                "dicom_instance_count": 0,
                "xml_annotation_file_count": len(xml_by_patient[patient_id]),
                "detected_series_count": 0,
                "included_ct_series_count": 0,
                "error": "No DICOM inventory records matched the XML Study/SeriesInstanceUID.",
            }
        )

    write_jsonl(manifests / "ct_series.jsonl", series_rows)
    write_jsonl(manifests / "patient_processing.jsonl", patient_rows)
    write_jsonl(manifests / "patient_failures.jsonl", patient_failures)

    metadata_rows = [
        {
            "patient_id": row.get("patient_id"),
            "study_instance_uid": row.get("study_instance_uid"),
            "series_instance_uid": row.get("series_instance_uid"),
            "included": row.get("included"),
            "instance_count": row.get("instance_count"),
            "modalities": row.get("modalities"),
            "rows": row.get("rows"),
            "columns": row.get("columns"),
            "orientation": row.get("orientation"),
            "slice_spacing_median": row.get("slice_spacing_median"),
            "slice_spacing_min": row.get("slice_spacing_min"),
            "slice_spacing_max": row.get("slice_spacing_max"),
            "series_descriptions": row.get("series_descriptions"),
            "image_type_values": row.get("image_type_values"),
            "exclusion_reasons": row.get("exclusion_reasons"),
        }
        for row in series_rows
    ]
    write_jsonl(manifests / "series_metadata.jsonl", metadata_rows)

    matches = match_sop_uids(ct_xml_rows, dicom_rows, Path(config["xml_root"]))
    write_jsonl(manifests / "sop_uid_matches.jsonl", matches)

    report = build_validation_report(
        dicom_rows,
        xml_rows,
        series_rows,
        matches,
        resolved_config,
        patient_rows=patient_rows,
        patient_failures=patient_failures,
    )
    write_reports(manifests, report)

    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2, sort_keys=True))
    return 0 if report["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
