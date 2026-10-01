"""Match XML annotation SOPInstanceUID values to DICOM inventory records."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import clean_text
from .xml_inventory import extract_xml_sop_uids, local_name

LOGGER = logging.getLogger("lidc_phase1.sop_matching")


def _parse_sop_uids(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return extract_xml_sop_uids(root)


def match_sop_uids(
    xml_rows: list[dict[str, Any]],
    dicom_rows: list[dict[str, Any]],
    xml_root: Path,
) -> list[dict[str, Any]]:
    dicom_by_sop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dicom_rows:
        if row.get("status") == "ok" and row.get("SOPInstanceUID"):
            dicom_by_sop[str(row["SOPInstanceUID"])].append(row)

    matches: list[dict[str, Any]] = []
    for xml_row in xml_rows:
        if xml_row.get("status") != "ok":
            continue
        xml_path = xml_root / str(xml_row["source_path"])
        try:
            sop_uids = _parse_sop_uids(xml_path)
        except Exception as exc:  # noqa: BLE001 - inventory already records parse status
            LOGGER.warning("Could not re-read XML for SOP matching %s: %s", xml_path, exc)
            continue
        for sop_uid in sop_uids:
            candidates = dicom_by_sop.get(sop_uid, [])
            candidate = candidates[0] if candidates else None
            if not candidate:
                status = "unmatched"
            elif len(candidates) > 1:
                status = "ambiguous_duplicate_dicom_sop_uid"
            elif xml_row.get("series_instance_uid") and candidate.get("SeriesInstanceUID") != xml_row.get("series_instance_uid"):
                status = "matched_wrong_series"
            elif xml_row.get("study_instance_uid") and candidate.get("StudyInstanceUID") != xml_row.get("study_instance_uid"):
                status = "matched_wrong_study"
            else:
                status = "matched"
            matches.append(
                {
                    "xml_source_path": xml_row["source_path"],
                    "xml_study_instance_uid": xml_row.get("study_instance_uid"),
                    "xml_series_instance_uid": xml_row.get("series_instance_uid"),
                    "xml_sop_instance_uid": sop_uid,
                    "match_status": status,
                    "dicom_source_path": candidate.get("source_path") if candidate else None,
                    "dicom_patient_id": candidate.get("PatientID") if candidate else None,
                    "dicom_study_instance_uid": candidate.get("StudyInstanceUID") if candidate else None,
                    "dicom_series_instance_uid": candidate.get("SeriesInstanceUID") if candidate else None,
                    "dicom_instance_number": candidate.get("InstanceNumber") if candidate else None,
                }
            )
    LOGGER.info(
        "Created %d XML-to-DICOM SOP matches: %d matched, %d unmatched",
        len(matches),
        sum(row["match_status"] == "matched" for row in matches),
        sum(row["match_status"] == "unmatched" for row in matches),
    )
    return matches

