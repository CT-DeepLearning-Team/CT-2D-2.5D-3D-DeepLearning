"""Inventory and lightweight structural parsing for LIDC XML annotations."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from .common import clean_text, relative_path, uid_matches_patient

LOGGER = logging.getLogger("lidc_phase1.xml_inventory")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def elements(root: ET.Element, name: str) -> Iterable[ET.Element]:
    expected = name.lower()
    return (element for element in root.iter() if local_name(element.tag).lower() == expected)


def first_text(root: ET.Element, names: tuple[str, ...]) -> str | None:
    wanted = {name.lower() for name in names}
    for element in root.iter():
        if local_name(element.tag).lower() in wanted and element.text:
            value = clean_text(element.text)
            if value:
                return value
    return None


def _patient_hint(path: Path) -> str | None:
    text = path.stem
    match = re.search(r"(\d{4})", text)
    return f"LIDC-IDRI-{match.group(1)}" if match else None


def _patient_hint_matches(hint: str | None, patient_ids: list[str]) -> bool:
    if not patient_ids:
        return True
    hint_digits = re.search(r"(\d{4})$", (hint or "").lower())
    for patient_id in patient_ids:
        normalized = patient_id.lower()
        if normalized == (hint or "").lower():
            return True
        patient_digits = re.search(r"(\d{4})$", normalized)
        if patient_digits and hint_digits and patient_digits.group(1) == hint_digits.group(1):
            return True
    return False


def extract_xml_sop_uids(root: ET.Element) -> list[str]:
    values: list[str] = []
    for element in root.iter():
        if local_name(element.tag).lower() == "imagesop_uid" and element.text:
            value = clean_text(element.text)
            if value:
                values.append(value)
    return values


def inventory_xml(xml_root: Path, patient_ids: list[str] | None = None) -> list[dict[str, Any]]:
    patient_ids = patient_ids or []
    if not xml_root.exists():
        raise FileNotFoundError(f"XML root does not exist: {xml_root}")
    paths = sorted(
        path
        for path in xml_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xml" and not path.name.startswith("._")
    )
    if patient_ids:
        candidates = [path for path in paths if uid_matches_patient(path, patient_ids)]
        if candidates:
            paths = candidates
    LOGGER.info("Scanning %d XML files under %s", len(paths), xml_root)

    rows: list[dict[str, Any]] = []
    for path in paths:
        row: dict[str, Any] = {
            "source_path": relative_path(path, xml_root),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "status": "unreadable",
            "error": None,
        }
        try:
            root = ET.parse(path).getroot()
            sop_uids = extract_xml_sop_uids(root)
            sessions = list(elements(root, "readingSession"))
            nodules = list(elements(root, "unblindedReadNodule"))
            non_nodules = list(elements(root, "nonNodule"))
            row.update(
                {
                    "status": "ok",
                    "root_element": local_name(root.tag),
                    "annotation_type": "lidc_ct" if local_name(root.tag).lower() == "lidcreadmessage" else "other",
                    "modality": first_text(root, ("Modality",)),
                    "patient_hint": _patient_hint(path),
                    "study_instance_uid": first_text(root, ("StudyInstanceUID", "StudyInstanceUid")),
                    "series_instance_uid": first_text(root, ("SeriesInstanceUID", "SeriesInstanceUid")),
                    "ct_series_instance_uid": first_text(root, ("CTSeriesInstanceUID", "CTSeriesInstanceUid")),
                    "xml_version": first_text(root, ("Version", "annotationVersion")),
                    "reading_session_count": len(sessions),
                    "nodule_count": len(nodules),
                    "non_nodule_count": len(non_nodules),
                    "sop_instance_uid_count": len(sop_uids),
                    "unique_sop_instance_uid_count": len(set(sop_uids)),
                    "sop_instance_uids": sorted(set(sop_uids)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - retain bad XML in the inventory
            row["error"] = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Unreadable XML %s: %s", path, exc)
        rows.append(row)
    # Some TCIA XML exports use numeric directory/file names that cannot be
    # mapped to LIDC patient IDs. In that case retain the full XML inventory;
    # the orchestrator will filter it by DICOM Study/SeriesInstanceUID.
    if patient_ids and paths != sorted(
        path
        for path in xml_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xml" and not path.name.startswith("._")
    ):
        before = len(rows)
        rows = [
            row
            for row in rows
            if _patient_hint_matches(row.get("patient_hint"), patient_ids)
            or uid_matches_patient(Path(str(row["source_path"])), patient_ids)
        ]
        LOGGER.info("Patient path filter retained %d/%d XML inventory records", len(rows), before)
    return rows
