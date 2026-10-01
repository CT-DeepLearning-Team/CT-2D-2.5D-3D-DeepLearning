"""DICOM file discovery and header inventory."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .common import clean_text, relative_path, sha256_file, uid_matches_patient

LOGGER = logging.getLogger("lidc_phase1.dicom_inventory")

try:
    import pydicom
except ImportError:  # pragma: no cover - exercised in the user's environment
    pydicom = None


TAGS: dict[str, str] = {
    "PatientID": "PatientID",
    "StudyInstanceUID": "StudyInstanceUID",
    "SeriesInstanceUID": "SeriesInstanceUID",
    "SOPInstanceUID": "SOPInstanceUID",
    "SOPClassUID": "SOPClassUID",
    "TransferSyntaxUID": "TransferSyntaxUID",
    "Modality": "Modality",
    "SeriesDescription": "SeriesDescription",
    "ProtocolName": "ProtocolName",
    "ImageType": "ImageType",
    "InstanceNumber": "InstanceNumber",
    "ImagePositionPatient": "ImagePositionPatient",
    "ImageOrientationPatient": "ImageOrientationPatient",
    "PixelSpacing": "PixelSpacing",
    "SliceThickness": "SliceThickness",
    "SpacingBetweenSlices": "SpacingBetweenSlices",
    "Rows": "Rows",
    "Columns": "Columns",
    "BitsAllocated": "BitsAllocated",
    "BitsStored": "BitsStored",
    "HighBit": "HighBit",
    "PixelRepresentation": "PixelRepresentation",
    "PhotometricInterpretation": "PhotometricInterpretation",
    "RescaleIntercept": "RescaleIntercept",
    "RescaleSlope": "RescaleSlope",
    "WindowCenter": "WindowCenter",
    "WindowWidth": "WindowWidth",
}
NUMERIC_SCALARS = {
    "InstanceNumber",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
}
NUMERIC_VECTORS = {
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
}
NUMERIC_FLOATS = {
    "SliceThickness",
    "SpacingBetweenSlices",
    "RescaleIntercept",
    "RescaleSlope",
}
SPECIFIC_TAGS = []
if pydicom is not None:
    SPECIFIC_TAGS = [
        pydicom.tag.Tag(tag)
        for tag in (
            0x00100020,  # PatientID
            0x0020000D,  # StudyInstanceUID
            0x0020000E,  # SeriesInstanceUID
            0x00080018,  # SOPInstanceUID
            0x00080016,  # SOPClassUID
            0x00080060,  # Modality
            0x0008103E,  # SeriesDescription
            0x00181030,  # ProtocolName
            0x00080008,  # ImageType
            0x00200013,  # InstanceNumber
            0x00200032,  # ImagePositionPatient
            0x00200037,  # ImageOrientationPatient
            0x00280030,  # PixelSpacing
            0x00180050,  # SliceThickness
            0x00180088,  # SpacingBetweenSlices
            0x00280010,  # Rows
            0x00280011,  # Columns
            0x00280100,  # BitsAllocated
            0x00280101,  # BitsStored
            0x00280102,  # HighBit
            0x00280103,  # PixelRepresentation
            0x00280004,  # PhotometricInterpretation
            0x00281052,  # RescaleIntercept
            0x00281053,  # RescaleSlope
            0x00281050,  # WindowCenter
            0x00281051,  # WindowWidth
        )
    ]


def _value(dataset: Any, name: str) -> Any:
    value = getattr(dataset, name, None)
    if value is None:
        return None
    if name in NUMERIC_VECTORS:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return [str(item) for item in value]
    if name in NUMERIC_SCALARS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)
    if name in NUMERIC_FLOATS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return clean_text(value)


def _require_pydicom() -> None:
    if pydicom is None:
        raise RuntimeError(
            "pydicom is required for DICOM inventory. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        )


def inventory_dicom(
    raw_root: Path,
    patient_ids: list[str] | None = None,
    compute_hashes: bool = False,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Read DICOM headers recursively without loading pixel arrays."""
    _require_pydicom()
    patient_ids = patient_ids or []
    if not raw_root.exists():
        raise FileNotFoundError(f"DICOM root does not exist: {raw_root}")

    scan_roots = [raw_root]
    if patient_ids:
        exact_roots = [raw_root / patient_id for patient_id in patient_ids if (raw_root / patient_id).is_dir()]
        if not exact_roots:
            # LIDC exports normally use LIDC-IDRI-#### directory names. Resolve a
            # requested numeric suffix without recursively enumerating the dataset.
            shallow_roots = [
                child
                for child in raw_root.iterdir()
                if child.is_dir() and uid_matches_patient(child, patient_ids)
            ]
            exact_roots = shallow_roots
        if exact_roots:
            scan_roots = exact_roots
    files = sorted(
        path
        for root in scan_roots
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
        and path.name not in {".DS_Store", "LICENSE"}
        and path.suffix.lower() not in {".xml", ".json", ".txt", ".csv", ".jpg", ".jpeg", ".png"}
    )
    if patient_ids and scan_roots == [raw_root]:
        # If directory naming does not expose the requested patient, scan all
        # files so PatientID can still be used as the authoritative filter.
        files = [path for path in files if uid_matches_patient(path, patient_ids)] or files
    LOGGER.info("Scanning %d candidate files under %s", len(files), raw_root)

    patient_id_set = {item.lower() for item in patient_ids}

    def inventory_one(path: Path) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source_path": relative_path(path, raw_root),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if compute_hashes else None,
            "status": "unreadable",
            "error": None,
        }
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=False,
                specific_tags=SPECIFIC_TAGS,
            )
            for output_name, dataset_name in TAGS.items():
                if output_name == "TransferSyntaxUID":
                    value = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
                    row[output_name] = clean_text(value)
                else:
                    row[output_name] = _value(dataset, dataset_name)
            row["status"] = "ok"
            row["has_image_geometry"] = bool(row.get("Rows") and row.get("Columns"))
            row["patient_filter_match"] = (
                not patient_ids
                or str(row.get("PatientID") or "").lower() in patient_id_set
                or uid_matches_patient(path, patient_ids)
            )
        except Exception as exc:  # noqa: BLE001 - inventory must record bad files and continue
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    rows: list[dict[str, Any]] = []
    worker_count = max(1, int(workers))
    LOGGER.info("Reading DICOM headers with %d worker(s)", worker_count)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dicom-header") as executor:
        for index, row in enumerate(executor.map(inventory_one, files), 1):
            if row.get("status") != "ok":
                LOGGER.warning("Unreadable DICOM candidate %s: %s", row["source_path"], row.get("error"))
            rows.append(row)
            if index % 1000 == 0:
                LOGGER.info("Scanned %d/%d files", index, len(files))

    if patient_ids:
        before = len(rows)
        rows = [row for row in rows if row.get("patient_filter_match", False)]
        LOGGER.info("Patient filter retained %d/%d inventory records", len(rows), before)
    return rows
