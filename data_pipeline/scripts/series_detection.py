"""CT series grouping, geometry checks, and inclusion decisions."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

LOGGER = logging.getLogger("lidc_phase1.series_detection")

CT_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.2",  # CT Image Storage
    "1.2.840.10008.5.1.4.1.1.2.1",  # Enhanced CT Image Storage
    "1.2.840.10008.5.1.4.1.1.2.2",  # Legacy Converted Enhanced CT Image Storage
}
LOCALIZER_WORDS = ("localizer", "scout", "topogram", "surview", "pilot")


def _float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        value = [value]
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _projection(row: dict[str, Any], normal: list[float]) -> float | None:
    position = _float_list(row.get("ImagePositionPatient"))
    return _dot(position, normal) if position and len(position) == 3 else None


def detect_ct_series(rows: list[dict[str, Any]], xml_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            str(row.get("PatientID") or ""),
            str(row.get("StudyInstanceUID") or ""),
            str(row.get("SeriesInstanceUID") or ""),
        )
        grouped[key].append(row)

    xml_series = {
        (row.get("study_instance_uid"), row.get("series_instance_uid"))
        for row in (xml_rows or [])
        if row.get("status") == "ok"
    }
    output: list[dict[str, Any]] = []
    for (patient_id, study_uid, series_uid), members in sorted(grouped.items()):
        modalities = sorted({str(row.get("Modality") or "") for row in members})
        sop_classes = sorted({str(row.get("SOPClassUID") or "") for row in members})
        descriptions = " ".join(
            str(value or "") for row in members for value in (row.get("SeriesDescription"), row.get("ProtocolName"), row.get("ImageType"))
        ).lower()
        is_ct = "CT" in modalities
        supported_sop_class = any(uid in CT_SOP_CLASS_UIDS for uid in sop_classes)
        localizer = any(word in descriptions for word in LOCALIZER_WORDS)

        orientation = _float_list(members[0].get("ImageOrientationPatient"))
        normal = _cross(orientation[:3], orientation[3:]) if orientation and len(orientation) == 6 else None
        normal_length = _norm(normal) if normal else 0.0
        if normal and normal_length:
            normal = [value / normal_length for value in normal]
        projections = sorted(value for row in members if (value := _projection(row, normal or [])) is not None)
        spacings = [projections[index + 1] - projections[index] for index in range(len(projections) - 1)]
        positive_spacings = [abs(value) for value in spacings if abs(value) > 1e-6]
        median_spacing = sorted(positive_spacings)[len(positive_spacings) // 2] if positive_spacings else None
        spacing_tolerance = max(0.05, (median_spacing or 1.0) * 0.15)
        spacing_anomalies = sum(1 for value in positive_spacings if median_spacing and abs(value - median_spacing) > spacing_tolerance)
        sop_uids = [row.get("SOPInstanceUID") for row in members if row.get("SOPInstanceUID")]
        duplicate_sop_count = len(sop_uids) - len(set(sop_uids))
        has_geometry = all(row.get("has_image_geometry") for row in members)
        xml_linked = (study_uid, series_uid) in xml_series

        reasons: list[str] = []
        if not is_ct:
            reasons.append("non_ct_modality")
        if not supported_sop_class:
            reasons.append("unsupported_or_missing_ct_sop_class")
        if not has_geometry:
            reasons.append("missing_image_geometry")
        if localizer:
            reasons.append("localizer_or_scout_description")
        if len(members) < 2:
            reasons.append("fewer_than_two_instances")
        if duplicate_sop_count:
            reasons.append("duplicate_sop_instance_uid")
        if spacing_anomalies:
            reasons.append("slice_spacing_anomalies")
        included = (
            is_ct
            and supported_sop_class
            and has_geometry
            and not localizer
            and len(members) >= 2
            and duplicate_sop_count == 0
        )
        output.append(
            {
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "series_instance_uid": series_uid,
                "instance_count": len(members),
                "modalities": modalities,
                "sop_class_uids": sop_classes,
                "supported_ct_sop_class": supported_sop_class,
                "series_descriptions": sorted({str(row.get("SeriesDescription") or "") for row in members}),
                "image_type_values": sorted({str(row.get("ImageType") or "") for row in members}),
                "rows": sorted({row.get("Rows") for row in members}),
                "columns": sorted({row.get("Columns") for row in members}),
                "orientation": orientation,
                "slice_position_count": len(projections),
                "slice_spacing_median": median_spacing,
                "slice_spacing_min": min(positive_spacings) if positive_spacings else None,
                "slice_spacing_max": max(positive_spacings) if positive_spacings else None,
                "slice_spacing_anomaly_count": spacing_anomalies,
                "duplicate_sop_instance_uid_count": duplicate_sop_count,
                "xml_linked_by_study_series_uid": xml_linked,
                "included": included,
                "decision": "include" if included else "exclude",
                "exclusion_reasons": reasons,
            }
        )
    LOGGER.info("Detected %d series; %d passed initial CT inclusion", len(output), sum(row["included"] for row in output))
    return output
