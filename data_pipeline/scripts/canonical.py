"""Phase 2 canonical volume and reader-level annotation creation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom

from .common import clean_text, json_default, read_jsonl, write_json, write_jsonl
from .xml_inventory import local_name

LOGGER = logging.getLogger("lidc_phase2")


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def safe_patient_id(patient_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in patient_id)


def text_of(element: ET.Element | None, name: str) -> str | None:
    if element is None:
        return None
    expected = name.lower()
    for child in element.iter():
        if local_name(child.tag).lower() == expected and child.text:
            value = clean_text(child.text)
            if value is not None:
                return value
    return None


def child_text(element: ET.Element, name: str) -> str | None:
    expected = name.lower()
    for child in element:
        if local_name(child.tag).lower() == expected and child.text:
            return clean_text(child.text)
    return None


def descendants(element: ET.Element, name: str) -> list[ET.Element]:
    expected = name.lower()
    return [child for child in element.iter() if local_name(child.tag).lower() == expected]


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def series_key(series_row: dict[str, Any]) -> str:
    identity = "|".join(
        str(series_row.get(field) or "")
        for field in ("patient_id", "study_instance_uid", "series_instance_uid")
    )
    return f"{safe_patient_id(str(series_row.get('patient_id') or 'UNKNOWN'))}__s{stable_id(identity)}"


def load_phase1_manifests(manifest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    names = {
        "dicom": "dicom_instances.jsonl",
        "series": "ct_series.jsonl",
        "xml": "xml_files.jsonl",
        "matches": "sop_uid_matches.jsonl",
    }
    manifests: dict[str, list[dict[str, Any]]] = {}
    for key, name in names.items():
        path = manifest_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 1 manifest is missing: {path}")
        manifests[key] = read_jsonl(path)
    report_path = manifest_dir / "validation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Required Phase 1 validation report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("counts", {}).get("failed_patients", 0):
        raise RuntimeError("Phase 1 contains failed patients; review the Phase 1 validation report before Phase 2.")
    return manifests


def select_series(
    series_rows: list[dict[str, Any]],
    patient_ids: list[str] | None = None,
    max_patients: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected = [row for row in series_rows if row.get("included") and "CT" in row.get("modalities", [])]
    available_patients = sorted({str(row.get("patient_id")) for row in selected})
    if patient_ids:
        requested = {str(patient_id) for patient_id in patient_ids}
        available_patients = [patient for patient in available_patients if patient in requested]
    if max_patients is not None:
        available_patients = available_patients[:max_patients]
    patient_set = set(available_patients)
    return [row for row in selected if str(row.get("patient_id")) in patient_set], available_patients


def _projection(row: dict[str, Any], normal: np.ndarray) -> float:
    position = np.asarray(row.get("ImagePositionPatient"), dtype=np.float64)
    return float(np.dot(position, normal))


def sort_instance_rows(instance_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    if not instance_rows:
        raise ValueError("No DICOM instances are available for the selected series.")
    orientation = np.asarray(instance_rows[0].get("ImageOrientationPatient"), dtype=np.float64)
    if orientation.shape != (6,) or not np.isfinite(orientation).all():
        raise ValueError("Series has invalid ImageOrientationPatient metadata.")
    row_cosines = orientation[:3]
    column_cosines = orientation[3:]
    normal = np.cross(row_cosines, column_cosines)
    norm = np.linalg.norm(normal)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Series has invalid slice-normal geometry.")
    normal = normal / norm
    for row in instance_rows:
        position = row.get("ImagePositionPatient")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"Missing ImagePositionPatient for SOP {row.get('SOPInstanceUID')}.")
    ordered = sorted(instance_rows, key=lambda row: _projection(row, normal))
    return ordered, row_cosines, column_cosines, normal


def reconstruct_volume(
    series_row: dict[str, Any],
    instance_rows: list[dict[str, Any]],
    raw_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, int]]:
    ordered, row_cosines, column_cosines, normal = sort_instance_rows(instance_rows)
    expected_rows = int(series_row["rows"][0])
    expected_columns = int(series_row["columns"][0])
    slices: list[np.ndarray] = []
    sop_to_index: dict[str, int] = {}
    source_paths: list[str] = []
    inversions = 0

    for index, manifest_row in enumerate(ordered):
        path = raw_root / str(manifest_row["source_path"])
        dataset = pydicom.dcmread(path, stop_before_pixels=False, force=False)
        pixels = np.asarray(dataset.pixel_array)
        if pixels.ndim != 2 or pixels.shape != (expected_rows, expected_columns):
            raise ValueError(
                f"Unexpected pixel shape for {manifest_row.get('SOPInstanceUID')}: "
                f"{pixels.shape}, expected {(expected_rows, expected_columns)}"
            )
        if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            bits_stored = int(getattr(dataset, "BitsStored", 16))
            pixels = ((1 << bits_stored) - 1) - pixels
            inversions += 1
        slope = parse_float(getattr(dataset, "RescaleSlope", manifest_row.get("RescaleSlope")), 1.0) or 1.0
        intercept = parse_float(getattr(dataset, "RescaleIntercept", manifest_row.get("RescaleIntercept")), 0.0) or 0.0
        hu = pixels.astype(np.float32) * slope + intercept
        slices.append(hu)
        sop_uid = str(manifest_row["SOPInstanceUID"])
        sop_to_index[sop_uid] = index
        source_paths.append(str(manifest_row["source_path"]))

    volume = np.stack(slices, axis=0).astype(np.float32, copy=False)
    positions = np.asarray([row["ImagePositionPatient"] for row in ordered], dtype=np.float64)
    if len(positions) > 1:
        steps = np.diff(positions, axis=0)
        slice_step = np.median(steps, axis=0)
        slice_spacing = float(np.linalg.norm(slice_step))
    else:
        slice_spacing = parse_float(ordered[0].get("SliceThickness"), 1.0) or 1.0
        slice_step = normal * slice_spacing
    pixel_spacing = np.asarray(ordered[0].get("PixelSpacing"), dtype=np.float64)
    if pixel_spacing.shape != (2,) or not np.isfinite(pixel_spacing).all() or (pixel_spacing <= 0).any():
        raise ValueError("Series has invalid PixelSpacing metadata.")

    affine = np.eye(4, dtype=np.float64)
    # Array axes are (slice, row, column). DICOM's first orientation vector is
    # the direction across columns; the second is the direction down rows.
    affine[:3, 0] = slice_step
    affine[:3, 1] = column_cosines * pixel_spacing[0]
    affine[:3, 2] = row_cosines * pixel_spacing[1]
    affine[:3, 3] = positions[0]
    metadata = {
        "patient_id": series_row.get("patient_id"),
        "study_instance_uid": series_row.get("study_instance_uid"),
        "series_instance_uid": series_row.get("series_instance_uid"),
        "shape_zyx": list(volume.shape),
        "dtype": str(volume.dtype),
        "hu_conversion": "HU = stored_pixel_value * RescaleSlope + RescaleIntercept",
        "spacing_zyx_mm": [slice_spacing, float(pixel_spacing[0]), float(pixel_spacing[1])],
        "image_orientation_patient": ordered[0].get("ImageOrientationPatient"),
        "origin_patient": positions[0].tolist(),
        "affine_array_zyx_to_patient": affine.tolist(),
        "slice_step_patient": slice_step.tolist(),
        "slice_sop_instance_uids_in_order": [str(row["SOPInstanceUID"]) for row in ordered],
        "source_paths_in_order": source_paths,
        "monochrome1_inversions": inversions,
    }
    return volume, affine, metadata, sop_to_index


def polygon_points(roi: ET.Element) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    edge_maps = descendants(roi, "edgeMap")
    if not edge_maps:
        edge_maps = [roi]
    for edge_map in edge_maps:
        xs = [parse_float(child.text) for child in descendants(edge_map, "xCoord")]
        ys = [parse_float(child.text) for child in descendants(edge_map, "yCoord")]
        points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if points:
            polygons.append(points)
    return polygons


def rasterize_polygon(mask: np.ndarray, points: list[tuple[float, float]]) -> int:
    """Rasterize an XML edge-map polygon into a (row, column) mask."""
    if not points:
        return 0
    height, width = mask.shape
    if len(points) == 1:
        x, y = points[0]
        column, row = int(round(x)), int(round(y))
        if 0 <= row < height and 0 <= column < width:
            mask[row, column] = True
            return 1
        return 0
    if len(points) == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        count = max(abs(int(round(x1 - x0))), abs(int(round(y1 - y0))), 1) + 1
        xs = np.rint(np.linspace(x0, x1, count)).astype(int)
        ys = np.rint(np.linspace(y0, y1, count)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[valid], xs[valid]] = True
        return int(valid.sum())

    vertices = np.asarray(points, dtype=np.float64)
    min_x = max(0, int(math.floor(vertices[:, 0].min())))
    max_x = min(width - 1, int(math.ceil(vertices[:, 0].max())))
    min_y = max(0, int(math.floor(vertices[:, 1].min())))
    max_y = min(height - 1, int(math.ceil(vertices[:, 1].max())))
    if min_x > max_x or min_y > max_y:
        return 0
    for row in range(min_y, max_y + 1):
        intersections: list[float] = []
        for index in range(len(vertices)):
            x0, y0 = vertices[index]
            x1, y1 = vertices[(index + 1) % len(vertices)]
            if (y0 <= row < y1) or (y1 <= row < y0):
                intersections.append(x0 + (row - y0) * (x1 - x0) / (y1 - y0))
        intersections.sort()
        for start in range(0, len(intersections) - 1, 2):
            left = max(min_x, int(math.ceil(intersections[start])))
            right = min(max_x, int(math.floor(intersections[start + 1])))
            if left <= right:
                mask[row, left : right + 1] = True
    return int(mask.sum())


def parse_reader_annotations(
    xml_path: Path,
    xml_row: dict[str, Any],
    volume_id: str,
    sop_to_index: dict[str, int],
    volume_shape: tuple[int, int, int],
    affine: np.ndarray,
) -> list[dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    sessions = descendants(root, "readingSession")
    records: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        reader_id = text_of(session, "servicingRadiologistID") or f"reader_{session_index + 1}"
        annotation_version = text_of(session, "annotationVersion")
        nodules = descendants(session, "unblindedReadNodule")
        for nodule_index, nodule in enumerate(nodules):
            nodule_id = child_text(nodule, "noduleID") or f"unnamed_{nodule_index + 1}"
            characteristics_element = next(iter(descendants(nodule, "characteristics")), None)
            characteristics = {
                local_name(child.tag): clean_text(child.text)
                for child in (list(characteristics_element) if characteristics_element is not None else [])
            }
            malignancy = characteristics.get("malignancy") or characteristics.get("cancerAssessment")
            identity = f"{volume_id}|{xml_row['source_path']}|{session_index}|{nodule_index}|{nodule_id}"
            annotation_key = f"{volume_id}__n{stable_id(identity)}"
            mask = np.zeros(volume_shape, dtype=np.uint8)
            contours: list[dict[str, Any]] = []
            matched_roi_count = 0
            unmatched_roi_count = 0
            out_of_bounds_polygon_count = 0
            for roi_index, roi in enumerate(descendants(nodule, "roi")):
                sop_uid = text_of(roi, "imageSOP_UID")
                slice_index = sop_to_index.get(str(sop_uid)) if sop_uid else None
                inclusion = text_of(roi, "inclusion")
                polygons = polygon_points(roi)
                contour = {
                    "roi_index": roi_index,
                    "sop_instance_uid": sop_uid,
                    "image_z_position": parse_float(text_of(roi, "imageZposition")),
                    "inclusion": inclusion,
                    "slice_index": slice_index,
                    "polygons": [[[float(x), float(y)] for x, y in polygon] for polygon in polygons],
                }
                contours.append(contour)
                if slice_index is None or str(inclusion).upper() == "FALSE":
                    unmatched_roi_count += 1 if slice_index is None else 0
                    continue
                matched_roi_count += 1
                slice_mask = mask[slice_index]
                before = int(slice_mask.sum())
                for polygon in polygons:
                    rasterize_polygon(slice_mask, polygon)
                if polygons and int(slice_mask.sum()) == before:
                    out_of_bounds_polygon_count += 1

            voxel_count = int(mask.sum())
            mask_indices = np.argwhere(mask > 0)
            centroid_voxel = mask_indices.mean(axis=0).tolist() if len(mask_indices) else None
            centroid_world = None
            if centroid_voxel is not None:
                homogeneous = np.asarray([*centroid_voxel, 1.0], dtype=np.float64)
                centroid_world = (affine @ homogeneous)[:3].tolist()
            records.append(
                {
                    "annotation_key": annotation_key,
                    "volume_id": volume_id,
                    "patient_id": xml_row.get("patient_id"),
                    "study_instance_uid": xml_row.get("study_instance_uid"),
                    "series_instance_uid": xml_row.get("series_instance_uid"),
                    "xml_source_path": xml_row.get("source_path"),
                    "reader_index": session_index,
                    "reader_id": reader_id,
                    "annotation_version": annotation_version,
                    "nodule_id": nodule_id,
                    "malignancy": parse_int(malignancy, None),
                    "malignancy_original": malignancy,
                    "characteristics": characteristics,
                    "roi_count": len(contours),
                    "matched_roi_count": matched_roi_count,
                    "unmatched_roi_count": unmatched_roi_count,
                    "out_of_bounds_polygon_count": out_of_bounds_polygon_count,
                    "mask_voxel_count": voxel_count,
                    "centroid_voxel_zyx": centroid_voxel,
                    "centroid_patient_xyz": centroid_world,
                    "contours": contours,
                    "mask_array": mask,
                    "annotation_status": "ok" if voxel_count > 0 else "empty_mask",
                }
            )
    return records


def save_nifti(array: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(array, affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=json_default, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def process_canonical_dataset(
    manifests: dict[str, list[dict[str, Any]]],
    raw_root: Path,
    xml_root: Path,
    output_root: Path,
    patient_ids: list[str] | None = None,
    max_patients: int | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("volumes", "masks", "nodules"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    selected_series, selected_patients = select_series(manifests["series"], patient_ids, max_patients)
    selected_pairs = {
        (row.get("study_instance_uid"), row.get("series_instance_uid")): row
        for row in selected_series
    }
    instance_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifests["dicom"]:
        if row.get("status") == "ok":
            pair = (row.get("StudyInstanceUID"), row.get("SeriesInstanceUID"))
            if pair in selected_pairs:
                instance_by_pair[pair].append(row)
    xml_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifests["xml"]:
        if row.get("status") == "ok" and row.get("annotation_type") == "lidc_ct":
            pair = (row.get("study_instance_uid"), row.get("series_instance_uid"))
            if pair in selected_pairs:
                xml_by_pair[pair].append(row)
    match_status: dict[tuple[str, str, str], str] = {
        (row.get("xml_source_path"), row.get("xml_sop_instance_uid"), row.get("xml_series_instance_uid")): row.get("match_status")
        for row in manifests["matches"]
    }

    metadata_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    nodule_json_paths: list[str] = []
    missing_annotation_series: list[dict[str, Any]] = []
    volumes_created = 0
    masks_created = 0
    malignancy_counts: Counter[str] = Counter()

    for index, series_row in enumerate(selected_series, 1):
        pair = (series_row.get("study_instance_uid"), series_row.get("series_instance_uid"))
        volume_id = series_key(series_row)
        patient_directory = safe_patient_id(str(series_row.get("patient_id") or "UNKNOWN"))
        volume_path = output_root / "volumes" / patient_directory / f"{volume_id}.nii.gz"
        metadata_path = output_root / "volumes" / patient_directory / f"{volume_id}.json"
        nodule_path = output_root / "nodules" / patient_directory / f"{volume_id}.jsonl"
        try:
            LOGGER.info("[%d/%d] Reconstructing %s", index, len(selected_series), volume_id)
            volume, affine, volume_metadata, sop_to_index = reconstruct_volume(
                series_row, instance_by_pair.get(pair, []), raw_root
            )
            save_nifti(volume, affine, volume_path)
            volume_metadata.update(
                {
                    "volume_id": volume_id,
                    "source_series_manifest": series_row,
                    "volume_path": str(volume_path.relative_to(output_root.parent.parent)),
                }
            )
            write_json(metadata_path, volume_metadata)
            volumes_created += 1

            xml_rows = xml_by_pair.get(pair, [])
            if not xml_rows:
                missing_annotation_series.append(
                    {"volume_id": volume_id, "patient_id": series_row.get("patient_id"), "series_instance_uid": series_row.get("series_instance_uid")}
                )
            series_nodules: list[dict[str, Any]] = []
            for xml_row in xml_rows:
                xml_row = {**xml_row, "patient_id": series_row.get("patient_id")}
                xml_path = xml_root / str(xml_row["source_path"])
                records = parse_reader_annotations(
                    xml_path,
                    xml_row,
                    volume_id,
                    sop_to_index,
                    tuple(volume.shape),
                    affine,
                )
                for record in records:
                    for contour in record["contours"]:
                        key = (record["xml_source_path"], contour.get("sop_instance_uid"), record["series_instance_uid"])
                        contour["phase1_match_status"] = match_status.get(key, "not_in_phase1_match_table")
                    mask_array = record.pop("mask_array")
                    mask_path = None
                    if record["mask_voxel_count"] > 0:
                        mask_path_obj = output_root / "masks" / patient_directory / volume_id / f"{record['annotation_key']}.nii.gz"
                        save_nifti(mask_array, affine, mask_path_obj)
                        mask_path = str(mask_path_obj.relative_to(output_root.parent.parent))
                        masks_created += 1
                    record["mask_path"] = mask_path
                    series_nodules.append(record)
                    malignancy_counts[str(record.get("malignancy_original"))] += 1
                    label_rows.append(
                        {
                            key: record.get(key)
                            for key in (
                                "annotation_key", "volume_id", "patient_id", "study_instance_uid", "series_instance_uid",
                                "xml_source_path", "reader_index", "reader_id", "annotation_version", "nodule_id",
                                "malignancy", "malignancy_original", "roi_count", "matched_roi_count", "unmatched_roi_count",
                                "out_of_bounds_polygon_count", "mask_voxel_count", "centroid_voxel_zyx", "centroid_patient_xyz",
                                "mask_path", "annotation_status",
                            )
                        }
                    )
            write_jsonl(nodule_path, series_nodules)
            nodule_json_paths.append(str(nodule_path.relative_to(output_root.parent.parent)))
            metadata_rows.append(
                {
                    "volume_id": volume_id,
                    "patient_id": series_row.get("patient_id"),
                    "study_instance_uid": series_row.get("study_instance_uid"),
                    "series_instance_uid": series_row.get("series_instance_uid"),
                    "volume_path": str(volume_path.relative_to(output_root.parent.parent)),
                    "metadata_path": str(metadata_path.relative_to(output_root.parent.parent)),
                    "nodule_jsonl_path": str(nodule_path.relative_to(output_root.parent.parent)),
                    "source_instance_count": len(instance_by_pair.get(pair, [])),
                    "shape_zyx": list(volume.shape),
                    "spacing_zyx_mm": volume_metadata["spacing_zyx_mm"],
                    "affine_array_zyx_to_patient": volume_metadata["affine_array_zyx_to_patient"],
                    "xml_annotation_file_count": len(xml_rows),
                    "nodule_annotation_count": len(series_nodules),
                    "mask_count": sum(record.get("mask_voxel_count", 0) > 0 for record in series_nodules),
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one series must not stop the run
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed canonical processing for %s", volume_id)
            failures.append(
                {
                    "volume_id": volume_id,
                    "patient_id": series_row.get("patient_id"),
                    "study_instance_uid": series_row.get("study_instance_uid"),
                    "series_instance_uid": series_row.get("series_instance_uid"),
                    "stage": "canonical_volume_or_annotation_processing",
                    "error": error,
                }
            )
            metadata_rows.append(
                {
                    "volume_id": volume_id,
                    "patient_id": series_row.get("patient_id"),
                    "study_instance_uid": series_row.get("study_instance_uid"),
                    "series_instance_uid": series_row.get("series_instance_uid"),
                    "volume_path": None,
                    "metadata_path": None,
                    "nodule_jsonl_path": None,
                    "source_instance_count": len(instance_by_pair.get(pair, [])),
                    "shape_zyx": None,
                    "spacing_zyx_mm": None,
                    "affine_array_zyx_to_patient": None,
                    "xml_annotation_file_count": len(xml_by_pair.get(pair, [])),
                    "nodule_annotation_count": 0,
                    "mask_count": 0,
                    "status": "failed",
                    "error": error,
                }
            )

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
    write_csv(output_root / "metadata.csv", metadata_rows, metadata_fields)
    write_csv(output_root / "labels.csv", label_rows, label_fields)
    write_jsonl(output_root / "failures.jsonl", failures)
    write_json(output_root / "run_config.json", {"selected_patients": selected_patients, "selected_series_count": len(selected_series)})

    actual_size_bytes = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    stats = {
        "selected_patients": len(selected_patients),
        "selected_patient_ids": selected_patients,
        "selected_series": len(selected_series),
        "volumes_created": volumes_created,
        "nodule_annotations_extracted": len(label_rows),
        "masks_created": masks_created,
        "missing_annotation_series": len(missing_annotation_series),
        "unmatched_roi_references": sum(row.get("unmatched_roi_count", 0) for row in label_rows),
        "failed_series": len(failures),
        "malignancy_original_counts": dict(malignancy_counts),
        "canonical_storage_bytes": actual_size_bytes,
        "canonical_storage_gib": actual_size_bytes / (1024**3),
        "full_dataset_volume_storage_estimate_gib": (
            (actual_size_bytes / volumes_created) * len([row for row in manifests["series"] if row.get("included")]) / (1024**3)
            if volumes_created else None
        ),
        "missing_annotation_details": missing_annotation_series,
    }
    write_json(output_root / "statistics.json", stats)
    return {
        "stats": stats,
        "metadata_rows": metadata_rows,
        "label_rows": label_rows,
        "failures": failures,
        "missing_annotation_series": missing_annotation_series,
    }


def write_validation_report(output_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    stats = result["stats"]
    issues: list[dict[str, str]] = []
    if result["failures"]:
        issues.append({"severity": "error", "code": "CANONICAL_PROCESSING_FAILURES", "message": f"{len(result['failures'])} selected CT series failed."})
    if result["missing_annotation_series"]:
        issues.append({"severity": "warning", "code": "MISSING_ANNOTATIONS", "message": f"{len(result['missing_annotation_series'])} selected CT series have no CT XML annotation file."})
    if stats["unmatched_roi_references"]:
        issues.append({"severity": "warning", "code": "UNMATCHED_ROI_SOPS", "message": f"{stats['unmatched_roi_references']} ROI references could not be mapped to volume slices."})
    status = "fail" if any(issue["severity"] == "error" for issue in issues) else ("warn" if issues else "pass")
    report = {
        "phase": "phase2_canonical_dataset",
        "status": status,
        "counts": {
            "patients": stats["selected_patients"],
            "selected_ct_series": stats["selected_series"],
            "volumes_created": stats["volumes_created"],
            "nodule_annotations_extracted": stats["nodule_annotations_extracted"],
            "masks_created": stats["masks_created"],
            "missing_annotation_series": stats["missing_annotation_series"],
            "unmatched_roi_references": stats["unmatched_roi_references"],
            "failed_series": stats["failed_series"],
        },
        "storage": {
            "canonical_storage_bytes": stats["canonical_storage_bytes"],
            "canonical_storage_gib": stats["canonical_storage_gib"],
            "full_dataset_volume_storage_estimate_gib": stats["full_dataset_volume_storage_estimate_gib"],
        },
        "issues": issues,
    }
    write_json(output_root / "validation_report.json", report)
    lines = [
        "# Phase 2 Canonical Dataset Validation Report", "", f"- Status: **{status.upper()}**", "",
        "## Counts", "", "| Metric | Count |", "|---|---:|",
    ]
    for key, value in report["counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Storage", "", f"- Canonical test output: `{stats['canonical_storage_gib']:.3f} GiB`", f"- Estimated full selected-volume output: `{stats['full_dataset_volume_storage_estimate_gib']:.3f} GiB`" if stats["full_dataset_volume_storage_estimate_gib"] is not None else "- Estimated full selected-volume output: unavailable", "", "## Issues", ""])
    if issues:
        lines.extend(f"- **{item['severity'].upper()}** `{item['code']}`: {item['message']}" for item in issues)
    else:
        lines.append("No validation issues were recorded.")
    (output_root / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
