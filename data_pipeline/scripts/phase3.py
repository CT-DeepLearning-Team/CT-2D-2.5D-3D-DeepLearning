"""Phase 3 consensus nodule and shared master-sample dataset creation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import math
import random
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .common import json_default, write_json


LOGGER = logging.getLogger("lidc_phase3")
REPO_ROOT = Path(__file__).resolve().parents[1]


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def parse_json_field(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def resolve_path(value: str | Path, canonical_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Phase 2 stores repository-relative paths such as processed/canonical/… .
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return canonical_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reader_session_key(record: dict[str, Any]) -> str:
    """Return a stable independent reading-session identity.

    The de-identified LIDC XML sometimes stores ``reader_id`` as ``anon``.
    The XML source file plus reading-session index remains the reliable
    session-level identity for consensus counting.
    """
    source = str(record.get("xml_source_path") or "")
    index = record.get("reader_index")
    if source and index not in (None, ""):
        return f"{source}::reader_index={index}"
    return f"reader_id={record.get('reader_id') or 'unknown'}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            converted = {}
            for key, value in row.items():
                converted[key] = json.dumps(value, default=json_default, sort_keys=True) if isinstance(value, (list, dict)) else value
            writer.writerow(converted)


def load_canonical(canonical_root: Path, patient_ids: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_rows = read_csv(canonical_root / "metadata.csv")
    selected_metadata = [
        row for row in metadata_rows
        if row.get("status") == "ok" and (not patient_ids or row.get("patient_id") in patient_ids)
    ]
    records: list[dict[str, Any]] = []
    for metadata in selected_metadata:
        nodule_path = resolve_path(metadata["nodule_jsonl_path"], canonical_root)
        if not nodule_path.exists():
            raise FileNotFoundError(f"Canonical nodule record is missing: {nodule_path}")
        with nodule_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["shape_zyx"] = parse_json_field(metadata.get("shape_zyx"), [])
                record["spacing_zyx_mm"] = parse_json_field(metadata.get("spacing_zyx_mm"), [])
                record["affine_array_zyx_to_patient"] = parse_json_field(metadata.get("affine_array_zyx_to_patient"), [])
                record["canonical_volume_path"] = resolve_path(metadata["volume_path"], canonical_root)
                record["canonical_volume_metadata_path"] = resolve_path(metadata["metadata_path"], canonical_root)
                record["canonical_nodule_path"] = nodule_path
                record["source_line_number"] = line_number
                records.append(record)
    return selected_metadata, records


def select_evenly_spaced_patients(all_patient_ids: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(all_patient_ids):
        return list(all_patient_ids)
    indices = np.linspace(0, len(all_patient_ids) - 1, count, dtype=int).tolist()
    return [all_patient_ids[index] for index in sorted(set(indices))]


def point_inside(value: float, lower: float, upper: float) -> bool:
    return lower - 1e-6 <= value <= upper + 1e-6


def rasterize_polygon(mask: np.ndarray, points: list[list[float]]) -> None:
    if not points:
        return
    height, width = mask.shape
    vertices = np.asarray(points, dtype=np.float64)
    if len(vertices) == 1:
        x, y = np.rint(vertices[0]).astype(int)
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = True
        return
    if len(vertices) == 2:
        x0, y0 = vertices[0]
        x1, y1 = vertices[1]
        count = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1) + 1
        xs = np.rint(np.linspace(x0, x1, count)).astype(int)
        ys = np.rint(np.linspace(y0, y1, count)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[valid], xs[valid]] = True
        return
    min_x = max(0, int(math.floor(vertices[:, 0].min())))
    max_x = min(width - 1, int(math.ceil(vertices[:, 0].max())))
    min_y = max(0, int(math.floor(vertices[:, 1].min())))
    max_y = min(height - 1, int(math.ceil(vertices[:, 1].max())))
    for row in range(min_y, max_y + 1):
        intersections: list[float] = []
        for index in range(len(vertices)):
            x0, y0 = vertices[index]
            x1, y1 = vertices[(index + 1) % len(vertices)]
            if (y0 <= row < y1) or (y1 <= row < y0):
                intersections.append(float(x0 + (row - y0) * (x1 - x0) / (y1 - y0)))
        intersections.sort()
        for start in range(0, len(intersections) - 1, 2):
            left = max(min_x, int(math.ceil(intersections[start])))
            right = min(max_x, int(math.floor(intersections[start + 1])))
            if left <= right:
                mask[row, left : right + 1] = True


def record_mask(record: dict[str, Any], shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for contour in record.get("contours", []):
        slice_index = contour.get("slice_index")
        if slice_index is None or str(contour.get("inclusion", "TRUE")).upper() == "FALSE":
            continue
        slice_index = int(slice_index)
        if not 0 <= slice_index < shape[0]:
            continue
        for polygon in contour.get("polygons", []):
            rasterize_polygon(mask[slice_index], polygon)
    if not mask.any() and record.get("mask_path"):
        source = resolve_path(record["mask_path"], REPO_ROOT / "processed" / "canonical")
        if source.exists():
            loaded = np.asarray(nib.load(source).dataobj)
            if loaded.shape == shape:
                mask = loaded > 0
    return mask


def geometry(record: dict[str, Any], shape: tuple[int, int, int], spacing: np.ndarray) -> dict[str, Any]:
    centroid = record.get("centroid_patient_xyz")
    if centroid is None:
        centroid = [None, None, None]
    centroid = [float(value) if value is not None else None for value in centroid]
    voxel_centroid = record.get("centroid_voxel_zyx") or [None, None, None]
    bbox = [[0, 0, 0], [int(value - 1) for value in shape]]
    points: list[tuple[int, int, int]] = []
    for contour in record.get("contours", []):
        z = contour.get("slice_index")
        if z is None:
            continue
        for polygon in contour.get("polygons", []):
            for point in polygon:
                if len(point) >= 2:
                    points.append((int(round(z)), int(round(point[1])), int(round(point[0]))))
    if points:
        array = np.asarray(points, dtype=int)
        minimum = np.maximum(array.min(axis=0), 0)
        maximum = np.minimum(array.max(axis=0), np.asarray(shape) - 1)
        bbox = [[int(value) for value in minimum], [int(value) for value in maximum]]
    extent = ((np.asarray(bbox[1]) - np.asarray(bbox[0]) + 1) * spacing).tolist()
    return {
        "centroid_patient_xyz": centroid,
        "centroid_voxel_zyx": voxel_centroid,
        "bbox_voxel_zyx": bbox,
        "bbox_extent_mm_zyx": [float(value) for value in extent],
    }


def bbox_iou(first: list[list[int]], second: list[list[int]]) -> float:
    lower = np.maximum(first[0], second[0])
    upper = np.minimum(first[1], second[1])
    intersection = np.maximum(upper - lower + 1, 0)
    intersection_volume = float(np.prod(intersection))
    first_volume = float(np.prod(np.maximum(np.asarray(first[1]) - first[0] + 1, 0)))
    second_volume = float(np.prod(np.maximum(np.asarray(second[1]) - second[0] + 1, 0)))
    denominator = first_volume + second_volume - intersection_volume
    return intersection_volume / denominator if denominator else 0.0


class UnionFind:
    def __init__(self, size: int, session_keys: list[str]) -> None:
        self.parent = list(range(size))
        self.session_sets = [{key} for key in session_keys]

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int, allow_same_session_duplicate: bool = False) -> bool:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            if self.session_sets[first_root] & self.session_sets[second_root] and not allow_same_session_duplicate:
                return False
            self.parent[second_root] = first_root
            self.session_sets[first_root].update(self.session_sets[second_root])
            self.session_sets[second_root] = set()
        return True


def cluster_annotations(records: list[dict[str, Any]], cluster_config: dict[str, Any]) -> list[list[dict[str, Any]]]:
    by_volume: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("annotation_status") != "ok" or int(record.get("mask_voxel_count") or 0) <= 0:
            continue
        shape = tuple(int(value) for value in record["shape_zyx"])
        spacing = np.asarray(record["spacing_zyx_mm"], dtype=float)
        record["geometry"] = geometry(record, shape, spacing)
        by_volume[record["volume_id"]].append(record)

    clusters: list[list[dict[str, Any]]] = []
    max_distance = float(cluster_config["max_centroid_distance_mm"])
    close_distance = float(cluster_config["close_distance_mm"])
    min_iou = float(cluster_config["min_bbox_iou"])
    for volume_id, volume_records in sorted(by_volume.items()):
        for record in volume_records:
            record["reader_session_id"] = reader_session_key(record)
        union_find = UnionFind(len(volume_records), [record["reader_session_id"] for record in volume_records])
        edges: list[tuple[float, float, int, int, bool]] = []
        for first in range(len(volume_records)):
            first_centroid = np.asarray(volume_records[first]["geometry"]["centroid_patient_xyz"], dtype=float)
            if not np.isfinite(first_centroid).all():
                continue
            for second in range(first + 1, len(volume_records)):
                second_centroid = np.asarray(volume_records[second]["geometry"]["centroid_patient_xyz"], dtype=float)
                if not np.isfinite(second_centroid).all():
                    continue
                distance = float(np.linalg.norm(first_centroid - second_centroid))
                overlap = bbox_iou(volume_records[first]["geometry"]["bbox_voxel_zyx"], volume_records[second]["geometry"]["bbox_voxel_zyx"])
                same_session = volume_records[first]["reader_session_id"] == volume_records[second]["reader_session_id"]
                same_session_duplicate = same_session and distance <= float(cluster_config.get("same_session_duplicate_distance_mm", 0.5)) and overlap >= float(cluster_config.get("same_session_duplicate_bbox_iou", 1.0))
                if same_session and not same_session_duplicate:
                    continue
                if distance <= max_distance and (distance <= close_distance or overlap >= min_iou) and (not same_session or same_session_duplicate):
                    edges.append((distance, -overlap, first, second, same_session_duplicate))
        # Process the strongest/closest links first and reject a link that
        # would place two annotations from one XML reading session in the
        # same consensus component. This prevents transitive inflation.
        for _, _, first, second, same_session_duplicate in sorted(edges):
            union_find.union(first, second, allow_same_session_duplicate=same_session_duplicate)
        components: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, record in enumerate(volume_records):
            components[union_find.find(index)].append(record)
        clusters.extend(sorted(components.values(), key=lambda group: min(item["annotation_key"] for item in group)))
    return clusters


def consensus_label(scores: list[int], num_readers: int, policy: dict[str, Any], suspicious_reasons: list[str]) -> dict[str, Any]:
    valid_scores = [score for score in scores if score in {1, 2, 3, 4, 5}]
    categories = sorted({"benign" if score <= 2 else "indeterminate" if score == 3 else "malignant" for score in valid_scores})
    rated_reader_count = len(valid_scores)
    minimum_readers = int(policy["minimum_readers_for_binary"])
    if not valid_scores:
        category = "uncertain"
        reasons = ["missing_malignancy_rating"]
    elif len(categories) != 1:
        category = "disagreement_uncertain"
        reasons = ["cross_category_disagreement"]
    else:
        median = float(statistics.median(valid_scores))
        if 1 <= median <= 2:
            category = "benign"
            reasons = []
        elif 4 <= median <= 5:
            category = "malignant"
            reasons = []
        elif median == 3:
            category = "indeterminate"
            reasons = ["indeterminate_rating"]
        else:
            category = "uncertain"
            reasons = ["undefined_median_midpoint"]
    score_range = max(valid_scores) - min(valid_scores) if valid_scores else None
    median_score = float(statistics.median(valid_scores)) if valid_scores else None
    if rated_reader_count < minimum_readers:
        reasons.append("insufficient_readers_for_strict_binary")
    if score_range is not None and score_range >= int(policy["strong_disagreement_range"]):
        reasons.append("strong_score_disagreement")
    reasons.extend(suspicious_reasons)
    binary_eligible = rated_reader_count >= minimum_readers and category in {"benign", "malignant"} and not any(
        reason in reasons for reason in (
            "cross_category_disagreement",
            "strong_score_disagreement",
            "suspicious_cluster_reader_count",
            "suspicious_cluster_annotation_count",
            "suspicious_cluster_diameter",
            "same_session_score_conflict",
        )
    )
    strict_3_reader_eligible = binary_eligible and rated_reader_count >= int(policy.get("strict_three_reader_minimum", 3))
    return {
        "label_category": category,
        "binary_label": 0 if binary_eligible and category == "benign" else 1 if binary_eligible and category == "malignant" else None,
        "binary_eligible": binary_eligible,
        "strict_3_reader_eligible": strict_3_reader_eligible,
        "exclusion_reasons": sorted(set(reasons)) if not binary_eligible else [],
        "rated_reader_count": rated_reader_count,
        "median_score_policy": median_score,
        "score_range": score_range,
        "score_categories": categories,
    }


def consensus_records(clusters: list[list[dict[str, Any]]], label_policy: dict[str, Any], cluster_config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for cluster in clusters:
        first = cluster[0]
        shape = tuple(int(value) for value in first["shape_zyx"])
        spacing = np.asarray(first["spacing_zyx_mm"], dtype=float)
        affine = np.asarray(first["affine_array_zyx_to_patient"], dtype=float)
        union_mask = np.zeros(shape, dtype=bool)
        for record in cluster:
            union_mask |= record_mask(record, shape)
        nonzero = np.argwhere(union_mask)
        if len(nonzero) == 0:
            continue
        centroid_voxel = nonzero.mean(axis=0)
        centroid_world = (affine @ np.asarray([*centroid_voxel, 1.0]))[:3]
        minimum = nonzero.min(axis=0).astype(int)
        maximum = nonzero.max(axis=0).astype(int)
        session_ids = sorted({record["reader_session_id"] for record in cluster})
        session_scores: dict[str, list[int]] = defaultdict(list)
        annotation_scores: list[int] = []
        reader_annotation_scores: list[dict[str, Any]] = []
        for record in cluster:
            value = record.get("malignancy")
            if str(value).isdigit() and int(value) in {1, 2, 3, 4, 5}:
                score = int(value)
                annotation_scores.append(score)
                session_scores[record["reader_session_id"]].append(score)
                reader_annotation_scores.append({"annotation_key": record["annotation_key"], "reader_session_id": record["reader_session_id"], "score": score})
        independent_scores = [values[0] for session_id, values in sorted(session_scores.items()) if values]
        suspicious_reasons: list[str] = []
        max_readers = int(cluster_config.get("max_consensus_readers", 4))
        max_annotations = int(cluster_config.get("max_consensus_annotations", 4))
        max_diameter = float(cluster_config.get("max_cluster_diameter_mm", 12.0))
        if len(session_ids) > max_readers:
            suspicious_reasons.append("suspicious_cluster_reader_count")
        if len(cluster) > max_annotations:
            suspicious_reasons.append("suspicious_cluster_annotation_count")
        centroids = [np.asarray(record["geometry"]["centroid_patient_xyz"], dtype=float) for record in cluster]
        diameter = max((float(np.linalg.norm(a - b)) for a, b in itertools.combinations(centroids, 2)), default=0.0)
        if diameter > max_diameter:
            suspicious_reasons.append("suspicious_cluster_diameter")
        if any(len(set(values)) > 1 for values in session_scores.values()):
            suspicious_reasons.append("same_session_score_conflict")
        label = consensus_label(independent_scores, len(session_ids), label_policy, suspicious_reasons)
        identity = "|".join(sorted(record["annotation_key"] for record in cluster))
        consensus_id = f"CN-{first['patient_id']}-{stable_id(identity)}"
        result = {
            "consensus_nodule_id": consensus_id,
            "patient_id": first["patient_id"],
            "volume_id": first["volume_id"],
            "study_instance_uid": first["study_instance_uid"],
            "series_instance_uid": first["series_instance_uid"],
            "contributing_annotation_keys": [record["annotation_key"] for record in cluster],
            "contributing_reader_ids": [record.get("reader_id") for record in cluster],
            "contributing_reader_session_ids": session_ids,
            "original_malignancy_scores": annotation_scores,
            "independent_malignancy_scores": independent_scores,
            "reader_annotation_scores": reader_annotation_scores,
            "num_annotations": len(cluster),
            "num_readers": len(session_ids),
            "malignancy_mean": float(statistics.mean(independent_scores)) if independent_scores else None,
            "malignancy_median": float(statistics.median(independent_scores)) if independent_scores else None,
            "malignancy_std": float(statistics.pstdev(independent_scores)) if len(independent_scores) > 1 else 0.0 if independent_scores else None,
            "malignancy_score_range": label["score_range"],
            "score_categories": label["score_categories"],
            "disagreement_flag": bool(label["exclusion_reasons"] and any("disagreement" in reason for reason in label["exclusion_reasons"])),
            "centroid_voxel_zyx": [float(value) for value in centroid_voxel],
            "centroid_patient_xyz": [float(value) for value in centroid_world],
            "bbox_voxel_zyx": [[int(value) for value in minimum], [int(value) for value in maximum]],
            "bbox_extent_mm_zyx": [float(value) for value in ((maximum - minimum + 1) * spacing)],
            "full_volume_shape_zyx": list(shape),
            "source_spacing_zyx_mm": [float(value) for value in spacing],
            "source_affine_array_zyx_to_patient": affine.tolist(),
            "source_volume_path": str(first["canonical_volume_path"]),
            "source_mask_voxel_count": int(union_mask.sum()),
            "duplicate_annotations_within_session": len(cluster) - len(session_ids),
            "cluster_diameter_mm": diameter,
            "suspicious_cluster": bool(suspicious_reasons),
            "suspicious_cluster_reasons": sorted(suspicious_reasons),
            "clustering_method": "constrained_connected_components_unique_xml_reading_session",
            "clustering_config": cluster_config,
            "union_mask": union_mask,
            **label,
        }
        results.append(result)
    return results


def crop_shape_from_statistics(consensus: list[dict[str, Any]], sample_config: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    target = sample_config.get("crop_fov_mm")
    extents = np.asarray([record["bbox_extent_mm_zyx"] for record in consensus], dtype=float)
    if target:
        fov = [float(value) for value in target]
        method = "configured_crop_fov_mm"
    else:
        margin = float(sample_config["context_margin_mm"])
        round_mm = float(sample_config["crop_round_mm"])
        observed_max = extents.max(axis=0) if len(extents) else np.asarray([32.0, 32.0, 32.0])
        fov = [float(max(round_mm, math.ceil((value + 2 * margin) / round_mm) * round_mm)) for value in observed_max]
        method = "max_observed_bbox_extent_plus_two_sided_context_margin_rounded_up"
    return fov, {
        "method": method,
        "observed_bbox_extent_max_mm_zyx": extents.max(axis=0).tolist() if len(extents) else [],
        "observed_bbox_extent_p95_mm_zyx": np.percentile(extents, 95, axis=0).tolist() if len(extents) else [],
        "context_margin_mm": sample_config.get("context_margin_mm"),
        "crop_round_mm": sample_config.get("crop_round_mm"),
    }


def resample_crop(volume: np.ndarray, source_affine: np.ndarray, center_world: np.ndarray, shape: tuple[int, int, int], spacing: np.ndarray, fill_value: float) -> tuple[np.ndarray, np.ndarray]:
    linear = source_affine[:3, :3]
    directions = linear / np.linalg.norm(linear, axis=0, keepdims=True)
    target_linear = directions * spacing.reshape(1, 3)
    target_origin = center_world - target_linear @ ((np.asarray(shape, dtype=float) - 1.0) / 2.0)
    inverse = np.linalg.inv(linear)
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    world = grid @ target_linear.T + target_origin
    source = (world - source_affine[:3, 3]) @ inverse.T
    valid = np.all((source >= 0) & (source <= (np.asarray(volume.shape) - 1)), axis=1)
    output = np.full(len(grid), fill_value, dtype=np.float32)
    if valid.any():
        coordinates = source[valid]
        lower = np.floor(coordinates).astype(int)
        upper = np.minimum(lower + 1, np.asarray(volume.shape) - 1)
        weights = coordinates - lower
        z0, y0, x0 = lower.T
        z1, y1, x1 = upper.T
        c000 = volume[z0, y0, x0]
        c001 = volume[z0, y0, x1]
        c010 = volume[z0, y1, x0]
        c011 = volume[z0, y1, x1]
        c100 = volume[z1, y0, x0]
        c101 = volume[z1, y0, x1]
        c110 = volume[z1, y1, x0]
        c111 = volume[z1, y1, x1]
        wz, wy, wx = weights.T
        output[valid] = (
            c000 * (1 - wz) * (1 - wy) * (1 - wx) + c001 * (1 - wz) * (1 - wy) * wx
            + c010 * (1 - wz) * wy * (1 - wx) + c011 * (1 - wz) * wy * wx
            + c100 * wz * (1 - wy) * (1 - wx) + c101 * wz * (1 - wy) * wx
            + c110 * wz * wy * (1 - wx) + c111 * wz * wy * wx
        )
    return output.reshape(shape).astype(np.float32), target_origin


def resample_mask(mask: np.ndarray, source_affine: np.ndarray, center_world: np.ndarray, shape: tuple[int, int, int], spacing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample a binary mask into the master-sample grid.

    A conventional nearest-neighbour pull resample can erase a very small
    nodule when the source mask occupies only one or two thick-slice voxels.
    Instead, map positive source voxel centres forward into the target grid
    (a label-preserving splat).  This keeps the mask physically aligned with
    the image grid while guaranteeing that every non-empty source annotation
    remains represented in the output whenever it falls inside the crop.
    """
    linear = source_affine[:3, :3]
    directions = linear / np.linalg.norm(linear, axis=0, keepdims=True)
    target_linear = directions * spacing.reshape(1, 3)
    target_origin = center_world - target_linear @ ((np.asarray(shape, dtype=float) - 1.0) / 2.0)
    output = np.zeros(shape, dtype=np.uint8)
    positive = np.argwhere(mask > 0)
    if len(positive):
        world = positive @ linear.T + source_affine[:3, 3]
        target = (world - target_origin) @ np.linalg.inv(target_linear).T
        indices = np.rint(target).astype(int)
        valid = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
        if valid.any():
            output[tuple(indices[valid].T)] = 1
    return output, target_origin


def crop_coverage(consensus: dict[str, Any], target_origin: np.ndarray, target_spacing: np.ndarray, source_affine: np.ndarray, crop_shape: tuple[int, int, int]) -> float:
    bbox = np.asarray(consensus["bbox_voxel_zyx"], dtype=float)
    corners = np.asarray([[z, y, x] for z in bbox[:, 0] for y in bbox[:, 1] for x in bbox[:, 2]], dtype=float)
    world = corners @ source_affine[:3, :3].T + source_affine[:3, 3]
    directions = source_affine[:3, :3] / np.linalg.norm(source_affine[:3, :3], axis=0, keepdims=True)
    # The target basis is orthonormal for valid DICOM affines; use a stable
    # inverse for oblique numerical tolerances.
    target_linear = directions * target_spacing.reshape(1, 3)
    target_coords = (world - target_origin) @ np.linalg.inv(target_linear).T
    inside = np.all((target_coords >= -0.5) & (target_coords <= (np.asarray(crop_shape) - 0.5)), axis=1)
    return float(inside.mean())


def split_patients(consensus: list[dict[str, Any]], split_config: dict[str, Any]) -> dict[str, str]:
    patients: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in consensus:
        patients[record["patient_id"]].append(record)
    groups: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for patient_id, records in patients.items():
        groups[(any(record.get("binary_label") == 0 for record in records), any(record.get("binary_label") == 1 for record in records))].append(patient_id)
    rng = random.Random(int(split_config["seed"]))
    assignments: dict[str, str] = {}
    names = ["train", "validation", "test"]
    ratios = [float(split_config[name]) for name in names]
    for group in groups.values():
        group.sort()
        rng.shuffle(group)
        counts = [int(math.floor(len(group) * ratio)) for ratio in ratios]
        for index in range(len(group) - sum(counts)):
            counts[index % 3] += 1
        cursor = 0
        for name, count in zip(names, counts):
            for patient_id in group[cursor : cursor + count]:
                assignments[patient_id] = name
            cursor += count
    return assignments


def build_patient_consensus(canonical_root: Path, patient_id: str, config: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Load, cluster, and summarize one patient, releasing source records afterwards."""
    _, reader_records = load_canonical(canonical_root, {patient_id})
    clusters = cluster_annotations(reader_records, config["clustering"])
    consensus = consensus_records(clusters, config["label_policy"], config["clustering"])
    return len(reader_records), consensus


def process_phase3_streaming(canonical_root: Path, output_root: Path, config: dict[str, Any], patient_ids: list[str] | None = None) -> dict[str, Any]:
    """Run Phase 3 in patient-sized passes to avoid loading the full corpus in memory."""
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("samples", "masks"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    canonical_metadata = read_csv(canonical_root / "metadata.csv")
    available_patients = sorted({row["patient_id"] for row in canonical_metadata if row.get("status") == "ok"})
    selected_patients = sorted(set(patient_ids)) if patient_ids else available_patients
    summaries: list[dict[str, Any]] = []
    reader_annotation_count = 0
    for patient_index, patient_id in enumerate(selected_patients, 1):
        count, patient_consensus = build_patient_consensus(canonical_root, patient_id, config)
        reader_annotation_count += count
        summaries.extend({key: value for key, value in record.items() if key != "union_mask"} for record in patient_consensus)
        if patient_index % 25 == 0:
            LOGGER.info("Indexed patients: %d/%d", patient_index, len(selected_patients))

    fov_mm, crop_statistics = crop_shape_from_statistics(summaries, config["sampling"])
    target_spacing = np.asarray(config["sampling"]["target_spacing_mm"], dtype=float)
    crop_shape = tuple(int(math.ceil(value / spacing)) for value, spacing in zip(fov_mm, target_spacing))
    assignments = split_patients(summaries, config["splits"])
    LOGGER.info("Phase 3 %s: patients=%d reader_annotations=%d physical_nodules=%d crop_shape=%s", config.get("run_scope", "pilot"), len(selected_patients), reader_annotation_count, len(summaries), crop_shape)

    metadata_rows: list[dict[str, Any]] = []
    labels_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    validation_issues: list[dict[str, Any]] = []
    processed_count = 0
    for patient_index, patient_id in enumerate(selected_patients, 1):
        _, patient_consensus = build_patient_consensus(canonical_root, patient_id, config)
        volume_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        for record in sorted(patient_consensus, key=lambda item: item["consensus_nodule_id"]):
            processed_count += 1
            index = processed_count
            consensus_id = record["consensus_nodule_id"]
            source_path = Path(record["source_volume_path"])
            try:
                if source_path not in volume_cache:
                    image = nib.load(source_path)
                    volume_cache[source_path] = (np.asarray(image.dataobj, dtype=np.float32), np.asarray(image.affine, dtype=float))
                volume, affine = volume_cache[source_path]
                center = np.asarray(record["centroid_patient_xyz"], dtype=float)
                sample, target_origin = resample_crop(volume, affine, center, crop_shape, target_spacing, float(config["sampling"]["image_fill_hu"]))
                mask, _ = resample_mask(record["union_mask"], affine, center, crop_shape, target_spacing)
                coverage = crop_coverage(record, target_origin, target_spacing, affine, crop_shape)
                record["crop_coverage_fraction_bbox"] = coverage
                record["crop_coverage_ok"] = coverage >= float(config["sampling"]["minimum_crop_coverage"])
                if not np.isfinite(sample).all():
                    raise ValueError("sample_contains_nonfinite_values")
                if not record["crop_coverage_ok"]:
                    record["exclusion_reasons"].append("crop_out_of_bounds")
                    record["binary_eligible"] = False
                    record["strict_3_reader_eligible"] = False
                    record["binary_label"] = None
                    validation_issues.append({"severity": "warning", "code": "CROP_OUT_OF_BOUNDS", "consensus_nodule_id": consensus_id})
                if not mask.any():
                    raise ValueError("resampled_mask_is_empty")
                np.save(output_root / "samples" / f"{consensus_id}.npy", sample.astype(np.float32))
                np.save(output_root / "masks" / f"{consensus_id}.npy", mask.astype(np.uint8))
                record["sample_path"] = f"samples/{consensus_id}.npy"
                record["mask_path"] = f"masks/{consensus_id}.npy"
                record["sample_shape_zyx"] = list(crop_shape)
                record["sample_spacing_zyx_mm"] = [float(value) for value in target_spacing]
                record["sample_origin_patient_xyz"] = [float(value) for value in target_origin]
            except Exception as exc:  # noqa: BLE001 - retain a per-nodule exclusion
                reason = f"sample_generation_error:{type(exc).__name__}"
                record.setdefault("exclusion_reasons", []).append(reason)
                record["binary_eligible"] = False
                record["strict_3_reader_eligible"] = False
                record["sample_path"] = None
                record["mask_path"] = None
                validation_issues.append({"severity": "error", "code": "SAMPLE_GENERATION_FAILURE", "consensus_nodule_id": consensus_id, "message": str(exc)})
                LOGGER.exception("Failed sample generation for %s", consensus_id)
            split = assignments.get(record["patient_id"], "unassigned")
            record["split"] = split
            exclusion_reasons = sorted(set(record.get("exclusion_reasons", [])))
            if not record.get("binary_eligible"):
                excluded_rows.append({
                    "consensus_nodule_id": consensus_id,
                    "patient_id": record["patient_id"],
                    "label_category": record["label_category"],
                    "num_readers": record["num_readers"],
                    "rated_reader_count": record["rated_reader_count"],
                    "original_malignancy_scores": record["original_malignancy_scores"],
                    "strict_3_reader_eligible": record["strict_3_reader_eligible"],
                    "suspicious_cluster": record["suspicious_cluster"],
                    "suspicious_cluster_reasons": record["suspicious_cluster_reasons"],
                    "exclusion_reasons": exclusion_reasons,
                })
            metadata_rows.append({key: value for key, value in record.items() if key not in {"union_mask"}})
            labels_rows.append({
                "consensus_nodule_id": consensus_id,
                "patient_id": record["patient_id"],
                "series_instance_uid": record["series_instance_uid"],
                "split": split,
                "label_category": record["label_category"],
                "binary_label": record["binary_label"],
                "binary_eligible": record["binary_eligible"],
                "strict_3_reader_eligible": record["strict_3_reader_eligible"],
                "original_malignancy_scores": record["original_malignancy_scores"],
                "contributing_reader_session_ids": record["contributing_reader_session_ids"],
                "num_readers": record["num_readers"],
                "rated_reader_count": record["rated_reader_count"],
                "num_annotations": record["num_annotations"],
                "median_score_policy": record["median_score_policy"],
                "malignancy_mean": record["malignancy_mean"],
                "malignancy_median": record["malignancy_median"],
                "malignancy_std": record["malignancy_std"],
                "malignancy_score_range": record["malignancy_score_range"],
                "score_categories": record["score_categories"],
                "disagreement_flag": record["disagreement_flag"],
                "exclusion_reasons": exclusion_reasons,
                "sample_path": record.get("sample_path"),
                "mask_path": record.get("mask_path"),
            })
            split_rows.append({"consensus_nodule_id": consensus_id, "patient_id": record["patient_id"], "split": split})
            if index % 100 == 0:
                LOGGER.info("Processed consensus nodules: %d/%d", index, len(summaries))
        volume_cache.clear()
        del patient_consensus
        if patient_index % 25 == 0:
            LOGGER.info("Processed patients: %d/%d", patient_index, len(selected_patients))

    consensus = summaries
    metadata = canonical_metadata
    reader_records = [None] * reader_annotation_count

    metadata_fields = sorted({key for row in metadata_rows for key in row})
    for field in ("consensus_nodule_id", "patient_id", "volume_id", "series_instance_uid", "sample_path", "mask_path"):
        if field in metadata_fields:
            metadata_fields.remove(field)
        metadata_fields.insert(0, field)
    write_csv(output_root / "metadata.csv", metadata_rows, list(dict.fromkeys(metadata_fields)))
    write_csv(output_root / "labels.csv", labels_rows, list(labels_rows[0].keys()) if labels_rows else [])
    write_csv(output_root / "splits.csv", split_rows, ["consensus_nodule_id", "patient_id", "split"])
    write_csv(output_root / "excluded_nodules.csv", excluded_rows, [
        "consensus_nodule_id", "patient_id", "label_category", "num_readers", "rated_reader_count",
        "original_malignancy_scores", "strict_3_reader_eligible", "suspicious_cluster",
        "suspicious_cluster_reasons", "exclusion_reasons",
    ])

    split_patient_ids = {name: sorted({row["patient_id"] for row in split_rows if row["split"] == name}) for name in ("train", "validation", "test")}
    class_counts = Counter(row["label_category"] for row in labels_rows)
    binary_counts = Counter(str(row["binary_label"]) for row in labels_rows if row["binary_eligible"])
    reader_counts = [record["num_readers"] for record in consensus]
    primary_excluded = [row for row in labels_rows if not row["binary_eligible"]]
    stats = {
        "run_scope": config.get("run_scope", "pilot"),
        "patients_included": len({row["patient_id"] for row in metadata_rows}),
        "reader_annotations_used": len(reader_records),
        "consensus_physical_nodules": len(consensus),
        "rated_consensus_nodules": sum(record["rated_reader_count"] > 0 for record in consensus),
        "consensus_samples_created": sum(bool(row.get("sample_path")) for row in metadata_rows),
        "benign_nodules": class_counts["benign"],
        "malignant_nodules": class_counts["malignant"],
        "indeterminate_nodules": class_counts["indeterminate"],
        "disagreement_uncertain_nodules": class_counts["disagreement_uncertain"] + class_counts["uncertain"],
        "strict_binary_eligible_nodules": sum(row["binary_eligible"] for row in labels_rows),
        "strict_3_reader_eligible_nodules": sum(row["strict_3_reader_eligible"] for row in labels_rows),
        "strict_binary_benign": binary_counts["0"],
        "strict_binary_malignant": binary_counts["1"],
        "excluded_nodules": len(excluded_rows),
        "exclusion_reason_counts": dict(Counter(reason for row in excluded_rows for reason in row["exclusion_reasons"])),
        "indeterminate_excluded": sum(row["label_category"] == "indeterminate" for row in primary_excluded),
        "disagreement_excluded": sum(any(reason in row["exclusion_reasons"] for reason in ("cross_category_disagreement", "strong_score_disagreement")) for row in primary_excluded),
        "insufficient_reader_excluded": sum("insufficient_readers_for_strict_binary" in row["exclusion_reasons"] for row in primary_excluded),
        "missing_rating_excluded": sum("missing_malignancy_rating" in row["exclusion_reasons"] for row in primary_excluded),
        "suspicious_clusters": sum(record["suspicious_cluster"] for record in consensus),
        "crop_fov_mm_zyx": fov_mm,
        "crop_shape_zyx": list(crop_shape),
        "target_spacing_mm_zyx": [float(value) for value in target_spacing],
        "crop_statistics": crop_statistics,
        "split_patient_counts": {key: len(value) for key, value in split_patient_ids.items()},
        "split_nodule_counts": {key: sum(row["split"] == key for row in split_rows) for key in ("train", "validation", "test")},
        "reader_count_min": min(reader_counts) if reader_counts else 0,
        "reader_count_median": statistics.median(reader_counts) if reader_counts else 0,
        "reader_count_max": max(reader_counts) if reader_counts else 0,
        "validation_issue_count": len(validation_issues),
        "validation_issues": validation_issues,
    }
    write_json(output_root / "statistics.json", stats)
    config_copy = json.loads(json.dumps(config, default=json_default))
    write_json(output_root / "phase3_config.json", config_copy)
    write_readme(output_root, config, stats)
    report = validate_phase3(output_root, metadata_rows, labels_rows, split_rows, stats, config)
    write_json(output_root / "validation_report.json", report)
    write_validation_markdown(output_root / "validation_report.md", report)
    return {"stats": stats, "report": report}


def process_phase3(canonical_root: Path, output_root: Path, config: dict[str, Any], patient_ids: list[str] | None = None) -> dict[str, Any]:
    return process_phase3_streaming(canonical_root, output_root, config, patient_ids)


def write_readme(output_root: Path, config: dict[str, Any], stats: dict[str, Any]) -> None:
    text = f"""# Final Team Dataset — Phase 3 ({stats['run_scope']})

This directory is the shared master-sample dataset for the study comparing 2D, 2.5D, and 3D pulmonary-nodule malignancy-risk classification.

This run is a **{stats['run_scope']}**. It creates one common 3D `.npy` sample and aligned `.npy` mask per consensus physical nodule. Downstream 2D, 2.5D, and 3D experiments must use `splits.csv` and derive all inputs from these same samples.

## Consensus policy

Reader-level annotations are clustered only within the same CT series. Reader identity is the XML source path plus reading-session index. An edge is created when centroid distance is at most `{config['clustering']['max_centroid_distance_mm']} mm` and either centroid distance is at most `{config['clustering']['close_distance_mm']} mm` or 3D native-voxel bounding-box IoU is at least `{config['clustering']['min_bbox_iou']}`. Components are joined only when their reader-session sets remain disjoint, preventing same-session reader inflation. Components exceeding the configured reader, annotation, or diameter limits are flagged and excluded from the primary cohort.

## Label policy

Original 1–5 ratings are preserved. Median scores 1–2 map to benign and 4–5 to malignant; median 3 is indeterminate and undefined midpoints are uncertain. Cross-category or strong score-range disagreement is excluded. Primary binary eligibility requires at least `{config['label_policy']['minimum_readers_for_binary']}` independent rated reading sessions. `strict_3_reader_eligible` marks the stricter three-reader sensitivity cohort.

## Master samples

Samples use target spacing `{stats['target_spacing_mm_zyx']} mm` in the canonical array-axis order (z, y, x) and crop shape `{stats['crop_shape_zyx']}`. The crop field of view was derived from observed consensus-mask extents using: `{stats['crop_statistics']['method']}`.

## Important files

- `samples/`: common 3D image samples, float32 HU
- `masks/`: aligned uint8 nodule masks
- `metadata.csv`: provenance, geometry, consensus, crop, and reader information
- `labels.csv`: label policy and original malignancy ratings
- `splits.csv`: fixed patient-level split assignment
- `excluded_nodules.csv`: nodules excluded from strict binary classification
- `dataset_card.md`: dataset scope and limitations
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")
    suffix = "\n## Pilot limitation\n\nThis pilot must be reviewed before a full-dataset run.\n" if stats["run_scope"] == "pilot" else "\n## Release status\n\nThis is the full Phase 3 shared team dataset. Model-specific 2D, 2.5D, and 3D datasets have not been created.\n"
    (output_root / "dataset_card.md").write_text(text + suffix, encoding="utf-8")


def validate_phase3(output_root: Path, metadata_rows: list[dict[str, Any]], labels_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]], stats: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = list(stats.get("validation_issues", []))
    split_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        split_by_patient[row["patient_id"]].add(row["split"])
    leakage = {patient: sorted(splits) for patient, splits in split_by_patient.items() if len(splits) != 1}
    if leakage:
        issues.append({"severity": "error", "code": "PATIENT_LEAKAGE", "patients": leakage})
    ids = [row["consensus_nodule_id"] for row in metadata_rows]
    if len(ids) != len(set(ids)):
        issues.append({"severity": "error", "code": "DUPLICATE_CONSENSUS_IDS"})
    if len({row["consensus_nodule_id"] for row in split_rows}) != len(split_rows):
        issues.append({"severity": "error", "code": "DUPLICATE_SPLIT_ROWS"})
    sample_hashes: dict[str, list[str]] = defaultdict(list)
    for row in metadata_rows:
        session_ids = parse_json_field(row.get("contributing_reader_session_ids"), [])
        try:
            recorded_readers = int(row.get("num_readers", 0))
        except (TypeError, ValueError):
            recorded_readers = -1
        if len(session_ids) != recorded_readers or len(session_ids) != len(set(session_ids)):
            issues.append({"severity": "error", "code": "SAME_SESSION_READER_INFLATION", "consensus_nodule_id": row["consensus_nodule_id"]})
        if parse_json_field(row.get("suspicious_cluster_reasons"), []) and row.get("binary_eligible") in (True, "True"):
            issues.append({"severity": "error", "code": "SUSPICIOUS_CLUSTER_IN_PRIMARY_COHORT", "consensus_nodule_id": row["consensus_nodule_id"]})
        sample_path = output_root / row["sample_path"] if row.get("sample_path") else None
        mask_path = output_root / row["mask_path"] if row.get("mask_path") else None
        if sample_path and not sample_path.exists():
            issues.append({"severity": "error", "code": "MISSING_SAMPLE", "consensus_nodule_id": row["consensus_nodule_id"]})
        if mask_path and not mask_path.exists():
            issues.append({"severity": "error", "code": "MISSING_MASK", "consensus_nodule_id": row["consensus_nodule_id"]})
        if sample_path and mask_path and sample_path.exists() and mask_path.exists():
            try:
                sample = np.load(sample_path, allow_pickle=False)
                mask = np.load(mask_path, allow_pickle=False)
                if sample.shape != mask.shape:
                    issues.append({"severity": "error", "code": "SAMPLE_MASK_SHAPE_MISMATCH", "consensus_nodule_id": row["consensus_nodule_id"]})
                if not np.isfinite(sample).all():
                    issues.append({"severity": "error", "code": "NONFINITE_SAMPLE", "consensus_nodule_id": row["consensus_nodule_id"]})
                if not set(np.unique(mask).tolist()).issubset({0, 1}):
                    issues.append({"severity": "error", "code": "INVALID_MASK_VALUES", "consensus_nodule_id": row["consensus_nodule_id"]})
                if not mask.any():
                    issues.append({"severity": "error", "code": "EMPTY_MASK", "consensus_nodule_id": row["consensus_nodule_id"]})
                sample_hashes[hashlib.sha256(sample.tobytes()).hexdigest()].append(row["consensus_nodule_id"])
            except Exception as exc:  # noqa: BLE001 - report corrupt output explicitly
                issues.append({"severity": "error", "code": "OUTPUT_ARRAY_READ_FAILURE", "consensus_nodule_id": row["consensus_nodule_id"], "message": str(exc)})
    for digest, duplicate_ids in sample_hashes.items():
        if len(duplicate_ids) > 1:
            issues.append({"severity": "error", "code": "DUPLICATE_SAMPLE_PAYLOAD", "consensus_nodule_ids": duplicate_ids, "sha256": digest})
    valid_categories = {"benign", "malignant", "indeterminate", "disagreement_uncertain", "uncertain"}
    for row in labels_rows:
        category = row.get("label_category")
        if category not in valid_categories:
            issues.append({"severity": "error", "code": "INVALID_LABEL_CATEGORY", "consensus_nodule_id": row["consensus_nodule_id"], "label_category": category})
        eligible = bool(row.get("binary_eligible"))
        strict_three = bool(row.get("strict_3_reader_eligible"))
        rated_count = int(row.get("rated_reader_count", 0))
        if eligible and category not in {"benign", "malignant"}:
            issues.append({"severity": "error", "code": "INVALID_BINARY_CATEGORY", "consensus_nodule_id": row["consensus_nodule_id"]})
        if eligible and row.get("binary_label") not in {0, 1}:
            issues.append({"severity": "error", "code": "MISSING_BINARY_LABEL", "consensus_nodule_id": row["consensus_nodule_id"]})
        if eligible and rated_count < int(config["label_policy"]["minimum_readers_for_binary"]):
            issues.append({"severity": "error", "code": "INSUFFICIENT_PRIMARY_RATINGS", "consensus_nodule_id": row["consensus_nodule_id"]})
        if strict_three and rated_count < int(config["label_policy"].get("strict_three_reader_minimum", 3)):
            issues.append({"severity": "error", "code": "INSUFFICIENT_STRICT_THREE_RATINGS", "consensus_nodule_id": row["consensus_nodule_id"]})
        if strict_three and not eligible:
            issues.append({"severity": "error", "code": "STRICT_THREE_NOT_PRIMARY_ELIGIBLE", "consensus_nodule_id": row["consensus_nodule_id"]})
    split_checksum = hashlib.sha256("\n".join(f"{row['consensus_nodule_id']}|{row['patient_id']}|{row['split']}" for row in sorted(split_rows, key=lambda item: item["consensus_nodule_id"])).encode()).hexdigest()
    status = "fail" if any(issue.get("severity") == "error" for issue in issues) else "warn" if issues or stats["excluded_nodules"] else "pass"
    return {
        "phase": "phase3_shared_team_dataset",
        "run_scope": stats["run_scope"],
        "status": status,
        "counts": {key: stats[key] for key in ("patients_included", "reader_annotations_used", "consensus_physical_nodules", "rated_consensus_nodules", "consensus_samples_created", "benign_nodules", "malignant_nodules", "indeterminate_nodules", "disagreement_uncertain_nodules", "strict_binary_eligible_nodules", "strict_3_reader_eligible_nodules", "indeterminate_excluded", "disagreement_excluded", "insufficient_reader_excluded", "excluded_nodules")},
        "split_patient_counts": stats["split_patient_counts"],
        "split_nodule_counts": stats["split_nodule_counts"],
        "split_checksum_sha256": split_checksum,
        "crop": {key: stats[key] for key in ("crop_fov_mm_zyx", "crop_shape_zyx", "target_spacing_mm_zyx", "crop_statistics")},
        "issues": issues,
        "config": {"clustering": config["clustering"], "label_policy": config["label_policy"], "splits": config["splits"]},
    }


def write_validation_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [f"# Phase 3 Validation Report ({report['run_scope']})", "", f"- Status: **{report['status'].upper()}**", "", "## Counts", "", "| Metric | Count |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in report["counts"].items())
    lines.extend(["", "## Splits", "", "| Split | Patients | Samples |", "|---|---:|---:|"])
    for split in ("train", "validation", "test"):
        lines.append(f"| {split} | {report['split_patient_counts'].get(split, 0)} | {report['split_nodule_counts'].get(split, 0)} |")
    lines.extend(["", "## Master sample geometry", "", f"- Crop FOV (z, y, x): `{report['crop']['crop_fov_mm_zyx']} mm`", f"- Crop shape (z, y, x): `{report['crop']['crop_shape_zyx']}`", f"- Target spacing (z, y, x): `{report['crop']['target_spacing_mm_zyx']} mm`", "", "## Issues", ""])
    if report["issues"]:
        lines.extend(f"- **{issue.get('severity', 'warning').upper()}** `{issue.get('code', 'UNSPECIFIED')}`: {issue}" for issue in report["issues"])
    else:
        lines.append("No validation issues were recorded.")
    lines.extend(["", f"- Split checksum (SHA-256): `{report['split_checksum_sha256']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
