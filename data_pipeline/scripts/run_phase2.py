"""Create canonical LIDC-IDRI volumes and reader-level annotation masks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.canonical import (  # noqa: E402
    load_phase1_manifests,
    process_canonical_dataset,
    write_validation_report,
)
from scripts.common import configure_logging, write_json  # noqa: E402

LOGGER = logging.getLogger("lidc_phase2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--patient-id", action="append", dest="patient_ids", help="Restrict Phase 2 to one or more patients.")
    parser.add_argument("--max-patients", type=int, help="Process only the first N selected patients in sorted order.")
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
    canonical_root = Path(config.get("canonical_output_root", "processed/canonical")).expanduser()
    if not canonical_root.is_absolute():
        canonical_root = (base / canonical_root).resolve()
    config["canonical_output_root"] = str(canonical_root)
    return config


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    log_path = REPO_ROOT / "logs" / "phase2.log"
    configure_logging(log_path, verbose=args.verbose, logger_name="lidc_phase2")
    manifests = load_phase1_manifests(REPO_ROOT / "processed/manifests")
    patient_ids = args.patient_ids or config.get("phase2_patient_ids", [])
    result = process_canonical_dataset(
        manifests,
        Path(config["raw_dicom_root"]),
        Path(config["xml_root"]),
        Path(config["canonical_output_root"]),
        patient_ids=patient_ids,
        max_patients=args.max_patients,
    )
    output_root = Path(config["canonical_output_root"])
    write_json(output_root / "run_config.json", {**config, "patient_ids": patient_ids, "max_patients": args.max_patients})
    report = write_validation_report(output_root, result)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "storage": report["storage"]}, indent=2, sort_keys=True))
    return 0 if report["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
