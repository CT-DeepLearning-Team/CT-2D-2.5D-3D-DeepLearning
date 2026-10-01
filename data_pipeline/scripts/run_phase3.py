"""Run the configurable Phase 3 pilot or full shared-team dataset build."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import configure_logging  # noqa: E402
from scripts.phase3 import process_phase3, read_csv, select_evenly_spaced_patients  # noqa: E402


LOGGER = logging.getLogger("lidc_phase3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--patient-id", action="append", dest="patient_ids")
    parser.add_argument("--max-patients", type=int, help="Pilot size. Patients are selected evenly from sorted canonical patient IDs.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing Phase 3 output directory.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent.parent if path.parent.name == "config" else Path.cwd()
    phase3 = config["phase3"]
    canonical_root = Path(phase3["canonical_root"]).expanduser()
    if not canonical_root.is_absolute():
        canonical_root = (base / canonical_root).resolve()
    output_root = Path(phase3["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = (base / output_root).resolve()
    phase3["canonical_root"] = str(canonical_root)
    phase3["output_root"] = str(output_root)
    return phase3


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    canonical_root = Path(config["canonical_root"])
    output_root = args.output_root.expanduser().resolve() if args.output_root else Path(config["output_root"])
    if not (canonical_root / "metadata.csv").exists():
        raise FileNotFoundError(f"Canonical metadata.csv is missing: {canonical_root / 'metadata.csv'}")

    metadata = read_csv(canonical_root / "metadata.csv")
    all_patients = sorted({row["patient_id"] for row in metadata if row.get("status") == "ok"})
    requested = sorted(set(args.patient_ids or []))
    if requested:
        unknown = sorted(set(requested) - set(all_patients))
        if unknown:
            raise ValueError(f"Requested patients are absent from canonical metadata: {unknown}")
        selected_patients = requested
    else:
        count = args.max_patients or int(config.get("pilot_patient_count", 20))
        selected_patients = select_evenly_spaced_patients(all_patients, count)

    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing and not args.overwrite:
            raise FileExistsError(f"Phase 3 output is not empty; pass --overwrite to replace it: {output_root}")
        if args.overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = REPO_ROOT / "logs" / "phase3.log"
    configure_logging(log_path, verbose=args.verbose, logger_name="lidc_phase3")
    run_config = json.loads(json.dumps(config))
    run_config["run_scope"] = "pilot" if len(selected_patients) < len(all_patients) else "full"
    run_config["pilot_patient_ids"] = selected_patients
    LOGGER.info("Starting Phase 3 %s: %d/%d patients", run_config["run_scope"], len(selected_patients), len(all_patients))
    result = process_phase3(canonical_root, output_root, run_config, selected_patients)
    report = result["report"]
    print(json.dumps({"status": report["status"], "counts": report["counts"], "crop": report["crop"], "split_patient_counts": report["split_patient_counts"]}, indent=2, sort_keys=True))
    LOGGER.info("Phase 3 complete: %s", json.dumps(report["counts"], sort_keys=True))
    return 0 if report["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
