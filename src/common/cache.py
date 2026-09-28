"""Build a central-slice cache so 2D training does not re-read 17 GB of volumes.

Each stored sample is a (104, 72, 80) float32 volume (~2.4 MB). The 2D model
only ever needs slice `CENTRAL_SLICE`, so we extract it once into a single
array of shape (N, 72, 80) -- about 170 MB, which fits comfortably in RAM.

Values are kept as RAW Hounsfield units here; windowing happens in the
transform, so the cache never has to be rebuilt if the window changes.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.common import config as C

CACHE_ARRAY = C.CACHE_DIR / "central_slices.npy"
CACHE_INDEX = C.CACHE_DIR / "central_slices_index.json"


def build(force: bool = False) -> None:
    if CACHE_ARRAY.exists() and CACHE_INDEX.exists() and not force:
        print(f"cache already present: {CACHE_ARRAY}")
        return

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.METADATA / "all_nodules_v2.csv", low_memory=False)
    ids = df[C.ID_COL].tolist()
    paths = df[C.PATH_COL].tolist()
    n = len(ids)
    print(f"caching central slice {C.CENTRAL_SLICE} for {n} nodules")

    out = np.lib.format.open_memmap(
        CACHE_ARRAY, mode="w+", dtype=np.float32, shape=(n, *C.SLICE_HW)
    )
    bad: list[str] = []
    for i, (nid, rel) in enumerate(zip(ids, paths)):
        f = C.DATA_ROOT / rel
        try:
            vol = np.load(f, mmap_mode="r")
            if tuple(vol.shape) != C.SAMPLE_SHAPE_ZYX:
                bad.append(f"{nid}: shape {vol.shape}")
                continue
            out[i] = np.asarray(vol[C.CENTRAL_SLICE], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{nid}: {exc}")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    out.flush()
    del out
    CACHE_INDEX.write_text(json.dumps({nid: i for i, nid in enumerate(ids)}))
    print(f"wrote {CACHE_ARRAY} ({CACHE_ARRAY.stat().st_size / 1e6:.0f} MB)")
    if bad:
        print(f"WARNING: {len(bad)} problem files", file=sys.stderr)
        for b in bad[:10]:
            print("  " + b, file=sys.stderr)


def load() -> tuple[np.ndarray, dict[str, int]]:
    if not CACHE_ARRAY.exists():
        raise FileNotFoundError(
            f"{CACHE_ARRAY} missing -- run: python scripts/build_cache.py"
        )
    return np.load(CACHE_ARRAY, mmap_mode="r"), json.loads(CACHE_INDEX.read_text())


if __name__ == "__main__":
    build(force="--force" in sys.argv)
