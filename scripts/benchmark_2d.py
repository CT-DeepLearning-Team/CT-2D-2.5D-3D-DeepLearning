"""Computational cost of the 2D model, for the accuracy-vs-cost comparison.

Reports parameters, forward-pass latency (batch 1 and batched) and peak memory.
Run the same script shape for 2.5D/3D by changing the input tensor so the three
numbers are measured identically.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "2d"))
from src.common import config as C
from src.common.seed import device
from model import ResNet18Classifier, n_params  # noqa: E402


def bench(model, x, iters: int = 100, warmup: int = 20) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if x.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if x.device.type == "mps":
            torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    dev = device()
    model = ResNet18Classifier(in_channels=1).to(dev).eval()
    out = {
        "approach": "2D",
        "encoder": "ResNet-18",
        "input_shape": [1, *C.SLICE_HW],
        "device": str(dev),
        "parameters": n_params(model),
    }

    for bs in (1, 32, 128):
        x = torch.randn(bs, 1, *C.SLICE_HW, device=dev)
        per_batch = bench(model, x)
        out[f"latency_batch{bs}_ms"] = round(per_batch * 1e3, 3)
        out[f"per_nodule_batch{bs}_ms"] = round(per_batch / bs * 1e3, 4)

    if dev.type == "mps":
        x = torch.randn(64, 1, *C.SLICE_HW, device=dev)
        model.train()
        loss = model(x).sum()
        loss.backward()
        torch.mps.synchronize()
        out["peak_mps_memory_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)

    p = C.RESULTS_DIR / "2d" / "computational_cost.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
