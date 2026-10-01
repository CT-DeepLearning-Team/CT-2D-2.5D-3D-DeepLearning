"""MoCo v2-style 3D SSL model and resumable trainer."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .model import CommonHUResNet3D18


class ProjectionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))
    def forward(self, x: Tensor) -> Tensor: return self.net(x)


class MoCoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.backbone = CommonHUResNet3D18(num_classes=3, dropout=0.20); self.projector = ProjectionMLP()
    def forward(self, x: Tensor) -> Tensor: return nn.functional.normalize(self.projector(self.backbone.forward_features(x)), dim=1)


class MoCo3D(nn.Module):
    def __init__(self, queue_size: int = 4096, momentum: float = 0.999, temperature: float = 0.07) -> None:
        super().__init__(); self.momentum = momentum; self.temperature = temperature
        self.encoder_q = MoCoEncoder(); self.encoder_k = copy.deepcopy(self.encoder_q)
        for parameter in self.encoder_k.parameters(): parameter.requires_grad_(False)
        queue = nn.functional.normalize(torch.randn(128, queue_size), dim=0)
        self.register_buffer("queue", queue); self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self) -> None:
        for query, key in zip(self.encoder_q.parameters(), self.encoder_k.parameters()): key.data.mul_(self.momentum).add_(query.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_enqueue(self, keys: Tensor) -> None:
        batch = keys.shape[0]; pointer = int(self.queue_ptr.item()); size = self.queue.shape[1]
        if size % batch != 0: raise ValueError("Queue size must be divisible by batch size")
        self.queue[:, pointer:pointer + batch] = keys.T; self.queue_ptr[0] = (pointer + batch) % size

    def forward(self, view_q: Tensor, view_k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query = self.encoder_q(view_q)
        with torch.no_grad():
            self.momentum_update_key_encoder(); key = self.encoder_k(view_k)
        positive = torch.sum(query * key, dim=1, keepdim=True)
        queue_snapshot = self.queue.detach().clone()
        negatives = query @ queue_snapshot
        logits = torch.cat((positive, negatives), dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_enqueue(key.detach())
        return logits, labels, query


@dataclass
class MoCoConfig:
    epochs: int = 50; learning_rate: float = 3e-4; weight_decay: float = 1e-4; queue_size: int = 4096; momentum: float = 0.999; temperature: float = 0.07; output_dir: Path = Path("runs_final_common_hu/moco_pretrain_3d_resnet18"); latest_name: str = "latest_checkpoint.pt"; final_name: str = "moco_final.pt"; history_name: str = "history.json"


def _rng() -> dict[str, Any]: return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
def _restore(state: dict[str, Any]) -> None: random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])


def fit_moco(model: MoCo3D, loader: DataLoader, *, config: MoCoConfig, device: torch.device, resume: bool = False) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    model.to(device); out = Path(config.output_dir); out.mkdir(parents=True, exist_ok=True); latest = out / config.latest_name; final = out / config.final_name; history_path = out / config.history_name
    history: list[dict[str, Any]] = []; start = 1
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); scheduler.load_state_dict(state["scheduler_state_dict"]); start = int(state["completed_epoch"]) + 1; history = list(state["history"]); _restore(state["rng_state"])
    for epoch in range(start, config.epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            q = batch["view_q"].to(device); k = batch["view_k"].to(device); optimizer.zero_grad(set_to_none=True); logits, labels, _ = model(q, k); loss = nn.functional.cross_entropy(logits, labels); loss.backward(); optimizer.step(); total += float(loss.detach()) * q.shape[0]; count += q.shape[0]
        scheduler.step(); record = {"epoch": epoch, "ssl_loss": total / count, "learning_rate": optimizer.param_groups[0]["lr"]}; history.append(record); history_path.write_text(json.dumps(history, indent=2) + "\n")
        state = {"epoch": epoch, "completed_epoch": epoch, "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "queue": model.queue.detach().cpu().clone(), "queue_ptr": model.queue_ptr.detach().cpu().clone(), "history": history, "rng_state": _rng(), "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}}
        torch.save(state, latest); print(f"epoch={epoch:03d} ssl_loss={record['ssl_loss']:.6f} lr={record['learning_rate']:.3e}", flush=True)
    if history: torch.save(state, final)
    return {"history": history, "latest_checkpoint": str(latest), "final_checkpoint": str(final)}
