"""Resumable supervised training for final A, B, C, and binary runs."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics


@dataclass
class SupervisedConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    output_dir: Path = Path("runs_final_common_hu")
    checkpoint_name: str = "best_model.pt"
    latest_name: str = "latest_checkpoint.pt"
    history_name: str = "history.json"


def seed_all(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def class_weights(dataset: Any, num_classes: int) -> tuple[Tensor, list[int]]:
    if getattr(dataset, "split", None) != "train": raise ValueError("Weights require train dataset")
    labels = [int(record.label) for record in dataset.records]
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
    if torch.any(counts == 0): raise ValueError(f"Missing class in train: {counts.tolist()}")
    total = counts.sum().float()
    return total / (num_classes * counts.float()), counts.tolist()


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])


def _batch(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    return batch["image"].to(device), batch["label"].to(device)


def _epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None, names: tuple[str, ...]) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0; total = 0; targets = []; predictions = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images, labels = _batch(batch, device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels)
            if training: loss.backward(); optimizer.step()
            count = labels.shape[0]; total_loss += float(loss.detach().item()) * count; total += count
            targets.extend(labels.detach().cpu().tolist()); predictions.extend(logits.detach().argmax(1).cpu().tolist())
    metrics = classification_metrics(targets, predictions, names); metrics["loss"] = total_loss / total
    return metrics


def _checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, *, epoch: int, best_epoch: int, best_f1: float, counter: int, history: list[dict[str, Any]], weights: Tensor, counts: list[int], config: SupervisedConfig, names: tuple[str, ...]) -> dict[str, Any]:
    return {"epoch": epoch, "completed_epoch": epoch, "best_epoch": best_epoch, "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "optimizer_state_dict": optimizer.state_dict(), "best_validation_macro_f1": best_f1, "early_stopping_counter": counter, "history": history, "rng_state": _rng_state(), "class_names": list(names), "class_weights": weights.cpu(), "train_class_counts": counts, "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}}


def fit_supervised(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, *, names: tuple[str, ...], config: SupervisedConfig, device: torch.device, resume: bool = False) -> dict[str, Any]:
    if getattr(train_loader.dataset, "split", None) != "train" or getattr(validation_loader.dataset, "split", None) != "validation":
        raise ValueError("Only train and validation loaders are permitted")
    model.to(device); weights, counts = class_weights(train_loader.dataset, len(names)); criterion = nn.CrossEntropyLoss(weight=weights.to(device)); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    out = Path(config.output_dir); out.mkdir(parents=True, exist_ok=True); best_path = out / config.checkpoint_name; latest_path = out / config.latest_name; history_path = out / config.history_name
    history: list[dict[str, Any]] = []; best_f1 = float("-inf"); best_epoch = 0; counter = 0; start = 1
    if resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=False); model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); start = int(state["completed_epoch"]) + 1; best_epoch = int(state["best_epoch"]); best_f1 = float(state["best_validation_macro_f1"]); counter = int(state["early_stopping_counter"]); history = list(state["history"]); _restore_rng(state["rng_state"])
    for epoch in range(start, config.epochs + 1):
        train_metrics = _epoch(model, train_loader, criterion, device, optimizer, names); val_metrics = _epoch(model, validation_loader, criterion, device, None, names)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}; history.append(record); history_path.write_text(json.dumps(history, indent=2) + "\n")
        f1 = val_metrics["macro_f1"]
        if f1 > best_f1 + config.min_delta:
            best_f1 = f1; best_epoch = epoch; counter = 0
            torch.save(_checkpoint(model, optimizer, epoch=epoch, best_epoch=best_epoch, best_f1=best_f1, counter=counter, history=history, weights=weights, counts=counts, config=config, names=names), best_path)
        else: counter += 1
        torch.save(_checkpoint(model, optimizer, epoch=epoch, best_epoch=best_epoch, best_f1=best_f1, counter=counter, history=history, weights=weights, counts=counts, config=config, names=names), latest_path)
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_macro_f1={f1:.4f}", flush=True)
        if counter >= config.patience: break
    return {"best_epoch": best_epoch, "best_validation_macro_f1": best_f1, "best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path), "history": history}


def load_transferred_classifier(checkpoint_path: Path, num_classes: int) -> nn.Module:
    from .model import CommonHUResNet3D18
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CommonHUResNet3D18(num_classes=num_classes, dropout=0.20)
    source = checkpoint["model_state_dict"]
    backbone = {key[len("encoder_q.backbone."):]: value for key, value in source.items() if key.startswith("encoder_q.backbone.") and key[len("encoder_q.backbone."):].startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}
    if not backbone: raise ValueError("No query encoder backbone in MoCo checkpoint")
    model.load_state_dict(backbone, strict=False)
    return model
