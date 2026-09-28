"""Training curves and confusion-matrix figures (overfitting evidence)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import config as C


def training_curves(history: dict, out: Path, title: str) -> None:
    ep = history["epoch"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_loss"], label="train")
    ax[0].plot(ep, history["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(ep, history["train_macro_f1"], label="train")
    ax[1].plot(ep, history["val_macro_f1"], label="validation")
    best = int(np.argmax(history["val_macro_f1"]))
    ax[1].axvline(ep[best], ls="--", c="k", lw=1,
                  label=f"selected (epoch {ep[best]})")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro F1")
    ax[1].set_title("Macro F1"); ax[1].legend(); ax[1].grid(alpha=.3)

    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def confusion_figure(cm: list[list[int]], out: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.NUM_CLASSES), C.CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(C.NUM_CLASSES), C.CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > .5 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
