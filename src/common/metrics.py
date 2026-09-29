"""The frozen metric set. Identical for 2D, 2.5D and 3D."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from src.common import config as C


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                class_names: tuple[str, ...] = C.CLASS_NAMES) -> dict:
    """y_prob: (N, K) softmax probabilities. Works for K = 3 and K = 2."""
    k = len(class_names)
    m: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    per_p = precision_score(y_true, y_pred, average=None,
                            labels=range(k), zero_division=0)
    per_r = recall_score(y_true, y_pred, average=None,
                         labels=range(k), zero_division=0)
    per_f = f1_score(y_true, y_pred, average=None,
                     labels=range(k), zero_division=0)
    for i, name in enumerate(class_names):
        m[f"precision_{name}"] = float(per_p[i])
        m[f"recall_{name}"] = float(per_r[i])
        m[f"f1_{name}"] = float(per_f[i])

    m["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=range(k)
    ).tolist()

    try:
        if k == 2:
            # Binary task: a single ROC-AUC on P(malignant).
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            m["roc_auc_ovr_macro"] = m["roc_auc"]
        else:
            m["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
            m["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            )
            for i, name in enumerate(class_names):
                m[f"auc_{name}_vs_rest"] = float(
                    roc_auc_score((y_true == i).astype(int), y_prob[:, i])
                )
    except ValueError as exc:
        m["roc_auc_error"] = str(exc)
    return m


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "macro_f1",
                 n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI. With only 396 test nodules the sampling error is
    large, so every headline number should carry one."""
    rng = np.random.default_rng(seed)
    fn = {"macro_f1": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
          "accuracy": accuracy_score,
          "balanced_accuracy": balanced_accuracy_score}[metric]
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        vals.append(fn(y_true[s], y_pred[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def majority_baseline(y_true: np.ndarray,
                      class_names: tuple[str, ...] = C.CLASS_NAMES) -> dict:
    """Always predict the most frequent class. Any model must beat this."""
    k = len(class_names)
    maj = np.bincount(y_true, minlength=k).argmax()
    y_pred = np.full_like(y_true, maj)
    prob = np.zeros((len(y_true), k)); prob[:, maj] = 1.0
    return compute_all(y_true, y_pred, prob, class_names)


def format_report(name: str, m: dict,
                  class_names: tuple[str, ...] = C.CLASS_NAMES) -> str:
    L = [f"=== {name} ==="]
    L.append(f"  accuracy           {m['accuracy']:.4f}")
    L.append(f"  balanced accuracy  {m['balanced_accuracy']:.4f}")
    L.append(f"  macro precision    {m['macro_precision']:.4f}")
    L.append(f"  macro recall       {m['macro_recall']:.4f}")
    L.append(f"  macro F1           {m['macro_f1']:.4f}")
    if "roc_auc" in m:
        L.append(f"  ROC-AUC            {m['roc_auc']:.4f}")
    elif "roc_auc_ovr_macro" in m:
        L.append(f"  ROC-AUC (OvR macro) {m['roc_auc_ovr_macro']:.4f}")
    L.append("  per class:      precision  recall      F1")
    for n_ in class_names:
        L.append(f"    {n_:<14}{m['precision_'+n_]:>9.4f}{m['recall_'+n_]:>8.4f}"
                 f"{m['f1_'+n_]:>8.4f}")
    L.append("  confusion matrix (rows=true " + "/".join(class_names) + "):")
    for r in m["confusion_matrix"]:
        L.append("    " + "".join(f"{v:>6d}" for v in r))
    return "\n".join(L)
