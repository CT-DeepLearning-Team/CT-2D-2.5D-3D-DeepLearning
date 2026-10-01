"""Dependency-light final-pipeline classification metrics."""

from __future__ import annotations

import numpy as np


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_metrics(targets: list[int], predictions: list[int], names: tuple[str, ...]) -> dict:
    n = len(names); confusion = np.zeros((n, n), dtype=np.int64)
    for target, pred in zip(targets, predictions): confusion[target, pred] += 1
    per_class = {}; precisions = []; recalls = []; f1s = []
    for i, name in enumerate(names):
        tp = float(confusion[i, i]); fp = float(confusion[:, i].sum() - tp); fn = float(confusion[i, :].sum() - tp)
        precision = _div(tp, tp + fp); recall = _div(tp, tp + fn); f1 = _div(2 * precision * recall, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(confusion[i, :].sum())}
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
    total = int(confusion.sum())
    return {"accuracy": _div(float(np.trace(confusion)), total), "balanced_accuracy": float(np.mean(recalls)), "macro_precision": float(np.mean(precisions)), "macro_recall": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)), "per_class": per_class, "confusion_matrix": confusion.tolist(), "num_samples": total}


def roc_auc_ovr(targets: list[int], probabilities: np.ndarray, names: tuple[str, ...]) -> dict:
    target = np.asarray(targets); result = {}; values = []
    for i, name in enumerate(names):
        positive = target == i; p = int(positive.sum()); q = int((~positive).sum())
        if not p or not q: result[name] = None; continue
        scores = probabilities[:, i]; order = np.argsort(scores, kind="mergesort"); sorted_scores = scores[order]; ranks = np.empty(scores.size)
        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]: end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0; start = end
        auc = (ranks[positive].sum() - p * (p + 1) / 2.0) / (p * q); result[name] = float(auc); values.append(float(auc))
    return {"macro_ovr_auc": float(np.mean(values)) if values else None, "per_class": result}
