"""Generate the reproducible final 3D result figures from saved artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "3d"
FIGURES = RESULTS / "figures"
CLASS_NAMES_3 = ("benign", "indeterminate", "malignant")
CLASS_NAMES_BINARY = ("benign", "malignant")


def read_metrics() -> list[dict[str, str]]:
    with (RESULTS / "final_3d_metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_history(directory: str) -> list[dict]:
    return json.loads((RESULTS / directory / "history.json").read_text())


def save_figure(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def grouped_bars(filename: str, labels: list[str], series: dict[str, list[float]], ylabel: str) -> None:
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series.items()):
        plt.bar(x + (index - (len(series) - 1) / 2) * width, values, width, label=name)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.legend()
    save_figure(filename)


def plot_summary(metrics: list[dict[str, str]]) -> None:
    labels = [row["experiment"] for row in metrics]
    grouped_bars(
        "abc_test_metrics.png",
        ["A", "B", "C"],
        {
            "Accuracy": [float(metrics[i]["test_accuracy"]) for i in range(3)],
            "Balanced accuracy": [float(metrics[i]["test_balanced_accuracy"]) for i in range(3)],
            "Macro-F1": [float(metrics[i]["test_macro_f1"]) for i in range(3)],
            "ROC-AUC": [float(metrics[i]["roc_auc"]) for i in range(3)],
        },
        "Test score",
    )
    grouped_bars(
        "train_validation_test_accuracy.png",
        labels,
        {
            "Train (post hoc)": [float(row["train_accuracy"]) for row in metrics],
            "Validation": [float(row["validation_accuracy"]) for row in metrics],
            "Test": [float(row["test_accuracy"]) for row in metrics],
        },
        "Accuracy",
    )
    grouped_bars(
        "validation_vs_test_macro_f1.png",
        labels,
        {
            "Validation Macro-F1": [float(row["validation_macro_f1"]) for row in metrics],
            "Test Macro-F1": [float(row["test_macro_f1"]) for row in metrics],
        },
        "Macro-F1",
    )


def plot_learning_curve(directory: str, filename: str, title: str) -> None:
    history = read_history(directory)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    validation_loss = [row["validation"]["loss"] for row in history]
    validation_f1 = [row["validation"]["macro_f1"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, validation_loss, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, validation_f1, label="Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    figure.suptitle(title)
    save_figure(filename)


def read_confusion(directory: str) -> tuple[list[str], np.ndarray]:
    with (RESULTS / directory / "test_confusion_matrix.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    names = rows[0][1:]
    matrix = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]], dtype=int)
    return names, matrix


def plot_confusion(directory: str, filename: str, title: str) -> None:
    names, matrix = read_confusion(directory)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    save_figure(filename)


def main() -> None:
    metrics = read_metrics()
    plot_summary(metrics)
    for directory, filename, title in (
        ("model_a", "model_a_learning_curves.png", "Model A learning curves"),
        ("model_b", "model_b_learning_curves.png", "Model B learning curves"),
        ("model_c", "model_c_learning_curves.png", "Model C learning curves"),
        ("binary", "binary_learning_curves.png", "Binary learning curves"),
    ):
        plot_learning_curve(directory, filename, title)
    for directory, filename, title in (
        ("model_a", "model_a_confusion_matrix.png", "Model A test confusion matrix"),
        ("model_b", "model_b_confusion_matrix.png", "Model B test confusion matrix"),
        ("model_c", "model_c_confusion_matrix.png", "Model C test confusion matrix"),
        ("binary", "binary_confusion_matrix.png", "Binary test confusion matrix"),
    ):
        plot_confusion(directory, filename, title)
    selected = [metrics[2], metrics[3]]
    grouped_bars(
        "three_class_vs_binary.png",
        ["Model C 3-class", "Binary"],
        {
            "Accuracy": [float(row["test_accuracy"]) for row in selected],
            "Balanced accuracy": [float(row["test_balanced_accuracy"]) for row in selected],
            "Macro-F1": [float(row["test_macro_f1"]) for row in selected],
            "ROC-AUC": [float(row["roc_auc"]) for row in selected],
        },
        "Test score",
    )
    print(f"generated_figures={FIGURES}")


if __name__ == "__main__":
    main()
