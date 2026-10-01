#!/usr/bin/env python3
"""Generate final 2D/2.5D/3D comparison figures from the committed CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPRESENTATIONS = ["2D", "2.5D", "3D"]
EXPERIMENTS = ["A", "B", "C", "Binary"]


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in (
                "validation_macro_f1",
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "roc_auc",
            ):
                row[key] = float(str(row[key]))
            std = str(row["test_macro_f1_std"]).strip()
            row["test_macro_f1_std"] = float(std) if std else None
            rows.append(row)
    return rows


def value(rows: list[dict[str, object]], representation: str, experiment: str, key: str) -> float:
    match = next(
        row
        for row in rows
        if row["representation"] == representation and row["experiment"] == experiment
    )
    return float(match[key])


def std_value(rows: list[dict[str, object]], representation: str, experiment: str) -> float | None:
    match = next(
        row
        for row in rows
        if row["representation"] == representation and row["experiment"] == experiment
    )
    return match["test_macro_f1_std"]  # type: ignore[return-value]


WIDTH = 1000
HEIGHT = 650
LEFT = 105
TOP = 90
RIGHT = 950
BOTTOM = 540
BLUE = (42, 95, 160)
GREEN = (47, 143, 131)
ORANGE = (219, 132, 50)
RED = (201, 92, 92)
GRID = (220, 225, 230)
TEXT = (35, 40, 45)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)


def chart(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.text((LEFT, 28), title, fill=TEXT, font=font(27))
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = BOTTOM - int(tick * (BOTTOM - TOP))
        draw.line((LEFT, y, RIGHT, y), fill=GRID, width=1)
        draw.text((48, y - 9), f"{tick:.2f}", fill=TEXT, font=font(16))
    draw.line((LEFT, TOP, LEFT, BOTTOM), fill=TEXT, width=2)
    draw.line((LEFT, BOTTOM, RIGHT, BOTTOM), fill=TEXT, width=2)
    draw.text((16, TOP - 28), "score", fill=TEXT, font=font(16))
    return image, draw


def y_coord(value: float) -> int:
    return BOTTOM - int(max(0.0, min(1.0, value)) * (BOTTOM - TOP))


def x_positions(labels: list[str]) -> list[int]:
    if len(labels) == 1:
        return [(LEFT + RIGHT) // 2]
    spacing = (RIGHT - LEFT) / (len(labels) - 1)
    return [int(LEFT + i * spacing) for i in range(len(labels))]


def save(image: Image.Image, output: Path, name: str) -> None:
    image.save(output / name, format="PNG", optimize=True)


def label_x(draw: ImageDraw.ImageDraw, labels: list[str], positions: list[int]) -> None:
    for label, x in zip(labels, positions):
        box = draw.textbbox((0, 0), label, font=font(18))
        draw.text((x - (box[2] - box[0]) // 2, BOTTOM + 18), label, fill=TEXT, font=font(18))


def line_chart(rows: list[dict[str, object]], experiment: str, output: Path) -> None:
    means = [value(rows, rep, experiment, "test_macro_f1") for rep in REPRESENTATIONS]
    errors = [std_value(rows, rep, experiment) for rep in REPRESENTATIONS]
    image, draw = chart(f"Model {experiment} Macro-F1 by representation")
    positions = x_positions(REPRESENTATIONS)
    points = [(x, y_coord(y)) for x, y in zip(positions, means)]
    draw.line(points, fill=BLUE, width=4)
    for (x, y), err in zip(points, errors):
        if err is not None:
            top = y_coord(means[points.index((x, y))] + err)
            bottom = y_coord(means[points.index((x, y))] - err)
            draw.line((x, top, x, bottom), fill=BLUE, width=3)
            draw.line((x - 7, top, x + 7, top), fill=BLUE, width=3)
            draw.line((x - 7, bottom, x + 7, bottom), fill=BLUE, width=3)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=BLUE, outline="white", width=2)
    label_x(draw, REPRESENTATIONS, positions)
    save(image, output, f"{experiment.lower()}_macro_f1_by_representation.png")


def grouped_abc(rows: list[dict[str, object]], output: Path) -> None:
    image, draw = chart("Three-class Macro-F1 across representations")
    positions = x_positions(REPRESENTATIONS)
    colors = {"A": BLUE, "B": ORANGE, "C": GREEN}
    for index, experiment in enumerate(("A", "B", "C")):
        for x, rep in zip(positions, REPRESENTATIONS):
            center = x + (index - 1) * 25
            top = y_coord(value(rows, rep, experiment, "test_macro_f1"))
            draw.rectangle((center - 10, top, center + 10, BOTTOM), fill=colors[experiment])
    label_x(draw, REPRESENTATIONS, positions)
    draw.text((LEFT + 20, TOP - 30), "A", fill=BLUE, font=font(17))
    draw.text((LEFT + 70, TOP - 30), "B", fill=ORANGE, font=font(17))
    draw.text((LEFT + 120, TOP - 30), "C", fill=GREEN, font=font(17))
    save(image, output, "abc_macro_f1_grouped.png")


def delta_chart(rows: list[dict[str, object]], first: str, second: str, title: str, filename: str, output: Path) -> None:
    deltas = [
        value(rows, rep, second, "test_macro_f1") - value(rows, rep, first, "test_macro_f1")
        for rep in REPRESENTATIONS
    ]
    low = min(-0.12, min(deltas) - 0.03)
    high = max(0.12, max(deltas) + 0.03)
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.text((LEFT, 28), title, fill=TEXT, font=font(27))

    def delta_y(value_: float) -> int:
        fraction = (value_ - low) / (high - low)
        return BOTTOM - int(fraction * (BOTTOM - TOP))

    for tick in (low, (low + high) / 2, high):
        y = delta_y(tick)
        draw.line((LEFT, y, RIGHT, y), fill=GRID, width=1)
        draw.text((35, y - 9), f"{tick:+.2f}", fill=TEXT, font=font(16))
    draw.line((LEFT, TOP, LEFT, BOTTOM), fill=TEXT, width=2)
    draw.line((LEFT, BOTTOM, RIGHT, BOTTOM), fill=TEXT, width=2)
    draw.text((16, TOP - 28), "change", fill=TEXT, font=font(16))
    positions = x_positions(REPRESENTATIONS)
    baseline = delta_y(0.0)
    for x, delta in zip(positions, deltas):
        top = delta_y(max(delta, 0.0))
        bottom = delta_y(min(delta, 0.0))
        draw.rectangle((x - 28, top, x + 28, max(bottom, baseline)), fill=GREEN if delta >= 0 else RED)
        draw.text((x - 25, min(top, bottom) - 25, ), f"{delta:+.3f}", fill=TEXT, font=font(16))
    label_x(draw, REPRESENTATIONS, positions)
    draw.text((LEFT + 20, BOTTOM + 55), "positive = improvement", fill=TEXT, font=font(16))
    save(image, output, filename)


def model_c_metrics(rows: list[dict[str, object]], output: Path) -> None:
    metrics = [
        ("test_accuracy", "Accuracy"),
        ("test_balanced_accuracy", "Balanced accuracy"),
        ("test_macro_f1", "Macro-F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    image, draw = chart("Model C metrics by representation")
    positions = x_positions(REPRESENTATIONS)
    colors = [BLUE, ORANGE, GREEN, RED]
    for index, (key, label) in enumerate(metrics):
        for x, rep in zip(positions, REPRESENTATIONS):
            center = x + (index - 1.5) * 18
            top = y_coord(value(rows, rep, "C", key))
            draw.rectangle((center - 7, top, center + 7, BOTTOM), fill=colors[index])
        draw.text((LEFT + 20 + (index % 2) * 230, TOP - 30 + (index // 2) * 22), label, fill=colors[index], font=font(15))
    label_x(draw, REPRESENTATIONS, positions)
    save(image, output, "model_c_metrics_by_representation.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = args.input or repo_root / "results/team_comparison/final_team_metrics.csv"
    output = args.output or repo_root / "results/team_comparison/figures"
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path)
    for experiment in EXPERIMENTS:
        line_chart(rows, experiment, output)
    grouped_abc(rows, output)
    delta_chart(rows, "A", "C", "Augmentation change: Model A → Model C", "augmentation_gain_macro_f1.png", output)
    delta_chart(rows, "A", "B", "MoCo change: Model A → Model B", "moco_change_macro_f1.png", output)
    delta_chart(rows, "C", "Binary", "Binary versus three-class Model C", "three_class_vs_binary_gain.png", output)
    model_c_metrics(rows, output)
    for path in sorted(output.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
