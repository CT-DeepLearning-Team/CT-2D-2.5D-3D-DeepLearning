"""Aggregate 2D results across seeds into the team report format.

Produces mean +/- std over seeds for Model A and Model B, the seed-pooled
confusion matrix, bootstrap CIs, a majority-class baseline for reference, and
a paired A-vs-B comparison. Writes results/2d/REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import config as C
from src.common.data import load_splits
from src.common.metrics import majority_baseline
from src.common.plots import confusion_figure

HEADLINE = ["macro_f1", "accuracy", "balanced_accuracy",
            "macro_precision", "macro_recall", "roc_auc_ovr_macro"]
PERCLASS = [f"f1_{n}" for n in C.CLASS_NAMES]


def collect(tag: str) -> list[dict]:
    base = C.RESULTS_DIR / "2d" / tag
    out = []
    for d in sorted(base.glob("seed*")):
        f = d / "result.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("test"):
                out.append(r)
    return out


def ms(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    return f"{a.mean():.4f} ± {a.std(ddof=0):.4f}" if len(a) > 1 else f"{a.mean():.4f}"


def pooled_cm(runs: list[dict]) -> np.ndarray:
    return np.sum([np.asarray(r["test"]["confusion_matrix"]) for r in runs], axis=0)


def section(name: str, runs: list[dict]) -> list[str]:
    L = [f"### {name}  (n = {len(runs)} seeds: "
         f"{', '.join(str(r['seed']) for r in runs)})", ""]
    L += ["| metric | validation | test |", "|---|---|---|"]
    for k in HEADLINE:
        if k not in runs[0]["test"]:
            continue
        L.append(f"| {k} | {ms([r['validation'][k] for r in runs])} | "
                 f"**{ms([r['test'][k] for r in runs])}** |")
    L.append(f"| best val macro F1 (selection) | "
             f"{ms([r['best_val_macro_f1'] for r in runs])} | — |")
    L += ["", "Per-class test F1:", "",
          "| class | test F1 | test precision | test recall |", "|---|---|---|---|"]
    for n in C.CLASS_NAMES:
        L.append(f"| {n} | {ms([r['test']['f1_'+n] for r in runs])} | "
                 f"{ms([r['test']['precision_'+n] for r in runs])} | "
                 f"{ms([r['test']['recall_'+n] for r in runs])} |")
    cis = [r["test"]["macro_f1_ci95"] for r in runs]
    L += ["", f"Test macro-F1 bootstrap 95% CI (per seed): " +
          "; ".join(f"[{a:.3f}, {b:.3f}]" for a, b in cis), ""]
    cm = pooled_cm(runs)
    L += [f"Seed-pooled test confusion matrix (rows = true "
          f"{'/'.join(C.CLASS_NAMES)}):", "", "```"]
    for r in cm:
        L.append("".join(f"{v:>7d}" for v in r))
    L += ["```", ""]
    ep = [r["best_epoch"] for r in runs]
    tr = [r["train_minutes"] for r in runs]
    L += [f"Selected epoch: {ep} | training time: "
          f"{np.mean(tr):.1f} min/seed | params: {runs[0]['params']:,}", ""]
    return L


def main() -> None:
    a, b = collect("model_a"), collect("model_b")
    if not a:
        raise SystemExit("no Model A results with test metrics yet")

    splits = load_splits()
    y_test = splits["test"][C.LABEL_COL].to_numpy()
    base = majority_baseline(y_test)

    L = ["# 2D Results — ResNet-18 on the central CT slice", "",
         "**Owner:** Bahodir (2D track)  ",
         f"**Input:** single slice, index {C.CENTRAL_SLICE} of the "
         f"{C.SAMPLE_SHAPE_ZYX} 1 mm crop → {C.SLICE_HW[0]}×{C.SLICE_HW[1]}  ",
         f"**HU window:** clip [{C.HU_MIN:.0f}, {C.HU_MAX:.0f}] → [0,1], then "
         "standardised with train-split mean/std  ",
         "**Task:** 3-class (benign / indeterminate / malignant), "
         "`class_index_v2`  ",
         f"**Splits:** train {len(splits['train'])} / val {len(splits['val'])} "
         f"/ test {len(splits['test'])} — shared CSVs, unmodified  ",
         "**Selection:** best validation macro-F1, early stopping  ",
         "**Model A:** random init.  **Model B:** MoCo v2 pretrain "
         "(`ssl_pretrain_train.csv`, train patients only) → same fine-tuning",
         "",
         "All numbers are mean ± std over seeds. Test split was evaluated only "
         "after the configuration was frozen on validation.", "",
         "## Reference baseline", "",
         f"Always predicting the majority class (*indeterminate*, "
         f"{int(np.bincount(y_test).max())}/{len(y_test)} of test) gives "
         f"macro F1 **{base['macro_f1']:.4f}**, accuracy "
         f"**{base['accuracy']:.4f}**, balanced accuracy "
         f"**{base['balanced_accuracy']:.4f}**.", "",
         "## Results", ""]
    L += section("Model A — random initialisation", a)
    if b:
        L += section("Model B — MoCo v2 pretraining → fine-tuning", b)
        L += ["### Model A vs Model B (test)", "",
              "| metric | Model A | Model B | Δ (B − A) |", "|---|---|---|---|"]
        for k in HEADLINE + PERCLASS:
            if k not in a[0]["test"]:
                continue
            va = np.mean([r["test"][k] for r in a])
            vb = np.mean([r["test"][k] for r in b])
            L.append(f"| {k} | {va:.4f} | {vb:.4f} | "
                     f"{'+' if vb >= va else ''}{vb - va:.4f} |")
        L.append("")
    else:
        L += ["### Model B", "", "_Not yet available._", ""]

    out = C.RESULTS_DIR / "2d" / "REPORT.md"
    out.write_text("\n".join(L))
    for tag, runs in (("model_a", a), ("model_b", b)):
        if runs:
            confusion_figure(
                pooled_cm(runs).tolist(),
                C.RESULTS_DIR / "2d" / tag / "confusion_test_pooled.png",
                f"2D {tag} — test, pooled over {len(runs)} seeds")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
