"""Build the final 2D report in the structure the team agreed.

Covers all three deliverables:
  1. 3-class supervised baseline      (results/2d/model_a)
  2. 3-class MoCo + fine-tuning       (results/2d/model_b)
  3. Binary supervised                (results/2d/binary_supervised)

Every number is mean +/- std over seeds. Writes results/2d/REPORT.md and
results/2d/summary.json (machine-readable, for Zaineb's final tables).
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

TAGS = {
    "model_a": ("3-class supervised baseline (Model A)", "3class"),
    "model_b": ("3-class MoCo v2 + fine-tuning (Model B)", "3class"),
    "binary_supervised": ("Binary supervised (sensitivity analysis)", "binary"),
}


def collect(tag: str) -> list[dict]:
    """Load per-seed results, back-filling fields added after a run was made."""
    base = C.RESULTS_DIR / "2d" / tag
    runs = []
    for d in sorted(base.glob("seed*")):
        f = d / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if not r.get("test"):
            continue
        if "class_names" not in r:
            # Runs produced before the binary task existed: infer from the
            # confusion-matrix size rather than re-training.
            k = len(r["test"]["confusion_matrix"])
            r["class_names"] = list(
                C.BINARY_CLASS_NAMES if k == 2 else C.CLASS_NAMES)
            r.setdefault("task", "binary" if k == 2 else "3class")
            r.setdefault("num_classes", k)
        runs.append(r)
    return runs


def ms(vals) -> str:
    a = np.asarray(list(vals), dtype=float)
    return f"{a.mean():.4f} ± {a.std(ddof=0):.4f}" if len(a) > 1 else f"{a.mean():.4f}"


def mean(runs, where, key) -> float:
    return float(np.mean([r[where][key] for r in runs]))


def pooled_cm(runs) -> np.ndarray:
    return np.sum([np.asarray(r["test"]["confusion_matrix"]) for r in runs], axis=0)


def auc_key(runs) -> str:
    return "roc_auc" if "roc_auc" in runs[0]["test"] else "roc_auc_ovr_macro"


def overfit_gap(runs) -> float:
    """train macro-F1 minus val macro-F1 at the selected epoch."""
    g = []
    for r in runs:
        i = r["best_epoch"] - 1
        g.append(r["history"]["train_macro_f1"][i] - r["history"]["val_macro_f1"][i])
    return float(np.mean(g))


def final_overfit_gap(runs) -> float:
    """train macro-F1 minus val macro-F1 at the LAST epoch run."""
    return float(np.mean([r["history"]["train_macro_f1"][-1]
                          - r["history"]["val_macro_f1"][-1] for r in runs]))


def section(tag: str, runs: list[dict]) -> list[str]:
    title, task = TAGS[tag]
    names = tuple(runs[0]["class_names"])
    ak = auc_key(runs)
    a0 = runs[0]["args"]
    L = [f"## {title}", "",
         f"Seeds: {', '.join(str(r['seed']) for r in runs)} "
         f"(n = {len(runs)}), reported as mean ± std.", "",
         "| metric | validation | test |", "|---|---|---|",
         f"| **macro F1** | {ms(r['validation']['macro_f1'] for r in runs)} | "
         f"**{ms(r['test']['macro_f1'] for r in runs)}** |",
         f"| accuracy | {ms(r['validation']['accuracy'] for r in runs)} | "
         f"**{ms(r['test']['accuracy'] for r in runs)}** |",
         f"| balanced accuracy | "
         f"{ms(r['validation']['balanced_accuracy'] for r in runs)} | "
         f"**{ms(r['test']['balanced_accuracy'] for r in runs)}** |",
         f"| macro precision | {ms(r['validation']['macro_precision'] for r in runs)} | "
         f"{ms(r['test']['macro_precision'] for r in runs)} |",
         f"| macro recall | {ms(r['validation']['macro_recall'] for r in runs)} | "
         f"{ms(r['test']['macro_recall'] for r in runs)} |",
         f"| ROC-AUC{'' if ak == 'roc_auc' else ' (OvR macro)'} | "
         f"{ms(r['validation'][ak] for r in runs)} | "
         f"**{ms(r['test'][ak] for r in runs)}** |", ""]
    L += [f"**Best validation macro-F1 (selection metric):** "
          f"{ms(r['best_val_macro_f1'] for r in runs)}  ",
          f"**Best epoch:** {[r['best_epoch'] for r in runs]} "
          f"(of {[r['epochs_run'] for r in runs]} run)", ""]

    L += ["Per-class test metrics:", "",
          "| class | precision | recall | F1 |", "|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | {ms(r['test']['precision_'+n] for r in runs)} | "
                 f"{ms(r['test']['recall_'+n] for r in runs)} | "
                 f"**{ms(r['test']['f1_'+n] for r in runs)}** |")
    L.append("")

    cis = [r["test"]["macro_f1_ci95"] for r in runs]
    L += ["Test macro-F1 bootstrap 95% CI per seed: " +
          "; ".join(f"[{a:.3f}, {b:.3f}]" for a, b in cis), ""]

    cm = pooled_cm(runs)
    L += [f"Test confusion matrix, pooled over {len(runs)} seeds "
          f"(rows = true, columns = predicted; order {', '.join(names)}):", "",
          "```",
          "                " + "".join(f"{n[:9]:>10}" for n in names)]
    for i, row in enumerate(cm):
        L.append(f"  true {names[i][:10]:<10}" + "".join(f"{v:>10d}" for v in row))
    L += ["```", ""]
    L += [f"Single-seed (seed {runs[0]['seed']}) confusion matrix, for a "
          "per-run view:", "", "```"]
    for i, row in enumerate(runs[0]["test"]["confusion_matrix"]):
        L.append(f"  true {names[i][:10]:<10}" + "".join(f"{v:>10d}" for v in row))
    L += ["```", "",
          f"Overfitting: train − validation macro-F1 was "
          f"{overfit_gap(runs):+.3f} at the selected epoch and "
          f"{final_overfit_gap(runs):+.3f} by the last epoch run.  ",
          f"Validation → test drift in macro-F1: "
          f"{mean(runs, 'test', 'macro_f1') - mean(runs, 'validation', 'macro_f1'):+.4f}.",
          "",
          f"Training settings: lr {a0['lr']}, weight decay {a0['wd']}, dropout "
          f"{a0['dropout']}, augmentation `{a0['aug']}`, label smoothing "
          f"{a0['label_smoothing']}, batch size {a0['batch_size']}, AdamW + cosine "
          f"schedule, inverse-frequency class weights, early stopping patience "
          f"{a0['patience']} on validation macro-F1, max {a0['epochs']} epochs.  ",
          f"Training time: {np.mean([r['train_minutes'] for r in runs]):.1f} min/seed "
          f"| parameters: {runs[0]['params']:,}", ""]
    return L


def main() -> None:
    found = {t: collect(t) for t in TAGS}
    found = {t: r for t, r in found.items() if r}
    if not found:
        raise SystemExit("no completed runs with test metrics")

    s3 = load_splits("3class")
    sb = load_splits("binary")
    base3 = majority_baseline(s3["test"][C.LABEL_COL].to_numpy(), C.CLASS_NAMES)
    baseb = majority_baseline(sb["test"][C.BINARY_LABEL_COL].to_numpy(),
                              C.BINARY_CLASS_NAMES)
    cost_f = C.RESULTS_DIR / "2d" / "computational_cost.json"
    cost = json.loads(cost_f.read_text()) if cost_f.exists() else {}
    moco_f = C.RESULTS_DIR / "2d" / "moco" / "moco_history.json"
    moco = json.loads(moco_f.read_text()) if moco_f.exists() else {}

    L = ["# 2D Results — Bahodir", "",
         "ResNet-18 on the defined central CT slice. Final numbers for the team "
         "comparison.", "",
         "## Setup", "",
         "| | |", "|---|---|",
         "| Representation | **2D**, single central slice |",
         f"| Input | slice {C.CENTRAL_SLICE} of the {C.SAMPLE_SHAPE_ZYX} 1 mm crop "
         f"→ {C.SLICE_HW[0]}×{C.SLICE_HW[1]}, 1 channel |",
         "| Architecture | torchvision **ResNet-18**, stock, `conv1` changed to "
         "1 input channel, dropout 0.5 before the classifier |",
         f"| Parameters | **{cost.get('parameters', 11171779):,}** (11.17 M) |",
         f"| HU window | clip [{C.HU_MIN:.0f}, {C.HU_MAX:.0f}] → [0,1], then "
         "standardised with 3-class train-split mean/std (0.3651 / 0.3145) |",
         "| Optimiser | AdamW, lr 1e-3, weight decay 0.05, cosine schedule |",
         "| Regularisation | dropout 0.5, strong train-only augmentation, label "
         "smoothing 0.10, early stopping (patience 20) |",
         "| Class imbalance | inverse-frequency weights in the loss |",
         "| Selection | best **validation macro-F1**; test evaluated only after "
         "the configuration was frozen |",
         "| Seeds | 0, 1, 2 — all numbers are mean ± std |",
         f"| Device | Apple M2, PyTorch MPS |", ""]
    if cost:
        L += [f"Inference: {cost.get('per_nodule_batch128_ms', '?')} ms/nodule "
              f"(batch 128), {cost.get('latency_batch1_ms', '?')} ms for a single "
              f"nodule; peak GPU memory {cost.get('peak_mps_memory_mb', '?')} MB.", ""]
    if moco:
        h = moco.get("history", {})
        L += ["**MoCo v2 pretraining (for Model B):** "
              f"{moco['args']['epochs']} epochs on {moco.get('ssl_nodules')} nodules "
              f"from {moco.get('ssl_patients')} training patients (labels unused), "
              f"K={moco['args']['K']}, T={moco['args']['T']}, m={moco['args']['m']}, "
              f"SGD lr {moco['args']['lr']}, batch {moco['args']['batch_size']}. "
              f"InfoNCE loss {h.get('loss', [0])[0]:.3f} → {h.get('loss', [0])[-1]:.3f}; "
              f"instance-discrimination accuracy → {h.get('acc', [0])[-1]:.3f}. "
              f"Took {moco.get('minutes', 0):.0f} min.", ""]

    L += ["## Reference baselines (always predict the majority class)", "",
          "| task | macro F1 | accuracy | balanced accuracy |", "|---|---|---|---|",
          f"| 3-class (predict *indeterminate*) | {base3['macro_f1']:.4f} | "
          f"{base3['accuracy']:.4f} | {base3['balanced_accuracy']:.4f} |",
          f"| binary (predict *benign*) | {baseb['macro_f1']:.4f} | "
          f"{baseb['accuracy']:.4f} | {baseb['balanced_accuracy']:.4f} |", "",
          "These matter for reading the numbers: 3-class accuracy near 0.50 is "
          "only marginally above the majority rate, whereas macro-F1 near 0.50 is "
          "a genuine result.", ""]

    for tag, runs in found.items():
        L += section(tag, runs)

    # ---- comparisons --------------------------------------------------------
    a, b = found.get("model_a"), found.get("model_b")
    bino = found.get("binary_supervised")

    if a and b:
        L += ["## Comparison 1 — supervised vs MoCo on the 3-class task", "",
              "| metric | Model A (supervised) | Model B (MoCo) | Δ (B − A) |",
              "|---|---|---|---|"]
        keys = [("macro_f1", "test macro F1"), ("accuracy", "test accuracy"),
                ("balanced_accuracy", "test balanced accuracy"),
                ("roc_auc_ovr_macro", "test ROC-AUC (OvR macro)")] + \
               [(f"f1_{n}", f"test F1 {n}") for n in C.CLASS_NAMES]
        for k, label in keys:
            va, vb = mean(a, "test", k), mean(b, "test", k)
            L.append(f"| {label} | {va:.4f} | {vb:.4f} | "
                     f"{vb - va:+.4f} |")
        va, vb = (float(np.mean([r["best_val_macro_f1"] for r in a])),
                  float(np.mean([r["best_val_macro_f1"] for r in b])))
        L.append(f"| best validation macro F1 | {va:.4f} | {vb:.4f} | {vb - va:+.4f} |")
        L.append("")

    if a and bino:
        L += ["## Comparison 2 — 3-class vs binary (sensitivity analysis)", "",
              "Both supervised, same architecture, same settings, same seeds. The "
              "binary cohort is the same images with *indeterminate* removed "
              "(1000 / 219 / 209 nodules).", "",
              "| metric | 3-class | binary | Δ |", "|---|---|---|---|"]
        for k, label in [("macro_f1", "test macro F1"),
                         ("accuracy", "test accuracy"),
                         ("balanced_accuracy", "test balanced accuracy")]:
            va, vb = mean(a, "test", k), mean(bino, "test", k)
            L.append(f"| {label} | {va:.4f} | {vb:.4f} | {vb - va:+.4f} |")
        va = mean(a, "test", "roc_auc_ovr_macro")
        vb = mean(bino, "test", auc_key(bino))
        L.append(f"| test ROC-AUC | {va:.4f} (OvR macro) | {vb:.4f} (binary) | "
                 f"{vb - va:+.4f} |")
        for n in ("benign", "malignant"):
            va, vb = mean(a, "test", f"f1_{n}"), mean(bino, "test", f"f1_{n}")
            L.append(f"| test F1 {n} | {va:.4f} | {vb:.4f} | {vb - va:+.4f} |")
        L += ["", "Note the two tasks are not directly comparable on macro-F1 "
              "alone — a 2-class macro-F1 has a higher chance level (0.33 vs "
              "0.21 for the majority baseline). The per-class F1 for benign and "
              "malignant, and the ROC-AUC, are the fairer comparison.", ""]

    L += ["## Analysis and conclusion", "",
          "See **`results/2d/ANALYSIS.md`** for the written analysis requested in "
          "`README_TEAM_EXPERIMENTS.txt`: supervised vs MoCo, whether SSL helped, "
          "easiest/hardest class, what the confusion matrix shows, overfitting, "
          "validation-to-test agreement, 3-class vs binary, and limitations.", "",
          "## Files", "",
          "- `results/2d/REPORT.md` — this report",
          "- `results/2d/ANALYSIS.md` — written analysis and conclusion",
          "- `results/2d/summary.json` — machine-readable numbers",
          "- `results/2d/<tag>/seed<N>/result.json` — per-seed metrics and history",
          "- `results/2d/<tag>/seed<N>/curves.png` — train vs validation curves",
          "- `results/2d/<tag>/confusion_test_pooled.png` — pooled confusion matrix",
          "- `results/2d/computational_cost.json` — parameters, latency, memory", ""]

    out = C.RESULTS_DIR / "2d" / "REPORT.md"
    out.write_text("\n".join(L))

    summary = {}
    for tag, runs in found.items():
        ak = auc_key(runs)
        summary[tag] = {
            "title": TAGS[tag][0], "task": TAGS[tag][1],
            "seeds": [r["seed"] for r in runs],
            "architecture": "ResNet-18 (2D, 1-channel central slice)",
            "parameters": runs[0]["params"],
            "best_epoch": [r["best_epoch"] for r in runs],
            "best_val_macro_f1_mean": float(np.mean(
                [r["best_val_macro_f1"] for r in runs])),
            "class_names": runs[0]["class_names"],
            "test_confusion_matrix_pooled": pooled_cm(runs).tolist(),
            "settings": runs[0]["args"],
        }
        for where in ("validation", "test"):
            for k in ("macro_f1", "accuracy", "balanced_accuracy",
                      "macro_precision", "macro_recall", ak):
                vals = [r[where][k] for r in runs]
                summary[tag][f"{where}_{k}_mean"] = float(np.mean(vals))
                summary[tag][f"{where}_{k}_std"] = float(np.std(vals))
            for n in runs[0]["class_names"]:
                for m_ in ("precision", "recall", "f1"):
                    vals = [r[where][f"{m_}_{n}"] for r in runs]
                    summary[tag][f"{where}_{m_}_{n}_mean"] = float(np.mean(vals))
    summary["baselines"] = {"three_class_majority": base3, "binary_majority": baseb}
    summary["computational_cost"] = cost
    (C.RESULTS_DIR / "2d" / "summary.json").write_text(json.dumps(summary, indent=2))

    for tag, runs in found.items():
        confusion_figure(pooled_cm(runs).tolist(),
                         C.RESULTS_DIR / "2d" / tag / "confusion_test_pooled.png",
                         f"2D {TAGS[tag][0]} — test, {len(runs)} seeds pooled",
                         tuple(runs[0]["class_names"]))

    print("\n".join(L))
    print(f"\nwrote {out} and summary.json")


if __name__ == "__main__":
    main()
