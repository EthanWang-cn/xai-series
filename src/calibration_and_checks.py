"""Calibration and split-quality analysis for the trained classifier.

Computes expected calibration error (ECE) and a reliability diagram from the
saved test predictions, and re-scores accuracy on the subset of test images
that have no byte-identical copy in the training split (a duplicate check
result stored in dataset_checks.json).
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from model import CLASSES, RESULTS

FIGDIR = RESULTS.parent / "figures"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "savefig.dpi": 200})


def main():
    preds = json.loads((RESULTS / "test_predictions.json").read_text(encoding="utf-8"))
    y = np.array([CLASSES.index(p["label"]) for p in preds])
    conf = np.array([max(p["probs"].values()) for p in preds])
    pred = np.array([CLASSES.index(p["pred"]) for p in preds])
    correct = (y == pred).astype(float)

    bins = np.linspace(0, 1, 16)
    idx = np.digitize(conf, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    ece, rows = 0.0, []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        ece += m.sum() / len(conf) * abs(acc_b - conf_b)
        rows.append({"bin": f"{bins[b]:.2f}-{bins[b+1]:.2f}", "n": int(m.sum()),
                     "accuracy": round(float(acc_b), 3), "confidence": round(float(conf_b), 3)})

    # duplicate-aware subset
    checks = json.loads((RESULTS / "dataset_checks.json").read_text(encoding="utf-8"))
    twins = {f.split("/", 1)[1] for g in checks["duplicates_across_split"] for f in g
             if f.startswith("testing/")}
    clean = [i for i, p in enumerate(preds) if f"{p['label']}/{p['file']}" not in twins]
    clean_acc = float((y[clean] == pred[clean]).mean())
    clean_f1 = float(f1_score(y[clean], pred[clean], average="macro"))
    leak_idx = [i for i in range(len(preds)) if i not in clean]
    leak_acc = float((y[leak_idx] == pred[leak_idx]).mean()) if leak_idx else float("nan")

    out = {
        "ece_15bin": round(float(ece), 4),
        "reliability_rows": rows,
        "mean_confidence": round(float(conf.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "n_test": len(preds),
        "n_test_with_train_twin": len(leak_idx),
        "accuracy_clean_subset": round(clean_acc, 4),
        "macro_f1_clean_subset": round(clean_f1, 4),
        "n_clean_subset": len(clean),
        "accuracy_duplicated_subset": round(leak_acc, 4),
        "mean_confidence_correct": round(float(conf[correct == 1].mean()), 4),
        "mean_confidence_wrong": round(float(conf[correct == 0].mean()), 4)
        if (correct == 0).any() else None,
    }
    (RESULTS / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # reliability diagram
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    centres = [r["confidence"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    sizes = [r["n"] for r in rows]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.scatter(centres, accs, s=[max(6, n * 1.2) for n in sizes], color="0.35",
               label="empirical bins")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_reliability.png")
    print(json.dumps({k: v for k, v in out.items() if k != "reliability_rows"}, indent=1))


if __name__ == "__main__":
    main()
