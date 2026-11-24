"""Train the brain MRI classifier used for both XAI methods (Task 3).

Run:
    python src/train_cnn.py --epochs 14

Outputs (in ../results):
    brain_mri_cnn.pt        best checkpoint (state dict)
    train_history.json      per-epoch loss / accuracy / macro F1
    test_metrics.json       accuracy, macro F1, per-class P/R/F1
    test_predictions.json   per-image label, prediction, class probabilities
    confusion_matrix.npy    raw counts, class order as CLASSES
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from model import (CLASSES, RESULTS, SEED, TEST_ROOT, TRAIN_ROOT, BrainMRIDataset,
                   BrainTumorCNN, eval_transform, set_seed, stratified_split,
                   train_transform)


def make_loaders(batch_size=32):
    train_full = BrainMRIDataset(TRAIN_ROOT, transform=train_transform(), classes=CLASSES)
    train_eval = BrainMRIDataset(TRAIN_ROOT, transform=eval_transform(), classes=CLASSES)
    test_ds = BrainMRIDataset(TEST_ROOT, transform=eval_transform(), classes=CLASSES)

    tr_idx, va_idx = stratified_split(train_full.targets(), val_frac=0.15, seed=SEED)
    train_sub = torch.utils.data.Subset(train_full, tr_idx)
    val_sub = torch.utils.data.Subset(train_eval, va_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, train_eval, test_ds, tr_idx, va_idx


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    ys, ps, probs = [], [], []
    total, loss_sum = 0, 0.0
    for x, y in loader:
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * x.size(0)
        total += x.size(0)
        p = torch.softmax(logits, dim=1)
        probs.append(p.numpy())
        ps.extend(p.argmax(1).tolist())
        ys.extend(y.tolist())
    ys, ps = np.array(ys), np.array(ps)
    out = {
        "accuracy": float((ys == ps).mean()),
        "macro_f1": float(f1_score(ys, ps, average="macro")),
        "n": int(total),
    }
    if criterion is not None:
        out["loss"] = loss_sum / total
    return out, ys, ps, np.concatenate(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    set_seed(SEED)
    torch.set_num_threads(8)
    RESULTS.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_eval, test_ds, tr_idx, va_idx = make_loaders(args.batch_size)
    print(f"train {len(tr_idx)} | val {len(va_idx)} | test {len(test_ds)}")

    model = BrainTumorCNN(len(CLASSES))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par:,}")

    # inverse-frequency class weights for the imbalanced no_tumor class
    y_tr = np.array([train_eval.targets()[i] for i in tr_idx])
    counts = np.bincount(y_tr, minlength=len(CLASSES)).astype(float)
    # square-root of inverse frequency: full inverse frequency over-weights the
    # no_tumor class and the model then defaults to it on shifted images
    weights = np.sqrt(counts.sum() / (len(CLASSES) * counts))
    print("class weights:", dict(zip(CLASSES, np.round(weights, 3).tolist())))

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.25, div_factor=10.0, final_div_factor=100.0)

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x, y in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            scheduler.step()          # OneCycleLR is stepped per batch
            running += loss.item() * x.size(0)
            seen += x.size(0)
        tr_loss = running / seen
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)
        rec = {"epoch": epoch, "train_loss": tr_loss, "lr": optimiser.param_groups[0]["lr"],
               "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
               "val_macro_f1": val_metrics["macro_f1"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "classes": CLASSES, "epoch": epoch,
                        "val_macro_f1": best_f1}, RESULTS / "brain_mri_cnn.pt")

    model.load_state_dict(best_state)
    test_metrics, ys, ps, probs = evaluate(model, test_loader, criterion)
    test_metrics["macro_precision"] = float(
        classification_report(ys, ps, output_dict=True, zero_division=0)["macro avg"]["precision"])
    test_metrics["macro_recall"] = float(
        classification_report(ys, ps, output_dict=True, zero_division=0)["macro avg"]["recall"])
    test_metrics["per_class"] = {
        c: {k: float(v) for k, v in classification_report(ys, ps, target_names=CLASSES,
                                                          output_dict=True, zero_division=0)[c].items()
            if k in ("precision", "recall", "f1-score", "support")}
        for c in CLASSES
    }
    test_metrics["parameters"] = int(n_par)
    test_metrics["best_val_macro_f1"] = float(best_f1)

    cm = confusion_matrix(ys, ps, labels=list(range(len(CLASSES))))
    np.save(RESULTS / "confusion_matrix.npy", cm)

    preds = [{"file": test_ds.samples[i][0].split("\\")[-1],
              "path": test_ds.samples[i][0],
              "label": CLASSES[ys[i]], "pred": CLASSES[ps[i]],
              "correct": bool(ys[i] == ps[i]),
              "probs": {c: float(probs[i][j]) for j, c in enumerate(CLASSES)}}
             for i in range(len(ys))]
    (RESULTS / "test_predictions.json").write_text(json.dumps(preds, indent=1), encoding="utf-8")
    (RESULTS / "train_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    (RESULTS / "test_metrics.json").write_text(json.dumps(test_metrics, indent=1), encoding="utf-8")
    (RESULTS / "split_indices.json").write_text(
        json.dumps({"train": tr_idx, "val": va_idx}, indent=0), encoding="utf-8")

    print(json.dumps(test_metrics, indent=1))
    print("confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
