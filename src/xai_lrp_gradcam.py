"""LRP and Grad-CAM explanations for the brain MRI classifier (Task 3).

The LRP routines adapt apply_lrp_on_vgg16() from the supplied 05_lrp.py
starter script: the same layer walk, the same rho/incr transformation of
weights, and the same z-rule backpropagation. The differences are that the
layer list is built from this project's own network, max pooling is treated as
average pooling, and the relevance at the output is initialised with the
softmax probability of the explained class instead of a one-hot of the maximum
logit.

Run (after train_cnn.py):
    python src/xai_lrp_gradcam.py

Outputs
    ../results/xai_metrics.json          agreement and deletion metrics
    ../results/case_sheets.json          per-case predictions and scores
    ../figures/fig_confusion.png
    ../figures/fig_training.png
    ../figures/fig_case_*.png            3-panel case sheets
    ../figures/fig_deletion.png
    ../figures/fig_agreement.png
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr

from model import (CLASSES, IMG_SIZE, RESULTS, SEED, TEST_ROOT, BrainMRIDataset,
                   BrainTumorCNN, denormalise, eval_transform, fold_batch_norm, set_seed)

PROJECT = RESULTS.parent
FIGDIR = PROJECT / "figures"
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- #
# LRP
# --------------------------------------------------------------------------- #
class LRPVGG:
    """Layer-wise relevance propagation for the compact VGG-style classifier."""

    def __init__(self, model, gamma=0.25, eps=0.25):
        self.model = model.eval()
        self.gamma = gamma
        self.eps = eps
        # flat layer list: features -> global average pool -> classifier
        self.ops = [m for m in model.features] + [model.pool] + [m for m in model.classifier]
        self.conv_positions = [i for i, m in enumerate(self.ops) if isinstance(m, nn.Conv2d)]
        self.linear_positions = [i for i, m in enumerate(self.ops) if isinstance(m, nn.Linear)]

    # -- rule selection ----------------------------------------------------- #
    def _rule(self, idx):
        """Return (rho, incr) for the layer at flat index idx.

        Shallow convolutions use the gamma rule (positive weights favoured),
        the middle convolutions use LRP-epsilon with a variance-based
        stabiliser, and the last convolution plus both fully connected layers
        use plain LRP-0, following the depth heuristics of the starter script.
        """
        n_conv = len(self.conv_positions)
        rank = self.conv_positions.index(idx) if idx in self.conv_positions else None
        if rank is not None and rank < 3:
            return (lambda p: p + self.gamma * p.clamp(min=0)), (lambda z: z + 1e-9)
        if rank is not None and rank < n_conv - 1:
            return (lambda p: p), (lambda z: z + 1e-9 + self.eps * ((z ** 2).mean() ** 0.5).data)
        return (lambda p: p), (lambda z: z + 1e-9)

    # -- z-rule ------------------------------------------------------------- #
    def _z_rule(self, idx, a, R_next):
        module = self.ops[idx]
        if isinstance(module, nn.MaxPool2d):
            module = nn.AvgPool2d(module.kernel_size, module.stride)
        rho, incr = self._rule(idx)
        layer = copy.deepcopy(module)
        if getattr(layer, "weight", None) is not None:
            layer.weight = nn.Parameter(rho(layer.weight.detach()))
        if getattr(layer, "bias", None) is not None:
            layer.bias = nn.Parameter(rho(layer.bias.detach()))
        a = a.detach().clone().requires_grad_(True)
        z = incr(layer(a))
        s = (R_next / z).detach()   # detach: keeps z * s non-constant in a
        (z * s).sum().backward()
        return (a * a.grad).detach()

    # -- forward / backward ------------------------------------------------- #
    def _forward(self, x):
        acts, h = [x], x
        with torch.no_grad():
            for m in self.ops:
                h = m(h)
                acts.append(h)
        return acts

    def explain(self, x, target):
        """Return the relevance map as a (H, W) array and the raw channel sum."""
        acts = self._forward(x)
        R = [None] * (len(self.ops) + 1)
        R[-1] = torch.zeros_like(acts[-1])
        R[-1][0, target] = torch.softmax(acts[-1], dim=1)[0, target]  # explained score
        for i in range(len(self.ops) - 1, -1, -1):
            m, a = self.ops[i], acts[i]
            if isinstance(m, (nn.ReLU, nn.Dropout)):
                R[i] = R[i + 1]
            elif isinstance(m, nn.Flatten):
                R[i] = R[i + 1].reshape(a.shape)
            elif isinstance(m, (nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Conv2d, nn.Linear)):
                R[i] = self._z_rule(i, a, R[i + 1])
            else:
                raise TypeError(type(m))
        relevance = R[0][0]
        return relevance.sum(dim=0).cpu().numpy(), relevance


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM:
    def __init__(self, model, layer):
        self.model = model.eval()
        self.layer = layer
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        # keep the activation and hook its gradient (avoids full backward hooks,
        # which clash with in-place activations elsewhere in the network)
        if not out.requires_grad:
            return  # LRP forward passes run under torch.no_grad()
        self.activations = out.detach()
        out.register_hook(lambda grad: setattr(self, "gradients", grad.detach()))

    def explain(self, x, target):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, target].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        return cam / (cam.max() + 1e-12)


# --------------------------------------------------------------------------- #
# visualisation
# --------------------------------------------------------------------------- #
def overlay(img_rgb, heat, cmap="seismic", signed_range=None, alpha=0.55):
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(img_rgb, cmap="gray")
    if signed_range is not None:
        norm = TwoSlopeNorm(vmin=-signed_range, vcenter=0.0, vmax=signed_range)
        im = ax.imshow(heat, cmap=cmap, norm=norm, alpha=alpha)
    else:
        im = ax.imshow(heat, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    return fig, ax, im


def case_figure(img_rgb, lrp_map, cam_map, title, out_path, subtitle=None):
    """Three aligned panels: input, signed LRP, Grad-CAM.

    The LRP colour limits are set from the 99th percentile of the positive
    relevance so that a few extreme pixels do not wash out the map; the raw
    values are unchanged and stored alongside the figure.
    """
    vmax = float(np.percentile(lrp_map[lrp_map > 0], 99)) if (lrp_map > 0).any() else 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.05))
    axes[0].imshow(img_rgb, cmap="gray", interpolation="bilinear")
    axes[0].set_title("(A) Input MRI", fontsize=9)
    axes[0].axis("off")

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    axes[1].imshow(img_rgb, cmap="gray", interpolation="bilinear")
    im1 = axes[1].imshow(lrp_map, cmap="seismic", norm=norm, alpha=0.6, interpolation="bilinear")
    axes[1].set_title("(B) LRP relevance", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(img_rgb, cmap="gray", interpolation="bilinear")
    im2 = axes[2].imshow(cam_map, cmap="inferno", alpha=0.6, vmin=0, vmax=1, interpolation="bilinear")
    axes[2].set_title("(C) Grad-CAM", fontsize=9)
    axes[2].axis("off")

    cb1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", fraction=0.05, pad=0.04, aspect=30)
    cb1.ax.tick_params(labelsize=6)
    cb1.set_label("signed relevance", fontsize=6)
    cb2 = fig.colorbar(im2, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.04, aspect=30)
    cb2.ax.tick_params(labelsize=6)
    cb2.set_label("normalised activation", fontsize=6)

    fig.suptitle(title, fontsize=9, y=1.02)
    if subtitle:
        fig.text(0.5, -0.06, subtitle, ha="center", fontsize=7.5)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# faithfulness metric (deletion test)
# --------------------------------------------------------------------------- #
def deletion_curve(model, x, target, attribution, fractions):
    """Mask the highest-attribution pixels with the baseline and track p(target)."""
    flat = attribution.reshape(-1)
    order = np.argsort(-flat)
    n = flat.size
    probs = []
    for frac in fractions:
        k = int(round(frac * n))
        xm = x.clone()
        if k > 0:
            idx = torch.from_numpy(order[:k])
            mask = torch.zeros(n, dtype=torch.bool)
            mask[idx] = True
            xm.view(1, x.shape[1], -1)[:, :, mask] = 0.0
        with torch.no_grad():
            probs.append(float(torch.softmax(model(xm), dim=1)[0, target]))
    return np.array(probs)


def main():
    set_seed(SEED)
    torch.set_num_threads(8)
    ckpt = torch.load(RESULTS / "brain_mri_cnn.pt", map_location="cpu")
    model = BrainTumorCNN(len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("checkpoint epoch", ckpt["epoch"], "val macro F1", round(ckpt["val_macro_f1"], 4))

    ds = BrainMRIDataset(TEST_ROOT, transform=eval_transform(), classes=CLASSES)
    preds = json.loads((RESULTS / "test_predictions.json").read_text(encoding="utf-8"))

    # LRP runs on a copy with batch normalisation folded into the convolutions
    lrp = LRPVGG(fold_batch_norm(model))
    cam = GradCAM(model, model.cam_layer())

    # ---- choose cases ----------------------------------------------------- #
    by_class = {c: [] for c in CLASSES}
    for i, p in enumerate(preds):
        by_class[p["label"]].append(i)
    correct = [i for i, p in enumerate(preds) if p["correct"]]
    errors = [i for i, p in enumerate(preds) if not p["correct"]]
    # pick high-confidence correct cases and the most confident error
    def best_correct(c):
        idx = [i for i in by_class[c] if preds[i]["correct"]]
        return max(idx, key=lambda i: preds[i]["probs"][c]) if idx else None
    cases = []
    for c in ["glioma_tumor", "pituitary_tumor", "meningioma_tumor", "no_tumor"]:
        i = best_correct(c)
        if i is not None:
            cases.append(("correct", i))
    if errors:
        err = max(errors, key=lambda i: max(preds[i]["probs"].values()))
        cases.append(("error", err))
    # folding check: the folded copy must reproduce the original logits
    x_probe, _ = ds[0]
    with torch.no_grad():
        diff = float((model(x_probe.unsqueeze(0)) - lrp.model(x_probe.unsqueeze(0))).abs().max())
    print("max |logit difference| after folding batch norm:", diff)

    print("cases:", [(k, preds[i]["file"], preds[i]["label"], preds[i]["pred"]) for k, i in cases])

    # ---- per-case figures -------------------------------------------------- #
    case_records = []
    letters = "ABCDEFGH"
    for n, (kind, i) in enumerate(cases):
        x, y = ds[i]
        x = x.unsqueeze(0)
        p = preds[i]
        target = CLASSES.index(p["pred"])
        rel, _ = lrp.explain(x, target)
        cam_map = cam.explain(x, target)
        img = denormalise(x[0])
        pretty = {"glioma_tumor": "glioma", "meningioma_tumor": "meningioma",
                  "no_tumor": "no tumour", "pituitary_tumor": "pituitary tumour"}
        title = (f"Case {letters[n]}: {p['file']}  |  label: {pretty[p['label']]}  |  "
                 f"prediction: {pretty[p['pred']]} ({p['probs'][p['pred']]:.3f})")
        out = FIGDIR / f"fig_case_{letters[n].lower()}.png"
        case_figure(img, rel, cam_map, title, out)
        case_records.append({
            "case": letters[n], "kind": kind, "file": p["file"],
            "label": p["label"], "prediction": p["pred"],
            "score": round(p["probs"][p["pred"]], 4),
            "target_index": target, "figure": out.name,
            "lrp_positive_mass_in_top10pct": float(
                np.sort(rel.reshape(-1))[::-1][: int(0.1 * rel.size)].clip(0).sum()
                / max(rel.clip(0).sum(), 1e-9)),
        })
        np.savez(RESULTS / f"maps_case_{letters[n].lower()}.npz", lrp=rel, cam=cam_map, img=img)

    (RESULTS / "case_sheets.json").write_text(json.dumps(case_records, indent=1), encoding="utf-8")

    # ---- agreement and deletion test over many images ---------------------- #
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(correct, size=min(60, len(correct)), replace=False).tolist())
    if os.environ.get("XAI_FIGURES_ONLY"):
        print("figures only: skipping deletion test")
        return
    fractions = np.linspace(0, 1, 21)
    rhos, ious, curves_lrp, curves_cam, conserve = [], [], [], [], []
    for i in sample:
        x, _ = ds[i]
        x = x.unsqueeze(0)
        target = CLASSES.index(preds[i]["pred"])
        rel, _ = lrp.explain(x, target)
        cam_map = cam.explain(x, target)
        conserve.append(float(rel.sum()) / max(preds[i]["probs"][preds[i]["pred"]], 1e-9))
        rhos.append(spearmanr(np.abs(rel).reshape(-1), cam_map.reshape(-1)).statistic)
        k = int(0.1 * rel.size)
        top_l = set(np.argsort(-np.abs(rel).reshape(-1))[:k].tolist())
        top_c = set(np.argsort(-cam_map.reshape(-1))[:k].tolist())
        ious.append(len(top_l & top_c) / len(top_l | top_c))
        curves_lrp.append(deletion_curve(model, x, target, np.clip(rel, 0, None), fractions))
        curves_cam.append(deletion_curve(model, x, target, cam_map, fractions))
        if len(curves_lrp) % 10 == 0:
            print("processed", len(curves_lrp), flush=True)

    curves_lrp, curves_cam = np.array(curves_lrp), np.array(curves_cam)
    auc = lambda c: float(np.trapezoid(c.mean(0), fractions))
    metrics = {
        "fold_logit_maxdiff": diff,
        "n_images": int(len(sample)),
        "fractions": fractions.round(3).tolist(),
        "spearman_mean": float(np.mean(rhos)),
        "spearman_median": float(np.median(rhos)),
        "spearman_iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
        "top10_iou_mean": float(np.mean(ious)),
        "top10_iou_median": float(np.median(ious)),
        "lrp_conservation_ratio": float(np.mean(conserve)),
        "deletion_auc_lrp": auc(curves_lrp),
        "deletion_auc_gradcam": auc(curves_cam),
        "deletion_curve_lrp": curves_lrp.mean(0).round(4).tolist(),
        "deletion_curve_gradcam": curves_cam.mean(0).round(4).tolist(),
    }
    (RESULTS / "xai_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=1))

    # ---- deletion figure --------------------------------------------------- #
    plt.figure(figsize=(4.4, 3.2))
    plt.plot(fractions * 100, curves_lrp.mean(0), "-o", ms=3, lw=1.4, label=f"LRP (AUC {auc(curves_lrp):.3f})")
    plt.plot(fractions * 100, curves_cam.mean(0), "-s", ms=3, lw=1.4, label=f"Grad-CAM (AUC {auc(curves_cam):.3f})")
    plt.xlabel("Highest-attribution pixels masked (%)")
    plt.ylabel(f"Mean p(target class), n = {len(sample)}")
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.25, lw=0.5)
    plt.savefig(FIGDIR / "fig_deletion.png")
    plt.close()

    # ---- agreement figure -------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    axes[0].hist(rhos, bins=14, color="0.55", edgecolor="white")
    axes[0].axvline(np.mean(rhos), color="crimson", lw=1.2, ls="--")
    axes[0].set_xlabel("Spearman rho between |LRP| and Grad-CAM")
    axes[0].set_ylabel("Images")
    axes[0].set_title(f"Agreement (mean {np.mean(rhos):.2f})", fontsize=9)
    axes[1].hist(ious, bins=14, color="0.55", edgecolor="white")
    axes[1].axvline(np.mean(ious), color="crimson", lw=1.2, ls="--")
    axes[1].set_xlabel("Jaccard overlap of top-10% pixels")
    axes[1].set_ylabel("Images")
    axes[1].set_title(f"Top-decile overlap (mean {np.mean(ious):.2f})", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_agreement.png")
    plt.close()

    # ---- confusion matrix + training curves -------------------------------- #
    cm = np.load(RESULTS / "confusion_matrix.npy")
    fig, ax = plt.subplots(figsize=(4.1, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), [c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_yticks(range(len(CLASSES)), [c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Dataset label")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Images")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_confusion.png")
    plt.close()

    hist = json.loads((RESULTS / "train_history.json").read_text(encoding="utf-8"))
    ep = [h["epoch"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(4.6, 3.0))
    ax1.plot(ep, [h["train_loss"] for h in hist], label="train loss", lw=1.3)
    ax1.plot(ep, [h["val_loss"] for h in hist], label="validation loss", lw=1.3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.grid(alpha=0.25, lw=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_macro_f1"] for h in hist], ":", color="crimson", lw=1.3, label="val macro F1")
    ax2.set_ylabel("Validation macro F1", color="crimson")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_training.png")
    plt.close()
    print("done")


if __name__ == "__main__":
    main()