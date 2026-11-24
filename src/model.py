"""Shared model, data loading and preprocessing for Assignment 2 (brain MRI XAI).

Adapted from the 05_lrp.py starter script supplied with xai-series. The
architecture is a compact VGG-style CNN rather than torchvision VGG16 because
training is CPU only; see Section 4.1 of the report.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

IMG_SIZE = 128
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "brain_mri"
TRAIN_ROOT = DATA_ROOT / "training"
TEST_ROOT = DATA_ROOT / "testing"
RESULTS = PROJECT.parent / "results"
SEED = 42

NORM_MEAN = (0.0, 0.0, 0.0)
NORM_STD = (1.0, 1.0, 1.0)


class PercentileNorm:
    """Per-image intensity standardisation.

    The downloaded collection mixes scanners and export settings, and the test
    split is visibly brighter than the training split (Section 2.2). Clipping
    at the 1st and 99th percentile and rescaling each image to zero mean and
    unit variance removes the global brightness level, so the network cannot
    use it as a class cue.
    """

    def __init__(self, lo=1, hi=99):
        self.lo, self.hi = lo, hi

    def __call__(self, img):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(a, self.lo), np.percentile(a, self.hi)
        a = np.clip(a, lo, max(hi, lo + 1.0))
        a = (a - a.mean()) / (a.std() + 1e-6)
        a = np.clip(a, -3.0, 3.0)
        return Image.fromarray(a.astype(np.float32), mode="F")


def _to_three(x):
    return x.repeat(3, 1, 1)


def _gamma_tensor(t, g=0.75):
    lo, hi = float(t.min()), float(t.max())
    return ((t - lo) / (hi - lo + 1e-6)) ** g * (hi - lo) + lo


def eval_transform(size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        PercentileNorm(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(_to_three),
    ])


def train_transform(size: int = IMG_SIZE) -> transforms.Compose:
    # No horizontal flip: laterality carries anatomical meaning in MRI.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        PercentileNorm(),
        # The two splits differ in field of view as well as in brightness
        # (Section 2.2), so the crop simulates strongly different zoom levels.
        transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.8, 1.25),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Lambda(_to_three),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.RandomApply([transforms.Lambda(_gamma_tensor)], p=0.4),
    ])


class BrainMRIDataset(Dataset):
    """ImageFolder-like loader that keeps the original file paths."""

    def __init__(self, root, transform=None, classes=None):
        self.root = Path(root)
        self.classes = list(classes) if classes is not None else sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            for p in sorted((self.root / c).glob("*.jpg")):
                self.samples.append((str(p), self.class_to_idx[c]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("L").convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def targets(self):
        return [l for _, l in self.samples]


def stratified_split(targets, val_frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


class BrainTumorCNN(nn.Module):
    """Compact VGG-style CNN (2 conv layers in the first three blocks)."""

    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 64 x 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 32 x 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16 x 16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  # CAM layer
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def cam_layer(self):
        return [m for m in self.features if isinstance(m, nn.Conv2d)][-1]


def fold_batch_norm(model):
    """Return a copy of the model with every BatchNorm2d folded into the
    preceding convolution, so LRP propagates through plain conv layers.

    In evaluation mode a BatchNorm2d is an affine per-channel transform, so
    folding is exact: the copy produces the same logits as the original.
    """
    m = copy.deepcopy(model).eval()
    new_layers, prev_conv = [], None
    for mod in m.features:
        if isinstance(mod, nn.BatchNorm2d):
            with torch.no_grad():
                scale = mod.weight / torch.sqrt(mod.running_var + mod.eps)
                w = prev_conv.weight.data * scale.view(-1, 1, 1, 1)
                b = (prev_conv.bias.data - mod.running_mean) * scale + mod.bias
                prev_conv.weight = nn.Parameter(w)
                prev_conv.bias = nn.Parameter(b)
            continue
        new_layers.append(mod)
        if isinstance(mod, nn.Conv2d):
            prev_conv = mod
    m.features = nn.Sequential(*new_layers)
    return m


def _gamma(a, g):
    """Gamma transform on a float array with negative values shifted to 0..1."""
    lo, hi = a.min(), a.max()
    b = (a - lo) / (hi - lo + 1e-6)
    return (b ** g) * (hi - lo) + lo


def denormalise(tensor):
    """(3,H,W) tensor -> (H,W,3) float array in [0,1] for display."""
    x = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return x


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)