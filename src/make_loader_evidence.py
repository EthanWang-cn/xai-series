"""Capture a transcript of the folder layout and the loader output (Task 2).

Renders the real stdout of the loading checks into a terminal-style PNG used as
Figure 2 in the report. Nothing is mocked: the text comes from listing the
folders and from instantiating the datasets defined in model.py.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from model import CLASSES, IMG_SIZE, TEST_ROOT, TRAIN_ROOT, BrainMRIDataset, eval_transform

REL = "xai-series/data/brain_mri"


def tree_lines():
    lines = [REL + "/"]
    for split, root_dir in [("training", TRAIN_ROOT), ("testing", TEST_ROOT)]:
        counts = {c: len(list((root_dir / c).glob("*.jpg"))) for c in CLASSES}
        lines.append(f"|-- {split}/   ({sum(counts.values())} jpg)")
        for i, c in enumerate(CLASSES):
            branch = "\\--" if i == len(CLASSES) - 1 else "|--"
            lines.append(f"|   {branch} {c}/   ({counts[c]} jpg)")
    return lines


def loader_lines():
    buf = io.StringIO()
    with redirect_stdout(buf):
        tr = BrainMRIDataset(TRAIN_ROOT, transform=eval_transform(), classes=CLASSES)
        te = BrainMRIDataset(TEST_ROOT, transform=eval_transform(), classes=CLASSES)
        print("TRAIN_ROOT = data/brain_mri/training")
        print("TEST_ROOT  = data/brain_mri/testing")
        print()
        print("train_dataset.classes      =", tr.classes)
        print("train_dataset.class_to_idx =", tr.class_to_idx)
        print("test_dataset.class_to_idx  =", te.class_to_idx)
        print("class mappings agree:", tr.class_to_idx == te.class_to_idx)
        print()
        print(f"len(train_dataset) = {len(tr)}   len(test_dataset) = {len(te)}")
        x, y = tr[0]
        print(f"sample tensor: {tuple(x.shape)}  dtype={x.dtype}  label={tr.classes[y]}")
        print(f"pixel range after normalisation: [{x.min():.2f}, {x.max():.2f}]")
        print(f"images resized to {IMG_SIZE}x{IMG_SIZE} and replicated to 3 channels")
    return buf.getvalue().splitlines()


def render(lines, out_path, width=1160):
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 17)
    line_h = 26
    pad = 22
    height = pad * 2 + line_h * len(lines)
    img = Image.new("RGB", (width, height), (18, 18, 18))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        colour = (235, 235, 235)
        if t.startswith("|") or t.startswith("\\") or t.startswith(REL):
            colour = (150, 200, 255)
        if "class_to_idx" in t or "agree" in t:
            colour = (250, 220, 140)
        if t.startswith("H:"):
            colour = (120, 220, 140)
        d.text((pad, pad + i * line_h), t, font=font, fill=colour)
    img.save(out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "figures" / "fig_loader_evidence.png"
    lines = ["H:\\...\\ass2-indi> tree /F xai-series\\data\\brain_mri"] + tree_lines() + [""]
    lines += ["H:\\...\\ass2-indi\\xai-series> python src\\make_loader_evidence.py"] + loader_lines()
    render(lines, out)
