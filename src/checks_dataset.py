"""Inventory and integrity checks for the downloaded dataset (Tasks 1 and 2).

Counts usable images, verifies that every file opens, records image modes and
sizes, and looks for exact duplicate files (MD5) within and across the
training and testing splits.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from model import CLASSES, RESULTS, TEST_ROOT, TRAIN_ROOT

SPLITS = {"training": TRAIN_ROOT, "testing": TEST_ROOT}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    inventory, hashes, unreadable, modes, sizes = {}, {}, [], Counter(), Counter()
    per_class = defaultdict(dict)
    for split, root in SPLITS.items():
        for c in CLASSES:
            files = sorted((root / c).glob("*.jpg"))
            n_ok = 0
            for f in files:
                try:
                    with Image.open(f) as im:
                        im.verify()
                    with Image.open(f) as im:
                        modes[im.mode] += 1
                        sizes[im.size] += 1
                    n_ok += 1
                    hashes.setdefault(md5(f), []).append(f"{split}/{c}/{f.name}")
                except Exception as exc:  # noqa: BLE001
                    unreadable.append({"file": str(f), "error": str(exc)})
            per_class[split][c] = n_ok

    dup_within, dup_across = {}, []
    for h, files in hashes.items():
        if len(files) > 1:
            splits = {f.split("/")[0] for f in files}
            if len(splits) > 1:
                dup_across.append(files)
            else:
                dup_within[h] = files

    out = {
        "per_class": per_class,
        "totals": {s: sum(per_class[s].values()) for s in per_class},
        "unreadable": unreadable,
        "image_modes": dict(modes),
        "top_image_sizes": {str(k): v for k, v in sizes.most_common(5)},
        "distinct_files": len(hashes),
        "duplicates_within_split": dup_within,
        "duplicates_across_split": dup_across,
    }
    (RESULTS / "dataset_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("duplicates_within_split",)}, indent=1)[:2000])


if __name__ == "__main__":
    main()
