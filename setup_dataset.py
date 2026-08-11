"""
setup_dataset.py
────────────────
Run this ONCE after downloading the Kaggle dataset.

Usage:
    python setup_dataset.py --src /path/to/your/downloaded/garbage_classification_v2

It will copy (or symlink) images into the project's dataset/ folder,
mapping Kaggle folder names → our class names.
"""

import os
import sys
import shutil
import argparse

# Kaggle folder name  →  our class folder name
KAGGLE_TO_CLASS = {
    "battery":         "Battery",
    "cardboard":       "Cardboard",
    "e-waste":         "E-Waste",
    "ewaste":          "E-Waste",
    "glass":           "Glass",
    "metal":           "Metal",
    "organic":         "Organic Waste",
    "organic waste":   "Organic Waste",
    "organic_waste":   "Organic Waste",
    "paper":           "Paper",
    "plastic":         "Plastic",
    "textile":         "Textile-Clothes",
    "textile-clothes": "Textile-Clothes",
    "clothes":         "Textile-Clothes",
    "trash":           "Trash",
    "garbage":         "Trash",
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def copy_dataset(src_root: str):
    if not os.path.isdir(src_root):
        print(f"[ERROR] Source folder not found: {src_root}")
        sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)
    total = 0

    for folder in os.listdir(src_root):
        folder_path = os.path.join(src_root, folder)
        if not os.path.isdir(folder_path):
            continue

        key = folder.lower().strip()
        target_cls = KAGGLE_TO_CLASS.get(key)

        # Try partial match if exact not found
        if not target_cls:
            for k, v in KAGGLE_TO_CLASS.items():
                if k in key or key in k:
                    target_cls = v
                    break

        if not target_cls:
            print(f"[SKIP] Unknown folder: '{folder}' — add mapping in KAGGLE_TO_CLASS")
            continue

        dest_dir = os.path.join(DATASET_DIR, target_cls)
        os.makedirs(dest_dir, exist_ok=True)

        imgs = [f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

        for img in imgs:
            src  = os.path.join(folder_path, img)
            dest = os.path.join(dest_dir, img)
            if not os.path.exists(dest):
                shutil.copy2(src, dest)

        print(f"[OK] {folder:25s} → {target_cls:20s}  ({len(imgs)} images)")
        total += len(imgs)

    print(f"\n✅  Dataset setup complete — {total} images in {DATASET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", required=True,
        help="Path to the root of the downloaded Kaggle dataset folder"
    )
    args = parser.parse_args()
    copy_dataset(args.src)
