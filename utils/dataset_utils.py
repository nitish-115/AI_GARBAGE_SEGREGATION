# ============================================================
#  utils/dataset_utils.py
#  Helper utilities for dataset inspection and fake-data gen
# ============================================================

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CLASS_NAMES, DATASET_DIR, IMG_SIZE


# ── Scan dataset folder ──────────────────────────────────────
def get_class_counts(dataset_dir: str = DATASET_DIR) -> dict:
    """Return {class_name: image_count} for every subfolder found."""
    counts = {}
    if not os.path.isdir(dataset_dir):
        return counts
    for cls in sorted(os.listdir(dataset_dir)):
        cls_path = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_path):
            imgs = [f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
            counts[cls] = len(imgs)
    return counts


def get_total_images(dataset_dir: str = DATASET_DIR) -> int:
    return sum(get_class_counts(dataset_dir).values())


# ── Create synthetic dataset (for demo / CI) ────────────────
def create_synthetic_dataset(
    base_dir: str = DATASET_DIR,
    n_per_class: int = 60,
    img_size: tuple = IMG_SIZE,
) -> None:
    """
    Generate synthetic coloured-noise images so the full pipeline can run
    without a real dataset.  Each class gets a distinct hue so the network
    can actually learn something meaningful.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Hue offsets (0-180 in OpenCV HSV, but here we use PIL RGB directly)
    hue_map = {
        "Battery":         (220, 50,  50),
        "Cardboard":       (180, 140, 80),
        "E-Waste":         (60,  60,  200),
        "Glass":           (100, 200, 220),
        "Metal":           (160, 160, 160),
        "Organic Waste":   (60,  160, 60),
        "Paper":           (240, 240, 200),
        "Plastic":         (240, 120, 60),
        "Textile/Clothes": (200, 80,  160),
        "Trash":           (100, 100, 100),
    }

    print(f"[dataset_utils] Creating synthetic dataset → {base_dir}")
    for cls in CLASS_NAMES:
        safe_cls = cls.replace("/", "-")   # avoid path confusion on Windows/Linux
        cls_dir  = os.path.join(base_dir, safe_cls)
        os.makedirs(cls_dir, exist_ok=True)
        base_rgb = hue_map.get(cls, (128, 128, 128))
        for i in range(n_per_class):
            noise = np.random.randint(0, 60, (*img_size, 3), dtype=np.uint8)
            base  = np.array(base_rgb, dtype=np.uint8).reshape(1, 1, 3)
            img_arr = np.clip(base + noise - 30, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, "RGB")
            img.save(os.path.join(cls_dir, f"{safe_cls}_{i:04d}.jpg"))

    print("[dataset_utils] Synthetic dataset created ✓")


# ── Validate a single image path ────────────────────────────
def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
