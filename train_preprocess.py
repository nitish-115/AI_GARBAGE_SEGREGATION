# ============================================================
#  train_preprocess.py — Data Loading & Augmentation Pipeline
# ============================================================

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    DATASET_DIR, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, TEST_SPLIT, CLASS_NAMES, CLASS_COLORS
)
from utils.dataset_utils import get_class_counts, create_synthetic_dataset


# ── Auto-create synthetic data if folder is empty ───────────
def ensure_dataset():
    counts = get_class_counts()
    if not counts or sum(counts.values()) == 0:
        print("[preprocess] No dataset found — generating synthetic images …")
        create_synthetic_dataset(n_per_class=80)
    else:
        print(f"[preprocess] Dataset found: {sum(counts.values())} images "
              f"across {len(counts)} classes.")


# ── Verify image is readable by PIL ─────────────────────────
def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


# ── Collect all image paths + labels ────────────────────────
def load_file_paths(dataset_dir: str = DATASET_DIR):
    all_paths, all_labels = [], []
    available_classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(available_classes)}

    bad_count = 0
    for cls in available_classes:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(cls_dir, fname)
                if is_valid_image(full_path):
                    all_paths.append(full_path)
                    all_labels.append(class_to_idx[cls])
                else:
                    bad_count += 1

    if bad_count:
        print(f"[preprocess] Skipped {bad_count} corrupt/unreadable images.")
    print(f"[preprocess] Total valid images: {len(all_paths)}")
    return all_paths, all_labels, available_classes


# ── Safe image parser (handles decode errors gracefully) ─────
def _parse_image(path, label, augment: bool = False):
    try:
        raw = tf.io.read_file(path)
        # Use decode_jpeg with try_recover_truncated for robustness
        img = tf.image.decode_jpeg(raw, channels=3, try_recover_truncated=True)
    except Exception:
        try:
            raw = tf.io.read_file(path)
            img = tf.image.decode_png(raw, channels=3)
        except Exception:
            # Return a blank image on total failure
            img = tf.zeros([*IMG_SIZE, 3], dtype=tf.uint8)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_hue(img, 0.05)
        img = tf.clip_by_value(img, 0.0, 1.0)

    return img, label


def _safe_parse(path, label, augment=False):
    """Wraps _parse_image; returns a black image if decoding fails."""
    def _decode(p, l):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3,
                                     expand_animations=False,
                                     dtype=tf.uint8)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.7, 1.3)
            img = tf.image.random_hue(img, 0.05)
            img = tf.clip_by_value(img, 0.0, 1.0)
        return img, l

    result = tf.py_function(
        func=lambda p, l: _pil_load(p.numpy().decode(), l.numpy(), augment),
        inp=[path, label],
        Tout=[tf.float32, tf.int64]
    )
    result[0].set_shape([*IMG_SIZE, 3])
    result[1].set_shape([])
    return result[0], result[1]


def _pil_load(path: str, label: int, augment: bool):
    """PIL-based loader — handles JPEG, PNG, BMP reliably on Apple Silicon."""
    import numpy as np
    from PIL import ImageEnhance, ImageFilter
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
    except Exception:
        arr = np.zeros([*IMG_SIZE, 3], dtype=np.float32)
        return arr.astype(np.float32), np.int64(label)

    if augment:
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (-25 to 25 degrees)
        if np.random.rand() > 0.3:
            angle = np.random.uniform(-25, 25)
            img = img.rotate(angle, fillcolor=(255, 255, 255))

        # Random zoom / crop (simulates different distances from camera)
        if np.random.rand() > 0.3:
            zoom = np.random.uniform(0.8, 1.0)
            w, h = img.size
            nw, nh = int(w * zoom), int(h * zoom)
            left = np.random.randint(0, w - nw + 1)
            top = np.random.randint(0, h - nh + 1)
            img = img.crop((left, top, left + nw, top + nh)).resize(IMG_SIZE, Image.LANCZOS)

        # Brightness
        if np.random.rand() > 0.3:
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.7, 1.3))

        # Contrast
        if np.random.rand() > 0.3:
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.7, 1.3))

        # Saturation
        if np.random.rand() > 0.3:
            img = ImageEnhance.Color(img).enhance(np.random.uniform(0.7, 1.3))

        # Slight blur (mimics phone/webcam autofocus softness)
        if np.random.rand() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.astype(np.float32), np.int64(label)


def build_datasets(dataset_dir: str = DATASET_DIR):
    ensure_dataset()
    paths, labels, classes = load_file_paths(dataset_dir)

    paths  = np.array(paths)
    labels = np.array(labels)

    # Split: train / val / test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )
    val_ratio = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=42
    )

    print(f"[preprocess] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    def make_ds(X, y, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.map(
            lambda p, l: _safe_parse(p, l, augment),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(X_train, y_train, augment=True)
    val_ds   = make_ds(X_val,   y_val,   augment=False)
    test_ds  = make_ds(X_test,  y_test,  augment=False)

    return train_ds, val_ds, test_ds, classes, (X_test, y_test)


# ── Dataset distribution plot ────────────────────────────────
def plot_distribution(save_path: str = None):
    counts = get_class_counts()
    if not counts:
        print("[preprocess] No dataset to plot.")
        return

    classes = list(counts.keys())
    values  = list(counts.values())
    colors  = CLASS_COLORS[:len(classes)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Dataset Class Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Waste Category", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    plt.xticks(rotation=30, ha="right", color="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9, color="white")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[preprocess] Distribution plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    build_datasets()
    plot_distribution(save_path="models/dataset_distribution.png")