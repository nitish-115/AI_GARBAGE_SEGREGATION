# ============================================================
#  predict.py — Image Prediction & Result Formatting
# ============================================================

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import IMG_SIZE, MODEL_PATH, SAVED_MODEL_DIR, BIN_MAP, CLASS_NAMES


# ── Singleton model loader ───────────────────────────────────
_model   = None
_classes = None


def load_model_and_classes():
    global _model, _classes

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}.\n"
                "Run  python train.py  first to train the model."
            )
        print("[predict] Loading model …")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[predict] Model loaded ✓")

    if _classes is None:
        cls_path = os.path.join(SAVED_MODEL_DIR, "classes.json")
        if os.path.exists(cls_path):
            with open(cls_path) as f:
                _classes = json.load(f)
        else:
            _classes = CLASS_NAMES          # fall back to config

    return _model, _classes


# ── Core preprocessing ───────────────────────────────────────
def preprocess_image(img_input) -> np.ndarray:
    """
    Accept a PIL Image, NumPy array (H×W×3), or file path.
    Returns a (1, 224, 224, 3) float32 array normalised to [0, 1].
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img_input)}")

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Prediction ───────────────────────────────────────────────
def predict(img_input) -> dict:
    """
    Returns a dict with:
      class_name   – predicted waste category
      confidence   – top-class probability (0–1)
      bin_label    – recommended disposal bin string
      bin_color    – hex colour for UI
      bin_icon     – emoji icon
      top5         – list of (class, prob) for top-5 classes
    """
    model, classes = load_model_and_classes()
    arr   = preprocess_image(img_input)
    probs = model.predict(arr, verbose=0)[0]          # shape: (n_classes,)

    top_idx  = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    cls_name = classes[top_idx]

    bin_label, bin_color, bin_icon = BIN_MAP.get(
        cls_name, ("🗑️ General Waste Bin", "#607D8B", "⚫")
    )

    top5 = sorted(
        [(classes[i], float(probs[i])) for i in range(len(classes))],
        key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "class_name": cls_name,
        "confidence": top_prob,
        "bin_label":  bin_label,
        "bin_color":  bin_color,
        "bin_icon":   bin_icon,
        "top5":       top5,
        "all_probs":  {classes[i]: float(probs[i]) for i in range(len(classes))},
    }


# ── CLI demo ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Garbage Classifier — Single Image")
    parser.add_argument("image", help="Path to the image file")
    args = parser.parse_args()

    result = predict(args.image)
    print(f"\n{'='*50}")
    print(f"  Waste Type  : {result['class_name']}")
    print(f"  Confidence  : {result['confidence']*100:.1f}%")
    print(f"  Disposal Bin: {result['bin_label']}")
    print(f"{'='*50}")
    print("\nTop-5 Predictions:")
    for cls, prob in result["top5"]:
        bar = "█" * int(prob * 30)
        print(f"  {cls:<20} {prob*100:5.1f}%  {bar}")
