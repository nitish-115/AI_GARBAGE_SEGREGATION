# ============================================================
#  Garbage Segregation System — Configuration
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH      = os.path.join(SAVED_MODEL_DIR, "garbage_classifier_model.h5")
HISTORY_PATH    = os.path.join(SAVED_MODEL_DIR, "training_history.npy")

# ── Image settings ───────────────────────────────────────────
IMG_SIZE    = (224, 224)
IMG_SHAPE   = (224, 224, 3)
BATCH_SIZE  = 32

# ── Training ─────────────────────────────────────────────────
EPOCHS          = 30
LEARNING_RATE   = 1e-4
FINE_TUNE_LR    = 1e-5
VALIDATION_SPLIT = 0.15
TEST_SPLIT       = 0.15
DROPOUT_RATE     = 0.5
PATIENCE         = 7          # Early-stopping patience
FINE_TUNE_LAYERS = 20         # Unfreeze last N layers of base

# ── Classes & Bins ──────────────────────────────────────────
CLASS_NAMES = [
    "Organic Waste",
    "Textile-Clothes",
    "battery",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

BIN_MAP = {
    "Organic Waste":   ("🌿  Compost / Green Bin",   "#8BC34A", "🟢"),
    "Textile-Clothes": ("👕  Donation / Textile Bin", "#E91E63", "🔴"),
    "battery":         ("⚠️  Hazardous Waste Bin",   "#FF4B4B", "🔴"),
    "cardboard":       ("♻️  Recycling Bin",          "#4CAF50", "🟢"),
    "glass":           ("🫙  Glass Recycling Bin",    "#2196F3", "🔵"),
    "metal":           ("🥫  Metal Recycling Bin",    "#9C27B0", "🟣"),
    "paper":           ("📄  Paper Recycling Bin",    "#00BCD4", "🔵"),
    "plastic":         ("🧴  Plastic Recycling Bin",  "#FF5722", "🟠"),
    "trash":           ("🗑️  General Waste Bin",      "#607D8B", "⚫"),
}

CLASS_COLORS = [
    "#8BC34A", "#E91E63", "#FF4B4B", "#4CAF50", "#2196F3",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]

# Disposal bin mapping


