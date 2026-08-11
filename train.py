# ============================================================
#  train.py — Model Architecture, Training & Evaluation
# ============================================================

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
try:
    from tensorflow.keras.optimizers.legacy import Adam  # faster on Apple M1/M2
except ImportError:
    from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import (
    IMG_SHAPE, DROPOUT_RATE, LEARNING_RATE, FINE_TUNE_LR,
    EPOCHS, PATIENCE, FINE_TUNE_LAYERS,
    MODEL_PATH, HISTORY_PATH, MODELS_DIR, SAVED_MODEL_DIR,
    CLASS_COLORS
)
from train_preprocess import build_datasets


# ── Build MobileNetV2 Transfer-Learning Model ────────────────
def build_model(n_classes: int) -> Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SHAPE)
    base.trainable = False          # freeze during head training

    inputs  = tf.keras.Input(shape=IMG_SHAPE)
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.Dropout(DROPOUT_RATE)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="GarbageClassifier_v1")
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_top_layers(model, base_model, n_layers: int):
    """Unfreeze the last `n_layers` of the base for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[train] Fine-tuning: last {n_layers} base layers unfrozen.")


# ── Training ─────────────────────────────────────────────────
def train(dataset_dir: str = None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, classes, (X_test, y_test) = build_datasets(
        *([dataset_dir] if dataset_dir else [])
    )
    n_classes = len(classes)
    print(f"[train] Classes ({n_classes}): {classes}")

    model, base = build_model(n_classes)
    model.summary()

    ckpt_path = os.path.join(MODELS_DIR, "best_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    # ── Phase 1: Train head only ─────────────────────────────
    print("\n[train] ── Phase 1: Head training ─────────────────")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS // 2, callbacks=callbacks, verbose=1)

    # ── Phase 2: Fine-tune ───────────────────────────────────
    print("\n[train] ── Phase 2: Fine-tuning ────────────────────")
    unfreeze_top_layers(model, base, FINE_TUNE_LAYERS)
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=EPOCHS, initial_epoch=len(h1.history["loss"]),
                   callbacks=callbacks, verbose=1)

    # ── Merge histories ──────────────────────────────────────
    history = {}
    for key in h1.history:
        history[key] = h1.history[key] + h2.history[key]

    # ── Save model & history ─────────────────────────────────
    model.save(MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")
    np.save(HISTORY_PATH, history)
    # also save classes
    cls_path = os.path.join(SAVED_MODEL_DIR, "classes.json")
    with open(cls_path, "w") as f:
        json.dump(classes, f)
    print(f"[train] Classes saved → {cls_path}")

    # ── Evaluation ───────────────────────────────────────────
    evaluate_model(model, test_ds, X_test, y_test, classes, history)
    return model, history, classes


# ── Evaluation ───────────────────────────────────────────────
def evaluate_model(model, test_ds, X_test, y_test, classes, history=None):
    print("\n[eval] Running evaluation on test set …")
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    # Save metrics
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    with open(os.path.join(SAVED_MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_confusion_matrix(y_test, y_pred, classes)
    if history:
        _plot_training_curves(history)


def _plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[eval] Confusion matrix saved → {save_path}")
    plt.close()


def _plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#16213e")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Accuracy
    axes[0].plot(history["accuracy"],     color="#4CAF50", lw=2, label="Train Acc")
    axes[0].plot(history["val_accuracy"], color="#FF9800", lw=2, label="Val Acc",   ls="--")
    axes[0].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

    # Loss
    axes[1].plot(history["loss"],     color="#F44336", lw=2, label="Train Loss")
    axes[1].plot(history["val_loss"], color="#2196F3", lw=2, label="Val Loss",  ls="--")
    axes[1].set_title("Loss Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    save_path = os.path.join(MODELS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[eval] Training curves saved → {save_path}")
    plt.close()


if __name__ == "__main__":
    train()