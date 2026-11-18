"""
CNN classifier: was a photo taken by Alex or by Kelly?

Builds a small convolutional neural network with TensorFlow/Keras, trains it on
Data/Alex and Data/Kelly, evaluates it on a held-out validation split, and then
predicts the unlabeled images in Data/HoldoutSet01 and Data/HoldoutSet02.

Run from the project root:
    .venv/bin/python cnn.py

Outputs (written to results/):
    training_curves.png      accuracy / loss per epoch
    confusion_matrix.png     validation-set confusion matrix
    holdout_predictions.csv  predicted photographer for every holdout image
    alex_kelly_cnn.keras     the trained model
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow startup logs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 128          # images are center-cropped to a square, then resized
BATCH_SIZE = 32
EPOCHS = 100            # upper bound; early stopping usually ends sooner
VAL_FRACTION = 0.2

DATA_DIR = Path("Data")
CLASSES = ["Alex", "Kelly"]   # label 0 = Alex, label 1 = Kelly
HOLDOUT_DIRS = ["HoldoutSet01", "HoldoutSet02"]
OUT_DIR = Path("results")

keras.utils.set_random_seed(SEED)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_image(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    """Load one image as a (size, size, 3) uint8 array.

    The raw files differ between photographers in ways unrelated to the photo
    itself (Alex's PNGs are mostly RGBA/palette mode, Kelly's are RGB and often
    portrait). To keep the CNN from learning those shortcuts, every image is
    converted to RGB and center-cropped to a square before resizing.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        # Flatten any transparency onto white, then drop the alpha channel.
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_folder(folder: Path, size: int = IMG_SIZE) -> tuple[np.ndarray, list[str]]:
    paths = sorted(folder.glob("*.png"))
    images = np.stack([load_image(p, size) for p in paths])
    return images, [p.name for p in paths]


def load_labeled_data(size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        images, _ = load_folder(DATA_DIR / name, size)
        xs.append(images)
        ys.append(np.full(len(images), label, dtype=np.int32))
        print(f"  {name}: {len(images)} images")
    return np.concatenate(xs), np.concatenate(ys)


def stratified_split(y: np.ndarray, val_fraction: float, seed: int):
    """Return train/val index arrays with the same class balance in each."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.where(y == label)[0])
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return rng.permutation(train_idx), rng.permutation(val_idx)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)


def build_cnn() -> keras.Model:
    # Augmentation runs only during training; it helps a lot with ~400 images.
    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = conv_block(x, filters)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(photo is Kelly's)

    model = keras.Model(inputs, outputs, name="alex_vs_kelly_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------------
# Evaluation / plotting helpers
# ----------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_history(history: keras.callbacks.History, path: Path) -> None:
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(h[metric], label="train")
        ax.plot(h[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=14)
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading labeled images...")
    X, y = load_labeled_data()
    train_idx, val_idx = stratified_split(y, VAL_FRACTION, SEED)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train: {len(X_train)}  Validation: {len(X_val)}")

    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, OUT_DIR / "training_curves.png")

    # ---- Validation results ----
    val_prob = model.predict(X_val, verbose=0).ravel()
    val_pred = (val_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_val, val_pred)
    accuracy = (val_pred == y_val).mean()
    majority_baseline = max(np.mean(y_val), 1 - np.mean(y_val))

    print("\n=== Validation results ===")
    print(f"Accuracy:           {accuracy:.3f}")
    print(f"Majority baseline:  {majority_baseline:.3f}  (always guess the bigger class)")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    for i, name in enumerate(CLASSES):
        precision = cm[i, i] / max(cm[:, i].sum(), 1)
        recall = cm[i, i] / max(cm[i, :].sum(), 1)
        print(f"  {name:<5}  precision={precision:.3f}  recall={recall:.3f}")
    plot_confusion(cm, OUT_DIR / "confusion_matrix.png")

    # ---- Holdout predictions ----
    rows = []
    for folder in HOLDOUT_DIRS:
        images, names = load_folder(DATA_DIR / folder)
        probs = model.predict(images, verbose=0).ravel()
        for name, p in zip(names, probs):
            rows.append(
                {
                    "set": folder,
                    "file": name,
                    "prob_kelly": round(float(p), 4),
                    "predicted": CLASSES[int(p >= 0.5)],
                }
            )
    preds = pd.DataFrame(rows)
    preds.to_csv(OUT_DIR / "holdout_predictions.csv", index=False)

    print("\n=== Holdout predictions ===")
    print(preds.groupby(["set", "predicted"]).size().unstack(fill_value=0))

    model.save(OUT_DIR / "alex_kelly_cnn.keras")
    print(f"\nSaved plots, predictions, and model to {OUT_DIR}/")


if __name__ == "__main__":
    main()
