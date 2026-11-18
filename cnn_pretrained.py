"""
Transfer-learning version of the Alex-vs-Kelly classifier, evaluated with
stratified 5-fold cross-validation.

Instead of learning every filter from ~400 photos, this starts from MobileNetV2
pretrained on ImageNet (1.4M photos) and trains in two phases:

  1. Feature extraction: freeze the pretrained base, train only a new
     classification head on top.
  2. Fine-tuning: unfreeze the top layers of the base and keep training with a
     much smaller learning rate so the pretrained features adapt to these photos.

Cross-validation: the 485 labeled images are split into 5 folds that keep the
Alex/Kelly ratio. Each fold takes one turn as the validation set while a fresh
model trains on the other 4, so every image is scored exactly once by a model
that never trained on it. The reported accuracy is the mean +/- std across folds.

Holdout images are predicted by all 5 fold models and their probabilities are
averaged (a small ensemble).

Run from the project root:
    .venv/bin/python cnn_pretrained.py

Outputs (written to results/pretrained/):
    cv_summary.csv              per-fold accuracy (head-only and fine-tuned)
    oof_predictions.csv         out-of-fold prediction for every labeled image
    confusion_matrix.png        pooled confusion matrix over all 5 folds
    fold{k}_training_curves.png accuracy / loss per epoch for each fold
    holdout_predictions.csv     ensemble prediction for every holdout image
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from cnn import (
    CLASSES,
    DATA_DIR,
    HOLDOUT_DIRS,
    SEED,
    confusion_matrix,
    load_folder,
    plot_confusion,
    plot_history,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
IMG_SIZE = 224               # MobileNetV2's native ImageNet resolution
BATCH_SIZE = 32
N_FOLDS = 5
HEAD_EPOCHS = 60             # phase 1: frozen base (upper bound, early stopping)
FINE_TUNE_EPOCHS = 30        # phase 2: top of base unfrozen
FINE_TUNE_LAYERS = 40        # how many of the base's last layers to unfreeze
OUT_DIR = Path("results/pretrained")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_labeled_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, names = [], [], []
    for label, cls in enumerate(CLASSES):
        images, files = load_folder(DATA_DIR / cls, IMG_SIZE)
        xs.append(images)
        ys.append(np.full(len(images), label, dtype=np.int32))
        names.extend(f"{cls}/{f}" for f in files)
        print(f"  {cls}: {len(images)} images")
    return np.concatenate(xs), np.concatenate(ys), names


def stratified_kfold(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Return an array assigning each sample a fold id in [0, n_folds).

    Each class is shuffled and dealt round-robin into the folds, so every fold
    has (almost) the same Alex/Kelly ratio as the full dataset.
    """
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        idx = rng.permutation(np.where(y == label)[0])
        fold_of[idx] = np.arange(len(idx)) % n_folds
    return fold_of


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model() -> tuple[keras.Model, keras.Model]:
    """Return (full_model, pretrained_base)."""
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

    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    # MobileNetV2 expects pixels scaled to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    # training=False keeps the base's BatchNorm layers in inference mode, even
    # after unfreezing; otherwise small batches wreck the pretrained statistics.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(photo is Kelly's)

    return keras.Model(inputs, outputs, name="alex_vs_kelly_mobilenetv2"), base


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def callbacks(patience: int) -> list:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7
        ),
    ]


def merge_histories(*histories: keras.callbacks.History) -> keras.callbacks.History:
    """Concatenate per-epoch metrics from both phases for one training plot."""
    merged = keras.callbacks.History()
    merged.history = {}
    for h in histories:
        for key, values in h.history.items():
            merged.history.setdefault(key, []).extend(values)
    return merged


def train_one_fold(X_train, y_train, X_val, y_val, fold: int):
    """Train a fresh model on one fold. Returns (model, head_only_val_acc)."""
    keras.backend.clear_session()            # free the previous fold's model
    keras.utils.set_random_seed(SEED + fold)
    model, base = build_model()

    # ---- Phase 1: train only the new head ----
    compile_model(model, 1e-3)
    h1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=HEAD_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks(patience=10),
        verbose=2,
    )
    head_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"  phase 1 (head only) val accuracy: {head_acc:.3f}")

    # ---- Phase 2: fine-tune the top of the base ----
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:  # keep BatchNorm frozen; see note in build_model
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks(patience=8),
        verbose=2,
    )
    plot_history(merge_histories(h1, h2), OUT_DIR / f"fold{fold + 1}_training_curves.png")
    return model, head_acc


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading labeled images...")
    X, y, names = load_labeled_data()

    print("Loading holdout images...")
    holdout = {folder: load_folder(DATA_DIR / folder, IMG_SIZE) for folder in HOLDOUT_DIRS}

    fold_of = stratified_kfold(y, N_FOLDS, SEED)
    oof_prob = np.zeros(len(y))
    holdout_probs = {folder: [] for folder in HOLDOUT_DIRS}
    fold_rows = []

    for fold in range(N_FOLDS):
        val_mask = fold_of == fold
        X_train, y_train = X[~val_mask], y[~val_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        print(f"\n===== Fold {fold + 1}/{N_FOLDS}  "
              f"(train {len(X_train)}, validation {len(X_val)}) =====")

        model, head_acc = train_one_fold(X_train, y_train, X_val, y_val, fold)

        val_prob = model.predict(X_val, verbose=0).ravel()
        oof_prob[val_mask] = val_prob
        acc = ((val_prob >= 0.5).astype(int) == y_val).mean()
        print(f"  phase 2 (fine-tuned) val accuracy: {acc:.3f}")
        fold_rows.append(
            {"fold": fold + 1, "n_val": int(val_mask.sum()),
             "head_only_acc": round(head_acc, 4), "fine_tuned_acc": round(acc, 4)}
        )

        for folder, (images, _) in holdout.items():
            holdout_probs[folder].append(model.predict(images, verbose=0).ravel())

    # ---- Cross-validation summary ----
    summary = pd.DataFrame(fold_rows)
    summary.to_csv(OUT_DIR / "cv_summary.csv", index=False)

    oof_pred = (oof_prob >= 0.5).astype(int)
    cm = confusion_matrix(y, oof_pred)
    majority_baseline = max(np.mean(y), 1 - np.mean(y))

    print("\n=== 5-fold cross-validation results (MobileNetV2 transfer learning) ===")
    print(summary.to_string(index=False))
    for col, label in [("head_only_acc", "Head only "), ("fine_tuned_acc", "Fine-tuned")]:
        print(f"{label} accuracy: {summary[col].mean():.3f} +/- {summary[col].std():.3f}")
    print(f"Pooled out-of-fold accuracy: {(oof_pred == y).mean():.3f}")
    print(f"Majority baseline:           {majority_baseline:.3f}")
    print("Pooled confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    for i, cls in enumerate(CLASSES):
        precision = cm[i, i] / max(cm[:, i].sum(), 1)
        recall = cm[i, i] / max(cm[i, :].sum(), 1)
        print(f"  {cls:<5}  precision={precision:.3f}  recall={recall:.3f}")
    plot_confusion(cm, OUT_DIR / "confusion_matrix.png")

    pd.DataFrame(
        {
            "file": names,
            "actual": [CLASSES[t] for t in y],
            "fold": fold_of + 1,
            "prob_kelly": np.round(oof_prob, 4),
            "predicted": [CLASSES[p] for p in oof_pred],
        }
    ).to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    # ---- Holdout predictions (average of the 5 fold models) ----
    rows = []
    for folder, (_, files) in holdout.items():
        mean_prob = np.mean(holdout_probs[folder], axis=0)
        for name, p in zip(files, mean_prob):
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

    print("\n=== Holdout predictions (5-model ensemble) ===")
    print(preds.groupby(["set", "predicted"]).size().unstack(fill_value=0))
    print(f"\nSaved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
