# Alex vs. Kelly Photo Classifier — Results & Notes

Context file for teammates and their coding agents. Covers what the task is, how
the data behaves, what models exist, how to run them, what they scored, and
what to watch out for. Last updated 2026-11-18.

## TL;DR

- **Task:** binary image classification: was a photo taken by **Alex** or **Kelly**?
- **Best model:** `cnn_pretrained.py`, MobileNetV2 (ImageNet weights) + fine-tuning.
  **82.7% ± 3.4% accuracy** under stratified 5-fold cross-validation.
- **Baseline:** `cnn.py`, small CNN trained from scratch. **69.1%** on a single 80/20 split.
- **Majority-class baseline** (always guess Alex): ~52.7%.
- **Use `results/pretrained/holdout_predictions.csv`** for holdout-set answers.
  The two models agree on only 55% of holdout images, so don't mix their outputs.

## Data

| Folder | Count | Labeled? | Notes |
|---|---|---|---|
| `Data/Alex/` | 256 | yes (label 0) | Mostly RGBA / palette (`P`) PNGs; mostly landscape (400×300, 448×300, 533×300) |
| `Data/Kelly/` | 229 | yes (label 1) | All RGB PNGs; many portrait (225×300, 169×300) |
| `Data/HoldoutSet01/` | 20 | **no** | `TestSetImageNN.png`; mostly RGB |
| `Data/HoldoutSet02/` | 46 | **no** | `TestSet03-ImageNN.png`; mostly RGBA, many portrait |

`Data/` is git-ignored; each teammate needs a local copy.

Content is everyday travel/life photography (cities, buildings, nature, people,
night scenes). There is no obvious visual "signature" separating the two
photographers, which likely caps achievable accuracy.

### ⚠️ Shortcut / leakage risk

The raw files differ by photographer in ways unrelated to the photo itself:

- **Color mode:** Alex = RGBA/P, Kelly = RGB.
- **Aspect ratio:** Kelly has many portrait images; Alex is mostly landscape.

A naive pipeline (e.g. keeping the alpha channel, or padding/stretching to a
square) lets a CNN learn file format or shape instead of photographic content,
and its accuracy would be misleadingly high. Both scripts guard against this in
`cnn.load_image()`:

1. Flatten any transparency onto white and convert to **RGB**.
2. **Center-crop to a square** (removes the aspect-ratio cue).
3. Resize to the model's input size.

**Any new model should reuse `cnn.load_image` / `cnn.load_folder`** rather than
writing its own loader. Note that the holdout sets have their own format mix
(HoldoutSet02 is mostly RGBA), so a leaky model would also make biased holdout
predictions.

## Code

| File | What it does |
|---|---|
| `cnn.py` | From-scratch CNN. Also holds the shared helpers: `load_image`, `load_folder`, `load_labeled_data`, `stratified_split`, `confusion_matrix`, `plot_history`, `plot_confusion`, and constants `SEED=42`, `CLASSES=["Alex","Kelly"]`. |
| `cnn_pretrained.py` | MobileNetV2 transfer learning with 5-fold CV. Imports helpers from `cnn.py`. |
| `EDA.ipynb` | Placeholder (imports only). |

Framework: **TensorFlow 2.21 / Keras 3.15** (already in `.venv`). No PyTorch or
scikit-learn installed; metrics like the confusion matrix are hand-rolled with
NumPy.

### Running

```bash
.venv/bin/python cnn.py              # ~3-4 min on CPU (Apple Silicon, no GPU used)
.venv/bin/python cnn_pretrained.py   # ~20-25 min on CPU (5 folds × 2 phases)
```

The first `cnn_pretrained.py` run downloads ~9 MB of MobileNetV2 weights from
Keras's public storage. The `Ignoring clocale-0.0.4 ...` line printed by the
shell is an unrelated Ruby gem warning and can be ignored.

### Model details

**`cnn.py` (from scratch)**
- Input 128×128. Augmentation: horizontal flip, ±5% rotation, 15% zoom, 10% translation, 15% contrast.
- 4 conv blocks (32→64→128→256 filters, each 2× Conv3×3+BatchNorm+ReLU, then MaxPool) → GlobalAvgPool → Dropout 0.5 → Dense 64 → Dropout 0.3 → sigmoid. ~1.2M params.
- Adam 1e-3, up to 100 epochs, EarlyStopping (val_loss, patience 20, restores best weights), ReduceLROnPlateau.
- Evaluation: single stratified 80/20 split (388 train / 97 val), seed 42.

**`cnn_pretrained.py` (transfer learning)**
- Input 224×224, same augmentation, rescaled to [-1, 1].
- MobileNetV2 base (`include_top=False`, ImageNet) → GlobalAvgPool → Dropout 0.4 → sigmoid.
- Phase 1: base frozen, Adam 1e-3, up to 60 epochs (patience 10).
- Phase 2: unfreeze last 40 base layers (**BatchNorm layers stay frozen**, base called with `training=False`), Adam 1e-5, up to 30 epochs (patience 8).
- Evaluation: stratified 5-fold CV (fold sizes 98/97/97/97/96), fresh model per fold, seed `42 + fold`.
- Holdout predictions = mean probability across the 5 fold models.

## Results

### Headline comparison

| | `cnn.py` (scratch) | `cnn_pretrained.py` (5-fold CV) |
|---|---|---|
| Validation accuracy | **69.1%** (one 97-image split) | **82.7% ± 3.4%** (mean ± std over folds) |
| Alex recall | 74.5% (38/51) | 86.7% (222/256) |
| Kelly recall | 63.0% (29/46) | 78.2% (179/229) |
| Majority baseline | 52.6% | 52.8% |

### Pretrained model, per fold

| Fold | n_val | Phase 1 (head only) | Phase 2 (fine-tuned) |
|---|---|---|---|
| 1 | 98 | 76.5% | 82.7% |
| 2 | 97 | 80.4% | 84.5% |
| 3 | 97 | 82.5% | 82.5% |
| 4 | 97 | 74.2% | 77.3% |
| 5 | 96 | 85.4% | 86.5% |
| **Mean** | | **80.0%** | **82.7%** |

Pooled out-of-fold confusion matrix (rows = actual, cols = predicted):

|  | Alex | Kelly |
|---|---|---|
| **Alex** | 222 | 34 |
| **Kelly** | 50 | 179 |

### Holdout predictions

| Set | `cnn.py` (Alex / Kelly) | `cnn_pretrained.py` ensemble (Alex / Kelly) |
|---|---|---|
| HoldoutSet01 (20) | 9 / 11 | 8 / 12 |
| HoldoutSet02 (46) | 25 / 21 | 24 / 22 |

The class totals look similar, but the two models agree on **only 36 of 66
(55%)** individual holdout images, which is near chance. Prefer the pretrained
ensemble; treat `cnn.py`'s holdout predictions as a weak baseline.

## Interpretation

- **Transfer learning helps a lot.** Even with the base frozen, MobileNetV2
  features reach ~80%; fine-tuning adds ~3 points. With only ~390 training
  images per fold, learning filters from scratch is the bottleneck.
- **The gain is not split luck.** The worst fold (77.3%) still beats the
  from-scratch model (69.1%) by 8+ points.
- **Kelly is the harder class.** The model calls Kelly's photos Alex's more
  often (50 errors) than the reverse (34).
- **Longer phase 1 mattered.** An earlier single-split pretrained run with
  phase 1 capped at 30 epochs scored 73.2%; validation loss was still falling
  when it stopped. Raising the cap to 60 accounted for much of the jump.

## Caveats

- **Comparison isn't perfectly matched:** 69.1% is from one split; 82.7% is a
  5-fold average. The gap far exceeds the fold-to-fold spread, so the conclusion
  holds, but for a write-up consider adding 5-fold CV to `cnn.py` too.
- **Slightly optimistic estimates:** both scripts use the validation fold for
  early stopping and LR scheduling, so validation accuracy is mildly
  optimistic. A nested split (train / early-stop-val / test) would remove this.
- **Small validation sets:** ~97 images per fold, so one image ≈ 1 point.
- **Holdout sets are unlabeled.** No accuracy can be computed on them.
- **Possible processing-artifact shortcut.** Out-of-fold accuracy on Alex's
  RGBA PNGs is 95.5% (178 imgs) vs. 66.7% on Alex's palette PNGs (78 imgs,
  ≤256 colors) and 78.2% on Kelly (all RGB). No image has real transparency,
  so RGB conversion is lossless. Still, the model may be partly recognizing how
  Alex's files were processed. Holdout Set 2 is ~all RGBA and Holdout Set 1 ~all
  RGB. See `TASKS.md` for the planned robustness check.
- **Holdout Set 2 count mismatch:** the project PDF says 24 images; our
  `Data/HoldoutSet02/` has 46 (`TestSet03-*`). Confirm with the instructor.

## Gotchas learned the hard way

- **BatchNorm + tiny datasets:** the first `cnn.py` run had val accuracy stuck
  at exactly the majority rate (predicting all Alex) while train accuracy rose.
  BatchNorm's running statistics hadn't converged with only 13 batches/epoch.
  Fixed with `BatchNormalization(momentum=0.9)` and more early-stopping patience.
- **Fine-tuning pretrained nets:** keep the base's BatchNorm layers frozen and
  call the base with `training=False`, or small batches will wreck the
  pretrained statistics.
- **Stale files:** `results/pretrained/alex_kelly_mobilenetv2.keras` and
  `results/pretrained/training_curves.png` are left over from the earlier
  single-split pretrained run (73.2%) and do **not** correspond to the 5-fold
  results. The 5-fold script doesn't save models.

## Output files

```
results/
├── alex_kelly_cnn.keras          # cnn.py model (git-ignored; regenerate locally)
├── training_curves.png           # cnn.py
├── confusion_matrix.png          # cnn.py (97-image val split)
├── holdout_predictions.csv       # cnn.py: set, file, prob_kelly, predicted
└── pretrained/
    ├── cv_summary.csv            # fold, n_val, head_only_acc, fine_tuned_acc
    ├── oof_predictions.csv       # file, actual, fold, prob_kelly, predicted
    ├── confusion_matrix.png      # pooled over all 485 images
    ├── fold{1..5}_training_curves.png
    ├── holdout_predictions.csv   # 5-model ensemble: set, file, prob_kelly, predicted
    ├── alex_kelly_mobilenetv2.keras   # STALE (single-split run; git-ignored)
    └── training_curves.png            # STALE (single-split run)
```

`prob_kelly` is the sigmoid output; ≥ 0.5 → predicted Kelly.

`*.keras` model files are git-ignored (14–22 MB each). Everything else in
`results/` is committed, so teammates can see the metrics and plots without
retraining.

## Possible next steps

1. Add 5-fold CV to `cnn.py` for an apples-to-apples comparison.
2. Try a stronger backbone (EfficientNetB0, ResNet50, ConvNeXt-Tiny).
3. Test-time augmentation (average predictions over flipped/cropped copies).
4. Inspect `oof_predictions.csv` for consistently misclassified images and look
   for patterns (scene type, people vs. landscapes, lighting).
5. Train a final model on all 485 labeled images for holdout predictions, once
   the hyperparameters are settled.
