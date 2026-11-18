# Project 4 Task List

Based on `DATA4401-Project04.pdf`, checked against the current repo state on
2026-11-18. See `MODEL_RESULTS.md` for model details and numbers.

## Compliance check (current work vs. project rules)

| Requirement | Status | Notes |
|---|---|---|
| **No metadata** (no EXIF, no aspect ratio as a device proxy); pixel content only | ✅ Compliant | We never read EXIF. Images are converted to RGB and **center-cropped to square**, so aspect ratio/orientation is removed. No image has real transparency, so flattening RGBA → RGB changes no pixels. |
| Any tools allowed (pretrained models, packages) | ✅ | TensorFlow/Keras, MobileNetV2 ImageNet weights. |
| **Phase 1:** fit a deep learning model | ✅ Mostly done | MobileNetV2 transfer learning, 82.7% ± 3.4% (5-fold CV). Needs a robustness check (below) and must be **locked by Dec 6**. |
| **Phase 2:** ≥20 hand-labeled features, follow-up modeling | ❌ Not started | The out-of-fold predictions in `results/pretrained/oof_predictions.csv` (`prob_kelly`) are exactly the Phase 2 target the PDF asks for. |
| **Phase 3:** class-wide ensemble on Holdout Set 2 | ❌ Not started | Class coordination, due Dec 14 noon. |
| Holdout Set 1 predictions (20 images) | 🟡 Draft exists | `results/pretrained/holdout_predictions.csv`; needs the instructor's template. |

### ⚠️ Issues found during the check

1. **Holdout Set 2 doesn't match the PDF.** The PDF says Holdout Set 2 has
   **24** images; our `Data/HoldoutSet02/` has **46**, and the files are named
   `TestSet03-ImageNN.png`. Confirm with the instructor which folder or images
   are the real Holdout Set 2 before Dec 14.
2. **Possible processing-artifact shortcut.** Alex's photos come in two file
   types. Out-of-fold accuracy differs sharply by type:
   - RGBA PNGs (178 images): **95.5%** correct
   - palette PNGs (78 images, quantized to ≤256 colors): **66.7%** correct
   - Kelly's photos (all RGB): 78.2%

   The RGBA images have no transparency, so this isn't a direct leak. But the
   model may be partly recognizing how Alex's files were processed (resizing,
   compression, color depth) rather than the photography itself. The PDF notes
   that holdout images "have been resized and modified a few times," so such
   cues may not carry over. Note that Holdout Set 2 is almost all RGBA and
   Holdout Set 1 is almost all RGB.
3. **Presentation date typo.** The PDF lists "Tuesday, December 16," but
   Dec 16, 2026 is a Wednesday. Confirm the date.
4. **"200 photos" ambiguity.** Phase 2 says to hand-label "each of the 200
   photos," but we have 485 labeled images. Decide whether to label all of
   them or a stratified subset of at least 200 (e.g. 100 per photographer).

## Deadlines

| Date | Deliverable |
|---|---|
| **Mon Nov 30** (in class) | Check-in: informal ~5 min demo showing the images processed into a usable dataset |
| **Sun Dec 6** (~2–3 pm) | Modeling procedure locked; bullet-point **approach summary + assessment**; **Holdout Set 1 predictions** (instructor's template → PDF, Gradescope) |
| Mon Dec 7 | Holdout Set 1 answers revealed |
| **Mon Dec 14, noon** | Class-wide **Holdout Set 2** predictions (shared Google Doc) |
| **Dec 16** (confirm day) | 5–10 min presentation to Alex and Kelly |
| **Thu Dec 17** | Full report + slide deck (Gradescope) |

## Tasks

### Phase 1: finish and lock the CNN (before Dec 6)

- [ ] **Ask the instructor about Holdout Set 2** (24 vs. 46 images, `TestSet03` naming).
- [ ] **Robustness check for processing artifacts.** Add augmentation that
  makes file-processing cues unreliable during training: random JPEG
  re-compression, random downscale/upscale, random palette quantization to
  256 colors applied to *both* photographers. Re-run 5-fold CV and compare
  overall and per-file-type accuracy. If accuracy on Alex's RGBA images falls
  toward the others, the model was leaning on artifacts. Prefer the robust
  version either way.
- [ ] Optional: add 5-fold CV to `cnn.py` for an apples-to-apples comparison
  with the pretrained model (useful for the slide on "how we made decisions").
- [ ] Optional: try one stronger backbone (EfficientNetB0 or ResNet50) and
  test-time augmentation. Keep whichever wins under CV.
- [ ] Optional: nested evaluation (separate early-stopping split inside each
  fold) for an unbiased accuracy number in the report.
- [ ] Decide the final predictor: the 5-fold ensemble (current) or one model
  retrained on all 485 images. Document the choice.
- [ ] Delete stale files `results/pretrained/alex_kelly_mobilenetv2.keras` and
  `results/pretrained/training_curves.png`.
- [ ] Add `requirements.txt` so teammates can recreate the environment
  (tensorflow, pillow, numpy, pandas, matplotlib, plus scikit-learn for Phase 2).

### Check-in demo (Nov 30)

- [ ] Fill in `EDA.ipynb`: load the images with `cnn.load_folder`, show
  counts per class, a sample grid per photographer, and before/after
  preprocessing (RGB conversion + center crop).
- [ ] Be ready to explain *why* we center-crop (the no-metadata/aspect-ratio rule).

### Phase 2: feature extraction (start now; labeling takes the longest)

- [ ] **Define at least 20 binary features**, with **at least 5 content** and
  **at least 5 composition** features. Write a one-line definition for each
  (`FEATURES.md`) so labelers agree.
  - Content ideas: people visible, selfie, food, animal, water/ocean,
    mountains, architecture/buildings, text/signage, vehicle, plants/flowers,
    indoor, night.
  - Composition ideas: subject centered, background in focus, horizon visible,
    rule-of-thirds placement, close-up/macro, wide landscape, symmetry,
    strong leading lines, high contrast, warm color cast, tilted horizon.
- [ ] Choose which images to label (all 485 or a stratified ≥200 subset). Also
  label the **66 holdout images** if the features will feed a holdout predictor.
- [ ] Create a shared labeling sheet (`file, feature_1, …, feature_N`) and
  split the images among teammates.
- [ ] **Agreement check:** have 2 people label the same ~20 images and compute
  % agreement per feature. Tighten the definitions of features that disagree.
- [ ] Hand-label the images.
- [ ] **Follow-up modeling** (target = out-of-fold `prob_kelly` from
  `oof_predictions.csv`):
  - [ ] Correlation between each feature and `prob_kelly` (point-biserial),
    plus a regression of `prob_kelly` on all features.
  - [ ] Feature-only classifier (logistic regression and/or a decision tree)
    predicting the true photographer, with cross-validation.
  - [ ] CNN probability + features combined (stacked logistic regression);
    does accuracy improve over the CNN alone?
- [ ] **Grad-CAM heatmaps** for a sample of confident Alex and Kelly
  predictions, to show which regions the CNN looks at. This supports the
  "reverse engineer" goal and makes good slides.

### Dec 6 submission

- [ ] Run the locked model on Holdout Set 1 and fill the instructor's
  20-row template → PDF → Gradescope.
- [ ] Write the bullet-point approach summary and assessment (CV accuracy,
  per-class recall, caveats). `MODEL_RESULTS.md` has most of it.

### Phase 3: class ensemble (Dec 7–14)

- [ ] After Dec 7, score our Holdout Set 1 predictions against the answers and
  report our accuracy to the class.
- [ ] Share our Holdout Set 2 probabilities (not just labels) in the agreed
  class format.
- [ ] Help pick the ensemble rule (e.g. average probabilities, weighted by each
  team's Holdout Set 1 accuracy).
- [ ] Get the final Holdout Set 2 predictions into the class Google Doc before **noon Dec 14**.

### Presentation (Dec 16) and report (Dec 17)

- [ ] Slides covering what the PDF asks for:
  - **Classification procedure:** model, why, complications (BatchNorm
    collapse, file-format/aspect-ratio leak risk, artifact check).
  - **Results:** CV accuracy, confusion matrix, Holdout Set 1 score.
  - **Follow-up:** which features correlate with Alex vs. Kelly predictions.
  - **Conclusions:** what we'd do differently.
  - **Predictions:** Holdout Set 2.
- [ ] Audience is Alex and Kelly: casual or fun is fine, but keep it accurate and complete.
- [ ] Full report (see Labs 1 & 2 for expected content) plus the slide deck to Gradescope.
