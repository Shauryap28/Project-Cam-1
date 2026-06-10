# DESIGN.md — Why this project is built the way it is

This document explains every meaningful choice in the project. If you come back to this repo in six months (or someone asks you about it in an interview), read this first.

The format is deliberately Q&A — each section answers a question someone could plausibly ask.

---

## 1. The problem and the framing

### Q: What is the actual task?
Given an image containing a soldier, decide whether the soldier is **effectively camouflaged** (hard to spot, `cam`) or **clearly visible** (`no_cam`).

This is a binary image classification problem on the surface. The complication is that the subject is *designed to be hard to see* and the background is correlated with the label.

### Q: Why is this harder than normal image classification?
Three reasons:

1. **The subject hides.** In cat-vs-dog, both classes are clearly visible. Here, a well-camouflaged soldier may occupy a tiny, low-contrast region.
2. **Background correlates with label but should not decide it.** Forest backgrounds appear frequently with `cam` examples, but a soldier standing openly in a forest is still `no_cam`. A naive classifier will learn *"green = camo"* and look right on the test set while being scientifically wrong.
3. **Standard object detectors fail.** YOLO and friends are trained to find visible objects. They are poor at finding subjects engineered to be invisible.

### Q: Why two stages instead of one classifier on raw images?
Because a single classifier on raw images consistently learns the background shortcut. Cropping the soldier region first removes most of the noise the classifier could otherwise exploit. Single-stage attempts plateaued around 65–70% accuracy in early experiments. Two-stage pushed that up by roughly 25–30 percentage points.

> Caveat to remember: this also creates a *train/inference mismatch* — the classifier sees crops during training but might see full images at inference if Stage 1 fails. The fusion logic in `src/inference.py` is a patch for this. See Section 6.

---

## 2. Stage 1 — Detection (ZoomNeXt)

### Q: Why ZoomNeXt and not YOLO / Faster R-CNN / Mask R-CNN?
ZoomNeXt is a **Camouflaged Object Detection** (COD) model. It's trained on datasets (CHAMELEON, CAMO, COD10K) full of objects that try not to be found — animals using natural camouflage, hidden insects, and yes, camouflaged people. General-purpose detectors are trained on visible objects and don't transfer well to this regime.

### Q: How does ZoomNeXt actually work, at a high level?
Three ideas worth being able to articulate:

1. **Multi-scale ("zoom") inference.** The image is processed at 0.5x, 1.0x, and 1.5x resolution. This mimics how a human searches a scene — first the big picture for shape, then zooming in for detail. In the code: see the `image_s`, `image_m`, `image_l` tensors in `src/detection.py::predict_mask`.
2. **Edge-aware processing.** Camouflage works by disrupting outlines. Edge-sensitive features help find places where a body boundary subtly breaks the background pattern.
3. **Hierarchical features from the backbone.** The model fuses low-level (texture) and high-level (shape) cues.

### Q: Which backbone is loaded and why?
**PVTv2-B4.** The code calls `PvtV2B4_ZoomNeXt` and loads `pvtv2-b4-zoomnext.pth`.

PVTv2 (Pyramid Vision Transformer v2) is a transformer backbone with a pyramid structure — it produces features at multiple resolutions, which fits ZoomNeXt's multi-scale design. The B4 variant is the second-largest, and the publicly available pretrained weights for ZoomNeXt include this configuration.

> **Honesty note.** The original README mentioned EfficientNet-B4 as the chosen backbone. The actual code loads PVTv2-B4. If asked in an interview, the truthful answer is "the code loads PVTv2-B4 — the EfficientNet variant was discussed but not the one ultimately used in the final pipeline."

### Q: What does ZoomNeXt output? A box or a mask?
A **segmentation mask** (per-pixel probability that each pixel belongs to a camouflaged object). We then post-process: threshold the probabilities, find the connected region, and convert that to a bounding box with `BBOX_PADDING` (15%) breathing room. The classifier then crops on that box.

### Q: Why threshold at 0.5 and require coverage > 0.5%?
- `0.5` on the sigmoid is the standard default — anything above is "more likely object than background."
- `0.5%` minimum coverage filters out spurious tiny detections (single pixels lit up by noise). Below that fraction we treat the image as "no detection" and fall back to the full image.

These are conservative defaults. They could be tuned per-domain.

---

## 3. Stage 2 — Classification (Swin V2)

### Q: Why a transformer (Swin V2) instead of a CNN (ResNet, EfficientNet)?
For this task we need to read two things at once:

1. **Local texture** — does the uniform pattern match the surrounding pattern?
2. **Global context** — does the soldier-body silhouette stand out from the scene?

Swin's **shifted-window attention** processes overlapping patches that each see local context, while the hierarchical structure of the network builds up global context across layers. That combination is well-suited here. CNNs can do this too, but for fine-grained texture-vs-context tasks transformers tend to be stronger when pretrained on enough data.

### Q: Why the Small variant specifically?
| Model | Params | Trade-off |
|---|---|---|
| Swin V2-Tiny | 28M | Light, slightly weaker fine-grained discrimination |
| **Swin V2-Small** | ~50M | **Selected — sweet spot** |
| Swin-Base V1 | 88M | Marginal gains, much heavier, slower to train |

Beyond a certain size, **training strategy** (the augmentation pipeline + two-phase fine-tuning) mattered more than model size. A well-trained 50M model beat a poorly trained 88M one in early experiments.

### Q: Why pretrained on ImageNet-22K, not ImageNet-1K?
22K has more visual variety — more textures, lighting conditions, and scene types. Better starting features for transfer learning to an out-of-distribution domain like battlefield imagery.

---

## 4. Training strategy

### Q: Why two-phase (head-only, then full fine-tune)?
This is a standard transfer-learning recipe and exists to avoid a specific failure: if you unfreeze everything from epoch 1 with a randomly-initialized classification head, the large loss signal from the random head flows backward and **corrupts the pretrained features** before they can be useful.

**Phase 1 (8 epochs, LR 1e-3, backbone frozen):** the random head learns to map Swin's frozen features to `cam`/`no_cam`. Cheap, fast.

**Phase 2 (up to 20 epochs, LR 5e-5, everything unfrozen):** the whole network fine-tunes gently. Lower LR so we don't destroy the pretrained representations. Early stopping with patience 5 to avoid overfitting on a small dataset.

### Q: Why AdamW and not Adam or SGD?
AdamW is Adam with **decoupled weight decay** — meaning weight decay actually behaves like L2 regularization instead of being entangled with the adaptive learning-rate normalization (which is what happens with Adam). For transformers AdamW is the de facto choice; the empirical evidence for it is strong.

### Q: Why cosine annealing LR schedule?
Cosine decay starts high (fast initial learning) and smoothly decays to a low value (fine refinement near the end). It's simple, has no extra hyperparameters to tune, and works well empirically. Step schedules and ReduceLROnPlateau are alternatives — cosine is just the most common default for transformer fine-tuning.

### Q: Why label smoothing = 0.1?
Normal cross-entropy targets are hard: `[1.0, 0.0]`. The model is pushed to output extreme confidence (logits → ±∞) which leads to overconfidence on ambiguous cases. Label smoothing softens the target to `[0.95, 0.05]`. This teaches the model **to express uncertainty on borderline images** — exactly what we want for partially-camouflaged subjects.

0.1 is the standard value from the original paper. Higher values smooth too aggressively and hurt accuracy.

### Q: Why gradient clipping at norm = 1.0?
Transformers can have unstable gradient spikes early in fine-tuning. Clipping the gradient norm to 1.0 caps any individual update so a bad batch can't blow up the weights. It's a cheap safety net with effectively no downside.

### Q: Why batch size 24 and input size 256?
- Input 256 is what Swin V2-Small expects (`swinv2_small_window8_256` — the 256 is in the name).
- Batch 24 is what fits on a Colab T4 GPU (~16 GB) with this model and input size. Larger batches OOM; smaller batches give noisier gradients.

---

## 5. Augmentations — the most important choice in the project

### Q: Why augmentations at all?
Two reasons:
1. The dataset is small (~5,000 images). Augmentation effectively multiplies it.
2. **More importantly: to break the background shortcut.** This is the *actual* point.

### Q: Why these specific augmentations and not, say, the standard AutoAugment / RandAugment pipeline?
Generic augmentation policies (RandAugment, AutoAugment, AugMix) are designed for ImageNet-style classification where the subject is centered and the background is uninformative. They don't target our specific failure mode.

Our augmentations are **adversarially designed against the background shortcut**: every one of them either destroys background reliability or focuses the model on the centered body region.

| Augmentation | What it attacks |
|---|---|
| `CenterCrop70` | Removes background-heavy outer regions (soldier always centered after ZoomNeXt crop) |
| `EdgeDarken` (vignette) | Suppresses peripheral background brightness |
| `EdgeCutout` (black border strips) | Forces model to use center content only |
| `HueShift ±30°` | Breaks "green hue = forest = camo" correlation |
| `BilateralFilter` | Smooths background texture, keeps body edges sharp |
| `CLAHE` | Boosts local contrast where body meets environment |
| `MedianBlur k=7` | Destroys fine leaf texture; body silhouette survives |

### Q: Why not just stack lots of augmentations?
Early experiments did exactly that — aggressive multi-augmentation pipelines actually **hurt performance**. Each augmentation adds noise and reduces effective signal. A targeted, smaller set focused on the actual failure mode outperformed longer chains.

This is a general principle: augmentations are a tool, not a quantity to maximize.

### Q: Why probabilities like 0.3–0.5 instead of always-on?
If an augmentation always fires, the model only ever sees augmented images and the val/test distribution (clean images) becomes out-of-distribution. Probabilistic application means the model sees a mix and generalizes to both clean and perturbed inputs.

### Q: Why ImageNet normalization mean/std?
Because Swin V2 was pretrained on ImageNet, and its weights expect inputs normalized with ImageNet's per-channel statistics (`[0.485, 0.456, 0.406]` mean, `[0.229, 0.224, 0.225]` std). Mismatched normalization would silently degrade everything.

---

## 6. Inference — fusion and confidence

### Q: What does the inference pipeline actually do at runtime?
For an input image:
1. Run ZoomNeXt → segmentation mask + bounding box.
2. Classify the **full image** with Swin V2.
3. If a detection exists, also classify the **cropped region**.
4. Blend the two predictions using coverage-weighted averaging.
5. Emit a confidence verdict.

### Q: Why do both full and crop predictions?
This is the train/inference mismatch patch. The classifier was trained on crops, so:
- If detection works, the crop is the in-distribution input — trust it.
- If detection fails or finds something tiny, the full image is all we have — trust it.
- In between, blend.

### Q: How does the fusion weight work?
```
crop_weight = min(coverage / 50.0, 0.7)
fused = full_probs * (1 - crop_weight) + crop_probs * crop_weight
```

This means:
- Coverage 5% → crop_weight = 0.1 (mostly full image)
- Coverage 25% → crop_weight = 0.5 (equal blend)
- Coverage ≥35% → crop_weight = 0.7 (capped, mostly crop)

### Q: Are these numbers (50, 0.7) principled?
**No — they are hand-tuned heuristics.** This is worth being honest about. The 0.7 cap exists because we never want the full image's prediction completely ignored (the full image has scene context the crop loses). The /50 divisor was chosen so meaningful coverage (~25%) gives roughly equal weight. Both are tuned by intuition, not by data.

A proper solution would be to learn the fusion weights, or to train the classifier on a mix of full and crop inputs so it doesn't matter. This is a known limitation and future-work item.

### Q: Why three confidence buckets (75 / 65)?
- **≥75% (HIGH):** trust the prediction.
- **65–75% (LOW):** flag as unreliable.
- **<65% (UNCERTAIN):** route to human review.

These thresholds come from the label-smoothing setting. With smoothing=0.1, the model's maximum achievable confidence on training data is around 90–95% (not 100%), so 75% as the "trusted" line is meaningful, not stratospheric. They were chosen by inspecting validation prediction distributions.

For production use, **calibration** (e.g., temperature scaling, Platt scaling) would replace these hard thresholds with proper probabilistic confidences. That's a learning direction worth exploring.

---

## 7. GradCAM — the actual evaluation

### Q: Why is GradCAM treated as the primary metric instead of accuracy?
Because **accuracy is corruptible**. A model that learned "green pixels = camo" can hit high accuracy on a test set where that correlation holds, while being scientifically useless. GradCAM reveals what the model is actually attending to. For a research project, *interpretability is the credibility check.*

A 96% model focusing on trees is worse than a 90% model focusing on the soldier's body. The first will fail in deployment the moment the background shifts.

### Q: How does GradCAM work in one sentence?
It backprops the gradient of the predicted class score through to a chosen feature map, weighting each channel by how much it contributed to the prediction, then visualizing the weighted activation as a heatmap on the original image.

### Q: Why is the `reshape_transform` needed for Swin?
GradCAM was originally built for CNNs whose feature maps have shape `[batch, channels, height, width]`. Swin outputs tokens shaped `[batch, num_tokens, channels]` — a flat sequence, not a 2D grid. The `reshape_transform` un-flattens this back into a 2D feature map so GradCAM can produce a spatial heatmap.

> **Fragility warning.** Different `timm` versions output tokens in different layouts (`[B, HW, C]` vs `[B, H, W, C]`). The current code assumes the first. If you upgrade `timm` and GradCAM throws a "too many values to unpack" error, this is why. Pin your timm version (`timm==0.9.x`).

### Q: Which Swin layer is targeted and why?
`model.layers[-1].blocks[-1].norm2` — the LayerNorm before the MLP of the **last block in the last stage**. This is the deepest spatial-feature layer in the network; semantically the richest place to look.

---

## 8. Dataset

### Q: Why build a custom dataset?
There is no off-the-shelf labeled dataset for cam-vs-no_cam soldiers. COD datasets (CAMO/COD10K) have camouflaged objects but no "non-camouflaged" counterpart and the subjects are mostly animals. Building one was the only option.

### Q: Why class-balanced splits?
With imbalanced classes the model learns to predict the majority class to minimize loss, then "achieves" high accuracy by being useless. Balance forces the model to actually discriminate.

### Q: How big is too small?
~5,000 images is enough for fine-tuning a pretrained model, but it caps the achievable accuracy and limits how well the model generalizes to under-represented environments (desert, arctic, urban-night). More data is the single biggest lever for improving this project.

---

## 9. Limitations — be ready to volunteer these

A good engineer names their project's weaknesses before being asked. Here are this project's:

1. **Background bias likely persists.** The augmentations attack it; GradCAM checks it; but ~5K images can't eliminate it entirely.
2. **Fusion is a heuristic.** Two hard-coded numbers (50, 0.7) controlling inference behavior is a code smell. A learned or calibrated alternative is the right next step.
3. **Train/inference distribution mismatch.** Classifier trained on crops, sometimes inferred on full images.
4. **Dataset is small and skewed.** Forest-dominated; desert and arctic under-represented.
5. **No external benchmark comparison.** Results are self-reported on a custom test set, so absolute accuracy numbers can't be compared to published work.
6. **Single-frame only.** Real surveillance is video; temporal coherence would help.
7. **Two-stage means two failure points.** If ZoomNeXt misses, the classifier sees the full image and the train/inference mismatch hurts more.

---

## 10. What I would do next, given more time

Roughly in order of expected return on effort:

1. **Train the classifier on a mix of crops and full images** so the fusion heuristic isn't needed.
2. **Calibrate confidences** (temperature scaling on a held-out set) — replaces the 75/65 thresholds with real probabilities.
3. **Triple the dataset**, with deliberate desert/arctic/urban coverage.
4. **Use semantic segmentation of the soldier body** for classification (input is the masked region, not a bounding box) — this would more directly destroy background information.
5. **Cross-attention between soldier features and background features** as an explicit architectural component, instead of relying on data augmentation to force the relationship.
6. **Thermal/IR imagery** — camouflage that hides in visible light is often visible in thermal. Different but adjacent problem.

---

## 11. Things I would say in an interview

When asked "what was hardest about this project?" — the answer is not the model, it's the *evaluation discipline*. Getting comfortable with the idea that 96% accuracy can mean the model is wrong, and that GradCAM is the real signal, was the conceptual shift the project demanded.

When asked "did you train Swin / ZoomNeXt from scratch?" — no, both are loaded with pretrained weights and Swin was fine-tuned. That's the standard and correct approach; training a transformer from scratch on 5K images would be pointless.

When asked "what would you redo?" — the fusion heuristic. It works, but it's the part of the project I'm least proud of.

When asked "why this project?" — because camouflage detection forced me to confront a problem most classification tasks let you ignore: that a model can be right for the wrong reasons. Everything else in the design follows from taking that seriously.
