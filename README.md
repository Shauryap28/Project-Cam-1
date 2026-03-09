# Project-Cam-1
My first git repo

# Camouflaged Soldier Detection and Classification

A deep learning pipeline that detects and classifies soldiers as camouflaged or non-camouflaged in real-world battlefield images. This was built from scratch as a research project exploring how computer vision can be used to identify hidden soldiers in natural environments.

---

## What This Project Does

Military camouflage is designed to make soldiers blend into their surroundings — matching colors, breaking body outlines, and disrupting visual patterns. The goal of this project is to build an AI system that can look at an image and determine whether a soldier is effectively camouflaged (hard to spot) or clearly visible (easy to spot).

### Definitions

**Camouflaged (cam):** A soldier who is hidden or difficult to spot in the environment. Their uniform, gear, or ghillie suit blends with the surrounding vegetation, terrain, or structures. Even a human observer might struggle to locate them at first glance.

**Non-Camouflaged (no_cam):** A soldier who is barely hidden or easy to spot. Their uniform or gear stands out from the background — whether due to color contrast, visible equipment, or simply being in an open area where blending is not effective.

The line between these two categories is not always clean. A soldier might be partially hidden, or their camouflage might work in one environment but fail in another. This ambiguity is what makes the problem genuinely interesting.

---

## Why This Approach?

### The Problem with Simple Classification

The obvious first instinct would be to throw images into a classifier and call it done. But camouflage detection is fundamentally different from typical image classification:

1. **The subject is intentionally hidden.** Unlike classifying cats vs dogs where both are clearly visible, a well-camouflaged soldier might occupy only a tiny, blurry region of the image.

2. **Background matters — but shouldn't dominate.** A forest background doesn't automatically mean camouflage. A soldier in combat gear standing openly in a forest is NOT camouflaged. The model needs to understand the relationship between the soldier and their environment.

3. **Standard object detectors struggle.** YOLO, Faster R-CNN, and similar detectors are trained to find visible objects. They perform poorly on subjects that are designed to be invisible.

### Two-Stage Pipeline

To handle this, the project uses a two-stage approach:

```
Raw Image -> Stage 1: DETECT the soldier -> Stage 2: CLASSIFY cam vs no_cam
```

**Stage 1 — ZoomNeXt (Camouflaged Object Detection)**
Finds the soldier in the image, even if they're hidden. Outputs a segmentation mask showing where the camouflaged object is.

**Stage 2 — Swin Transformer V2 (Classification)**
Takes the detected region and classifies whether the soldier is effectively camouflaged or clearly visible.

---

## Stage 1: ZoomNeXt — Finding the Hidden

### Why ZoomNeXt?

A model specifically designed to find objects that don't want to be found was needed here. Regular object detectors fail in this context because camouflaged subjects share color, texture, and pattern with their background.

[ZoomNeXt](https://github.com/lartpang/ZoomNeXt) is a state-of-the-art Camouflaged Object Detection (COD) model that works by:

- Looking at images at multiple zoom levels (similar to how a human would scan an area — first the big picture, then zooming into suspicious regions)
- Using edge-aware processing to find subtle boundary disruptions where a body meets the background
- Leveraging hierarchical features to detect objects that blend with surroundings

### Backbone Selection

ZoomNeXt supports multiple backbones. Several were tested and compared:

| Backbone | Parameters | Performance | Selected |
|----------|-----------|-------------|----------|
| ResNet-50 | 25M | Good baseline | No |
| PVTv2-B4 | 62M | Excellent detail | No |
| EfficientNet-B4 | 19M | Best balance | Yes |

**EfficientNet-B4** (referred to as ZoomNeXt-B4 throughout) was selected because:
- Best accuracy-to-speed ratio for this use case
- Strong multi-scale feature extraction
- Good at detecting both large and small camouflaged regions
- Efficient enough to run on consumer GPUs

### Pre-trained Weights

ZoomNeXt weights pre-trained on standard COD benchmarks (CHAMELEON, CAMO, COD10K) are used. These datasets contain various camouflaged objects — animals, insects, and soldiers — which gives the model a strong foundation for finding hidden subjects.

---

## Stage 2: Swin Transformer V2 — Making the Call

### Why Swin Transformer?

Once the soldier region is found and cropped, a classifier is needed that can pick up on subtle visual cues:
- Is the clothing texture matching the background?
- Are body edges disrupted or clear?
- Does the color palette blend or contrast?

**Swin Transformer V2** was chosen for a few reasons:

1. **Shifted window attention** — processes the image in overlapping patches, capturing both local texture details (camo pattern) and global context (how the soldier relates to surroundings)

2. **Hierarchical features** — builds understanding from fine details (fabric texture) to broad patterns (body shape vs environment)

3. **Pre-trained on ImageNet-22K** — starts with a strong understanding of natural scenes, textures, and objects

### Model Variants Tested

| Model | Parameters | Val Acc | Test Acc | Notes |
|-------|-----------|---------|----------|-------|
| Swin-Base V1 | 88M | 95.2% | — | Too heavy, marginal gain |
| Swin V2-Tiny | 28M | 94.3% | 96.3% | Lightweight, excellent |
| Swin V2-Small | 50M | 94.8% | 95.7% | Better body focus (GradCAM) |

### Training Strategy

A two-phase training approach is used:

**Phase 1 — Head Only (backbone frozen)**
- Only the classification head is trained
- Backbone (pre-trained on ImageNet) stays frozen
- This lets the head learn to map Swin's features to the cam/no_cam classes
- Prevents the backbone from forgetting useful general features too early
- Typically runs for 8 epochs

**Phase 2 — Full Fine-tuning (everything unfrozen)**
- All layers become trainable
- Lower learning rate to fine-tune without destroying pre-trained knowledge
- Early stopping with patience of 5 epochs
- Saves the best model based on validation accuracy
- Typically runs for 10-15 epochs before early stopping kicks in

### Augmentation Philosophy

The augmentations here are specifically designed to force the model to learn soldier body features rather than taking shortcuts based on background patterns:

| Augmentation | Purpose |
|-------------|---------|
| BilateralFilter | Smooths vegetation texture while keeping body edges sharp |
| CLAHE | Enhances local contrast so the soldier body stands out |
| MedianBlur (k=7) | Removes fine leaf details, preserves body shape |
| HueShift | Randomly changes colors — breaks the "green = forest = camo" shortcut |
| CenterCrop | Removes background-heavy edges, focuses on soldier in center |
| EdgeDarken | Vignette effect suppresses peripheral background |
| EdgeCutout | Blacks out border strips — forces the model to use center content |
| Label Smoothing (0.1) | Teaches the model to express uncertainty on ambiguous cases |

Without these, the model tends to learn background shortcuts — for example, "forest background = camouflage." These augmentations destroy background reliability and force the model to focus on the soldier-environment relationship instead.

---

## Dataset

### Custom Dataset

This dataset was built from scratch. There is no pre-existing labeled dataset for camouflaged vs non-camouflaged soldiers.

**Collection process:**
- Sourced soldier images from various environments (forests, deserts, urban areas, snow)
- Manually labeled each image as `cam` (camouflaged) or `no_cam` (non-camouflaged)
- Used ZoomNeXt to detect and crop soldier regions from raw images
- Created clean train/val/test splits

**Dataset structure:**
```
dataset/
├── train/
│   ├── cam/
│   └── no_cam/
├── val/
│   ├── cam/
│   └── no_cam/
└── test/
    ├── cam/
    └── no_cam/
```

**Dataset statistics:**

| Split | Cam | No_Cam | Total |
|-------|-----|--------|-------|
| Train | ~2,000 | ~2,000 | ~4,000 |
| Val | ~250 | ~250 | ~500 |
| Test | ~258 | ~259 | ~517 |

The dataset is balanced — roughly equal numbers of cam and no_cam images in each split. This prevents the model from being biased toward predicting one class.

---

## Results

### Classification Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~95-96% |
| Cam Recall | ~95-96% |
| No_Cam Recall | ~96-97% |
| Total Test Errors | ~19-22 / 517 |

### GradCAM Analysis

GradCAM (Gradient-weighted Class Activation Mapping) is used to visualize what the model is actually looking at when making predictions. This is arguably the most important part of a research project like this — high accuracy means nothing if the model is cheating by looking at background instead of the soldier.

**What to look for:**
- Red/yellow on soldier body → model learned the correct features
- Red on uniform/gear → model is correctly identifying clothing patterns
- Red on trees/sky → model learned the wrong features (background shortcut)
- Same heatmap for both classes → model is confused

The augmentation strategy — center focus, hue shift, edge removal — is specifically designed to push GradCAM attention toward the soldier body.

---

## Getting Started

### Requirements

```
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8 (GPU recommended)
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
grad-cam>=1.4.0
```

### Pre-trained Weights

- **ZoomNeXt-B4:** Available from the [ZoomNeXt repository](https://github.com/lartpang/ZoomNeXt) — EfficientNet-B4 backbone weights
- **Swin V2-Small:** Available in this repo under `checkpoints/` (trained on the custom dataset)

### Quick Inference

```python
# Load models
zoomnext = load_zoomnext("path/to/zoomnext_b4.pth")
swin = load_swin("path/to/swin_v2_small_best.pth")

# Run pipeline
mask, prob_map, has_object = predict_mask("soldier_image.jpg", zoomnext)
bbox = get_bounding_box(mask)
cropped = Image.open("soldier_image.jpg").crop(bbox)
prediction, confidence = classify(cropped, swin)

print(f"Result: {prediction} ({confidence:.1f}%)")
```

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE

```

---

## Technical Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Size | 256 x 256 |
| Batch Size | 24 |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 5e-5 |
| Weight Decay | 0.05 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Early Stopping Patience | 5 |
| Label Smoothing | 0.1 |
| Gradient Clipping | max_norm=1.0 |

### Hardware Used

- **Training:** Google Colab (T4/A100 GPU)
- **Training Time:** Approximately 30-45 minutes total (Phase 1 + Phase 2)
- **Inference:** Roughly 0.5 seconds per image (ZoomNeXt + Swin combined)

---

## Research Insights

### What This Project Revealed

**1. Background bias is a real problem.**
Without careful augmentation, models consistently learned "forest = camouflage" instead of actually analyzing the soldier. GradCAM heatmaps made this failure mode immediately obvious. This is probably the most important thing the project confirmed experimentally.

**2. Two-stage beats one-stage by a large margin.**
A single classifier trained directly on raw images performed at around 65-70% accuracy. Detecting first, cropping the soldier region, then classifying improved accuracy by roughly 25-30 percentage points. The crop removes most of the noise.

**3. Model size matters less than training strategy.**
The 28M parameter Swin V2-Tiny performed comparably to the 88M Swin-Base V1. For this kind of fine-grained task, a well-trained smaller model can match or beat a poorly trained larger one. The two-phase training and augmentation setup mattered much more than simply using a bigger backbone.

**4. Targeted augmentation beats stacking many augmentations.**
Early experiments with aggressive augmentation pipelines actually hurt performance. A smaller set of augmentations specifically designed to break background shortcuts — bilateral filter, CLAHE, hue shift — consistently outperformed longer augmentation chains.

**5. Accuracy is not the right final metric for this task.**
A model with 96% accuracy that consistently focuses on background is less scientifically valid than a model with 90% accuracy that correctly attends to the soldier-environment relationship. GradCAM analysis should be treated as the primary evaluation criterion for research credibility.

### Known Limitations

- The model can struggle when full raw images are used instead of pre-cropped soldier regions, due to training-inference mismatch
- Dense vegetation combined with ghillie suits tends to produce lower confidence (~80%), though the predictions are usually still correct
- Dataset size (~5000 images total) limits the accuracy ceiling
- Some background bias likely persists despite the augmentation strategy

### Future Directions

- Segment the soldier body from background before classification using semantic segmentation, rather than relying on bounding box crops
- Collect more diverse data across different terrains and seasons, particularly desert and arctic environments which are underrepresented
- Explore attention mechanisms that explicitly model the soldier-background relationship rather than treating the crop as a standalone image
- Investigate thermal and infrared imagery where standard camouflage behaves very differently
- Real-time inference from video input using a streaming pipeline

---

## Citation

If this work is useful for your research, please cite:

```
@misc{camouflage-detection-2026,
  title={Camouflaged Soldier Detection and Classification using ZoomNeXt and Swin Transformer V2},
  author={Shaurya Pratap Singh},
  year={2026},
  url={https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

---

## Acknowledgments

- [ZoomNeXt](https://github.com/lartpang/ZoomNeXt) for the camouflaged object detection backbone
- [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer) for the classification architecture
- [timm](https://github.com/huggingface/pytorch-image-models) for the PyTorch Image Models library
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for the GradCAM implementation

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

The MIT License is a short, permissive open-source license that allows anyone to freely use, copy, modify, and distribute this code — including for commercial purposes — as long as the original copyright notice is included. It does not require derivative works to be open-sourced. For a research project like this, MIT is a reasonable choice: it lowers the barrier for other students and researchers to build on the work without legal friction.

---

## Author

**Shaurya Pratap Singh**  
B.Tech, Amity University Noida  

