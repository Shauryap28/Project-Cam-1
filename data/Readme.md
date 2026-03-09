# Dataset

## About

This is a custom dataset built from scratch for camouflaged soldier detection and classification.

The dataset is NOT included in this repository due to size. Contact the author for access.

## Structure

```
dataset/
├── train/
│   ├── cam/       # ~2000 cropped camouflaged soldier images
│   └── no_cam/    # ~2000 cropped non-camouflaged soldier images
├── val/
│   ├── cam/       # ~250 images
│   └── no_cam/    # ~250 images
└── test/
    ├── cam/       # ~258 images
    └── no_cam/    # ~259 images
```

## How Images Were Created

1. Raw battlefield/military images collected from various sources
2. ZoomNeXt-B4 used to detect soldier regions (camouflaged object detection)
3. Detected regions cropped using bounding boxes from segmentation masks
4. Each crop manually verified and labeled as `cam` or `no_cam`
5. Split into train/val/test with stratified sampling

## Labeling Criteria

- **cam**: Soldier is wearing camouflage that effectively blends with the environment. A human observer would need more than a casual glance to spot them.
- **no_cam**: Soldier is clearly visible. Their uniform or positioning makes them easy to identify against the background .
