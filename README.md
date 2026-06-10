# Camouflaged Soldier Detection & Classification

A two-stage deep-learning pipeline that finds a soldier in an image and decides whether they are **effectively camouflaged** (hard to spot) or **clearly visible**. Built from scratch on a custom dataset, with a deliberate focus on *why* the model makes its decision — not just accuracy.

```
Raw image  ─►  Stage 1: DETECT (ZoomNeXt)  ─►  crop  ─►  Stage 2: CLASSIFY (Swin V2)  ─►  cam / no_cam
```

---

## Why two stages?

Camouflage classification is not normal image classification. The subject is *designed* to be hard to see, often occupying a small, low-contrast region, and the background (e.g. a forest) is correlated with the label but must **not** be the deciding factor. A soldier standing openly in a forest is `no_cam`.

A single classifier on raw images tends to learn the shortcut *"green background = camouflage"*. Detecting and cropping the soldier first removes most of that background noise before classification.

| Stage | Model | Job |
|-------|-------|-----|
| 1 | **ZoomNeXt** (PVTv2-B4 backbone) | Camouflaged object detection → segmentation mask → bounding box |
| 2 | **Swin Transformer V2 (Small)** | Classify the cropped region: `cam` vs `no_cam` |

> **Note on the detector backbone:** this code loads `PvtV2B4_ZoomNeXt` with the `pvtv2-b4-zoomnext.pth` weights, i.e. the **PVTv2-B4** backbone. If you intend to use a different backbone, change `ZOOMNEXT_WEIGHTS` / the import in `config.py` and `src/detection.py`.

---

## Repository structure

```
camo-soldier-detection/
├── config.py              # ALL paths + hyperparameters live here
├── requirements.txt
├── src/
│   ├── detection.py       # Stage 1: ZoomNeXt load / mask / bbox
│   ├── augmentations.py   # background-breaking transforms + pipeline
│   ├── dataset.py         # CamoDataset + dataloaders
│   ├── model.py           # Swin V2 classifier builder
│   ├── engine.py          # train/eval loops + two-phase trainer
│   ├── inference.py       # full pipeline + coverage fusion + verdict
│   └── gradcam.py         # GradCAM (the real evaluation tool)
└── scripts/
    ├── crop_dataset.py    # 1. run ZoomNeXt over split dataset
    ├── train.py           # 2. train the classifier
    ├── evaluate.py        # 3. test-set metrics
    └── predict.py         # 4. single-image inference
```

## Quick start

```bash
pip install -r requirements.txt

# edit paths in config.py first, then:
python scripts/crop_dataset.py        # Stage 1 → cropped dataset
python scripts/train.py               # two-phase fine-tune
python scripts/evaluate.py            # confusion matrix + report
python scripts/predict.py image.jpg   # one image
```

---

## Stage 2 training strategy

**Phase 1 — head only (8 epochs):** backbone frozen, only the classification head trains, so the ImageNet features aren't destroyed early.

**Phase 2 — full fine-tune (up to 20 epochs):** everything unfrozen at a low LR, early stopping (patience 5), best checkpoint saved on validation accuracy.

### Augmentations (the important part)

These are chosen specifically to **break the background shortcut**, not to generically inflate the dataset:

| Augmentation | Purpose |
|---|---|
| `CenterCrop70` | drop background-heavy edges |
| `EdgeDarken` / `EdgeCutout` | suppress / remove peripheral background |
| `HueShift` | breaks "green = camo" colour cue |
| `BilateralFilter` / `CLAHE` / `MedianBlur` | smooth texture, keep body edges |
| Label smoothing (0.1) | express uncertainty on ambiguous cases |

---

## Inference: detection → classification → fusion

At inference the classifier sees a domain it wasn't trained on when detection fails (it was trained on *crops*). To handle that, `src/inference.py` classifies **both** the full image and the crop, then blends them with a coverage-weighted average (`crop_weight` grows with mask coverage, capped at 0.7), and emits a confidence verdict (high / low / human-review).

> This fusion is a **hand-tuned heuristic**, not a learned component. It's a pragmatic patch for the train/inference mismatch — treat it as such, and see *Limitations*.

---

## Evaluation — read this before trusting accuracy

**GradCAM is the primary metric here, not accuracy.** A 96%-accurate model that lights up trees is less scientifically valid than a 90% model that attends to the soldier.

- Heat on the soldier body / uniform → correct features learned ✅
- Heat on trees / sky → background shortcut ❌
- Identical heatmap for both classes → model is confused ❌

> Report exact numbers from a single fixed run (seeded) rather than ranges. Always pair the test accuracy with GradCAM examples for at least a few `cam` and `no_cam` images.

---

## Dataset

Custom-built (no pre-existing labelled cam/no_cam soldier dataset exists). Images sourced across forest / desert / urban / snow, manually labelled, split into train/val/test, then cropped via ZoomNeXt. Layout:

```
dataset/{train,val,test}/{cam,no_cam}/
```

The set is class-balanced to avoid prediction bias.

## Limitations

- Train/inference mismatch when fed full raw images (the fusion mitigates, doesn't fix, this).
- Dense vegetation + ghillie suits lower confidence.
- ~5k images caps the accuracy ceiling; desert/arctic are under-represented.
- Some background bias likely persists despite the augmentations — verify with GradCAM.

## Future work

Semantic segmentation before classification; more terrain diversity; attention that explicitly models the soldier-background relationship; thermal/IR imagery; real-time video.

## Citation

```bibtex
@misc{camouflage-detection-2026,
  title  = {Camouflaged Soldier Detection and Classification using ZoomNeXt and Swin Transformer V2},
  author = {Shaurya Pratap Singh},
  year   = {2026}
}
```

**Acknowledgments:** ZoomNeXt (detection backbone), Swin Transformer V2 (classifier), `timm`, `pytorch-grad-cam`.
