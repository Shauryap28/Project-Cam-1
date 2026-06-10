"""
Stage 1 — Camouflaged Object Detection with ZoomNeXt.

Loads the pretrained ZoomNeXt model, runs multi-scale inference to produce a
segmentation mask, and converts that mask into a padded bounding box.

NOTE: this loads the PVTv2-B4 backbone (`PvtV2B4_ZoomNeXt`), which is what your
original notebook actually used. If you genuinely trained/selected the
EfficientNet-B4 variant, swap the import and the weights path in config.py and
fix the README table to match.
"""
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import config


def load_zoomnext(weights_path: str = config.ZOOMNEXT_WEIGHTS,
                  zoomnext_dir: str = config.ZOOMNEXT_DIR,
                  device=config.DEVICE):
    """Load ZoomNeXt (PVTv2-B4) with pretrained COD weights."""
    sys.path.insert(0, zoomnext_dir)
    from methods import PvtV2B4_ZoomNeXt  # noqa: E402  (path is set above)

    model = PvtV2B4_ZoomNeXt(pretrained=False).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("net") or ckpt
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ZoomNeXt loaded ({n_params:,} params)")
    return model


@torch.no_grad()
def predict_mask(image_path: str, model, threshold: float = config.DET_THRESHOLD,
                 device=config.DEVICE):
    """
    Multi-scale (0.5x / 1.0x / 1.5x) forward pass -> probability map -> binary mask.
    Returns (binary_mask, prob_map, has_object).
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)

    bw, bh = config.DET_BASE_W, config.DET_BASE_H
    scales = {
        "image_s": cv2.resize(img_np, (int(bw * 0.5), int(bh * 0.5))),
        "image_m": cv2.resize(img_np, (bw, bh)),
        "image_l": cv2.resize(img_np, (int(bw * 1.5), int(bh * 1.5))),
    }
    data = {
        k: torch.from_numpy(v).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
        for k, v in scales.items()
    }

    logits = model(data)
    prob_map = torch.sigmoid(logits)
    prob_map = F.interpolate(prob_map, size=(orig_h, orig_w),
                             mode="bilinear", align_corners=False)
    prob_map = prob_map.squeeze().cpu().numpy()

    mask = (prob_map > threshold).astype(np.uint8)
    has_object = (mask.sum() / mask.size) > config.DET_MIN_COVERAGE
    return mask, prob_map, has_object


def get_bounding_box(mask: np.ndarray, padding: float = config.BBOX_PADDING):
    """Tight box around the mask, expanded by `padding` on each side. None if empty."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h, w = mask.shape
    py = int((y2 - y1) * padding)
    px = int((x2 - x1) * padding)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))
