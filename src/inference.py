"""
Full inference: ZoomNeXt detect -> Swin classify (full + crop) -> coverage-weighted
fusion -> confidence verdict.

The "fusion" exists because the classifier is trained on cropped soldiers, but at
inference time detection can fail or be partial. When there is a confident crop we
lean on it; otherwise we fall back to the full image. The weighting is a heuristic
(crop_weight scales with coverage), not a learned component — see README caveats.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from src.augmentations import build_eval_transform
from src.detection import predict_mask, get_bounding_box

_EVAL_TF = build_eval_transform()


def classify(img_pil, model, device=config.DEVICE):
    """Return softmax probabilities over CLASS_NAMES for a single PIL image."""
    resized = img_pil.resize((config.INPUT_SIZE, config.INPUT_SIZE), Image.LANCZOS)
    tensor = _EVAL_TF(resized).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    return probs, resized, tensor


def run_pipeline(image_path, detector, classifier, device=config.DEVICE):
    """Returns a dict with the prediction, probabilities, coverage, fusion info, verdict."""
    mask, prob_map, has_object = predict_mask(image_path, detector, device=device)
    original = Image.open(image_path).convert("RGB")
    coverage = mask.sum() / mask.size * 100

    full_probs, _, _ = classify(original, classifier, device)

    crop_probs, bbox = None, None
    if has_object:
        bbox = get_bounding_box(mask)
        if bbox:
            crop_probs, _, _ = classify(original.crop(bbox), classifier, device)

    # Coverage-weighted fusion
    if crop_probs is not None and coverage >= config.FUSION_MIN_COVERAGE:
        crop_w = min(coverage / 50.0, config.FUSION_MAX_CROP_WEIGHT)
        fused = full_probs * (1 - crop_w) + crop_probs * crop_w
        fused /= fused.sum()
        fusion_type = f"FUSED (full {1-crop_w:.0%} + crop {crop_w:.0%})"
    else:
        fused = full_probs
        fusion_type = "FULL ONLY"

    pred_idx = int(fused.argmax())
    confidence = fused[pred_idx] * 100
    if confidence >= config.HIGH_CONF:
        verdict = "HIGH CONFIDENCE — trusted"
    elif confidence >= config.LOW_CONF:
        verdict = "LOW CONFIDENCE — may be unreliable"
    else:
        verdict = "UNCERTAIN — human review needed"

    return {
        "prediction": config.CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probs": fused,
        "full_probs": full_probs,
        "crop_probs": crop_probs,
        "coverage": coverage,
        "fusion_type": fusion_type,
        "verdict": verdict,
        "mask": mask,
        "prob_map": prob_map,
        "bbox": bbox,
        "has_object": has_object,
    }
