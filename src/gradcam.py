"""
GradCAM for the Swin V2 classifier — your most important sanity check.
High accuracy is meaningless if the heatmap lights up trees instead of the soldier.

CAVEAT: `swin_reshape_transform` assumes the target layer outputs tokens shaped
[B, H*W, C]. Some timm versions output [B, H, W, C] instead, which will raise a
"too many values to unpack" error. If that happens, reshape from the 4D layout
instead. Verify the shape for YOUR installed timm version before trusting results.
"""
import numpy as np
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from src.augmentations import build_eval_transform

_EVAL_TF = build_eval_transform()


def swin_reshape_transform(tensor, height=8, width=8):
    B, HW, C = tensor.size()           # expects [B, tokens, channels]
    h = w = int(HW ** 0.5)
    result = tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
    return F.interpolate(result, size=(height, width), mode="bilinear", align_corners=False)


def get_target_layer(model):
    return [model.layers[-1].blocks[-1].norm2]


def gradcam_overlay(model, img_pil, class_idx, device=config.DEVICE):
    """Return an RGB heatmap overlay for `class_idx` on the given image."""
    resized = img_pil.resize((config.INPUT_SIZE, config.INPUT_SIZE), Image.LANCZOS)
    rgb = np.array(resized).astype(np.float32) / 255.0
    tensor = _EVAL_TF(resized).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=get_target_layer(model),
                  reshape_transform=swin_reshape_transform)
    grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0, :]
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
    cam.__del__()
    return overlay
