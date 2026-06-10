"""
Background-breaking augmentations for Stage 2.

The whole point of this file: stop the classifier from learning the shortcut
"green background = camouflage". Each transform either removes background at the
edges, breaks colour cues, or smooths texture while keeping body edges. They are
plain PIL/numpy/cv2 callables that drop straight into torchvision Compose.
"""
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

import config


class CenterCrop70:
    """Randomly crop to the centre 60-85% — the soldier is centred after ZoomNeXt,
    so the edges are mostly background and safe to throw away."""
    def __init__(self, p=0.5, min_ratio=0.6, max_ratio=0.85):
        self.p, self.min_ratio, self.max_ratio = p, min_ratio, max_ratio

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        w, h = img.size
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        nw, nh = int(w * ratio), int(h * ratio)
        left, top = (w - nw) // 2, (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh))


class EdgeDarken:
    """Vignette: centre stays bright, edges fade toward black so peripheral
    background carries little signal."""
    def __init__(self, p=0.4, strength=0.4):
        self.p, self.strength = p, strength

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.clip(np.sqrt((X - cx) ** 2 / cx ** 2 + (Y - cy) ** 2 / cy ** 2), 0, 1)
        mask = (1.0 - self.strength * dist)[:, :, None]
        return Image.fromarray((arr * mask).clip(0, 255).astype(np.uint8))


class EdgeCutout:
    """Black out 1-2 random border strips so the model can't lean on sky or
    side vegetation."""
    def __init__(self, p=0.5, max_strip=0.25):
        self.p, self.max_strip = p, max_strip

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        for edge in random.sample(["top", "bottom", "left", "right"], random.randint(1, 2)):
            s = random.uniform(0.08, self.max_strip)
            if edge == "top":
                arr[:int(h * s), :] = 0
            elif edge == "bottom":
                arr[int(h * (1 - s)):, :] = 0
            elif edge == "left":
                arr[:, :int(w * s)] = 0
            else:
                arr[:, int(w * (1 - s)):] = 0
        return Image.fromarray(arr)


class HueShift:
    """Rotate hue so 'green = forest = camo' stops being reliable."""
    def __init__(self, max_shift=30, p=0.4):
        self.max_shift, self.p = max_shift, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-self.max_shift, self.max_shift)) % 180
        return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


class BilateralFilter:
    """Smooth flat regions, keep body edges sharp."""
    def __init__(self, d=9, sigma_color=75, sigma_space=75, p=0.3):
        self.d, self.sigma_color, self.sigma_space, self.p = d, sigma_color, sigma_space, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return Image.fromarray(
            cv2.bilateralFilter(np.array(img), self.d, self.sigma_color, self.sigma_space))


class CLAHETransform:
    """Local contrast boost so the body region pops out of the background."""
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8), p=0.3):
        self.clip_limit, self.tile_grid, self.p = clip_limit, tile_grid, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class MedianBlurTransform:
    """Remove fine leaf texture, keep overall body shape."""
    def __init__(self, kernel_size=7, p=0.25):
        self.kernel_size, self.p = kernel_size, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return Image.fromarray(cv2.medianBlur(np.array(img), self.kernel_size))


def build_train_transform(input_size=config.INPUT_SIZE):
    return transforms.Compose([
        transforms.Resize((input_size + 20, input_size + 20)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        CenterCrop70(p=0.5, min_ratio=0.65, max_ratio=0.85),
        EdgeDarken(p=0.4, strength=0.4),
        EdgeCutout(p=0.4, max_strip=0.2),
        HueShift(max_shift=30, p=0.4),
        BilateralFilter(p=0.3),
        CLAHETransform(p=0.3),
        MedianBlurTransform(kernel_size=7, p=0.25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def build_eval_transform(input_size=config.INPUT_SIZE):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])
