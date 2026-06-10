"""
Stage 1 batch job: run ZoomNeXt over the split dataset and save cropped soldier
regions. Falls back to the full image (resized) when nothing is detected.

Usage:  python scripts/crop_dataset.py
"""
import os
import sys
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.detection import load_zoomnext, predict_mask, get_bounding_box

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    detector = load_zoomnext()
    if os.path.exists(config.CROPPED_ROOT):
        shutil.rmtree(config.CROPPED_ROOT)

    stats = {"cropped": 0, "full": 0, "errors": 0}
    for split in ["train", "val", "test"]:
        for cls in ["cam", "no_cam"]:
            src_dir = os.path.join(config.SPLIT_DATASET_ROOT, split, cls)
            dst_dir = os.path.join(config.CROPPED_ROOT, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            images = [f for f in Path(src_dir).iterdir() if f.suffix.lower() in IMG_EXTS]

            for img_path in tqdm(images, desc=f"{split}/{cls}"):
                try:
                    mask, _, has_object = predict_mask(str(img_path), detector)
                    img = Image.open(str(img_path)).convert("RGB")
                    bbox = get_bounding_box(mask) if has_object else None
                    if bbox:
                        out = img.crop(bbox)
                        stats["cropped"] += 1
                    else:
                        out = img
                        stats["full"] += 1
                    out = out.resize((config.CROP_SIZE, config.CROP_SIZE), Image.LANCZOS)
                    out.save(os.path.join(dst_dir, img_path.name), quality=95)
                except Exception as e:
                    print(f"  ERROR {img_path.name}: {e}")
                    stats["errors"] += 1

    print(f"\nCropped: {stats['cropped']} | Full: {stats['full']} | Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
