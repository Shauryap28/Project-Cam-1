"""Dataset + dataloaders for the cropped cam/no_cam images."""
import os
from pathlib import Path
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config
from src.augmentations import build_train_transform, build_eval_transform

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CamoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images, self.labels = [], []
        self.class_to_idx = {"cam": 0, "no_cam": 1}
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for f in Path(cls_dir).iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.images.append(str(f))
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_loaders(cropped_root=config.CROPPED_ROOT, batch_size=config.BATCH_SIZE):
    train_tf, eval_tf = build_train_transform(), build_eval_transform()
    train_ds = CamoDataset(os.path.join(cropped_root, "train"), train_tf)
    val_ds = CamoDataset(os.path.join(cropped_root, "val"), eval_tf)
    test_ds = CamoDataset(os.path.join(cropped_root, "test"), eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Train class balance -> {dict(Counter(train_ds.labels))}")
    return train_loader, val_loader, test_loader
