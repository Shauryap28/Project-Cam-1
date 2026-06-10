"""
Train the Stage 2 classifier (two-phase).

Usage:  python scripts/train.py
"""
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import build_loaders
from src.model import build_classifier
from src.engine import two_phase_train


def set_seed(seed=config.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    print(f"Device: {config.DEVICE}")
    train_loader, val_loader, _ = build_loaders()
    model = build_classifier()
    two_phase_train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
