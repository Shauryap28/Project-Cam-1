"""
Evaluate the best checkpoint on the test split.

Usage:  python scripts/evaluate.py
"""
import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import build_loaders
from src.model import build_classifier
from src.engine import evaluate


def main():
    _, _, test_loader = build_loaders()
    model = build_classifier(pretrained=False)
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.1f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\nTest accuracy: {test_acc:.2f}%   Test loss: {test_loss:.4f}\n")
    names = ["camouflage", "no_camouflage"]
    print(classification_report(labels, preds, target_names=names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
