"""
Run the full pipeline on one image and print the verdict.

Usage:  python scripts/predict.py path/to/image.jpg
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import build_classifier
from src.detection import load_zoomnext
from src.inference import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]

    detector = load_zoomnext()
    classifier = build_classifier(pretrained=False)
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.eval()

    r = run_pipeline(image_path, detector, classifier)
    print("\n" + "=" * 50)
    print(f"  PREDICTION : {r['prediction'].upper()}  ({r['confidence']:.1f}%)")
    print(f"  cam {r['probs'][0]*100:.1f}%  |  no_cam {r['probs'][1]*100:.1f}%")
    print(f"  coverage   : {r['coverage']:.1f}%")
    print(f"  fusion     : {r['fusion_type']}")
    print(f"  verdict    : {r['verdict']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
