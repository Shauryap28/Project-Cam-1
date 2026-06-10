"""Stage 2 classifier — Swin Transformer V2 (small) via timm."""
import timm
import config


def build_classifier(name=config.CLASSIFIER_NAME,
                     num_classes=config.NUM_CLASSES,
                     pretrained=True,
                     device=config.DEVICE):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name} loaded ({n_params:,} params, {num_classes} classes)")
    return model
