"""Training engine: one-epoch loop, evaluation, freeze helpers, two-phase trainer."""
import os
import torch
import torch.nn as nn

import config


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = "head" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Frozen — trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True
    print(f"  Unfrozen — trainable: {sum(p.numel() for p in model.parameters()):,} (100%)")


def train_one_epoch(model, loader, criterion, optimizer, device=config.DEVICE):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device=config.DEVICE):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total * 100, all_preds, all_labels


def two_phase_train(model, train_loader, val_loader, device=config.DEVICE):
    """Phase 1 = head only (backbone frozen). Phase 2 = full fine-tune with early stopping."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phase": []}

    # ---- Phase 1 ----
    print("\n=== PHASE 1: head only ===")
    freeze_backbone(model)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.PHASE1_LR, weight_decay=0.01)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=config.PHASE1_EPOCHS, eta_min=1e-5)
    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, opt1, device)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        sched1.step()
        for k, v in zip(history, [tl, ta, vl, va, 1]):
            history[k].append(v)
        print(f"  E{epoch:02d}  train {ta:5.1f}%  val {va:5.1f}%")

    # ---- Phase 2 ----
    print("\n=== PHASE 2: full fine-tune ===")
    unfreeze_all(model)
    opt2 = torch.optim.AdamW(model.parameters(), lr=config.PHASE2_LR, weight_decay=config.WEIGHT_DECAY)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=config.PHASE2_EPOCHS, eta_min=1e-7)
    best_val, patience = 0.0, 0
    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, opt2, device)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        sched2.step()
        for k, v in zip(history, [tl, ta, vl, va, 2]):
            history[k].append(v)

        if va > best_val:
            best_val, patience = va, 0
            torch.save({"epoch": epoch, "phase": 2,
                        "model_state_dict": model.state_dict(),
                        "val_acc": va, "val_loss": vl}, config.BEST_MODEL_PATH)
            tag = "BEST (saved)"
        else:
            patience += 1
            tag = f"patience {patience}/{config.PATIENCE}"
        print(f"  E{epoch:02d}  train {ta:5.1f}%  val {va:5.1f}%  {tag}")
        if patience >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"\nBest val acc: {best_val:.1f}%  ->  {config.BEST_MODEL_PATH}")
    return history, best_val
