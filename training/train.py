#!/usr/bin/env python3
"""
Fine-tune the Prithvi change-detection head on Rondônia tiles.

Designed to run on Kaggle free GPU (T4 / P100).  On a local CPU it will
still run but slowly — useful for smoke-testing with a tiny epoch count.

Usage (from project root):
    python -m training.train [--epochs N] [--batch-size B] [--tiles-dir PATH]
                             [--output-dir PATH] [--no-pretrained] [--cpu]

Outputs:
    training/checkpoints/best_model.pth   (gitignored)
    training/checkpoints/train_log.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.model   import PrithviChangeDetector
from training.dataset import make_splits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    """BCE with positive-class weight to handle deforestation class imbalance."""
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1  = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"f1": f1.item(), "iou": iou.item(),
            "precision": precision.item(), "recall": recall.item()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for before, after, labels in loader:
        before, after, labels = before.to(device), after.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(before, after)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for before, after, labels in loader:
        before, after, labels = before.to(device), after.to(device), labels.to(device)
        logits = model(before, after)
        total_loss += criterion(logits, labels).item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    metrics = binary_metrics(
        torch.cat(all_logits), torch.cat(all_labels)
    )
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="BCE positive-class weight (deforestation pixels are rare)")
    parser.add_argument("--tiles-dir",  type=str,
                        default="data_pipeline/tiles")
    parser.add_argument("--output-dir", type=str,
                        default="training/checkpoints")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip loading Prithvi weights (for quick smoke tests)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU is available")
    args = parser.parse_args()

    # Device
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Dirs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"Loading tiles from {args.tiles_dir} ...")
    train_ds, val_ds = make_splits(args.tiles_dir)
    print(f"  Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model
    print("Building model...")
    model = PrithviChangeDetector(
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    ).to(device)
    print(f"  Trainable params : {model.trainable_param_count():,}")
    print(f"  Total params     : {model.total_param_count():,}")

    # Optimizer: only head parameters
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = WeightedBCELoss(pos_weight=args.pos_weight)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_f1 = 0.0
    log = []

    print(f"\nTraining for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        t0       = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"F1={val_metrics['f1']:.4f} | "
            f"IoU={val_metrics['iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        log.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"  ✓ Saved best model (F1={best_f1:.4f})")

    # Save final checkpoint and log
    torch.save(model.state_dict(), out_dir / "last_model.pth")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete.  Best val F1 = {best_f1:.4f}")
    print(f"Checkpoints → {out_dir}/")


if __name__ == "__main__":
    main()
