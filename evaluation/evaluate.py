#!/usr/bin/env python3
"""
Evaluate all three models on the validation split and save metrics + charts.

Usage (from project root):
    python -m evaluation.evaluate [--tiles-dir PATH] [--ckpt PATH] [--output-dir PATH]

Requires:
  - data_pipeline/tiles/   (run data pipeline first)
  - training/checkpoints/best_model.pth   (download from Kaggle, or omit for baselines only)

Outputs:
  - evaluation/results/metrics.json
  - assets/metrics_comparison.png
  - assets/sample_predictions.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, jaccard_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset  import ChangeDetectionDataset, make_splits
from training.baseline import NDVIDiffBaseline, RandomForestBaseline, eval_on_dataset
from evaluation.visualize import plot_metrics_comparison, plot_sample_predictions


# ---------------------------------------------------------------------------
# Deep model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_deep_model(model, dataset, device, batch_size: int = 8):
    """Return (flat_probs, flat_labels) arrays for the entire dataset."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_probs, all_labels = [], []
    for before, after, labels in loader:
        before, after = before.to(device), after.to(device)
        logits = model(before, after)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        all_labels.append(labels.numpy().flatten())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_full_metrics(probs_or_preds: np.ndarray,
                          labels: np.ndarray,
                          threshold: float = 0.5,
                          model_name: str = "") -> dict:
    """Compute F1, IoU, precision, recall; optionally AP if probs given."""
    is_prob = probs_or_preds.dtype == np.float32 or probs_or_preds.max() <= 1.0
    preds   = (probs_or_preds >= threshold).astype(np.uint8)
    labels  = labels.astype(np.uint8)

    metrics = {
        "model":     model_name,
        "f1":        round(float(f1_score(labels, preds,        zero_division=0)), 4),
        "iou":       round(float(jaccard_score(labels, preds,   zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds,    zero_division=0)), 4),
        "threshold": threshold,
        "n_pixels":  int(len(labels)),
        "pos_rate":  round(float(labels.mean()), 4),
    }
    if is_prob and probs_or_preds.max() < 1.0 + 1e-6:
        try:
            metrics["ap"] = round(float(average_precision_score(labels, probs_or_preds)), 4)
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir",  default="data_pipeline/tiles")
    parser.add_argument("--ckpt",       default="training/checkpoints/best_model.pth",
                        help="Prithvi checkpoint. Omit to evaluate baselines only.")
    parser.add_argument("--rf-model",   default="training/checkpoints/rf_baseline.pkl",
                        help="Saved RF model from training/baseline.py")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out_dir    = Path(args.output_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = Path(args.assets_dir);  assets_dir.mkdir(parents=True, exist_ok=True)

    print("Loading validation split...")
    _, val_ds = make_splits(args.tiles_dir)
    print(f"  Val tiles: {len(val_ds)}")

    all_metrics = []
    sample_data = {}  # store a few tiles for the sample-prediction plot

    # ── NDVI Baseline ─────────────────────────────────────────────────────────
    print("\n[1/3] NDVI Difference Baseline...")
    ndvi = NDVIDiffBaseline()
    ndvi_preds, val_labels = eval_on_dataset(ndvi.predict, val_ds)
    ndvi_metrics = compute_full_metrics(ndvi_preds.astype(np.float32),
                                        val_labels, model_name="NDVI Baseline")
    all_metrics.append(ndvi_metrics)
    print(f"  F1={ndvi_metrics['f1']}  IoU={ndvi_metrics['iou']}  "
          f"P={ndvi_metrics['precision']}  R={ndvi_metrics['recall']}")

    # ── RF Baseline ───────────────────────────────────────────────────────────
    print("\n[2/3] Random Forest Baseline...")
    rf_path = Path(args.rf_model)
    if rf_path.exists():
        rf = RandomForestBaseline.load(rf_path)
    else:
        print("  No saved RF model found — training a fresh one on val set "
              "(for eval purposes only; for fair comparison run baseline.py first)")
        rf = RandomForestBaseline()
        train_ds_temp, _ = make_splits(args.tiles_dir)
        rf.fit(train_ds_temp)

    rf_preds, _ = eval_on_dataset(rf.predict, val_ds)
    rf_metrics  = compute_full_metrics(rf_preds.astype(np.float32),
                                       val_labels, model_name="Random Forest")
    all_metrics.append(rf_metrics)
    print(f"  F1={rf_metrics['f1']}  IoU={rf_metrics['iou']}  "
          f"P={rf_metrics['precision']}  R={rf_metrics['recall']}")

    # ── Prithvi fine-tuned ────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        print(f"\n[3/3] Prithvi Change Detector (from {ckpt_path})...")
        from training.model import PrithviChangeDetector

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"  Device: {device}")

        model = PrithviChangeDetector(pretrained=False, freeze_backbone=False)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        prithvi_probs, _ = run_deep_model(model, val_ds, device, args.batch_size)
        prithvi_metrics  = compute_full_metrics(prithvi_probs, val_labels,
                                                model_name="Prithvi Fine-tuned")
        all_metrics.append(prithvi_metrics)
        print(f"  F1={prithvi_metrics['f1']}  IoU={prithvi_metrics['iou']}  "
              f"P={prithvi_metrics['precision']}  R={prithvi_metrics['recall']}")

        # Collect one sample tile for visualisation
        before, after, label = val_ds[0]
        with torch.no_grad():
            logit = model(before.unsqueeze(0).to(device),
                          after.unsqueeze(0).to(device))
            pred_map = torch.sigmoid(logit)[0, 0].cpu().numpy()
        sample_data["prithvi_pred"] = pred_map

    else:
        print(f"\n[3/3] Prithvi checkpoint not found at {ckpt_path} — "
              "run Kaggle notebook and place best_model.pth there to include.")
        prithvi_metrics = None

    # Collect a sample tile for visualisation regardless
    if not sample_data:
        before, after, label = val_ds[0]
        sample_data["before"] = before.numpy()
        sample_data["after"]  = after.numpy()
        sample_data["label"]  = label.numpy()[0]
        ndvi_pred_tile = ndvi.predict(before.numpy(), after.numpy())
        sample_data["ndvi_pred"] = ndvi_pred_tile.astype(np.float32)

    # ── Save metrics ──────────────────────────────────────────────────────────
    results = {
        "val_tiles": len(val_ds),
        "models": all_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_path = assets_dir / "metrics_comparison.png"
    plot_metrics_comparison(all_metrics, save_path=str(chart_path))
    print(f"Comparison chart → {chart_path}")

    sample_path = assets_dir / "sample_predictions.png"
    plot_sample_predictions(
        before=val_ds[0][0].numpy(),
        after=val_ds[0][1].numpy(),
        label=val_ds[0][2].numpy()[0],
        ndvi_pred=ndvi.predict(val_ds[0][0].numpy(), val_ds[0][1].numpy()),
        prithvi_pred=sample_data.get("prithvi_pred"),
        save_path=str(sample_path),
    )
    print(f"Sample predictions → {sample_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"{'Model':<22} {'F1':>6} {'IoU':>6} {'Prec':>6} {'Recall':>6}")
    print("-" * 58)
    for m in all_metrics:
        print(f"{m['model']:<22} {m['f1']:>6.4f} {m['iou']:>6.4f} "
              f"{m['precision']:>6.4f} {m['recall']:>6.4f}")
    print("=" * 58)


if __name__ == "__main__":
    main()
