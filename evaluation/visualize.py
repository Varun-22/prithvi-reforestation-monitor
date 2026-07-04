"""Plotting utilities for evaluation — comparison charts and prediction overlays."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from data_pipeline.preprocess import denormalize_tile, PRITHVI_MEANS, PRITHVI_STDS


# ---------------------------------------------------------------------------
# Metrics comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    models_metrics: list[dict],
    save_path: str = "assets/metrics_comparison.png",
) -> None:
    """
    Side-by-side grouped bar chart: F1 and IoU for each model.
    Handles 2 or 3 models.  Saves to save_path.
    """
    if not models_metrics:
        return

    names  = [m["model"]    for m in models_metrics]
    f1s    = [m["f1"]       for m in models_metrics]
    ious   = [m["iou"]      for m in models_metrics]
    precs  = [m["precision"] for m in models_metrics]
    recalls= [m["recall"]   for m in models_metrics]

    x     = np.arange(len(names))
    width = 0.20
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]   # Blue, Green, Orange, Purple

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Deforestation Change Detection — Model Comparison\n"
                 "Rondônia, Brazil  |  Sentinel-2 (2019 → 2022)",
                 fontsize=13, fontweight="bold", y=1.02)

    # ── Left: F1 and IoU grouped bars ────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - width/2, f1s,  width, label="F1",  color=colours[0], alpha=0.85)
    b2 = ax.bar(x + width/2, ious, width, label="IoU", color=colours[1], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("F1 and IoU")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5)

    # ── Right: Precision / Recall grouped bars ───────────────────────────────
    ax2 = axes[1]
    b3 = ax2.bar(x - width/2, precs,   width, label="Precision", color=colours[2], alpha=0.85)
    b4 = ax2.bar(x + width/2, recalls, width, label="Recall",    color=colours[3], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Score"); ax2.set_title("Precision and Recall")
    ax2.legend(); ax2.grid(axis="y", alpha=0.4)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Sample prediction overlay
# ---------------------------------------------------------------------------

def _rgb_preview(tile_norm: np.ndarray, percentile: int = 98) -> np.ndarray:
    """Return uint8 RGB from normalised (6, H, W) tile (bands R-G-B = 2-1-0)."""
    raw = denormalize_tile(tile_norm.astype(np.float32))
    rgb = np.stack([raw[2], raw[1], raw[0]], axis=-1)   # Red, Green, Blue → (H,W,3)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, percentile)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _ndvi_colormap(tile_norm: np.ndarray) -> np.ndarray:
    """Return uint8 NDVI image (RdYlGn colormap) from normalised tile."""
    raw  = denormalize_tile(tile_norm.astype(np.float32))
    red, nir = raw[2], raw[3]
    denom = nir + red
    ndvi_arr = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    cmap  = plt.get_cmap("RdYlGn")
    normed = (np.clip(ndvi_arr, -0.2, 0.8) + 0.2) / 1.0   # map [-0.2, 0.8] → [0, 1]
    rgba  = cmap(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def plot_sample_predictions(
    before:       np.ndarray,          # (6, H, W) normalised
    after:        np.ndarray,          # (6, H, W) normalised
    label:        np.ndarray,          # (H, W) binary float
    ndvi_pred:    np.ndarray,          # (H, W) binary uint8
    prithvi_pred: Optional[np.ndarray] = None,  # (H, W) float probs (optional)
    save_path:    str = "assets/sample_predictions.png",
) -> None:
    """
    Panel:  Before RGB | After RGB | NDVI before | NDVI after |
            GT label   | NDVI pred | [Prithvi pred if available]
    """
    panels = [
        (_rgb_preview(before),     "Before (RGB)"),
        (_rgb_preview(after),      "After (RGB)"),
        (_ndvi_colormap(before),   "NDVI Before"),
        (_ndvi_colormap(after),    "NDVI After"),
        (_binary_to_rgb(label),    "Ground Truth\n(NDVI pseudo-label)"),
        (_binary_to_rgb(ndvi_pred),"NDVI Baseline\nPrediction"),
    ]
    if prithvi_pred is not None:
        panels.append((_binary_to_rgb((prithvi_pred >= 0.5).astype(np.float32)),
                        "Prithvi\nPrediction"))

    n   = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    fig.suptitle("Sample Tile — Rondônia Change Detection",
                 fontsize=11, fontweight="bold")

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Legend for binary maps
    patches = [
        mpatches.Patch(color="#d32f2f", label="Deforested"),
        mpatches.Patch(color="#1b5e20", label="Forest / unchanged"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def _binary_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Render a binary (H, W) mask: red=1, dark-green=0."""
    H, W = mask.shape
    img  = np.zeros((H, W, 3), dtype=np.uint8)
    img[mask >= 0.5] = [211, 47,  47]    # red  — deforested
    img[mask <  0.5] = [ 27, 94, 32]    # dark green — forest/unchanged
    return img
