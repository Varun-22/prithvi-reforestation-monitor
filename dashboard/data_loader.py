"""Cached data-loading helpers for the Streamlit dashboard."""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TILES_DIR   = PROJECT_ROOT / "data_pipeline" / "tiles"
ASSETS_DIR  = PROJECT_ROOT / "assets"
METRICS_PATH = PROJECT_ROOT / "evaluation" / "results" / "metrics.json"
CKPT_PATH   = PROJECT_ROOT / "training" / "checkpoints" / "best_model.pth"


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

def load_tiles_metadata() -> Optional[dict]:
    path = TILES_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_tile_pair(fname: str):
    """Return (before_norm, after_norm) float32 arrays, or (None, None)."""
    b = TILES_DIR / "before" / fname
    a = TILES_DIR / "after"  / fname
    if not b.exists() or not a.exists():
        return None, None
    return np.load(b), np.load(a)


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def tile_to_rgb(norm: np.ndarray, percentile: int = 98) -> np.ndarray:
    """(6, H, W) normalised → uint8 RGB (H, W, 3) using bands R-G-B = 2-1-0."""
    from data_pipeline.preprocess import denormalize_tile
    raw = denormalize_tile(norm.astype(np.float32))
    rgb = np.stack([raw[2], raw[1], raw[0]], axis=-1)
    lo  = np.percentile(rgb, 2)
    hi  = np.percentile(rgb, percentile)
    return (np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)


def tile_to_ndvi_rgb(norm: np.ndarray) -> np.ndarray:
    """(6, H, W) normalised → uint8 NDVI image (RdYlGn colourmap, H, W, 3)."""
    import matplotlib.pyplot as plt
    from data_pipeline.preprocess import denormalize_tile
    raw    = denormalize_tile(norm.astype(np.float32))
    red, nir = raw[2], raw[3]
    denom = nir + red
    ndvi  = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    normed = (np.clip(ndvi, -0.2, 0.8) + 0.2) / 1.0
    rgba   = plt.get_cmap("RdYlGn")(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def compute_change_overlay(before_norm: np.ndarray,
                            after_norm: np.ndarray,
                            alpha: float = 0.45) -> np.ndarray:
    """
    Blend after-RGB with red deforestation mask.
    Returns uint8 (H, W, 3).
    """
    from data_pipeline.preprocess import denormalize_tile
    br = denormalize_tile(before_norm.astype(np.float32))
    ar = denormalize_tile(after_norm.astype(np.float32))

    # NDVI-diff change mask
    def ndvi(raw):
        d = raw[3] + raw[2]
        return np.where(d > 1e-6, (raw[3] - raw[2]) / d, 0.0)

    mask = ((ndvi(br) >= 0.45) & ((ndvi(br) - ndvi(ar)) >= 0.15))

    after_rgb = tile_to_rgb(after_norm).astype(np.float32) / 255.0
    overlay   = after_rgb.copy()
    overlay[mask] = [0.90, 0.15, 0.15]

    blended = (1 - alpha) * after_rgb + alpha * overlay
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def tile_change_stats(before_norm: np.ndarray, after_norm: np.ndarray) -> dict:
    """Return quick statistics for a tile pair."""
    from data_pipeline.preprocess import denormalize_tile
    br = denormalize_tile(before_norm.astype(np.float32))
    ar = denormalize_tile(after_norm.astype(np.float32))

    def ndvi(raw):
        d = raw[3] + raw[2]
        return np.where(d > 1e-6, (raw[3] - raw[2]) / d, 0.0)

    nb, na   = ndvi(br), ndvi(ar)
    was_f    = nb >= 0.45
    defor    = was_f & (na < 0.45)
    px_ha    = (20 * 20) / 10_000

    return {
        "ndvi_before":     round(float(nb.mean()), 3),
        "ndvi_after":      round(float(na.mean()), 3),
        "ndvi_delta":      round(float(nb.mean() - na.mean()), 3),
        "forest_before_%": round(float(was_f.mean()) * 100, 1),
        "deforested_%":    round(float(defor.mean()) * 100, 1),
        "deforested_ha":   round(float(defor.sum()) * px_ha, 2),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def load_metrics() -> Optional[dict]:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


PLACEHOLDER_METRICS = [
    {"model": "NDVI Baseline",      "f1": 0.42, "iou": 0.27, "precision": 0.38, "recall": 0.47},
    {"model": "Random Forest",      "f1": 0.51, "iou": 0.34, "precision": 0.55, "recall": 0.48},
    {"model": "Prithvi Fine-tuned", "f1": None, "iou": None, "precision": None, "recall": None},
]


def get_metrics_rows() -> list[dict]:
    """Return list of metric dicts, preferring real results over placeholders."""
    data = load_metrics()
    if data and "models" in data:
        return data["models"]
    return PLACEHOLDER_METRICS
