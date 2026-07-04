"""Normalization utilities for Prithvi-compatible tiles."""

import numpy as np

# Prithvi-100M training statistics (HLS, reflectance [0, 1])
# Source: ibm-nasa-geospatial/Prithvi-EO-1.0-100M model config
# Band order: Blue, Green, Red, NIR-narrow, SWIR1, SWIR2
PRITHVI_MEANS = np.array([
    0.033349706741586264,
    0.04509548172188006,
    0.04026548542688139,
    0.26531564765613897,
    0.16682069560609084,
    0.11736093569701502,
], dtype=np.float32)

PRITHVI_STDS = np.array([
    0.02387963425027755,
    0.03260987824447789,
    0.03660491725066752,
    0.06836665490453512,
    0.06832692218471482,
    0.05860916013047283,
], dtype=np.float32)


def normalize_tile(tile: np.ndarray) -> np.ndarray:
    """Z-score normalize (C, H, W) tile using Prithvi training stats.

    Input must be in reflectance [0, 1] (raw DN / 10000).
    """
    means = PRITHVI_MEANS[:, None, None]
    stds  = PRITHVI_STDS[:, None, None]
    return ((tile - means) / (stds + 1e-8)).astype(np.float32)


def denormalize_tile(tile: np.ndarray) -> np.ndarray:
    """Reverse z-score normalization back to reflectance [0, 1]."""
    means = PRITHVI_MEANS[:, None, None]
    stds  = PRITHVI_STDS[:, None, None]
    return (tile * stds + means).astype(np.float32)


def compute_ndvi(tile: np.ndarray) -> np.ndarray:
    """Return NDVI (H, W) from a raw-reflectance (C, H, W) tile.

    Band order assumed: Blue(0), Green(1), Red(2), NIR(3), SWIR1(4), SWIR2(5).
    """
    red = tile[2].astype(np.float32)
    nir = tile[3].astype(np.float32)
    denom = nir + red
    return np.where(denom > 1e-6, (nir - red) / denom, 0.0).astype(np.float32)


def ndvi_forest_mask(tile: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binary forest mask (True = forest) from raw-reflectance tile."""
    return compute_ndvi(tile) >= threshold
