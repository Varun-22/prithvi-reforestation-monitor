#!/usr/bin/env python3
"""
Tile raw Sentinel-2 bands into 224×224 patches.

Applies cloud masking (SCL), scales to reflectance, and z-score normalises
using Prithvi training statistics.  Only tiles where both time points have
≥50% valid pixels are kept.

Usage (from project root):
    python -m data_pipeline.tile_imagery

Outputs:
    data_pipeline/tiles/before/tile_XXXX.npy   shape (6, 224, 224) float32
    data_pipeline/tiles/after/tile_XXXX.npy    shape (6, 224, 224) float32
    data_pipeline/tiles/metadata.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.config import (
    BAND_NAMES, MIN_VALID_FRAC, RAW_DIR, REFLECTANCE_SCALE,
    S2_BANDS, TILE_SIZE, TILE_STRIDE, TILES_DIR, VALID_SCL,
)
from data_pipeline.preprocess import normalize_tile


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_scene_stack(time_label: str):
    """Load all spectral bands → (C, H, W) float32 array in raw DN."""
    scene_dir = RAW_DIR / time_label
    arrays = []
    H = W = None

    for band in S2_BANDS:
        path = scene_dir / f"{band}.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.  Run fetch_imagery.py first."
            )
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if H is None:
                H, W = src.height, src.width
            arrays.append(arr)

    # 10m bands (B02/B03/B04) resampled to 20m can differ by ±1 px from
    # native 20m bands (B8A/B11/B12) due to rounding — crop to minimum.
    min_h = min(a.shape[0] for a in arrays)
    min_w = min(a.shape[1] for a in arrays)
    arrays = [a[:min_h, :min_w] for a in arrays]
    H, W = min_h, min_w

    return np.stack(arrays, axis=0), H, W  # (C, H, W)


def load_scl(time_label: str, H: int, W: int) -> np.ndarray:
    """Load SCL band, nearest-neighbour resampled to (H, W)."""
    path = RAW_DIR / time_label / "SCL.tif"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.  Run fetch_imagery.py first.")
    with rasterio.open(path) as src:
        return src.read(
            1, out_shape=(H, W), resampling=Resampling.nearest
        ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def valid_pixel_mask(scl: np.ndarray, valid_classes: set) -> np.ndarray:
    """Boolean mask: True where SCL class is in valid_classes."""
    mask = np.zeros(scl.shape, dtype=bool)
    for cls in valid_classes:
        mask |= scl == cls
    return mask


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def generate_tile_coords(H: int, W: int, tile_size: int, stride: int):
    """Yield (row, col) top-left corners for non-overlapping-then-strided grid."""
    for row in range(0, H - tile_size + 1, stride):
        for col in range(0, W - tile_size + 1, stride):
            yield row, col


def extract_tile(stack: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    return stack[:, row:row + size, col:col + size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    (TILES_DIR / "before").mkdir(exist_ok=True)
    (TILES_DIR / "after").mkdir(exist_ok=True)

    print("Loading spectral stacks...")
    before_stack, H, W = load_scene_stack("before")
    after_stack,  _, _ = load_scene_stack("after")

    print(f"  Image size: {H} × {W} px  |  {len(S2_BANDS)} bands each")

    print("Loading cloud masks (SCL)...")
    before_scl = load_scl("before", H, W)
    after_scl  = load_scl("after",  H, W)
    before_valid = valid_pixel_mask(before_scl, VALID_SCL)
    after_valid  = valid_pixel_mask(after_scl,  VALID_SCL)
    combined     = before_valid & after_valid
    pct_valid = combined.mean() * 100
    print(f"  Combined valid pixels: {pct_valid:.1f}%")

    # Scale raw DN → reflectance [0, 1]
    before_stack = np.clip(before_stack / REFLECTANCE_SCALE, 0.0, 1.0)
    after_stack  = np.clip(after_stack  / REFLECTANCE_SCALE, 0.0, 1.0)

    coords      = list(generate_tile_coords(H, W, TILE_SIZE, TILE_STRIDE))
    tiles_meta  = []
    kept = skipped = 0

    print(f"Tiling ({len(coords)} candidate positions, "
          f"min valid fraction = {MIN_VALID_FRAC:.0%})...")

    for row, col in tqdm(coords):
        mask_tile   = combined[row:row + TILE_SIZE, col:col + TILE_SIZE]
        valid_frac  = float(mask_tile.mean())
        if valid_frac < MIN_VALID_FRAC:
            skipped += 1
            continue

        idx  = len(tiles_meta)
        fname = f"tile_{idx:04d}.npy"

        before_t = normalize_tile(extract_tile(before_stack, row, col, TILE_SIZE))
        after_t  = normalize_tile(extract_tile(after_stack,  row, col, TILE_SIZE))

        np.save(TILES_DIR / "before" / fname, before_t)
        np.save(TILES_DIR / "after"  / fname, after_t)

        tiles_meta.append({
            "idx":        idx,
            "filename":   fname,
            "row":        row,
            "col":        col,
            "valid_frac": round(valid_frac, 4),
        })
        kept += 1

    metadata = {
        "n_tiles":       kept,
        "n_skipped":     skipped,
        "tile_size":     TILE_SIZE,
        "tile_stride":   TILE_STRIDE,
        "image_size":    [H, W],
        "bands":         S2_BANDS,
        "band_names":    BAND_NAMES,
        "min_valid_frac": MIN_VALID_FRAC,
        "tiles":         tiles_meta,
    }
    meta_path = TILES_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nKept {kept} tile pairs  |  Skipped {skipped} (cloud/shadow)")
    print(f"Tiles  →  {TILES_DIR}/before/  and  {TILES_DIR}/after/")
    print(f"Metadata →  {meta_path}")


if __name__ == "__main__":
    main()
