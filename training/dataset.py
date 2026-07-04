"""
PyTorch Dataset for paired before/after change-detection tiles.

Pseudo-labels are derived from NDVI difference (no external ground truth needed):
  - forest_before  = NDVI_before  ≥ FOREST_NDVI_THRESHOLD
  - forest_after   = NDVI_after   ≥ FOREST_NDVI_THRESHOLD
  - change label   = forest_before AND NOT forest_after  (deforestation)

This lets the model train on Rondônia tiles without manual annotation.
The NDVI-threshold baseline (baseline.py) is trained on the same signal,
ensuring a fair apples-to-apples comparison.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# Band indices in the (6, H, W) tile arrays
BAND_RED = 2   # Red  (B04)
BAND_NIR = 3   # NIR  (B8A)

FOREST_NDVI_THRESHOLD = 0.45  # tiles with NDVI ≥ this are "forest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndvi_from_raw(tile_norm: np.ndarray) -> np.ndarray:
    """
    Compute NDVI from a z-score normalised tile.
    Re-applies Prithvi mean/std to recover approximate reflectance.
    """
    from data_pipeline.preprocess import PRITHVI_MEANS, PRITHVI_STDS, denormalize_tile
    raw = denormalize_tile(tile_norm.astype(np.float32))  # back to reflectance
    red = raw[BAND_RED].astype(np.float64)
    nir = raw[BAND_NIR].astype(np.float64)
    denom = nir + red
    return np.where(denom > 1e-6, (nir - red) / denom, 0.0).astype(np.float32)


def _change_label(before_norm: np.ndarray, after_norm: np.ndarray,
                  threshold: float = FOREST_NDVI_THRESHOLD) -> np.ndarray:
    """Binary (H, W) change mask: 1 where forest→non-forest (deforestation)."""
    ndvi_b = _ndvi_from_raw(before_norm)
    ndvi_a = _ndvi_from_raw(after_norm)
    was_forest = ndvi_b >= threshold
    now_forest = ndvi_a >= threshold
    return (was_forest & ~now_forest).astype(np.float32)  # deforested pixels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChangeDetectionDataset(Dataset):
    """Paired before/after normalised tiles with NDVI pseudo-labels."""

    def __init__(
        self,
        tiles_dir: str | Path,
        augment: bool = False,
        ndvi_threshold: float = FOREST_NDVI_THRESHOLD,
    ):
        self.tiles_dir    = Path(tiles_dir)
        self.augment      = augment
        self.ndvi_threshold = ndvi_threshold

        meta_path = self.tiles_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — run data_pipeline/run_pipeline.py first."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self.tile_ids = [t["filename"] for t in meta["tiles"]]

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        fname = self.tile_ids[idx]
        before = np.load(self.tiles_dir / "before" / fname)  # (6, 224, 224)
        after  = np.load(self.tiles_dir / "after"  / fname)

        label = _change_label(before, after, self.ndvi_threshold)  # (224, 224)

        if self.augment:
            before, after, label = self._augment(before, after, label)

        return (
            torch.from_numpy(before),           # (6, 224, 224) float32
            torch.from_numpy(after),            # (6, 224, 224) float32
            torch.from_numpy(label).unsqueeze(0),  # (1, 224, 224) float32
        )

    @staticmethod
    def _augment(before: np.ndarray, after: np.ndarray,
                 label: np.ndarray):
        """Consistent random flips applied to both time points + label."""
        if np.random.rand() > 0.5:   # horizontal flip
            before = before[:, :, ::-1].copy()
            after  = after[:, :, ::-1].copy()
            label  = label[:, ::-1].copy()
        if np.random.rand() > 0.5:   # vertical flip
            before = before[:, ::-1, :].copy()
            after  = after[:, ::-1, :].copy()
            label  = label[::-1, :].copy()
        return before, after, label


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def make_splits(
    tiles_dir: str | Path,
    train_frac: float = 0.8,
    seed: int = 42,
):
    """Return (train_dataset, val_dataset) with augmentation on train only."""
    # Build full dataset without augment to get length; then split
    full_no_aug = ChangeDetectionDataset(tiles_dir, augment=False)
    n = len(full_no_aug)
    n_train = int(n * train_frac)
    n_val   = n - n_train

    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=gen)

    train_ds = _SubsetDataset(ChangeDetectionDataset(tiles_dir, augment=True),
                              list(train_idx))
    val_ds   = _SubsetDataset(ChangeDetectionDataset(tiles_dir, augment=False),
                              list(val_idx))
    return train_ds, val_ds


class _SubsetDataset(Dataset):
    """Wrap a Dataset with an explicit index list."""
    def __init__(self, ds: Dataset, indices: list[int]):
        self.ds      = ds
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[self.indices[i]]
