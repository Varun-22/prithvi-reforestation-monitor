#!/usr/bin/env python3
"""
Baseline change-detection models (no deep learning).

1. NDVIDiffBaseline  — threshold on |NDVI_after − NDVI_before|.
   Fast, interpretable, commonly used in deforestation monitoring.

2. RandomForestBaseline — scikit-learn RandomForest on 12-band feature
   vectors (6 bands × 2 time points) per pixel, with NDVI pseudo-labels.

Usage (from project root):
    python -m training.baseline [--tiles-dir PATH] [--output-dir PATH]

Outputs saved to training/checkpoints/:
    ndvi_baseline_preds.npy   — binary predictions on validation set
    rf_baseline_preds.npy
    rf_baseline.pkl           — saved RF model
    baseline_metrics.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.dataset import ChangeDetectionDataset, make_splits, _change_label
from data_pipeline.preprocess import denormalize_tile, PRITHVI_MEANS, PRITHVI_STDS

BAND_RED = 2
BAND_NIR = 3
FOREST_NDVI = 0.45


# ---------------------------------------------------------------------------
# NDVI difference baseline
# ---------------------------------------------------------------------------

class NDVIDiffBaseline:
    """
    Predict deforestation where NDVI dropped by at least `threshold` AND
    NDVI_before was above `forest_min` (was forest, now bare).
    """

    def __init__(self, drop_threshold: float = 0.15, forest_min: float = FOREST_NDVI):
        self.drop_threshold = drop_threshold
        self.forest_min     = forest_min

    def predict(self, before_norm: np.ndarray, after_norm: np.ndarray) -> np.ndarray:
        """
        before_norm, after_norm: (6, H, W) normalised tiles
        returns: (H, W) binary change map
        """
        b_raw = denormalize_tile(before_norm.astype(np.float32))
        a_raw = denormalize_tile(after_norm.astype(np.float32))

        ndvi_b = _ndvi(b_raw)
        ndvi_a = _ndvi(a_raw)

        was_forest    = ndvi_b >= self.forest_min
        ndvi_dropped  = (ndvi_b - ndvi_a) >= self.drop_threshold
        return (was_forest & ndvi_dropped).astype(np.uint8)


def _ndvi(raw: np.ndarray) -> np.ndarray:
    red, nir = raw[BAND_RED], raw[BAND_NIR]
    denom = nir + red
    return np.where(denom > 1e-6, (nir - red) / denom, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Random Forest baseline
# ---------------------------------------------------------------------------

class RandomForestBaseline:
    """
    Pixel-wise RF classifier on 12 spectral features (6 before + 6 after).
    Subsamples pixels to stay memory-efficient.
    """

    def __init__(self, n_estimators: int = 100, max_samples_per_tile: int = 500,
                 n_jobs: int = -1, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced",
        )
        self.max_samples = max_samples_per_tile

    def _extract_features(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Flatten spatial dims; concatenate both time points → (N, 12)."""
        C, H, W = before.shape
        b = before.reshape(C, -1).T  # (N, 6)
        a = after.reshape(C, -1).T
        return np.concatenate([b, a], axis=1)  # (N, 12)

    def fit(self, dataset: ChangeDetectionDataset) -> None:
        X_list, y_list = [], []
        for i in range(len(dataset)):
            before, after, label = dataset[i]
            before = before.numpy(); after = after.numpy(); label = label.numpy()[0]

            feats = self._extract_features(before, after)  # (N, 12)
            labs  = label.flatten()                         # (N,)

            # Subsample to keep training tractable
            n = feats.shape[0]
            idx = np.random.choice(n, min(self.max_samples, n), replace=False)
            X_list.append(feats[idx])
            y_list.append(labs[idx])

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        print(f"  RF training on {X.shape[0]:,} pixels ({y.mean()*100:.1f}% positive)...")
        self.model.fit(X, y)

    def predict(self, before_norm: np.ndarray, after_norm: np.ndarray) -> np.ndarray:
        """Returns (H, W) binary prediction."""
        feats = self._extract_features(before_norm, after_norm)
        preds = self.model.predict(feats)
        return preds.reshape(before_norm.shape[1], before_norm.shape[2]).astype(np.uint8)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestBaseline":
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_on_dataset(predict_fn, dataset):
    """Run predict_fn on every tile; return flat arrays of preds and labels."""
    all_preds, all_labels = [], []
    for i in range(len(dataset)):
        before, after, label = dataset[i]
        before = before.numpy(); after = after.numpy(); label = label.numpy()[0]
        pred  = predict_fn(before, after)
        all_preds.append(pred.flatten())
        all_labels.append(label.flatten().astype(np.uint8))
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds, labels) -> dict:
    f1  = f1_score(labels, preds, zero_division=0)
    iou = jaccard_score(labels, preds, zero_division=0)
    return {"f1": round(float(f1), 4), "iou": round(float(iou), 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir",  default="data_pipeline/tiles")
    parser.add_argument("--output-dir", default="training/checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_ds, val_ds = make_splits(args.tiles_dir)
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    results = {}

    # ---- NDVI diff baseline ----
    print("\n[1/2] NDVI Difference Baseline")
    ndvi_baseline = NDVIDiffBaseline()
    ndvi_preds, val_labels = eval_on_dataset(ndvi_baseline.predict, val_ds)
    ndvi_metrics = compute_metrics(ndvi_preds, val_labels)
    print(f"  F1={ndvi_metrics['f1']:.4f}  IoU={ndvi_metrics['iou']:.4f}")
    results["ndvi_baseline"] = ndvi_metrics
    np.save(out_dir / "ndvi_baseline_preds.npy", ndvi_preds)

    # ---- Random Forest baseline ----
    print("\n[2/2] Random Forest Baseline")
    rf_baseline = RandomForestBaseline()
    rf_baseline.fit(train_ds)
    rf_preds, _ = eval_on_dataset(rf_baseline.predict, val_ds)
    rf_metrics  = compute_metrics(rf_preds, val_labels)
    print(f"  F1={rf_metrics['f1']:.4f}  IoU={rf_metrics['iou']:.4f}")
    results["rf_baseline"] = rf_metrics
    np.save(out_dir / "rf_baseline_preds.npy", rf_preds)
    rf_baseline.save(out_dir / "rf_baseline.pkl")

    # Save
    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline metrics → {out_dir}/baseline_metrics.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
