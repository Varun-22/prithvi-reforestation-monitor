"""
Geospatial tools available to the ReAct agent.

Each tool is a plain Python function that returns a JSON-serialisable dict.
Tools work from locally processed tiles; they degrade gracefully if the data
pipeline hasn't been run or the Prithvi checkpoint isn't available yet.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

# Lazy imports — only load torch/rasterio if a function actually needs them
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TILES_DIR    = _PROJECT_ROOT / "data_pipeline" / "tiles"
_RAW_DIR      = _PROJECT_ROOT / "data_pipeline" / "raw"
_CKPT_PATH    = _PROJECT_ROOT / "training" / "checkpoints" / "best_model.pth"
_ASSETS_DIR   = _PROJECT_ROOT / "assets"

# Pixel area in hectares at 20m resolution
_PX_HA = (20 * 20) / 10_000  # 0.04 ha per pixel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tiles_meta() -> Optional[dict]:
    path = _TILES_DIR / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _ndvi_from_raw(raw: np.ndarray) -> np.ndarray:
    """NDVI from (6, H, W) reflectance array."""
    red, nir = raw[2].astype(np.float32), raw[3].astype(np.float32)
    denom = nir + red
    return np.where(denom > 1e-6, (nir - red) / denom, 0.0)


def _load_tile_pair(fname: str):
    """Return (before_norm, after_norm) ndarrays or None."""
    b = _TILES_DIR / "before" / fname
    a = _TILES_DIR / "after"  / fname
    if not b.exists() or not a.exists():
        return None, None
    return np.load(b), np.load(a)


def _ndvi_baseline_pred(before_norm: np.ndarray, after_norm: np.ndarray) -> np.ndarray:
    """Quick NDVI-diff binary prediction (no torch needed)."""
    from data_pipeline.preprocess import denormalize_tile
    br = denormalize_tile(before_norm.astype(np.float32))
    ar = denormalize_tile(after_norm.astype(np.float32))
    ndvi_b = _ndvi_from_raw(br)
    ndvi_a = _ndvi_from_raw(ar)
    return ((ndvi_b >= 0.45) & ((ndvi_b - ndvi_a) >= 0.15)).astype(np.float32)


def _deep_model_pred(before_norm: np.ndarray, after_norm: np.ndarray) -> np.ndarray:
    """Run Prithvi model on a single tile. Returns probability map."""
    import torch
    from training.model import PrithviChangeDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PrithviChangeDetector(pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(_CKPT_PATH, map_location=device))
    model.eval().to(device)

    b = torch.from_numpy(before_norm).unsqueeze(0).to(device)
    a = torch.from_numpy(after_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(b, a))[0, 0].cpu().numpy()
    return prob


# ---------------------------------------------------------------------------
# Tool 1 — run_inference
# ---------------------------------------------------------------------------

def run_inference(region: str = "default", date_range: str = "2019-2022") -> dict:
    """
    Run change-detection inference on all tiles for the study region.

    Uses the Prithvi fine-tuned checkpoint if available, otherwise the
    NDVI-difference baseline.  Returns aggregated forest-change statistics.

    Args:
        region:     'default' (Rondônia study area) or a description (informational)
        date_range: Date range string, e.g. '2019-2022' (informational; tiles are fixed)

    Returns dict with keys:
        model_used, n_tiles, forest_before_pct, forest_after_pct,
        forest_lost_pct, forest_lost_ha, changed_pixels, total_pixels
    """
    meta = _load_tiles_meta()
    if meta is None:
        return {"error": "No tile metadata found. Run data_pipeline/run_pipeline.py first."}

    use_deep = _CKPT_PATH.exists()
    model_used = "prithvi-finetuned" if use_deep else "ndvi-baseline"

    tile_files  = [t["filename"] for t in meta["tiles"]]
    all_changed, all_was_forest, total_px = 0, 0, 0

    for fname in tile_files:
        before_n, after_n = _load_tile_pair(fname)
        if before_n is None:
            continue

        # Pseudo-label (what "was forest" before)
        from data_pipeline.preprocess import denormalize_tile
        br = denormalize_tile(before_n.astype(np.float32))
        was_forest = (_ndvi_from_raw(br) >= 0.45).sum()

        if use_deep:
            try:
                pred = (_deep_model_pred(before_n, after_n) >= 0.5).astype(np.uint8)
            except Exception:
                pred = _ndvi_baseline_pred(before_n, after_n).astype(np.uint8)
        else:
            pred = _ndvi_baseline_pred(before_n, after_n).astype(np.uint8)

        all_changed  += int(pred.sum())
        all_was_forest += int(was_forest)
        total_px     += pred.size

    if total_px == 0:
        return {"error": "Tiles found in metadata but could not load tile files."}

    forest_before_pct = round(all_was_forest / total_px * 100, 2)
    forest_after_pct  = round((all_was_forest - all_changed) / total_px * 100, 2)
    forest_lost_pct   = round(all_changed / max(all_was_forest, 1) * 100, 2)
    forest_lost_ha    = round(all_changed * _PX_HA, 1)

    return {
        "region":           region,
        "date_range":       date_range,
        "model_used":       model_used,
        "n_tiles":          len(tile_files),
        "total_pixels":     total_px,
        "changed_pixels":   all_changed,
        "forest_before_pct": forest_before_pct,
        "forest_after_pct":  forest_after_pct,
        "forest_lost_pct":   forest_lost_pct,
        "forest_lost_ha":    forest_lost_ha,
    }


# ---------------------------------------------------------------------------
# Tool 2 — fetch_historical_data
# ---------------------------------------------------------------------------

def fetch_historical_data(region: str = "Rondônia") -> dict:
    """
    Return metadata about available imagery for the region.

    Reads from locally cached scene_meta.json files (created by the data pipeline).
    Does not require network access.

    Args:
        region: Region name (informational; always returns Rondônia study area data)

    Returns dict with keys:
        region, bbox, before_scene, after_scene, years_apart, tile_count,
        expected_change_context
    """
    result = {
        "region":   region,
        "bbox":     [-63.0, -10.7, -62.7, -10.4],
        "before_scene": None,
        "after_scene":  None,
    }

    for tp in ("before", "after"):
        meta_path = _RAW_DIR / tp / "scene_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                sm = json.load(f)
            result[f"{tp}_scene"] = {
                "item_id":      sm.get("item_id"),
                "datetime":     sm.get("datetime"),
                "cloud_cover":  sm.get("cloud_cover"),
                "resolution_m": sm.get("resolution_m"),
            }
        else:
            result[f"{tp}_scene"] = {
                "note": f"Run data_pipeline/run_pipeline.py to fetch '{tp}' scene",
                "expected_date_range": (
                    "2019-07-01 / 2019-09-30" if tp == "before"
                    else "2022-07-01 / 2022-09-30"
                ),
            }

    meta = _load_tiles_meta()
    result["tile_count"]  = meta["n_tiles"] if meta else 0
    result["years_apart"] = 3   # 2019 → 2022

    result["expected_change_context"] = (
        "Rondônia has lost ~4% of its remaining forest per year on average "
        "(PRODES/MapBiomas data). Over a 3-year window a ~10–15% loss in "
        "previously forested tiles is plausible for active frontier areas."
    )
    return result


# ---------------------------------------------------------------------------
# Tool 3 — compute_change_stats
# ---------------------------------------------------------------------------

def compute_change_stats(tile_index: int = 0) -> dict:
    """
    Compute detailed per-tile change statistics for a specific tile.

    Args:
        tile_index: Index of the tile to analyse (0-based; see metadata.json)

    Returns dict with keys:
        tile_file, ndvi_before_mean, ndvi_after_mean, ndvi_delta,
        forest_before_pct, deforested_pct, stable_forest_pct,
        estimated_area_ha, change_severity
    """
    meta = _load_tiles_meta()
    if meta is None:
        return {"error": "No tiles found. Run data_pipeline/run_pipeline.py first."}

    tiles = meta["tiles"]
    if tile_index >= len(tiles):
        return {"error": f"tile_index {tile_index} out of range (max {len(tiles)-1})"}

    fname = tiles[tile_index]["filename"]
    before_n, after_n = _load_tile_pair(fname)
    if before_n is None:
        return {"error": f"Could not load tile {fname}"}

    from data_pipeline.preprocess import denormalize_tile
    br = denormalize_tile(before_n.astype(np.float32))
    ar = denormalize_tile(after_n.astype(np.float32))

    ndvi_b = _ndvi_from_raw(br)
    ndvi_a = _ndvi_from_raw(ar)

    was_forest   = ndvi_b >= 0.45
    still_forest = ndvi_a >= 0.45
    deforested   = was_forest & ~still_forest

    total         = ndvi_b.size
    forest_before = int(was_forest.sum())
    defor_px      = int(deforested.sum())
    stable_forest = int((was_forest & still_forest).sum())

    ndvi_delta    = float(ndvi_b.mean() - ndvi_a.mean())
    severity      = ("high" if defor_px / max(forest_before, 1) > 0.20
                     else "medium" if defor_px / max(forest_before, 1) > 0.05
                     else "low")

    return {
        "tile_file":          fname,
        "tile_index":         tile_index,
        "ndvi_before_mean":   round(float(ndvi_b.mean()), 4),
        "ndvi_after_mean":    round(float(ndvi_a.mean()), 4),
        "ndvi_delta":         round(ndvi_delta, 4),
        "forest_before_pct":  round(forest_before / total * 100, 2),
        "deforested_pct":     round(defor_px / total * 100, 2),
        "stable_forest_pct":  round(stable_forest / total * 100, 2),
        "total_pixels":       total,
        "deforested_pixels":  defor_px,
        "estimated_area_ha":  round(defor_px * _PX_HA, 2),
        "change_severity":    severity,
    }


# ---------------------------------------------------------------------------
# Tool 4 — generate_visualization
# ---------------------------------------------------------------------------

def generate_visualization(region: str = "default",
                            tile_index: int = 0,
                            save_dir: Optional[str] = None) -> dict:
    """
    Generate a before/after change-detection visualisation for a tile.

    Creates assets/agent_viz_{tile_index}.png.

    Args:
        region:     Informational label for the plot title
        tile_index: Which tile to visualise
        save_dir:   Override output directory (defaults to assets/)

    Returns dict with keys:
        image_path, tile_index, description
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    meta = _load_tiles_meta()
    if meta is None:
        return {"error": "No tiles found. Run data_pipeline/run_pipeline.py first."}

    tiles = meta["tiles"]
    if tile_index >= len(tiles):
        tile_index = 0

    fname    = tiles[tile_index]["filename"]
    before_n, after_n = _load_tile_pair(fname)
    if before_n is None:
        return {"error": f"Could not load tile {fname}"}

    from data_pipeline.preprocess import denormalize_tile

    def rgb(norm):
        raw = denormalize_tile(norm.astype(np.float32))
        img = np.stack([raw[2], raw[1], raw[0]], -1)
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

    def bin_img(mask):
        h, w = mask.shape
        out = np.zeros((h, w, 3))
        out[mask > 0.5] = [0.83, 0.19, 0.19]
        out[mask <= 0.5] = [0.11, 0.37, 0.13]
        return out

    br = denormalize_tile(before_n.astype(np.float32))
    ar = denormalize_tile(after_n.astype(np.float32))
    change_mask = _ndvi_baseline_pred(before_n, after_n)

    # Semi-transparent red overlay on after RGB
    after_rgb = rgb(after_n).copy()
    overlay   = after_rgb.copy()
    overlay[change_mask > 0.5] = [0.9, 0.15, 0.15]
    blended   = 0.55 * after_rgb + 0.45 * overlay

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.suptitle(f"Change Detection — {region}  |  Tile {tile_index}  "
                 f"(2019 → 2022)", fontsize=11, fontweight="bold")

    axes[0].imshow(rgb(before_n)); axes[0].set_title("Before (2019)"); axes[0].axis("off")
    axes[1].imshow(rgb(after_n));  axes[1].set_title("After (2022)");  axes[1].axis("off")
    axes[2].imshow(blended);
    axes[2].set_title("Change Overlay\n(red = detected deforestation)")
    axes[2].axis("off")

    patches = [mpatches.Patch(color="#d32f2f", label="Deforested"),
               mpatches.Patch(color="#1b5e20", label="Forest / unchanged")]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()

    out_dir = Path(save_dir) if save_dir else _ASSETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"agent_viz_tile{tile_index:04d}.png"
    plt.savefig(img_path, dpi=130, bbox_inches="tight")
    plt.close()

    change_pct = round(float(change_mask.mean()) * 100, 2)
    return {
        "image_path":   str(img_path),
        "tile_index":   tile_index,
        "tile_file":    fname,
        "change_pct":   change_pct,
        "description":  (
            f"Three-panel visualisation saved to {img_path}. "
            f"Estimated {change_pct}% of this tile shows deforestation signal "
            f"(NDVI-based overlay; Prithvi overlay available after checkpoint download)."
        ),
    }


# ---------------------------------------------------------------------------
# Tool registry — maps name → callable for the agent dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "run_inference":       run_inference,
    "fetch_historical_data": fetch_historical_data,
    "compute_change_stats": compute_change_stats,
    "generate_visualization": generate_visualization,
}
