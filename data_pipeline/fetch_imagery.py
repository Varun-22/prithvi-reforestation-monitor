#!/usr/bin/env python3
"""
Download Sentinel-2 L2A imagery for Rondônia from Microsoft Planetary Computer.

Usage (from project root):
    python -m data_pipeline.fetch_imagery

Outputs per time point into data_pipeline/raw/{before,after}/:
    B02.tif  B03.tif  B04.tif  B8A.tif  B11.tif  B12.tif  SCL.tif
    scene_meta.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

try:
    import planetary_computer as pc
    import pystac_client
except ImportError:
    print("Install missing deps: pip install planetary-computer pystac-client")
    sys.exit(1)

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.config import (
    BBOX, MAX_CLOUD_COVER, RAW_DIR, S2_BANDS,
    TARGET_RESOLUTION, TIME_POINTS,
)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------

def search_scenes(catalog, date_range: tuple, bbox: list, max_cloud: int):
    """Search PC for S2 L2A items; return list sorted by ascending cloud cover."""
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=30,
    )
    items = list(search.items())
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return items


def find_best_scene(catalog, time_label: str, date_range: tuple,
                    bbox: list, max_cloud: int):
    """Return the least-cloudy scene, relaxing cloud threshold if needed."""
    for threshold in [max_cloud, max_cloud * 2, 80]:
        items = search_scenes(catalog, date_range, bbox, threshold)
        if items:
            item = items[0]
            cc = item.properties.get("eo:cloud_cover", "?")
            print(f"  [{time_label}] {item.id}  |  cloud: {cc:.1f}%"
                  f"  (threshold relaxed to {threshold}%)" if threshold > max_cloud else
                  f"  [{time_label}] {item.id}  |  cloud: {cc:.1f}%")
            return item
    raise RuntimeError(
        f"No Sentinel-2 scenes found for '{time_label}' "
        f"({date_range[0]} → {date_range[1]}) even at 80% cloud cover."
    )


# ---------------------------------------------------------------------------
# COG windowed read
# ---------------------------------------------------------------------------

def read_band_window(asset_href: str, bbox_wgs84: list, target_res_m: int):
    """
    Read one COG asset clipped to bbox_wgs84, resampled to target_res_m metres.
    Returns (ndarray H×W, rasterio profile dict).
    """
    with rasterio.open(asset_href) as src:
        west, south, east, north = transform_bounds(
            "EPSG:4326", src.crs, *bbox_wgs84
        )
        window = window_from_bounds(west, south, east, north, src.transform)
        window = window.round_lengths().round_offsets()

        native_res = abs(float(src.transform.a))
        scale = native_res / target_res_m
        out_h = max(1, int(round(window.height * scale)))
        out_w = max(1, int(round(window.width  * scale)))

        data = src.read(
            1,
            window=window,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        )

        out_transform = rasterio.transform.from_bounds(
            west, south, east, north, out_w, out_h
        )
        profile = {
            "driver":    "GTiff",
            "dtype":     data.dtype,
            "width":     out_w,
            "height":    out_h,
            "count":     1,
            "crs":       src.crs,
            "transform": out_transform,
            "compress":  "deflate",
            "predictor": 2,
        }
        return data, profile


# ---------------------------------------------------------------------------
# Download one scene
# ---------------------------------------------------------------------------

def download_scene(item, time_label: str, bbox: list,
                   target_res_m: int, bands: list, out_dir: Path) -> dict:
    scene_dir = out_dir / time_label
    scene_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for band in bands + ["SCL"]:
        if band not in item.assets:
            print(f"    Warning: asset '{band}' missing from {item.id} — skipping")
            continue

        href = item.assets[band].href
        print(f"    {band} ... ", end="", flush=True)
        data, profile = read_band_window(href, bbox, target_res_m)

        out_path = scene_dir / f"{band}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)

        saved[band] = str(out_path)
        print(f"{data.shape[1]}×{data.shape[0]} px  →  {out_path.name}")

    meta = {
        "item_id":      item.id,
        "cloud_cover":  item.properties.get("eo:cloud_cover"),
        "datetime":     item.properties.get("datetime"),
        "bbox_wgs84":   bbox,
        "resolution_m": target_res_m,
        "files":        saved,
    }
    meta_path = scene_dir / "scene_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    Metadata → {meta_path}")
    return meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to Microsoft Planetary Computer STAC...")
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)

    for time_label, date_range in TIME_POINTS.items():
        print(f"\n{'='*60}")
        print(f"Time point: {time_label}  ({date_range[0]} → {date_range[1]})")
        print(f"{'='*60}")

        item = find_best_scene(
            catalog, time_label, date_range, BBOX, MAX_CLOUD_COVER
        )
        download_scene(
            item, time_label, BBOX, TARGET_RESOLUTION, S2_BANDS, RAW_DIR
        )

    print(f"\nDone.  Raw bands saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
