#!/usr/bin/env python3
"""
Run the full data pipeline end-to-end.

    python -m data_pipeline.run_pipeline

Steps:
  1. fetch_imagery  — download S2 L2A bands from Planetary Computer
  2. tile_imagery   — cloud-mask, normalise, tile into 224×224 patches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    print("=" * 60)
    print("Prithvi Reforestation Monitor — Data Pipeline")
    print("=" * 60)

    print("\n[Step 1/2]  Fetching Sentinel-2 imagery...")
    from data_pipeline.fetch_imagery import main as fetch
    fetch()

    print("\n[Step 2/2]  Tiling + cloud-masking + normalising...")
    from data_pipeline.tile_imagery import main as tile
    tile()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
