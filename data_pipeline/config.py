"""Pipeline configuration — region, dates, bands, paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Study area: Ji-Paraná region, Rondônia, Brazil
# Active deforestation frontier; excellent Sentinel-2 dry-season coverage.
# 0.3° × 0.3° keeps local tile storage under ~200 MB for both time points.
# ---------------------------------------------------------------------------
BBOX = [-63.0, -10.7, -62.7, -10.4]  # [west, south, east, north] WGS84

# Two dry-season windows 3 years apart.  Dry season (Jun-Sep) maximises
# clear-sky probability in Rondônia; 3-year gap amplifies detectable change.
TIME_POINTS = {
    "before": ("2019-07-01", "2019-09-30"),
    "after":  ("2022-07-01", "2022-09-30"),
}

# ---------------------------------------------------------------------------
# Sentinel-2 L2A bands matching Prithvi-100M's HLS training channels
# Order matters: Blue, Green, Red, NIR-narrow, SWIR1, SWIR2
# ---------------------------------------------------------------------------
S2_BANDS   = ["B02", "B03", "B04", "B8A", "B11", "B12"]
BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]

# ---------------------------------------------------------------------------
# Download / preprocessing
# ---------------------------------------------------------------------------
MAX_CLOUD_COVER    = 20     # % — relax to 40 if no scenes found
TARGET_RESOLUTION  = 20     # metres (native res of SWIR/NIR bands)
REFLECTANCE_SCALE  = 10000.0

# Sentinel-2 Scene Classification Layer classes kept as valid land pixels
# 4=Vegetation  5=Bare soil/not-vegetated  6=Water  7=Unclassified
VALID_SCL = {4, 5, 6, 7}

# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
TILE_SIZE   = 224   # pixels — Prithvi input size
TILE_STRIDE = 200   # gives 24-px overlap between adjacent tiles
MIN_VALID_FRAC = 0.50  # discard tiles where >50% pixels are cloud/shadow

# ---------------------------------------------------------------------------
# Paths  (raw imagery and tiles are gitignored)
# ---------------------------------------------------------------------------
RAW_DIR   = PROJECT_ROOT / "data_pipeline" / "raw"
TILES_DIR = PROJECT_ROOT / "data_pipeline" / "tiles"
