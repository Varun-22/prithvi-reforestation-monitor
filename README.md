# Prithvi Reforestation Monitor

Fine-tuned NASA/IBM Prithvi-100M geospatial vision transformer for deforestation
and reforestation monitoring in Rondônia, Brazil — wrapped in an agentic reasoning
layer and an interactive Streamlit dashboard. Everything runs free: local dev on
Apple Silicon, GPU training on Kaggle/Colab free tier.

---

## What it does

1. **Data pipeline** — queries Sentinel-2 / HLS imagery via Microsoft Planetary
   Computer's STAC API for two time points ~1–2 years apart over Rondônia. Tiles
   imagery into 224×224 patches, cloud-masks, and normalizes.

2. **Model** — freezes Prithvi-100M's pretrained backbone and trains a lightweight
   change-detection head for binary forest / non-forest classification. A simple
   NDVI-threshold baseline provides a comparison floor.

3. **Evaluation** — IoU and F1 scores for both models, plus a visual comparison
   chart saved to `assets/`.

4. **Agent** — ReAct-style agent (Claude via Anthropic API) with tools:
   `run_inference`, `fetch_historical_data`, `compute_change_stats`,
   `generate_visualization`. Given a region + date range it reasons over model
   output and returns a plain-English change summary.

5. **Dashboard** — Streamlit app with before/after imagery, change overlay,
   metrics comparison, and a chat interface backed by the agent.

---

## Project structure

```
prithvi-reforestation-monitor/
├── data_pipeline/     # STAC query, tiling, cloud-masking, normalization
├── training/          # Fine-tuning script + Kaggle notebook
├── evaluation/        # IoU/F1 metrics + comparison chart
├── agent/             # ReAct agent with geospatial tools
├── dashboard/         # Streamlit app
├── notebooks/         # Exploratory notebooks
├── assets/            # Charts, screenshots, GIFs (committed)
├── requirements.txt
└── .env.example
```

Raw imagery tiles and model checkpoints are gitignored — see **Regenerating data**
below.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/prithvi-reforestation-monitor.git
cd prithvi-reforestation-monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (agent/dashboard) | Claude API key |
| `HF_TOKEN` | No | HuggingFace token for gated models |
| `REGION_BBOX` | No | Override default Rondônia bbox |

---

## Regenerating data

Data is **not** committed. Run the pipeline to recreate tiles locally:

```bash
cd data_pipeline
python fetch_imagery.py        # downloads raw HLS/Sentinel-2 scenes
python tile_imagery.py         # tiles into 224×224 patches
python preprocess.py           # cloud-mask + normalize
```

Tiles are saved to `data_pipeline/tiles/` (gitignored).

---

## Training on Kaggle (free GPU)

1. Run the data pipeline locally to generate tiles.
2. Upload `data_pipeline/tiles/` as a Kaggle Dataset named
   `prithvi-rondonia-tiles`.
3. Open `training/kaggle_notebook.ipynb` on Kaggle, attach the dataset,
   enable GPU accelerator (T4 × 2 or P100).
4. Run all cells. Download the saved checkpoint (`best_model.pth`) when done.
5. Place checkpoint at `training/checkpoints/best_model.pth` (gitignored).

---

## Running the dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## Results

*(Populated after Stage 4 — see `assets/` for charts.)*

| Model | IoU | F1 |
|---|---|---|
| NDVI Baseline | — | — |
| Prithvi fine-tuned | — | — |

---

## Architecture

```
Planetary Computer STAC
        │
        ▼
  data_pipeline/          ← fetch, tile, cloud-mask, normalize
        │
        ▼
  training/               ← frozen Prithvi-100M backbone + change-detection head
        │
        ▼
  evaluation/             ← IoU, F1 vs. NDVI baseline
        │
        ▼
  agent/                  ← ReAct loop (Claude) with geospatial tools
        │
        ▼
  dashboard/              ← Streamlit: imagery, overlay, metrics, chat
```

---

## Why Rondônia?

Rondônia, Brazil is one of the most documented active deforestation frontiers.
It has dense Sentinel-2/HLS coverage, published ground-truth change maps from
PRODES/MapBiomas, and is widely used in change-detection research — making it
straightforward to contextualize results.

---

## License

MIT
