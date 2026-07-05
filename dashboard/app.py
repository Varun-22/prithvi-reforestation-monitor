"""
Prithvi Reforestation Monitor — Streamlit Dashboard

Run from project root:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Prithvi Reforestation Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.data_loader import (
    load_tiles_metadata, load_tile_pair,
    tile_to_rgb, tile_to_ndvi_rgb, compute_change_overlay, tile_change_stats,
    get_metrics_rows, ASSETS_DIR,
)

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 Prithvi Reforestation Monitor")
    st.markdown(
        "Fine-tuned NASA/IBM **Prithvi-100M** geospatial ViT for "
        "deforestation monitoring over **Rondônia, Brazil** (2019 → 2022)."
    )
    st.markdown("---")

    meta = load_tiles_metadata()
    if meta:
        st.success(f"✅ {meta['n_tiles']} tile pairs loaded")
        tile_idx = st.slider("Tile index", 0, meta["n_tiles"] - 1, 0,
                             help="Select which 224×224 patch to inspect")
    else:
        st.warning("No tiles found. Run `data_pipeline/run_pipeline.py` first.")
        tile_idx = 0

    st.markdown("---")
    st.markdown(
        "**Study area:** Ji-Paraná region, Rondônia  \n"
        "**Before:** July–Sep 2019  \n"
        "**After:** July–Sep 2022  \n"
        "**Resolution:** 20 m (Sentinel-2)"
    )
    st.markdown("---")
    st.markdown(
        "[GitHub](https://github.com/Varun-22/prithvi-reforestation-monitor) · "
        "[Kaggle Notebook](training/kaggle_notebook.ipynb)"
    )


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_map, tab_metrics, tab_chat = st.tabs(
    ["🗺️ Change Detection", "📊 Model Metrics", "💬 Agent Chat"]
)


# ===========================================================================
# Tab 1 — Change Detection Map
# ===========================================================================
with tab_map:
    st.header("Sentinel-2 Change Detection — Rondônia")

    if meta is None:
        # No tiles available: show the static sample from assets/
        st.info(
            "Tile data not found locally. Showing a synthetic preview.  \n"
            "Run `python -m data_pipeline.run_pipeline` to fetch real imagery."
        )
        preview_path = ASSETS_DIR / "sample_predictions.png"
        if preview_path.exists():
            st.image(str(preview_path), use_column_width=True)
        else:
            st.warning("No preview image found in assets/ either.")
    else:
        fname = meta["tiles"][tile_idx]["filename"]
        before_n, after_n = load_tile_pair(fname)

        if before_n is None:
            st.error(f"Could not load tile {fname}")
        else:
            # ── Imagery row ───────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)

            before_rgb  = tile_to_rgb(before_n)
            after_rgb   = tile_to_rgb(after_n)
            overlay_rgb = compute_change_overlay(before_n, after_n)

            col1.image(before_rgb,  caption="Before (2019)",               use_column_width=True)
            col2.image(after_rgb,   caption="After (2022)",                 use_column_width=True)
            col3.image(overlay_rgb, caption="Change Overlay (red = deforested)", use_column_width=True)

            # ── NDVI row ──────────────────────────────────────────────────────
            with st.expander("NDVI maps", expanded=False):
                c1, c2 = st.columns(2)
                c1.image(tile_to_ndvi_rgb(before_n), caption="NDVI Before", use_column_width=True)
                c2.image(tile_to_ndvi_rgb(after_n),  caption="NDVI After",  use_column_width=True)

            # ── Stats ─────────────────────────────────────────────────────────
            st.markdown("#### Tile statistics")
            stats = tile_change_stats(before_n, after_n)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("NDVI Before",     f"{stats['ndvi_before']:.3f}")
            m2.metric("NDVI After",      f"{stats['ndvi_after']:.3f}",
                      delta=f"{stats['ndvi_delta']:+.3f}")
            m3.metric("Forest cover (before)", f"{stats['forest_before_%']:.1f}%")
            m4.metric("Deforested",      f"{stats['deforested_%']:.1f}%")
            m5.metric("Area lost",       f"{stats['deforested_ha']:.1f} ha")

            # ── Agent viz images if any exist ─────────────────────────────────
            agent_imgs = sorted(ASSETS_DIR.glob("agent_viz_tile*.png"))
            if agent_imgs:
                with st.expander(f"Agent-generated visualisations ({len(agent_imgs)} saved)"):
                    for p in agent_imgs[-3:]:   # show last 3
                        st.image(str(p), caption=p.name, use_column_width=True)


# ===========================================================================
# Tab 2 — Model Metrics
# ===========================================================================
with tab_metrics:
    st.header("Model Comparison — IoU and F1")

    metrics_img = ASSETS_DIR / "metrics_comparison.png"
    if metrics_img.exists():
        st.image(str(metrics_img), use_column_width=True)
    else:
        st.warning("Run `python -m evaluation.evaluate` to generate the comparison chart.")

    st.markdown("#### Metrics table")

    rows = get_metrics_rows()
    import pandas as pd

    df = pd.DataFrame(rows).rename(columns={
        "model": "Model", "f1": "F1", "iou": "IoU",
        "precision": "Precision", "recall": "Recall"
    })

    # Highlight Prithvi row if it has real values
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else "TBD"

    for col in ["F1", "IoU", "Precision", "Recall"]:
        if col in df.columns:
            df[col] = df[col].apply(_fmt)

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "Placeholder baseline values shown until `evaluation/evaluate.py` is run "
        "with `training/checkpoints/best_model.pth` present (download from Kaggle)."
    )


# ===========================================================================
# Tab 3 — Agent Chat
# ===========================================================================
with tab_chat:
    st.header("Geospatial Agent Chat")
    st.markdown(
        "Ask questions about deforestation in Rondônia. "
        "The agent uses your local tiles and the Prithvi model to reason and respond."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not set.** The agent requires a Claude API key.  \n"
            "Copy `.env.example` → `.env` and add your key, then restart the app."
        )
        st.code("cp .env.example .env\n# edit .env and add: ANTHROPIC_API_KEY=sk-ant-...")
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Suggested queries ─────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**Suggested queries:**")
        suggestions = [
            "What changed in Rondônia between 2019 and 2022?",
            "How much forest was lost? Give me the area in hectares.",
            "Show me a visualisation of the deforestation.",
            "Compare the NDVI values before and after.",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, key=f"sugg_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

    # ── Conversation history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── New input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about deforestation in Rondônia…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent reasoning…"):
                try:
                    from agent.react_agent import ReActAgent
                    agent   = ReActAgent(verbose=False)
                    answer  = agent.run(prompt)
                except Exception as e:
                    answer = f"⚠️ Agent error: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
