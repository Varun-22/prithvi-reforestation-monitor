"""
ReAct-style geospatial agent backed by the Anthropic Messages API.

The agent is given four tools (run_inference, fetch_historical_data,
compute_change_stats, generate_visualization) and reasons over their outputs
to answer questions about deforestation in Rondônia.

LLM calls are kept minimal:
  - Max MAX_ITERATIONS tool-use rounds per query.
  - Uses claude-haiku by default (fastest, cheapest); override via AGENT_MODEL env var.
  - Tools execute locally with no LLM involvement — only the reasoning step calls Claude.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.tools import TOOL_REGISTRY

MAX_ITERATIONS = 6   # hard cap on tool-call rounds per query

DEFAULT_MODEL  = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a geospatial change-detection analyst monitoring \
deforestation in Rondônia, Brazil. You have access to Sentinel-2 satellite \
imagery from 2019 and 2022 processed through a Prithvi-100M fine-tuned model.

When asked about a region or change event:
1. Call run_inference first to get aggregate change statistics.
2. Use fetch_historical_data to add scene-level context if useful.
3. Use compute_change_stats on a specific tile for pixel-level detail.
4. Call generate_visualization only if the user explicitly asks to see a map.
5. Synthesise findings into a concise plain-English summary.

Be direct. Report numbers (percentages, hectares). Flag events where \
forest_lost_pct > 5% as significant. If data is unavailable, say so clearly \
and tell the user which pipeline step to run."""


# ---------------------------------------------------------------------------
# Tool schema (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "run_inference",
        "description": (
            "Run change-detection inference on all available tiles for the study region. "
            "Returns aggregate statistics: forest coverage before/after, "
            "forest loss in % and hectares, model used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name, e.g. 'Rondônia' or 'default'.",
                },
                "date_range": {
                    "type": "string",
                    "description": "Date range description, e.g. '2019-2022'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "fetch_historical_data",
        "description": (
            "Return metadata about the available Sentinel-2 scenes for the region: "
            "scene IDs, acquisition dates, cloud cover, spatial coverage. "
            "Use this for context before running inference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name (e.g. 'Rondônia').",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compute_change_stats",
        "description": (
            "Compute detailed pixel-level statistics for a specific tile: "
            "mean NDVI before/after, deforestation percentage, estimated area lost, "
            "and a severity rating (low/medium/high)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tile_index": {
                    "type": "integer",
                    "description": "0-based tile index (use 0 for a representative sample).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_visualization",
        "description": (
            "Generate a before/after change-detection visualisation image for a tile "
            "and save it to assets/. Returns the image path and a description. "
            "Only call this when the user explicitly asks to see a map or image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region label for the plot title.",
                },
                "tile_index": {
                    "type": "integer",
                    "description": "Tile to visualise (default 0).",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ReActAgent:
    """ReAct agent: Reason → Act (tool) → Observe → repeat → Answer."""

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = True):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.model   = model
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tool_name: str, tool_input: dict) -> str:
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = fn(**tool_input)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def run(self, user_query: str) -> str:
        """
        Run the ReAct loop for a single user query.
        Returns the agent's final plain-English answer.
        """
        messages: list[dict] = [{"role": "user", "content": user_query}]
        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1
            if self.verbose:
                print(f"\n[Agent] Calling {self.model} (round {iterations})...", flush=True)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # --- End turn (final answer) ---
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return "(No text response from model.)"

            # --- Tool use ---
            if response.stop_reason != "tool_use":
                break   # unexpected stop reason — extract any text and return

            # Add the assistant's message (may contain text + tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if self.verbose:
                    print(f"  → Tool: {block.name}({json.dumps(block.input)})")

                result_text = self._dispatch(block.name, block.input)

                if self.verbose:
                    # Pretty-print truncated result
                    preview = result_text[:300] + "..." if len(result_text) > 300 else result_text
                    print(f"  ← {preview}")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        # Fallback: extract any text from the last response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return "Agent reached the iteration limit without a final answer."

    # ------------------------------------------------------------------
    # Interactive session
    # ------------------------------------------------------------------

    def chat(self) -> None:
        """Simple REPL — type 'exit' or Ctrl-C to quit."""
        print("Prithvi Geospatial Agent  |  type 'exit' to quit")
        print("─" * 52)
        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query:
                continue
            answer = self.run(query)
            print(f"\nAgent: {answer}")
