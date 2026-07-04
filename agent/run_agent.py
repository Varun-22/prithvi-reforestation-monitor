#!/usr/bin/env python3
"""
CLI entry point for the Prithvi Geospatial Agent.

Usage:
    # Interactive chat
    python -m agent.run_agent

    # One-shot query
    python -m agent.run_agent --query "What changed in Rondônia between 2019 and 2022?"

    # Dry-run: execute tools only, skip LLM (useful when ANTHROPIC_API_KEY not set)
    python -m agent.run_agent --dry-run

Environment variables (set in .env):
    ANTHROPIC_API_KEY  — required for LLM calls
    AGENT_MODEL        — override model (default: claude-haiku-4-5-20251001)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # python-dotenv optional; can set env vars manually

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dry_run() -> None:
    """Execute each tool once with default args and print results (no LLM)."""
    from agent.tools import (
        run_inference, fetch_historical_data, compute_change_stats, generate_visualization
    )

    print("=" * 60)
    print("Dry-run: executing tools without LLM")
    print("=" * 60)

    tests = [
        ("run_inference",          lambda: run_inference()),
        ("fetch_historical_data",  lambda: fetch_historical_data()),
        ("compute_change_stats",   lambda: compute_change_stats(tile_index=0)),
        ("generate_visualization", lambda: generate_visualization(tile_index=0)),
    ]

    for name, fn in tests:
        print(f"\n── {name} ──")
        try:
            result = fn()
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"  ERROR: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prithvi Geospatial Agent")
    parser.add_argument("--query",    type=str, default=None,
                        help="One-shot query (skips interactive mode)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run tools without LLM — useful for testing without API key")
    parser.add_argument("--model",    type=str, default=None,
                        help="Override AGENT_MODEL env var")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Suppress per-round logging")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.model:
        os.environ["AGENT_MODEL"] = args.model

    from agent.react_agent import ReActAgent
    agent = ReActAgent(verbose=not args.no_verbose)

    if args.query:
        print(f"\nQuery: {args.query}\n")
        answer = agent.run(args.query)
        print(f"\nAnswer:\n{answer}")
    else:
        agent.chat()


if __name__ == "__main__":
    main()
