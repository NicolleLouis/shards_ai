#!/usr/bin/env python3
"""Run a local statistical campaign and write JSON, CSV and HTML/SVG reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shards_ai.analysis import CampaignConfig, run_campaign, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Stop on the first game error.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("analysis_output") / "random_vs_random",
    )
    args = parser.parse_args()
    config = CampaignConfig(
        duration_seconds=args.duration_seconds,
        games=args.games,
        seed=args.seed,
        max_actions=args.max_actions,
        max_turns=args.max_turns,
        strict=args.strict,
    )
    result = run_campaign(config)
    report_path = write_report(result, args.output_dir)
    print(f"seed={result.root_seed}")
    print(f"completed={result.completed} wins={sum(result.wins.values())} draws={result.draws} errors={len(result.errors)}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
