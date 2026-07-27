"""Benchmark the simulation loop used by the analysis script.

Run with: poetry run python benchmarks/benchmark_analysis_campaign.py --games 10
Profile with: poetry run python -m cProfile -s cumulative benchmarks/benchmark_analysis_campaign.py --games 3
"""

import argparse
from time import perf_counter

from shards_ai.analysis import CampaignConfig, run_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")

    started_at = perf_counter()
    result = run_campaign(
        CampaignConfig(games=args.games, duration_seconds=3600, seed=42)
    )
    elapsed = perf_counter() - started_at
    print(
        f"games={result.completed} elapsed_seconds={elapsed:.3f} "
        f"games_per_second={result.completed / elapsed:.3f} "
        f"wins={sum(result.wins.values())} draws={result.draws} errors={len(result.errors)}"
    )


if __name__ == "__main__":
    main()
