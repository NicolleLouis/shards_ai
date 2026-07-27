"""Benchmark aggregation of winner deck snapshots.

Run with: poetry run python benchmarks/benchmark_analysis.py --games 100000
Profile with: poetry run python -m cProfile -s cumulative benchmarks/benchmark_analysis.py --games 10000
"""

import argparse
from time import perf_counter

from shards_ai.analysis import build_statistics
from shards_ai.game import CARD_CATALOG


def build_snapshots(games: int) -> list[dict[str, object]]:
    card_ids = tuple(CARD_CATALOG)
    snapshots = []
    for game_index in range(games):
        cards = {
            card_id: 1 + ((game_index + card_index) % 3)
            for card_index, card_id in enumerate(card_ids)
            if (game_index + card_index) % 4 != 0
        }
        snapshots.append({"game_index": game_index, "cards": cards})
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100_000)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")

    snapshots = build_snapshots(args.games)
    started_at = perf_counter()
    cards, factions, grouped = build_statistics(snapshots)
    elapsed = perf_counter() - started_at
    print(
        f"games={args.games} elapsed_seconds={elapsed:.3f} "
        f"snapshots_per_second={args.games / elapsed:.1f} "
        f"cards={len(cards)} factions={len(factions)} groups={len(grouped)}"
    )


if __name__ == "__main__":
    main()
