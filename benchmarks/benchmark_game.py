"""Minimal reference benchmark for the deterministic duel engine.

Run with: poetry run python benchmarks/benchmark_game.py
Profile with: poetry run python -m cProfile -s cumulative benchmarks/benchmark_game.py --games 1000
"""

import argparse
from time import perf_counter

from shards_ai.game import (
    AssignPower,
    Game,
    GameStatus,
    PassPlayPhase,
    PlayCard,
    StopBuying,
)


def play_game(seed: int) -> None:
    game = Game.new(seed=seed)
    while game.state.status is GameStatus.RUNNING:
        while game.active.hand:
            game.apply(PlayCard(game.active.hand[-1].instance_id))
        game.apply(PassPlayPhase())
        game.apply(StopBuying())
        game.apply(AssignPower(game.active.power))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")

    games = args.games
    started_at = perf_counter()
    for seed in range(games):
        play_game(seed)
    elapsed = perf_counter() - started_at
    print(f"games={games} elapsed_seconds={elapsed:.3f} games_per_second={games / elapsed:.1f}")


if __name__ == "__main__":
    main()
