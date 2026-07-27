"""Reference benchmark for two HeuristicPlayers.

Run with: PYTHONPATH=. poetry run python benchmarks/benchmark_heuristic_players.py --games 1000
Profile with: PYTHONPATH=. poetry run python -m cProfile -s cumulative benchmarks/benchmark_heuristic_players.py --games 100
"""

from __future__ import annotations

import argparse
from time import perf_counter

from shards_ai.ai import HeuristicPlayer
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


def play_game(seed: int) -> GameStatus:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    players = {
        player_id: HeuristicPlayer(player_id)
        for player_id in PlayerId
    }
    return GameRunner(game, players).run().status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_000)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")

    started_at = perf_counter()
    statuses = [play_game(seed) for seed in range(args.games)]
    elapsed = perf_counter() - started_at
    finished = statuses.count(GameStatus.FINISHED)
    draws = statuses.count(GameStatus.DRAW)
    print(
        f"games={args.games} elapsed_seconds={elapsed:.3f} "
        f"games_per_second={args.games / elapsed:.1f} "
        f"finished={finished} draws={draws}"
    )


if __name__ == "__main__":
    main()
