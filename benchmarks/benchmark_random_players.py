"""Reference benchmark for one HeuristicPlayer versus one RandomPlayer.

Run with: poetry run python -m benchmarks.benchmark_random_players --games 1000
Profile with: poetry run python -m cProfile -s cumulative -m benchmarks.benchmark_random_players --games 100
"""

import argparse
from time import perf_counter

from shards_ai.ai import HeuristicPlayer, RandomPlayer
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


def play_game(seed: int) -> GameStatus:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    heuristic_player_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    players = {
        player_id: (
            HeuristicPlayer(player_id)
            if player_id is heuristic_player_id
            else RandomPlayer(player_id, root_rng.derive(f"player-{player_id.value}"))
        )
        for player_id in PlayerId
    }
    runner = GameRunner(game, players)
    state = runner.run()
    return state.status


def main() -> None:
    parser = argparse.ArgumentParser()
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
