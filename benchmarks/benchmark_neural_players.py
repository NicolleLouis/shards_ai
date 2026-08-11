"""Benchmark a trained NeuralPlayer against RandomPlayer or HeuristicPlayer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer, build_neural_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


def play_game(
    seed: int,
    checkpoint: Path,
    opponent: str,
    profile_path: Path | None,
    opponent_checkpoint: Path | None,
    max_actions: int,
    max_turns: int | None,
    neural_scorer=None,
    opponent_scorer=None,
) -> dict[str, object]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = neural_id.opponent
    neural = build_neural_player(
        neural_id,
        game,
        root_rng.derive("neural"),
        checkpoint_path=checkpoint if neural_scorer is None else None,
        scorer=neural_scorer,
    )
    if opponent == "random":
        other = RandomPlayer(opponent_id, root_rng.derive("opponent"))
    elif opponent == "neural":
        other = build_neural_player(
            opponent_id,
            game,
            root_rng.derive("opponent"),
            checkpoint_path=opponent_checkpoint if opponent_scorer is None else None,
            scorer=opponent_scorer,
        )
    else:
        profile = load_profile(profile_path) if profile_path else None
        other = HeuristicPlayer(
            opponent_id,
            profile.weights if profile else None,
            profile.card_acquisition_weights if profile else None,
            profile.constraint_weights if profile else None,
        )
    runner = GameRunner(
        game,
        {neural_id: neural, opponent_id: other},
        max_actions=max_actions,
        max_turns=max_turns,
    )
    started = time.perf_counter()
    try:
        state = runner.run()
    except Exception as error:
        raise RuntimeError(f"Neural benchmark failed for seed={seed}") from error
    elapsed = time.perf_counter() - started
    return {
        "seed": seed,
        "status": state.status.value,
        "neural_won": state.winner is neural_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "actions": runner.actions_played,
        "neural_decisions": neural.decisions,
        "neural_inference_seconds": neural.total_inference_seconds,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("random", "heuristic", "neural"), default="random")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.opponent == "heuristic" and args.profile is not None and not args.profile.exists():
        parser.error(f"heuristic profile not found: {args.profile}")
    if args.opponent == "neural" and (args.opponent_checkpoint is None or not args.opponent_checkpoint.exists()):
        parser.error("--opponent-checkpoint is required for a neural opponent")

    torch.set_num_threads(args.torch_threads)
    neural_scorer = NeuralPlayer.load_scorer(args.checkpoint)
    opponent_scorer = (
        NeuralPlayer.load_scorer(args.opponent_checkpoint)
        if args.opponent == "neural"
        else None
    )

    results = [
        play_game(args.seed + index, args.checkpoint, args.opponent, args.profile, args.opponent_checkpoint,
                  args.max_actions, args.max_turns, neural_scorer, opponent_scorer)
        for index in range(args.games)
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "opponent": args.opponent,
        "opponent_checkpoint": str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
        "games": len(results),
        "neural_wins": sum(result["neural_won"] for result in results),
        "opponent_wins": sum(result["opponent_won"] for result in results),
        "draws": sum(result["draw"] for result in results),
        "total_actions": sum(result["actions"] for result in results),
        "neural_decisions": sum(result["neural_decisions"] for result in results),
        "neural_inference_seconds": sum(result["neural_inference_seconds"] for result in results),
        "elapsed_seconds": sum(result["elapsed_seconds"] for result in results),
    }
    summary["neural_win_rate"] = summary["neural_wins"] / args.games
    summary["draw_rate"] = summary["draws"] / args.games
    summary["average_neural_inference_ms"] = (
        1000 * summary["neural_inference_seconds"] / summary["neural_decisions"]
        if summary["neural_decisions"] else 0.0
    )
    payload = {"summary": summary, "games": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
