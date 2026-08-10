"""Compare two neural checkpoints on identical game seeds and opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.benchmark_neural_panel import (
    NEURAL_PROFILE_PATHS,
    OPPONENTS,
    play_game,
)
from benchmarks.benchmark_neural_mix import _summary
from shards_ai.ai import NeuralPlayer
from shards_ai.ai.heuristic_profiles import load_profile


QUALITY_WEIGHTS = {
    "random": 0.25,
    "v007": 1.0,
    "v008": 1.5,
    "neural:v001": 0.25,
    "neural:v002": 0.25,
    "neural:v003": 0.25,
    "neural:v004": 0.25,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path, name in (
        (args.candidate_checkpoint, "candidate"),
        (args.reference_checkpoint, "reference"),
    ):
        if not path.exists():
            parser.error(f"{name} checkpoint not found: {path}")
    if args.games <= 0 or args.torch_threads <= 0:
        parser.error("--games and --torch-threads must be positive")

    torch.set_num_threads(args.torch_threads)
    heuristic_profiles = {
        "v007": load_profile(Path("configs/heuristic_profiles/v007.yaml")),
        "v008": load_profile(Path("configs/heuristic_profiles/v008.yaml")),
    }
    neural_scorers = {
        profile_id: NeuralPlayer.load_scorer(path)
        for profile_id, path in NEURAL_PROFILE_PATHS.items()
    }
    candidate_scorer = NeuralPlayer.load_scorer(args.candidate_checkpoint)
    reference_scorer = NeuralPlayer.load_scorer(args.reference_checkpoint)

    by_opponent = {}
    for opponent in OPPONENTS:
        candidate_records = [
            play_game(
                args.seed + index,
                candidate_scorer,
                opponent,
                heuristic_profiles,
                neural_scorers,
                args.max_actions,
                args.max_turns,
            )
            for index in range(args.games)
        ]
        reference_records = [
            play_game(
                args.seed + index,
                reference_scorer,
                opponent,
                heuristic_profiles,
                neural_scorers,
                args.max_actions,
                args.max_turns,
            )
            for index in range(args.games)
        ]
        candidate_summary = _summary(candidate_records)
        reference_summary = _summary(reference_records)
        by_opponent[opponent] = {
            "candidate": candidate_summary,
            "reference": reference_summary,
            "delta_win_rate": candidate_summary["neural_win_rate"] - reference_summary["neural_win_rate"],
        }
        print(
            f"completed={opponent} candidate={candidate_summary['neural_win_rate']:.3f} "
            f"reference={reference_summary['neural_win_rate']:.3f}",
            flush=True,
        )

    total_weight = sum(QUALITY_WEIGHTS.values())
    weighted_delta = sum(
        QUALITY_WEIGHTS[opponent] * result["delta_win_rate"]
        for opponent, result in by_opponent.items()
    ) / total_weight
    report = {
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "reference_checkpoint": str(args.reference_checkpoint),
        "games_per_opponent": args.games,
        "seed": args.seed,
        "opponents": list(OPPONENTS),
        "results": by_opponent,
        "weighted_delta_win_rate": weighted_delta,
        "decision": "diagnostic_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"weighted_delta_win_rate": weighted_delta}, sort_keys=True))


if __name__ == "__main__":
    main()
