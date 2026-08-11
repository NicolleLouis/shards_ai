#!/usr/bin/env python3
"""Generate strategic macro demonstrations from Heuristic V8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shards_ai.ai import MacroDatasetCampaignConfig, generate_macro_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-profile", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--heuristic-opponent", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--neural-opponent", type=Path, default=Path("configs/neural_profiles/v004.pt"))
    parser.add_argument("--output", type=Path, required=True)
    volume = parser.add_mutually_exclusive_group(required=True)
    volume.add_argument("--games", type=int)
    volume.add_argument("--target-decisions", type=int)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_macro_dataset(
        MacroDatasetCampaignConfig(
            teacher_profile_path=args.teacher_profile,
            heuristic_opponent_profile_path=args.heuristic_opponent,
            neural_opponent_checkpoint_path=args.neural_opponent,
            output_path=args.output,
            seed=args.seed,
            games=args.games,
            target_decisions=args.target_decisions,
            max_games=args.max_games,
            max_actions=args.max_actions,
            max_turns=args.max_turns,
            strict_errors=not args.continue_on_error,
        )
    )
    print(json.dumps({
        "output": str(result.output_path),
        "manifest": str(result.manifest_path),
        "attempted_games": result.attempted_games,
        "completed_games": result.completed_games,
        "decision_count": result.decision_count,
        "macro_decision_count": result.macro_decision_count,
        "atomic_decision_count": result.atomic_decision_count,
        "error_count": result.error_count,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
