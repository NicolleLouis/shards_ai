#!/usr/bin/env python3
"""Generate round-robin imitation datasets for heuristic, neural and random players."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shards_ai.ai.league_dataset import (
    LeagueDatasetConfig,
    LeaguePlayerSpec,
    collect_league_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heuristic-profile",
        action="append",
        type=Path,
        default=[],
        help="Heuristic YAML path; repeat for each profile.",
    )
    parser.add_argument(
        "--neural-checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Neural checkpoint path; repeat for each profile.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _profile_id(path: Path) -> str:
    return path.stem


def main() -> None:
    args = build_parser().parse_args()
    players: list[LeaguePlayerSpec] = []
    if not args.no_random:
        players.append(LeaguePlayerSpec("random", "random"))
    players.extend(
        LeaguePlayerSpec("heuristic", _profile_id(path), path)
        for path in args.heuristic_profile
    )
    players.extend(
        LeaguePlayerSpec("neural", _profile_id(path), path)
        for path in args.neural_checkpoint
    )
    result = collect_league_dataset(
        LeagueDatasetConfig(
            players=tuple(players),
            output_dir=args.output_dir,
            seed=args.seed,
            games_per_matchup=args.games_per_matchup,
            max_actions=args.max_actions,
            max_turns=args.max_turns,
            strict_errors=not args.continue_on_error,
        )
    )
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "attempted_games": result.attempted_games,
        "completed_games": result.completed_games,
        "decision_count": result.decision_count,
        "error_count": result.error_count,
        "variants": {name: str(path) for name, path in result.variant_paths.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
