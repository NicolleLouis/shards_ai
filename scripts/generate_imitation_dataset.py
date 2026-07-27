#!/usr/bin/env python3
"""Generate a masked JSONL dataset from validated heuristic profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shards_ai.ai import DatasetCampaignConfig, MatchupSpec, generate_dataset
from shards_ai.ai.heuristic_profiles import load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        type=Path,
        required=True,
        help="Validated heuristic profile path; repeat for a profile pool.",
    )
    parser.add_argument(
        "--opponent-profile",
        dest="opponent_profiles",
        action="append",
        type=Path,
        help="Opponent-only profile; repeat to build targeted teacher matchups.",
    )
    parser.add_argument("--output", type=Path, required=True)
    volume = parser.add_mutually_exclusive_group(required=True)
    volume.add_argument("--games", type=int)
    volume.add_argument("--target-decisions", type=int)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Exclude failed games and continue the campaign.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    matchups = None
    record_profile_ids = None
    profile_paths = tuple(args.profiles)
    if args.opponent_profiles:
        all_profiles = {path: load_profile(path) for path in (*profile_paths, *args.opponent_profiles)}
        matchups = tuple(
            [MatchupSpec(teacher)]
            + [MatchupSpec(teacher, opponent) for opponent in args.opponent_profiles]
            for teacher in profile_paths
        )
        matchups = tuple(matchup for teacher_matchups in matchups for matchup in teacher_matchups)
        record_profile_ids = frozenset(all_profiles[path].profile_id for path in profile_paths)
    result = generate_dataset(
        DatasetCampaignConfig(
            profile_paths=tuple(dict.fromkeys(profile_paths + tuple(args.opponent_profiles or ()))),
            output_path=args.output,
            seed=args.seed,
            games=args.games,
            target_decisions=args.target_decisions,
            max_games=args.max_games,
            max_actions=args.max_actions,
            max_turns=args.max_turns,
            matchups=matchups,
            record_profile_ids=record_profile_ids,
            strict_errors=not args.continue_on_error,
        )
    )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "manifest": str(result.manifest_path),
                "attempted_games": result.attempted_games,
                "completed_games": result.completed_games,
                "excluded_games": result.excluded_games,
                "decision_count": result.decision_count,
                "error_count": result.error_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
