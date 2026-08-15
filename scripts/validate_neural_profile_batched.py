#!/usr/bin/env python3
"""Validate a neural checkpoint in resumable, bounded-size batches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from shards_ai.ai import NeuralPlayer, load_active_training_profile, load_training_profile
from shards_ai.game import GameRunner
from scripts.validate_neural_profile import (
    _aggregate,
    _panel,
    _play,
    acceptance_metrics,
    format_validation_line,
    promote_candidate,
)


def batch_ranges(games: int, batch_games: int) -> list[tuple[int, int]]:
    if games <= 0:
        raise ValueError("games must be positive")
    if batch_games <= 0:
        raise ValueError("batch_games must be positive")
    return [(start, min(start + batch_games, games)) for start in range(0, games, batch_games)]


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _config(args, candidate_profile_id: str, active_profile_id: str, opponents: list[str]) -> dict[str, object]:
    return {
        "candidate_profile": candidate_profile_id,
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "reference_profile": active_profile_id,
        "games_per_opponent": args.games,
        "batch_games": args.batch_games,
        "seed": args.seed,
        "max_actions": args.max_actions,
        "max_turns": args.max_turns,
        "opponents": opponents,
    }


def _initial_state(config: dict[str, object], opponents: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config": config,
        "completed_games": {opponent: 0 for opponent in opponents},
        "results": {
            opponent: {"candidate_records": [], "reference_records": []}
            for opponent in opponents
        },
    }


def _load_state(path: Path, config: dict[str, object], opponents: list[str]) -> dict[str, object]:
    if not path.exists():
        return _initial_state(config, opponents)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or state.get("config") != config:
        raise ValueError(f"Progress file does not match this validation: {path}")
    if set(state.get("completed_games", {})) != set(opponents):
        raise ValueError(f"Progress file has a different opponent panel: {path}")
    return state


def validate_batched(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.torch_threads)
    candidate_profile = load_training_profile(args.candidate_profile)
    candidate_checkpoint = args.candidate_checkpoint or candidate_profile.resolve_path(candidate_profile.output)
    if not candidate_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_checkpoint}")
    args.candidate_checkpoint = candidate_checkpoint
    active_profile = load_active_training_profile(args.active_profile)
    reference_checkpoint = active_profile.resolve_path(active_profile.output)
    if not reference_checkpoint.exists():
        raise FileNotFoundError(f"Active reference checkpoint not found: {reference_checkpoint}")

    opponents, heuristic_profiles, neural_profiles, hybrid_profiles = _panel(args, candidate_profile.profile_id)
    config = _config(args, candidate_profile.profile_id, active_profile.profile_id, opponents)
    state = _load_state(args.progress_output, config, opponents)
    candidate_scorer = NeuralPlayer.load_scorer(candidate_checkpoint)
    reference_scorer = NeuralPlayer.load_scorer(reference_checkpoint)
    neural_scorers = {
        profile_id: NeuralPlayer.load_scorer(checkpoint)
        for profile_id, (_path, _profile, checkpoint) in neural_profiles.items()
    }
    scorers_by_checkpoint = {}
    for profile in hybrid_profiles.values():
        checkpoint = profile.acquisition_checkpoint.resolve()
        scorers_by_checkpoint.setdefault(str(checkpoint), NeuralPlayer.load_scorer(checkpoint))
    hybrid_scorers = {
        profile_id: scorers_by_checkpoint[str(profile.acquisition_checkpoint.resolve())]
        for profile_id, profile in hybrid_profiles.items()
    }

    batches = batch_ranges(args.games, args.batch_games)
    for batch_number, (start, end) in enumerate(batches, start=1):
        pending = [opponent for opponent in opponents if state["completed_games"][opponent] < end]
        if not pending:
            continue
        for opponent in pending:
            records = state["results"][opponent]
            for index in range(start, end):
                seed = args.seed + index
                records["candidate_records"].append(
                    _play(seed, candidate_scorer, opponent, heuristic_profiles, neural_scorers, hybrid_profiles,
                          args.max_actions, args.max_turns, hybrid_scorers)
                )
                records["reference_records"].append(
                    _play(seed, reference_scorer, opponent, heuristic_profiles, neural_scorers, hybrid_profiles,
                          args.max_actions, args.max_turns, hybrid_scorers)
                )
            state["completed_games"][opponent] = end
        _atomic_write(args.progress_output, state)
        print(f"Batch {batch_number}/{len(batches)} terminé ({end}/{args.games} parties par adversaire)", flush=True)

    by_opponent = {}
    for opponent in opponents:
        records = state["results"][opponent]
        candidate_summary = _aggregate(records["candidate_records"])
        reference_summary = _aggregate(records["reference_records"])
        delta = candidate_summary["win_rate"] - reference_summary["win_rate"]
        by_opponent[opponent] = {
            "candidate": candidate_summary,
            "reference": reference_summary,
            "delta_win_rate": delta,
            "improved": delta > 0,
            "not_regressed": delta >= 0,
        }
    decision_metrics = acceptance_metrics(by_opponent)
    accepted = bool(decision_metrics["accepted"])
    report = {
        "candidate_profile": candidate_profile.profile_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "reference_profile": active_profile.profile_id,
        "reference_checkpoint": str(reference_checkpoint),
        "games_per_opponent": args.games,
        "batch_games": args.batch_games,
        "seed": args.seed,
        "opponents": opponents,
        "results": by_opponent,
        "decision_metrics": decision_metrics,
        "decision": "accepted" if accepted else "rejected",
        "promotion": promote_candidate(args, candidate_profile, active_profile, candidate_checkpoint)
        if accepted and args.promote else None,
    }
    _atomic_write(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--profile-dir", type=Path, default=Path("configs/neural_training_profiles"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("configs/neural_profiles"))
    parser.add_argument("--active-profile", type=Path, default=Path("configs/neural_training_profiles/active.yaml"))
    parser.add_argument("--active-neural-profile", type=Path, default=Path("configs/neural_profiles/active.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--profile-hybrid-v001", type=Path, default=Path("configs/hybrid_profiles/hybrid-v001.yaml"))
    parser.add_argument("--profile-hybrid-v003", type=Path, default=Path("configs/hybrid_profiles/hybrid-v003.yaml"))
    parser.add_argument("--profile-hybrid-v004", type=Path, default=Path("configs/hybrid_profiles/hybrid-v004.yaml"))
    parser.add_argument("--profile-hybrid-v005", type=Path, default=Path("configs/hybrid_profiles/hybrid-v005.yaml"))
    parser.add_argument("--profile-hybrid-v006", type=Path, default=Path("configs/hybrid_profiles/hybrid-v006.yaml"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--batch-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_validation/latest.json"))
    parser.add_argument("--progress-output", type=Path, default=Path("artifacts/neural_validation/latest.progress.json"))
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.games <= 0 or args.batch_games <= 0:
        parser.error("--games and --batch-games must be positive")
    report = validate_batched(args)
    print(f"Candidat : {report['candidate_profile']}")
    print(f"Référence : {report['reference_profile']}")
    for opponent, result in report["results"].items():
        print(format_validation_line(opponent, result, report["candidate_profile"], report["reference_profile"]))
    print(f"Décision : {report['decision'].upper()}")
    return 0 if report["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
