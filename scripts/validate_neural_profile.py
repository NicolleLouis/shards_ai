#!/usr/bin/env python3
"""Validate and optionally promote a neural checkpoint against a fixed opponent panel."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from shards_ai.ai import (
    HeuristicPlayer,
    NeuralPlayer,
    RandomPlayer,
    load_active_training_profile,
    load_training_profile,
    next_training_profile_id,
    save_training_profile,
    versioned_training_profiles,
)
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


def _play(
    seed: int,
    candidate_scorer,
    opponent: str,
    heuristic_profiles: dict[str, object],
    neural_scorers: dict[str, object],
    max_actions: int,
    max_turns: int | None,
    torch_threads: int,
) -> dict[str, object]:
    torch.set_num_threads(torch_threads)
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    candidate_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = candidate_id.opponent
    candidate = NeuralPlayer(candidate_id, None, root_rng.derive("candidate"), scorer=candidate_scorer)
    if opponent == "random":
        other = RandomPlayer(opponent_id, root_rng.derive("opponent"))
    elif opponent.startswith("neural:"):
        other = NeuralPlayer(
            opponent_id,
            None,
            root_rng.derive("opponent"),
            scorer=neural_scorers[opponent.removeprefix("neural:")],
        )
    else:
        profile = heuristic_profiles[opponent]
        other = HeuristicPlayer(
            opponent_id,
            profile.weights,
            profile.card_acquisition_weights,
            profile.constraint_weights,
        )
    state = GameRunner(
        game,
        {candidate_id: candidate, opponent_id: other},
        max_actions=max_actions,
        max_turns=max_turns,
    ).run()
    return {
        "seed": seed,
        "candidate_won": state.winner is candidate_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "status": state.status.value,
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    games = len(records)
    wins = sum(bool(record["candidate_won"]) for record in records)
    losses = sum(bool(record["opponent_won"]) for record in records)
    draws = sum(bool(record["draw"]) for record in records)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "records": records,
    }


def acceptance_decision(results: dict[str, dict[str, object]]) -> bool:
    """Apply the promotion rule to one aggregate result per opponent."""
    return bool(results) and all(item["delta_win_rate"] >= 0 for item in results.values()) and any(
        item["delta_win_rate"] > 0 for item in results.values()
    )


def _panel(args: argparse.Namespace, candidate_profile_id: str) -> tuple[list[str], dict[str, object], dict[str, object]]:
    heuristic_profiles = {
        "v007": load_profile(args.profile_v007),
        "v008": load_profile(args.profile_v008),
    }
    neural_profiles = {}
    for _number, path, profile in reversed(versioned_training_profiles(args.profile_dir)):
        if profile.profile_id == candidate_profile_id:
            continue
        checkpoint = profile.resolve_path(profile.output)
        if checkpoint.exists():
            neural_profiles[profile.profile_id] = (path, profile, checkpoint)
        if len(neural_profiles) == 2:
            break
    opponents = ["random", "v007", "v008", *(f"neural:{profile_id}" for profile_id in neural_profiles)]
    return opponents, heuristic_profiles, neural_profiles


def validate(args: argparse.Namespace) -> dict[str, object]:
    candidate_profile = load_training_profile(args.candidate_profile)
    candidate_checkpoint = args.candidate_checkpoint or candidate_profile.resolve_path(candidate_profile.output)
    if not candidate_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_checkpoint}")
    active_profile = load_active_training_profile(args.active_profile)
    reference_checkpoint = active_profile.resolve_path(active_profile.output)
    if not reference_checkpoint.exists():
        raise FileNotFoundError(f"Active reference checkpoint not found: {reference_checkpoint}")

    opponents, heuristic_profiles, neural_profiles = _panel(args, candidate_profile.profile_id)
    candidate_scorer = NeuralPlayer.load_scorer(candidate_checkpoint)
    reference_scorer = NeuralPlayer.load_scorer(reference_checkpoint)
    neural_scorers = {
        profile_id: NeuralPlayer.load_scorer(checkpoint)
        for profile_id, (_path, _profile, checkpoint) in neural_profiles.items()
    }
    by_opponent = {}
    for opponent in opponents:
        candidate_records = [
            _play(args.seed + index, candidate_scorer, opponent, heuristic_profiles, neural_scorers,
                  args.max_actions, args.max_turns, args.torch_threads)
            for index in range(args.games)
        ]
        reference_records = [
            _play(args.seed + index, reference_scorer, opponent, heuristic_profiles, neural_scorers,
                  args.max_actions, args.max_turns, args.torch_threads)
            for index in range(args.games)
        ]
        candidate_summary = _aggregate(candidate_records)
        reference_summary = _aggregate(reference_records)
        delta = candidate_summary["win_rate"] - reference_summary["win_rate"]
        by_opponent[opponent] = {
            "candidate": candidate_summary,
            "reference": reference_summary,
            "delta_win_rate": delta,
            "improved": delta > 0,
            "not_regressed": delta >= 0,
        }

    accepted = acceptance_decision(by_opponent)
    report = {
        "candidate_profile": candidate_profile.profile_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "reference_profile": active_profile.profile_id,
        "reference_checkpoint": str(reference_checkpoint),
        "games_per_opponent": args.games,
        "seed": args.seed,
        "opponents": opponents,
        "results": by_opponent,
        "decision": "accepted" if accepted else "rejected",
        "promotion": None,
    }
    if accepted and not args.no_promote:
        promoted_id = next_training_profile_id(args.profile_dir)
        promoted_checkpoint = args.checkpoint_dir / f"{promoted_id}.pt"
        if promoted_checkpoint.exists():
            raise FileExistsError(f"Promoted checkpoint already exists: {promoted_checkpoint}")
        promoted = replace(
            candidate_profile,
            profile_id=promoted_id,
            parent_profile_id=active_profile.profile_id,
            output=str(promoted_checkpoint),
        )
        promoted_path = args.profile_dir / f"{promoted_id}.yaml"
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
        checkpoint["profile_id"] = promoted.profile_id
        checkpoint["profile_fingerprint"] = promoted.fingerprint
        training_config = dict(checkpoint.get("training_config", {}))
        training_config.update({
            "profile_id": promoted.profile_id,
            "profile_fingerprint": promoted.fingerprint,
            "output": str(promoted_checkpoint),
        })
        checkpoint["training_config"] = training_config
        temporary_checkpoint = promoted_checkpoint.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, promoted_checkpoint)
        save_training_profile(promoted, promoted_path)
        args.active_profile.parent.mkdir(parents=True, exist_ok=True)
        args.active_profile.write_text(
            yaml.safe_dump({"schema_version": 1, "active_profile_id": promoted_id}, sort_keys=False),
            encoding="utf-8",
        )
        args.active_neural_profile.parent.mkdir(parents=True, exist_ok=True)
        args.active_neural_profile.write_text(
            yaml.safe_dump({"schema_version": 1, "active_profile_id": promoted_id}, sort_keys=False),
            encoding="utf-8",
        )
        report["promotion"] = {
            "profile_id": promoted_id,
            "profile_path": str(promoted_path),
            "checkpoint_path": str(promoted_checkpoint),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--profile-dir", type=Path, default=Path("configs/neural_training_profiles"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("configs/neural_profiles"))
    parser.add_argument("--active-profile", type=Path, default=Path("configs/neural_training_profiles/active.yaml"))
    parser.add_argument("--active-neural-profile", type=Path, default=Path("configs/neural_profiles/active.yaml"))
    parser.add_argument("--profile-v007", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_validation/latest.json"))
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Candidat : {report['candidate_profile']}")
    print(f"Référence : {report['reference_profile']}")
    for opponent, result in report["results"].items():
        candidate = result["candidate"]
        reference = result["reference"]
        print(
            f"{opponent}: candidat {candidate['wins']}/{candidate['games']} ({candidate['win_rate']:.1%}) | "
            f"référence {reference['wins']}/{reference['games']} ({reference['win_rate']:.1%}) | "
            f"delta {result['delta_win_rate']:+.1%} | "
            f"{'PROGRES' if result['improved'] else 'OK' if result['not_regressed'] else 'BAISSE'}"
        )
    print(f"Décision : {report['decision'].upper()}")
    if report["promotion"]:
        print(f"Profil actif : {report['promotion']['profile_id']}")
    return 0 if report["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
