#!/usr/bin/env python3
"""Validate a deckbuilding PPO checkpoint inside the complete Hybrid V3 composition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.ai.neural_training_profiles import load_training_profile
from shards_ai.ai.player_factory import build_neural_player
from shards_ai.ai.rl_training import (
    NeuralActorCritic,
    PPOTrainingAcquisitionPolicy,
    _build_hybrid_training_opponent,
)
from shards_ai.ai.composed_player import build_hybrid_player
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId
from scripts.validate_neural_profile import QUALITY_OPPONENT_WEIGHTS, acceptance_metrics


def _play(
    model: NeuralActorCritic,
    composition_profile: str,
    opponent_name: str,
    seed: int,
    heuristic_profiles,
    *,
    max_actions: int,
    max_turns: int,
) -> dict[str, object]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    learner_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = learner_id.opponent
    acquisition_policy = PPOTrainingAcquisitionPolicy(
        learner_id, game, model, stochastic=False,
    )
    candidate = build_hybrid_player(
        learner_id,
        game,
        root_rng.derive("hybrid-candidate"),
        profile=composition_profile,
        acquisition_policy=acquisition_policy,
    )
    opponent = _build_hybrid_training_opponent(
        opponent_name,
        opponent_id,
        game,
        root_rng.derive("opponent"),
        heuristic_profiles,
    )
    state = GameRunner(
        game,
        {learner_id: candidate, opponent_id: opponent},
        max_actions=max_actions,
        max_turns=max_turns,
    ).run()
    return {
        "seed": seed,
        "candidate_won": state.winner is learner_id,
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


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-best", action="store_true",
                        help="Evaluate best_actor_critic_state_dict instead of latest model state.")
    parser.add_argument("--reference-profile", type=Path, default=Path("configs/neural_training_profiles/v006.yaml"))
    parser.add_argument("--reference-composition-profile", type=Path, default=Path("configs/hybrid_profiles/hybrid-v003.yaml"))
    parser.add_argument("--reference-checkpoint", type=Path, default=Path("configs/neural_profiles/v006.pt"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--batch-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_validation/hybrid_deckbuilding.json"))
    args = parser.parse_args()
    if args.games <= 0 or args.batch_games <= 0:
        parser.error("--games and --batch-games must be positive")
    torch.set_num_threads(args.torch_threads)

    candidate_profile = load_training_profile(args.candidate_profile)
    reference_profile = load_training_profile(args.reference_profile)
    if not args.reference_composition_profile.exists():
        parser.error(f"Reference composition profile not found: {args.reference_composition_profile}")
    candidate_checkpoint = args.candidate_checkpoint or candidate_profile.resolve_path(candidate_profile.output)
    if not candidate_checkpoint.exists():
        parser.error(f"Candidate checkpoint not found: {candidate_checkpoint}")
    if not args.reference_checkpoint.exists():
        parser.error(f"Reference checkpoint not found: {args.reference_checkpoint}")
    if candidate_profile.decision_family != "acquisition":
        parser.error("Candidate profile must be an acquisition PPO profile")

    candidate_payload = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    candidate_source = "latest"
    if args.candidate_best:
        best_state = candidate_payload.get("best_actor_critic_state_dict")
        if best_state is None:
            parser.error("Candidate checkpoint does not contain best_actor_critic_state_dict")
        candidate_payload = dict(candidate_payload)
        candidate_payload["actor_critic_state_dict"] = best_state
        candidate_source = "best"
    candidate_model = NeuralActorCritic.from_checkpoint(candidate_payload)
    reference_model = NeuralActorCritic.from_checkpoint(
        torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
    )
    heuristic_profiles = {
        name: load_profile(Path(f"configs/heuristic_profiles/{name}.yaml"))
        for name in ("v007", "v008")
    }
    opponents = list(QUALITY_OPPONENT_WEIGHTS)
    results = {}
    for opponent in opponents:
        candidate_records = []
        reference_records = []
        for index in range(args.games):
            seed = args.seed + index
            candidate_records.append(_play(
                candidate_model, candidate_profile.composition_profile or "configs/hybrid_profiles/hybrid-v003.yaml",
                opponent, seed, heuristic_profiles,
                max_actions=10000, max_turns=200,
            ))
            reference_records.append(_play(
                reference_model, str(args.reference_composition_profile),
                opponent, seed, heuristic_profiles,
                max_actions=10000, max_turns=200,
            ))
            if (index + 1) % args.batch_games == 0:
                print(f"{opponent}: {index + 1}/{args.games}", flush=True)
        candidate_summary = _aggregate(candidate_records)
        reference_summary = _aggregate(reference_records)
        results[opponent] = {
            "candidate": candidate_summary,
            "reference": reference_summary,
            "delta_win_rate": candidate_summary["win_rate"] - reference_summary["win_rate"],
        }

    decision_metrics = acceptance_metrics(results)
    report = {
        "candidate_profile": candidate_profile.profile_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_source": candidate_source,
        "reference_profile": reference_profile.profile_id,
        "reference_checkpoint": str(args.reference_checkpoint),
        "reference_composition_profile": str(args.reference_composition_profile),
        "games_per_opponent": args.games,
        "batch_games": args.batch_games,
        "seed": args.seed,
        "opponents": opponents,
        "results": results,
        "decision_metrics": decision_metrics,
        "decision": "accepted" if decision_metrics["accepted"] else "rejected",
        "promotion": None,
    }
    _atomic_write(args.output, report)
    print(json.dumps({"decision": report["decision"], "decision_metrics": decision_metrics}, sort_keys=True))
    return 0 if report["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
