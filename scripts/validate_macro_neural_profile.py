#!/usr/bin/env python3
"""Apply the neural promotion gate to a macro candidate without promoting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shards_ai.ai import (
    HeuristicPlayer,
    MacroNeuralPlayer,
    NeuralModelConfig,
    RandomPlayer,
    build_neural_scorer,
    build_neural_player,
    load_active_training_profile,
    load_training_profile,
)
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId

from scripts.validate_neural_profile import acceptance_metrics, _panel


def _load_scorer(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", "independent_action")
    scorer = build_neural_scorer(
        architecture,
        NeuralModelConfig(**checkpoint["model_config"]),
    )
    scorer.load_state_dict(checkpoint["model_state_dict"])
    scorer.eval()
    return scorer, architecture


def _macro_player(player_id, game, macro_scorer):
    def choose_macro(_game, observation, candidates):
        with torch.inference_mode():
            return int(macro_scorer(observation, candidates).argmax().item())

    return MacroNeuralPlayer(
        player_id,
        game,
        candidate_scorer=choose_macro,
    )


def _play(
    seed: int,
    candidate_macro_scorer,
    opponent: str,
    heuristic_profiles,
    neural_scorers,
    max_actions: int,
    max_turns: int | None,
    candidate_is_macro: bool,
) -> dict[str, object]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    candidate_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = candidate_id.opponent
    if candidate_is_macro:
        candidate = _macro_player(
            candidate_id, game, candidate_macro_scorer,
        )
    else:
        candidate = build_neural_player(
            candidate_id, game, root_rng.derive("candidate"), scorer=candidate_macro_scorer,
        )
    if opponent == "random":
        other = RandomPlayer(opponent_id, root_rng.derive("opponent"))
    elif opponent.startswith("neural:"):
        other = build_neural_player(
            opponent_id,
            game,
            root_rng.derive("opponent"),
            scorer=neural_scorers[opponent.removeprefix("neural:")],
        )
    else:
        profile = heuristic_profiles[opponent]
        other = HeuristicPlayer(opponent_id, profile.weights, profile.card_acquisition_weights, profile.constraint_weights)
    runner = GameRunner(game, {candidate_id: candidate, opponent_id: other}, max_actions=max_actions, max_turns=max_turns)
    state = runner.run()
    return {
        "seed": seed,
        "candidate_won": state.winner is candidate_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "status": state.status.value,
        "actions": runner.actions_played,
        "decisions": candidate.decisions,
        "macro_decisions": getattr(candidate, "macro_decisions", 0),
    }


def _aggregate(records):
    games = len(records)
    wins = sum(bool(record["candidate_won"]) for record in records)
    return {
        "games": games,
        "wins": wins,
        "losses": sum(bool(record["opponent_won"]) for record in records),
        "draws": sum(bool(record["draw"]) for record in records),
        "win_rate": wins / games if games else 0.0,
        "mean_decisions": sum(record["decisions"] for record in records) / games if games else 0.0,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, default=Path("configs/neural_training_profiles"))
    parser.add_argument("--active-profile", type=Path, default=Path("configs/neural_training_profiles/active.yaml"))
    parser.add_argument("--profile-v007", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_validation/macro_latest.json"))
    args = parser.parse_args()
    if args.games <= 0 or args.torch_threads <= 0:
        parser.error("games and torch-threads must be positive")

    candidate_profile = load_training_profile(args.candidate_profile)
    active_profile = load_active_training_profile(args.active_profile)
    opponents, heuristic_profiles, neural_profiles = _panel(args, candidate_profile.profile_id)
    candidate_scorer, candidate_architecture = _load_scorer(args.candidate_checkpoint)
    reference_checkpoint = active_profile.resolve_path(active_profile.output)
    reference_scorer, reference_architecture = _load_scorer(reference_checkpoint)
    neural_scorers = {
        profile_id: _load_scorer(checkpoint)[0]
        for profile_id, (_path, _profile, checkpoint) in neural_profiles.items()
    }
    if candidate_architecture not in {
        "structured_semantic_v5_macro_root_action_v2",
        "structured_semantic_v5_macro_known_consequence_v1",
        "structured_semantic_v5_macro_tactical_action_v1",
    }:
        parser.error(f"candidate must be a supported macro architecture, got {candidate_architecture!r}")
    by_opponent = {}
    for opponent in opponents:
        candidate_records = [
            _play(args.seed + index, candidate_scorer, opponent, heuristic_profiles, neural_scorers, args.max_actions, args.max_turns, True)
            for index in range(args.games)
        ]
        reference_records = [
            _play(args.seed + index, reference_scorer, opponent, heuristic_profiles, neural_scorers, args.max_actions, args.max_turns, False)
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
        print(f"completed={opponent} games={args.games}", flush=True)

    decision_metrics = acceptance_metrics(by_opponent)
    report = {
        "candidate_profile": candidate_profile.profile_id,
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "candidate_architecture": candidate_architecture,
        "reference_profile": active_profile.profile_id,
        "reference_checkpoint": str(reference_checkpoint),
        "games_per_opponent": args.games,
        "seed": args.seed,
        "opponents": opponents,
        "results": by_opponent,
        "decision_metrics": decision_metrics,
        "decision": "accepted" if decision_metrics["accepted"] else "rejected",
        "promotion": None,
        "promotion_note": "Macro candidate was evaluated with the existing gate; automatic promotion is disabled until the macro player is the supported active inference contract.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Décision : {report['decision'].upper()}")
    return 0 if report["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
