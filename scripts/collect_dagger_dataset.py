#!/usr/bin/env python3
"""Collect v008 labels on states visited by the current NeuralPlayer."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer
from shards_ai.ai.action_representation import representation_for_neural_action
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.analysis.dagger_dataset import (
    SCHEMA_VERSION, raw_ranks, serialize_action, state_signature, teacher_play_phase_end,
)
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, Phase, PlayerId
from shards_ai.game.observation import OBSERVATION_SCHEMA_VERSION


def parse_opponents(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"opponent must use label=path: {value}")
        label, path = value.split("=", 1)
        result.append((label, Path(path)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--dagger-stage", default="dagger_1")
    parser.add_argument("--opponent", action="append", required=True, help="label=profile-or-checkpoint; repeatable")
    parser.add_argument("--games-per-opponent", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.games_per_opponent <= 0 or args.torch_threads <= 0:
        parser.error("games-per-opponent and torch-threads must be positive")
    opponents = parse_opponents(args.opponent)
    if not opponents or not args.checkpoint.exists() or not args.profile.exists():
        parser.error("checkpoint, profile and opponents must exist")
    for _label, path in opponents:
        if not path.exists():
            parser.error(f"opponent path not found: {path}")
    torch.set_num_threads(args.torch_threads)
    scorer = NeuralPlayer.load_scorer(args.checkpoint)
    opponent_scorers = {
        label: NeuralPlayer.load_scorer(path)
        for label, path in opponents
        if path.suffix == ".pt"
    }
    profile = load_profile(args.profile)
    if profile.profile_id != "v008":
        parser.error(f"DAgGER teacher profile must be v008, got {profile.profile_id}")
    opponent_profiles = {
        label: load_profile(path)
        for label, path in opponents
        if path.suffix != ".pt"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    records = 0
    completed = 0
    errors = []
    with temporary.open("w", encoding="utf-8") as stream:
        for game_index in range(args.games_per_opponent * len(opponents)):
            label, opponent_path = opponents[game_index % len(opponents)]
            seed = args.seed + game_index
            root_rng = GameRandom(seed)
            game = Game.new(seed=seed, rng=root_rng.derive("engine"))
            neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
            opponent_id = neural_id.opponent
            neural = NeuralPlayer(neural_id, None, root_rng.derive("neural"), scorer=scorer)
            if opponent_path.suffix == ".pt":
                other = NeuralPlayer(opponent_id, None, root_rng.derive("opponent"), scorer=opponent_scorers[label])
            else:
                other_profile = opponent_profiles[label]
                other = HeuristicPlayer(opponent_id, other_profile.weights, other_profile.card_acquisition_weights, other_profile.constraint_weights)
            teacher = HeuristicPlayer(neural_id, profile.weights, profile.card_acquisition_weights, profile.constraint_weights)
            runner = GameRunner(game, {neural_id: neural, opponent_id: other}, max_actions=args.max_actions, max_turns=args.max_turns)
            phase_context = None
            first_divergence = False
            game_records = []

            def observe(observation, legal_actions, chosen, player_id):
                nonlocal phase_context, first_divergence
                if player_id is not neural_id:
                    return
                phase = observation.phase
                if phase == Phase.PLAY.value and (phase_context is None or phase_context["phase"] != phase):
                    phase_context = {
                        "id": f"{seed}-play-{observation.turn_number}",
                        "phase": phase,
                        "start_state": copy.deepcopy(game.state),
                        "start_rng": copy.deepcopy(game._rng),
                        "records": [],
                    }
                representations = [representation_for_neural_action(action, observation) for action in legal_actions]
                teacher_action = teacher.choose_action(game.state, legal_actions)
                scores = [teacher.score_action(game.state, action) for action in legal_actions]
                ranks = raw_ranks(scores)
                neural_index = legal_actions.index(chosen)
                teacher_index = legal_actions.index(teacher_action)
                divergent = chosen != teacher_action
                after_first = first_divergence
                first_divergence = first_divergence or divergent
                record = {
                    "dataset_schema_version": SCHEMA_VERSION,
                    "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                    "game_id": f"{args.dagger_stage}-{seed}", "game_seed": seed,
                    "decision_index": len(game_records), "turn_number": observation.turn_number,
                    "acting_player": player_id.value, "teacher_profile_id": profile.profile_id,
                    "teacher_profile_path": str(args.profile), "opponent_type": label,
                    "observation": asdict(observation),
                    "legal_actions": [serialize_action(action, representation) for action, representation in zip(legal_actions, representations, strict=True)],
                    "action_representations": [representation.to_dict() for representation in representations],
                    "teacher_scores": scores, "heuristic_scores": scores,
                    "teacher_raw_ranks": ranks, "heuristic_raw_ranks": ranks,
                    "neural_action_index": neural_index, "teacher_action_index": teacher_index,
                    # Keep the historical training contract: chosen_action is the teacher label.
                    "chosen_action_index": teacher_index,
                    "chosen_action": serialize_action(teacher_action, representations[teacher_index]),
                    "neural_action": serialize_action(chosen, representations[neural_index]),
                    "teacher_action": serialize_action(teacher_action, representations[teacher_index]),
                    "neural_action_type": representations[neural_index].action_type,
                    "teacher_action_type": representations[teacher_index].action_type,
                    "teacher_rank": ranks[neural_index],
                    "teacher_top3": ranks[neural_index] <= 3,
                    "regret": scores[teacher_index] - scores[neural_index],
                    "first_divergence": divergent and not after_first,
                    "decision_after_first_divergence": after_first,
                    "play_phase_id": phase_context["id"] if phase_context else None,
                    "play_phase_equivalent": None, "strategic_divergence": None,
                }
                game_records.append(record)
                if phase_context:
                    phase_context["records"].append(record)

            def transition(before, action, after, player_id):
                nonlocal phase_context
                if player_id is not neural_id or phase_context is None:
                    return
                if game.state.phase is not Phase.PLAY:
                    teacher_end = teacher_play_phase_end(phase_context["start_state"], phase_context["start_rng"], neural_id, teacher)
                    actual_end = state_signature(game.state, game)
                    equivalent = actual_end == teacher_end
                    strategic = not equivalent
                    for record in phase_context["records"]:
                        record["play_phase_equivalent"] = equivalent
                        record["strategic_divergence"] = strategic
                    phase_context = None

            try:
                final_state = runner.run(decision_observer=observe, transition_observer=transition)
                for record in game_records:
                    record["final_outcome"] = "draw" if final_state.status is GameStatus.DRAW else ("win" if final_state.winner is neural_id else "loss")
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                records += len(game_records)
                completed += 1
            except Exception as error:
                errors.append({"seed": seed, "opponent": label, "error": repr(error)})
                raise
    temporary.replace(args.output)
    manifest = {
        "schema_version": SCHEMA_VERSION, "checkpoint": str(args.checkpoint),
        "teacher_profile": str(args.profile), "opponents": [{"label": label, "path": str(path)} for label, path in opponents],
        "dagger_stage": args.dagger_stage,
        "games_per_opponent": args.games_per_opponent, "attempted_games": args.games_per_opponent * len(opponents),
        "completed_games": completed, "decision_count": records, "errors": errors,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
