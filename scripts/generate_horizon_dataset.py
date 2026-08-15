#!/usr/bin/env python3
"""Generate canonical and turn-number-only horizon forecast datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from shards_ai.ai.card_visibility import forecast_card_visibility
from shards_ai.ai.heuristic_player import HeuristicPlayer
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.ai.horizon_forecast import (
    HORIZON_DATASET_SCHEMA_VERSION,
    HORIZON_FEATURE_SET_V1,
    features_from_state,
    features_to_record,
    horizon_class_for_remaining_turns,
    model_for_feature_set,
    project_baseline_dataset,
)
from shards_ai.ai.player_factory import build_neural_player
from shards_ai.ai.random_player import RandomPlayer
from shards_ai.game import BuyCard, Game, GameRandom, GameRunner, GameStatus, PlayerId


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--heuristic-profile", action="append", type=Path, default=[])
    value.add_argument("--neural-checkpoint", action="append", type=Path, default=[])
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--games-per-matchup", type=int, required=True)
    value.add_argument("--seed", type=int, required=True)
    value.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    value.add_argument("--max-turns", type=int, default=GameRunner.MAX_TURNS_PER_PLAYER * 2)
    value.add_argument("--no-random", action="store_true")
    value.add_argument(
        "--horizon-checkpoint",
        type=Path,
        help="Optional trained horizon V1 classifier used to create BuyCard visibility records.",
    )
    value.add_argument(
        "--visibility-output",
        type=Path,
        help="Optional JSONL path for deterministic BuyCard visibility records.",
    )
    return value


def game_seed(root_seed: int, index: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"horizon:{root_seed}:{index}".encode(), digest_size=8).digest(), "big")


def main() -> None:
    args = parser().parse_args()
    if args.games_per_matchup <= 0 or args.max_turns <= 0:
        raise SystemExit("games-per-matchup and max-turns must be positive")
    specs: list[tuple[str, str, Path | None]] = []
    if not args.no_random:
        specs.append(("random", "random", None))
    specs.extend(("heuristic", path.stem, path) for path in args.heuristic_profile)
    specs.extend(("neural", path.stem, path) for path in args.neural_checkpoint)
    if len(specs) < 2:
        raise SystemExit("At least two players are required")
    if len({profile_id for _, profile_id, _ in specs}) != len(specs):
        raise SystemExit("Player profile ids must be unique")
    if bool(args.horizon_checkpoint) != bool(args.visibility_output):
        raise SystemExit("--horizon-checkpoint and --visibility-output must be provided together")
    horizon_model = None
    if args.horizon_checkpoint:
        torch.set_num_threads(1)
        checkpoint = torch.load(args.horizon_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("feature_set") != HORIZON_FEATURE_SET_V1:
            raise SystemExit("Visibility generation requires an active-state V1 horizon checkpoint")
        horizon_model = model_for_feature_set(HORIZON_FEATURE_SET_V1)
        horizon_model.load_state_dict(checkpoint["model_state_dict"])
        horizon_model.eval()
    undirected = [(left, right) for index, left in enumerate(specs) for right in specs[index + 1:]]
    directed = undirected + [(right, left) for left, right in undirected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    visibility_records: list[dict] = []
    games = 0
    for matchup_index, (left, right) in enumerate(directed):
        for local_index in range(args.games_per_matchup):
            seed = game_seed(args.seed, matchup_index * args.games_per_matchup + local_index)
            root_rng = GameRandom(seed)
            game = Game.new(seed=seed, rng=root_rng.derive("engine"))
            players = {}
            for player_id, spec in ((PlayerId.PLAYER_1, left), (PlayerId.PLAYER_2, right)):
                kind, profile_id, path = spec
                rng = root_rng.derive(f"player-{player_id.value}")
                if kind == "random":
                    player = RandomPlayer(player_id, rng)
                elif kind == "heuristic":
                    profile = load_profile(path)
                    player = HeuristicPlayer(player_id, profile.weights, profile.card_acquisition_weights, profile.constraint_weights)
                else:
                    player = build_neural_player(player_id, game, rng, checkpoint_path=path)
                players[player_id] = player
            runner = GameRunner(game, players, max_actions=args.max_actions, max_turns=args.max_turns)
            game_records: list[dict] = []
            visibility_game_records: list[dict] = []
            turn_keys: list[tuple[int, str]] = []
            seen_turn: tuple[int, str] | None = None

            def on_decision(observation, legal_actions, _chosen, player_id) -> None:
                nonlocal seen_turn
                state = runner.game.state
                key = (state.turn_number, player_id.name)
                if key != seen_turn:
                    turn_keys.append(key)
                    seen_turn = key
                game_records.append({
                    "schema_version": HORIZON_DATASET_SCHEMA_VERSION,
                    "feature_set": "active_state_faction_counts_v1",
                    "game_id": f"horizon-game-{matchup_index:05d}-{local_index:05d}",
                    "game_seed": seed,
                    "decision_index": len(game_records),
                    "active_player": player_id.value,
                    "player_1_profile": left[1],
                    "player_2_profile": right[1],
                    "features": features_to_record(features_from_state(state)),
                    "_turn_position": len(turn_keys) - 1,
                })
                if horizon_model is not None:
                    with torch.inference_mode():
                        horizon_input = torch.tensor(
                            [features_from_state(state).v1_vector()], dtype=torch.float32
                        )
                        horizon_index = int(horizon_model(horizon_input).argmax(dim=1).item())
                    selected_horizon_class = "T6+" if horizon_index == 6 else f"T{horizon_index}"
                    for candidate_action in legal_actions:
                        if not isinstance(candidate_action, BuyCard):
                            continue
                        forecast = forecast_card_visibility(state, candidate_action, selected_horizon_class)
                        visibility_game_records.append({
                            "schema_version": HORIZON_DATASET_SCHEMA_VERSION,
                            "game_id": game_records[-1]["game_id"],
                            "game_seed": seed,
                            "decision_index": len(game_records) - 1,
                            "active_player": player_id.value,
                            "selected_horizon_class": selected_horizon_class,
                            "river_slot": candidate_action.river_slot,
                            "card_instance_id": candidate_action.card_instance_id,
                            "card_definition_id": state.river[candidate_action.river_slot].definition.card_id,
                            "visibility_class": forecast.visibility_class,
                            "effective_deck_size": forecast.snapshot.effective_deck_size,
                            "full_deck_cycle_turns": forecast.snapshot.full_deck_cycle_turns,
                            "remaining_draw_cycle_turns": forecast.snapshot.remaining_draw_cycle_turns,
                            "current_draw_pile_size": forecast.snapshot.current_draw_pile_size,
                            "future_deck_size": forecast.snapshot.future_deck_size,
                            "total_draw_amount": forecast.snapshot.total_draw_amount,
                        })

            try:
                final_state = runner.run(decision_observer=on_decision)
            except Exception as error:
                raise RuntimeError(f"failed game seed={seed}: {error}") from error
            if final_state.status is not GameStatus.FINISHED:
                continue
            for record in game_records:
                position = int(record.pop("_turn_position"))
                active_name = "PLAYER_1" if record["active_player"] == 1 else "PLAYER_2"
                remaining_turns = sum(
                    player == active_name for _, player in turn_keys[position + 1:]
                )
                record["target_horizon_class"] = horizon_class_for_remaining_turns(remaining_turns)
                records.append(record)
            visibility_records.extend(visibility_game_records)
            games += 1
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    baseline_path = args.output.with_name(args.output.stem + "_baseline.jsonl")
    baseline_records = project_baseline_dataset(args.output, baseline_path)
    visibility_path = None
    if args.visibility_output:
        visibility_path = args.visibility_output
        visibility_path.parent.mkdir(parents=True, exist_ok=True)
        with visibility_path.open("w", encoding="utf-8") as stream:
            for record in visibility_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema_version": HORIZON_DATASET_SCHEMA_VERSION,
        "canonical_dataset": str(args.output),
        "baseline_dataset": str(baseline_path),
        "games": games,
        "records": len(records),
        "baseline_records": baseline_records,
        "visibility_output": str(visibility_path) if visibility_path else None,
        "visibility_records": len(visibility_records),
        "players": [{"kind": kind, "profile_id": profile_id, "path": str(path) if path else None} for kind, profile_id, path in specs],
        "games_per_matchup": args.games_per_matchup,
        "seed": args.seed,
        "max_turns": args.max_turns,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
