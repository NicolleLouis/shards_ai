"""Round-robin league collection for multi-profile imitation datasets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from shards_ai.game import CARD_CATALOG, Game, GameRandom, GameRunner, GameStatus, PlayerId
from shards_ai.game.observation import NeuralObservation, OBSERVATION_SCHEMA_VERSION

from .action_representation import (
    ACTION_REPRESENTATION_SCHEMA_VERSION,
    representation_for_neural_action,
)
from .heuristic_player import HeuristicPlayer
from .heuristic_profiles import load_profile
from .neural_player import NeuralPlayer
from .random_player import RandomPlayer


LEAGUE_DATASET_SCHEMA_VERSION = 1
RESULT_WEIGHTS = {"loss": 0.75, "draw": 1.0, "win": 1.25}
TEACHER_WEIGHTS = {
    "random": 0.10,
    "v001": 0.50,
    "v002": 0.60,
    "v003": 0.75,
    "v004": 0.90,
    "v007": 1.00,
    "v008": 1.50,
}


@dataclass(frozen=True, slots=True)
class LeaguePlayerSpec:
    kind: str
    profile_id: str
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class LeagueDatasetConfig:
    players: tuple[LeaguePlayerSpec, ...]
    output_dir: Path
    seed: int
    games_per_matchup: int
    max_actions: int = GameRunner.DEFAULT_MAX_ACTIONS
    max_turns: int | None = None
    strict_errors: bool = True


@dataclass(frozen=True, slots=True)
class LeagueDatasetResult:
    output_dir: Path
    attempted_games: int
    completed_games: int
    decision_count: int
    error_count: int
    variant_paths: dict[str, Path]


def collect_league_dataset(config: LeagueDatasetConfig) -> LeagueDatasetResult:
    _validate_config(config)
    matchups = [(left, right) for left_index, left in enumerate(config.players)
                 for right in config.players[left_index + 1:]]
    directed_matchups = [(left, right) for left, right in matchups] + [
        (right, left) for left, right in matchups
    ]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_dir / "league_records.jsonl.tmp"
    attempted_games = completed_games = decision_count = error_count = 0
    errors: list[str] = []
    totals = {"control_full_unweighted": 0.0, "weighted_moderate": 0.0, "winner_only": 0.0}
    counts = {"control_full_unweighted": 0, "weighted_moderate": 0, "winner_only": 0}

    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for matchup_index, (left_spec, right_spec) in enumerate(directed_matchups):
                for local_game_index in range(config.games_per_matchup):
                    game_index = matchup_index * config.games_per_matchup + local_game_index
                    attempted_games += 1
                    game_id = f"league-game-{game_index:08d}"
                    game_seed = _game_seed(config.seed, game_index)
                    try:
                        runner, players, scorers = _build_runner(
                            game_seed, left_spec, right_spec, config
                        )
                        records: list[dict[str, Any]] = []

                        def on_decision(observation, legal_actions, chosen, player_id) -> None:
                            player_spec = left_spec if player_id is PlayerId.PLAYER_1 else right_spec
                            player = players[player_id]
                            neural_observation = (
                                observation
                                if isinstance(observation, NeuralObservation)
                                else runner.game.neural_observation_for(player_id)
                            )
                            representations = [
                                representation_for_neural_action(action, neural_observation)
                                for action in legal_actions
                            ]
                            scores = _scores_for_player(
                                player, player_spec, observation, neural_observation, legal_actions,
                                representations, scorers.get(player_id),
                            )
                            chosen_index = list(legal_actions).index(chosen)
                            opponent_spec = right_spec if player_id is PlayerId.PLAYER_1 else left_spec
                            records.append({
                                "dataset_schema_version": LEAGUE_DATASET_SCHEMA_VERSION,
                                "game_id": game_id,
                                "game_seed": game_seed,
                                "decision_index": len(records),
                                "turn_number": neural_observation.turn_number,
                                "acting_player": player_id.value,
                                "teacher_type": player_spec.kind,
                                "teacher_profile_id": player_spec.profile_id,
                                "teacher_checkpoint": str(player_spec.path) if player_spec.kind == "neural" and player_spec.path else None,
                                "opponent_type": opponent_spec.kind,
                                "opponent_profile_id": opponent_spec.profile_id,
                                "observation": asdict(neural_observation),
                                "action_representations": [representation.to_dict() for representation in representations],
                                "teacher_scores": scores,
                                "heuristic_scores": scores,
                                "teacher_scores_available": scores is not None,
                                "chosen_action_index": chosen_index,
                                "chosen_action": representations[chosen_index].to_dict(),
                            })

                        final_state = runner.run(decision_observer=on_decision)
                        for record in records:
                            record["final_outcome"] = _outcome_for_player(
                                final_state, PlayerId(record["acting_player"])
                            )
                            moderate = _moderate_weight(record)
                            record["raw_moderate_weight"] = moderate
                            totals["control_full_unweighted"] += 1.0
                            totals["weighted_moderate"] += moderate
                            counts["control_full_unweighted"] += 1
                            counts["weighted_moderate"] += 1
                            if record["final_outcome"] == "win":
                                totals["winner_only"] += 1.0
                                counts["winner_only"] += 1
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                        decision_count += len(records)
                        completed_games += 1
                    except Exception as error:
                        error_count += 1
                        errors.append(f"{game_id}: {error!r}")
                        if config.strict_errors:
                            raise
        variant_paths = _materialize_variants(output_dir, temporary_path, totals, counts)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    manifest = {
        "dataset_schema_version": LEAGUE_DATASET_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_representation_schema_version": ACTION_REPRESENTATION_SCHEMA_VERSION,
        "card_catalog_fingerprint": _catalog_fingerprint(),
        "seed": config.seed,
        "games_per_matchup": config.games_per_matchup,
        "attempted_games": attempted_games,
        "completed_games": completed_games,
        "decision_count": decision_count,
        "error_count": error_count,
        "errors": errors,
        "players": [
            {"kind": player.kind, "profile_id": player.profile_id, "path": str(player.path) if player.path else None}
            for player in config.players
        ],
        "result_weights": RESULT_WEIGHTS,
        "teacher_weights": TEACHER_WEIGHTS,
        "variants": {
            name: {"records": counts[name], "raw_weight_total": totals[name], "normalized_mean_weight": 1.0}
            for name in counts
        },
    }
    manifest_path = output_dir / "league_dataset.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in variant_paths.values():
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return LeagueDatasetResult(output_dir, attempted_games, completed_games, decision_count, error_count, variant_paths)


def _validate_config(config: LeagueDatasetConfig) -> None:
    if len(config.players) < 2:
        raise ValueError("At least two league players are required")
    if len({player.profile_id for player in config.players}) != len(config.players):
        raise ValueError("League player profile_id values must be unique")
    if config.games_per_matchup <= 0:
        raise ValueError("games_per_matchup must be positive")


def _build_runner(game_seed, left_spec, right_spec, config):
    root_rng = GameRandom(game_seed)
    game = Game.new(seed=game_seed, rng=root_rng.derive("engine"))
    players = {}
    scorers = {}
    for player_id, spec in ((PlayerId.PLAYER_1, left_spec), (PlayerId.PLAYER_2, right_spec)):
        rng = root_rng.derive(f"player-{player_id.value}")
        if spec.kind == "random":
            player = RandomPlayer(player_id, rng)
        elif spec.kind == "heuristic":
            profile = load_profile(spec.path)
            player = HeuristicPlayer(player_id, profile.weights, profile.card_acquisition_weights, profile.constraint_weights)
        elif spec.kind == "neural":
            checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
            checkpoint_profile_id = checkpoint.get("profile_id")
            if checkpoint_profile_id is not None and checkpoint_profile_id != spec.profile_id:
                raise ValueError(
                    f"Neural checkpoint {spec.path} belongs to {checkpoint_profile_id!r}, "
                    f"but was registered as {spec.profile_id!r}"
                )
            scorer = NeuralPlayer.load_scorer(spec.path)
            scorers[player_id] = scorer
            player = NeuralPlayer(player_id, spec.path, rng, scorer=scorer)
        else:
            raise ValueError(f"Unsupported league player kind: {spec.kind!r}")
        players[player_id] = player
    return GameRunner(game, players, max_actions=config.max_actions, max_turns=config.max_turns), players, scorers


def _scores_for_player(player, spec, observation, neural_observation, legal_actions, representations, scorer):
    if spec.kind == "heuristic":
        return [float(player.score_action(observation, action)) for action in legal_actions]
    if spec.kind == "neural":
        with torch.inference_mode():
            return [float(value) for value in scorer(neural_observation, representations).tolist()]
    return None


def _moderate_weight(record: dict[str, Any]) -> float:
    teacher = TEACHER_WEIGHTS.get(record["teacher_profile_id"], 1.0)
    return teacher * RESULT_WEIGHTS[record["final_outcome"]]


def _materialize_variants(output_dir, temporary_path, totals, counts):
    paths = {
        "control_full_unweighted": output_dir / "league_control_full_unweighted.jsonl",
        "weighted_moderate": output_dir / "league_weighted_moderate.jsonl",
        "winner_only": output_dir / "league_winner_only.jsonl",
    }
    streams = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    handles = {name: path.open("w", encoding="utf-8") for name, path in streams.items()}
    try:
        for line in temporary_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record.pop("raw_moderate_weight", None)
            control = dict(record, sample_weight=1.0)
            handles["control_full_unweighted"].write(json.dumps(control, sort_keys=True) + "\n")
            mean_weight = totals["weighted_moderate"] / counts["weighted_moderate"]
            moderate = dict(record, sample_weight=_moderate_weight(record) / mean_weight)
            handles["weighted_moderate"].write(json.dumps(moderate, sort_keys=True) + "\n")
            if record["final_outcome"] == "win":
                handles["winner_only"].write(json.dumps(dict(record, sample_weight=1.0), sort_keys=True) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    for name, temporary in streams.items():
        os.replace(temporary, paths[name])
    return paths


def _outcome_for_player(state, player_id):
    if state.status is GameStatus.DRAW or state.winner is None:
        return "draw"
    return "win" if state.winner == player_id else "loss"


def _game_seed(root_seed: int, game_index: int) -> int:
    payload = f"shards-ai-league:{root_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _catalog_fingerprint() -> str:
    payload = json.dumps(sorted(CARD_CATALOG), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
