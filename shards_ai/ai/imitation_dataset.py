from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .action_representation import (
    ACTION_REPRESENTATION_SCHEMA_VERSION,
    representation_for_action,
)
from .card_representation import CARD_REPRESENTATION_SCHEMA_VERSION
from .heuristic_player import HeuristicPlayer
from .heuristic_profiles import HeuristicProfile, load_profile
from .random_player import RandomPlayer
from shards_ai.game import (
    CARD_CATALOG,
    Game,
    GameRandom,
    GameRunner,
    GameStatus,
    OBSERVATION_SCHEMA_VERSION,
    PlayerId,
)


DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MatchupSpec:
    heuristic_profile_path: Path
    opponent_profile_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetCampaignConfig:
    profile_paths: tuple[Path, ...]
    output_path: Path
    seed: int
    games: int | None = None
    target_decisions: int | None = None
    max_games: int | None = None
    max_actions: int = GameRunner.DEFAULT_MAX_ACTIONS
    max_turns: int | None = None
    matchups: tuple[MatchupSpec, ...] | None = None
    record_profile_ids: frozenset[str] | None = None
    strict_errors: bool = True

    def __post_init__(self) -> None:
        if not self.profile_paths:
            raise ValueError("At least one heuristic profile is required")
        if self.games is None and self.target_decisions is None:
            raise ValueError("Set games or target_decisions")
        if self.games is not None and self.games <= 0:
            raise ValueError("games must be positive")
        if self.target_decisions is not None and self.target_decisions <= 0:
            raise ValueError("target_decisions must be positive")
        if self.max_games is not None and self.max_games <= 0:
            raise ValueError("max_games must be positive")
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("max_turns must be positive")


@dataclass(frozen=True, slots=True)
class DatasetGenerationResult:
    output_path: Path
    manifest_path: Path
    attempted_games: int
    completed_games: int
    excluded_games: int
    decision_count: int
    error_count: int


def default_matchups(profile_paths: tuple[Path, ...]) -> tuple[MatchupSpec, ...]:
    """Build a deterministic mix of heuristic/random and heuristic/heuristic games."""
    return tuple(
        [MatchupSpec(profile_path) for profile_path in profile_paths]
        + [
            MatchupSpec(left, right)
            for left in profile_paths
            for right in profile_paths
        ]
    )


def generate_dataset(config: DatasetCampaignConfig) -> DatasetGenerationResult:
    profiles = {
        Path(path): load_profile(path)
        for path in config.profile_paths
    }
    matchups = config.matchups or default_matchups(config.profile_paths)
    if not matchups:
        raise ValueError("At least one matchup is required")
    for matchup in matchups:
        if matchup.heuristic_profile_path not in profiles:
            raise ValueError(f"Unknown matchup profile: {matchup.heuristic_profile_path}")
        if matchup.opponent_profile_path is not None and matchup.opponent_profile_path not in profiles:
            raise ValueError(f"Unknown matchup profile: {matchup.opponent_profile_path}")

    output_path = Path(config.output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attempted_games = completed_games = excluded_games = decision_count = error_count = 0
    decisions_by_profile: dict[str, int] = {}
    decisions_by_matchup: dict[str, int] = {}
    errors: list[str] = []

    try:
        with temporary_path.open("w", encoding="utf-8") as dataset_stream:
            while _should_start_game(config, attempted_games, decision_count):
                matchup = matchups[attempted_games % len(matchups)]
                game_seed = _game_seed(config.seed, attempted_games)
                game_id = f"game-{attempted_games:08d}"
                attempted_games += 1
                try:
                    runner, heuristic_players, profile_ids, profile_paths = _build_runner(
                        game_seed,
                        attempted_games - 1,
                        matchup,
                        profiles,
                        config,
                    )
                    records: list[dict[str, Any]] = []

                    def on_decision(observation, legal_actions, chosen, player_id) -> None:
                        nonlocal records
                        heuristic_player = heuristic_players.get(player_id)
                        if heuristic_player is None:
                            return
                        profile_id = profile_ids[player_id]
                        if config.record_profile_ids is not None and profile_id not in config.record_profile_ids:
                            return
                        neural_observation = runner.game.neural_observation_for(player_id)
                        action_representations = [
                            representation_for_action(action, runner.game.state)
                            for action in legal_actions
                        ]
                        scores = [
                            heuristic_player.score_action(observation, action)
                            for action in legal_actions
                        ]
                        chosen_index = list(legal_actions).index(chosen)
                        opponent_id = player_id.opponent
                        opponent_profile_id = profile_ids.get(opponent_id)
                        record = {
                            "dataset_schema_version": DATASET_SCHEMA_VERSION,
                            "game_id": game_id,
                            "game_seed": game_seed,
                            "decision_index": len(records),
                            "turn_number": neural_observation.turn_number,
                            "acting_player": player_id.value,
                            "heuristic_profile_id": profile_id,
                            "heuristic_profile_path": str(profile_paths[player_id]),
                            "opponent_type": "heuristic" if opponent_profile_id else "random",
                            "opponent_profile_id": opponent_profile_id,
                            "observation": asdict(neural_observation),
                            "legal_actions": [
                                _serialize_action(action, representation)
                                for action, representation in zip(
                                    legal_actions,
                                    action_representations,
                                    strict=True,
                                )
                            ],
                            "action_representations": [
                                representation.to_dict() for representation in action_representations
                            ],
                            "heuristic_scores": scores,
                            "heuristic_raw_ranks": _raw_ranks(scores),
                            "chosen_action_index": chosen_index,
                            "chosen_action": _serialize_action(
                                chosen,
                                action_representations[chosen_index],
                            ),
                            "_acting_player": player_id,
                        }
                        records.append(record)

                    final_state = runner.run(decision_observer=on_decision)
                    outcome_by_player = {
                        player_id: _outcome_for_player(final_state, player_id)
                        for player_id in heuristic_players
                    }
                    for record in records:
                        record["final_outcome"] = outcome_by_player[record.pop("_acting_player")]
                        dataset_stream.write(json.dumps(record, sort_keys=True) + "\n")
                        profile_id = record["heuristic_profile_id"]
                        decisions_by_profile[profile_id] = decisions_by_profile.get(profile_id, 0) + 1
                        matchup_key = _matchup_key(matchup, profiles)
                        decisions_by_matchup[matchup_key] = decisions_by_matchup.get(matchup_key, 0) + 1
                    decision_count += len(records)
                    completed_games += 1
                except Exception as error:
                    excluded_games += 1
                    error_count += 1
                    errors.append(f"game-{attempted_games - 1:08d}: {error!r}")
                    if config.strict_errors:
                        raise
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "card_representation_schema_version": CARD_REPRESENTATION_SCHEMA_VERSION,
        "action_representation_schema_version": ACTION_REPRESENTATION_SCHEMA_VERSION,
        "card_catalog_fingerprint": _card_catalog_fingerprint(),
        "seed": config.seed,
        "target_decisions": config.target_decisions,
        "requested_games": config.games,
        "attempted_games": attempted_games,
        "completed_games": completed_games,
        "excluded_games": excluded_games,
        "decision_count": decision_count,
        "error_count": error_count,
        "profiles": [
            {"path": str(path), "profile_id": profiles[path].profile_id}
            for path in config.profile_paths
        ],
        "record_profile_ids": sorted(config.record_profile_ids) if config.record_profile_ids is not None else None,
        "decisions_by_profile": decisions_by_profile,
        "decisions_by_matchup": decisions_by_matchup,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DatasetGenerationResult(
        output_path=output_path,
        manifest_path=manifest_path,
        attempted_games=attempted_games,
        completed_games=completed_games,
        excluded_games=excluded_games,
        decision_count=decision_count,
        error_count=error_count,
    )


def _build_runner(game_seed, game_index, matchup, profiles, config):
    root_rng = GameRandom(game_seed)
    game = Game.new(seed=game_seed, rng=root_rng.derive("engine"))
    swap_sides = game_index % 2 == 1
    left_id = PlayerId.PLAYER_2 if swap_sides else PlayerId.PLAYER_1
    right_id = left_id.opponent
    heuristic_players = {}
    profile_ids = {}
    profile_paths = {}

    left_profile = profiles[matchup.heuristic_profile_path]
    heuristic_players[left_id] = HeuristicPlayer(
        left_id,
        left_profile.weights,
        left_profile.card_acquisition_weights,
        left_profile.constraint_weights,
    )
    profile_ids[left_id] = left_profile.profile_id
    profile_paths[left_id] = matchup.heuristic_profile_path

    if matchup.opponent_profile_path is None:
        right_player = RandomPlayer(right_id, root_rng.derive(f"player-{right_id.value}"))
    else:
        right_profile = profiles[matchup.opponent_profile_path]
        right_player = HeuristicPlayer(
            right_id,
            right_profile.weights,
            right_profile.card_acquisition_weights,
            right_profile.constraint_weights,
        )
        heuristic_players[right_id] = right_player
        profile_ids[right_id] = right_profile.profile_id
        profile_paths[right_id] = matchup.opponent_profile_path

    players = {left_id: heuristic_players[left_id], right_id: right_player}
    return (
        GameRunner(game, players, max_actions=config.max_actions, max_turns=config.max_turns),
        heuristic_players,
        profile_ids,
        profile_paths,
    )


def _serialize_action(action, representation) -> dict[str, Any]:
    return {
        "action_type": representation.action_type,
        "parameters": asdict(action),
    }


def _raw_ranks(scores: list[float]) -> list[int]:
    return [1 + sum(other > score for other in scores) for score in scores]


def _outcome_for_player(state, player_id: PlayerId) -> str:
    if state.status is GameStatus.DRAW or state.winner is None:
        return "draw"
    return "win" if state.winner == player_id else "loss"


def _game_seed(root_seed: int, game_index: int) -> int:
    payload = f"shards-ai-imitation:{root_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _should_start_game(config, attempted_games: int, decision_count: int) -> bool:
    if config.games is not None and attempted_games >= config.games:
        return False
    if config.target_decisions is not None and decision_count >= config.target_decisions:
        return False
    if config.max_games is not None and attempted_games >= config.max_games:
        return False
    return True


def _matchup_key(matchup: MatchupSpec, profiles: dict[Path, HeuristicProfile]) -> str:
    left = profiles[matchup.heuristic_profile_path].profile_id
    right = "random" if matchup.opponent_profile_path is None else profiles[matchup.opponent_profile_path].profile_id
    return f"{left}_vs_{right}"


def _card_catalog_fingerprint() -> str:
    payload = json.dumps(sorted(CARD_CATALOG), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
