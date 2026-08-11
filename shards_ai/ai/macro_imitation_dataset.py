"""Collection of strategic-decision demonstrations for the macro PLAY policy."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shards_ai.game import CARD_CATALOG, Game, GameRandom, GameRunner, GameStatus, PlayerId
from shards_ai.game.actions import Action
from shards_ai.game.enums import Phase

from .action_representation import representation_for_neural_action
from .card_representation import CARD_REPRESENTATION_SCHEMA_VERSION
from .heuristic_player import HeuristicPlayer
from .heuristic_profiles import load_profile
from .macro_player import MacroNeuralPlayer
from .neural_player import NeuralPlayer
from .play_turn_solver import (
    PlayTurnCandidate,
    PlayTurnSolver,
)


MACRO_DATASET_SCHEMA_VERSION = 3
MACRO_CANDIDATE_SCHEMA_VERSION = 4
CANONICALIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MacroDatasetCampaignConfig:
    teacher_profile_path: Path
    heuristic_opponent_profile_path: Path
    neural_opponent_checkpoint_path: Path
    output_path: Path
    seed: int
    games: int | None = None
    target_decisions: int | None = None
    max_games: int | None = None
    max_actions: int = GameRunner.DEFAULT_MAX_ACTIONS
    max_turns: int | None = None
    strict_errors: bool = True

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class MacroDatasetGenerationResult:
    output_path: Path
    manifest_path: Path
    attempted_games: int
    completed_games: int
    excluded_games: int
    decision_count: int
    macro_decision_count: int
    atomic_decision_count: int
    error_count: int


def heuristic_macro_selector(teacher: HeuristicPlayer):
    """Return a selector that applies the exact V8 heuristic to branch roots."""

    def select(game: Game, _observation, candidates: tuple[PlayTurnCandidate, ...]) -> int:
        if not candidates:
            raise ValueError("Cannot select from an empty macro candidate list")
        legal = game.legal_actions()
        first_actions = [candidate.atomic_trace[0] for candidate in candidates]
        chosen = teacher.choose_action(game.observation_for(teacher.player_id), first_actions)
        return first_actions.index(chosen)

    return select


def generate_macro_dataset(config: MacroDatasetCampaignConfig) -> MacroDatasetGenerationResult:
    teacher_profile = load_profile(config.teacher_profile_path)
    opponent_profile = load_profile(config.heuristic_opponent_profile_path)
    output_path = Path(config.output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attempted = completed = excluded = decisions = macro_decisions = atomic_decisions = errors_count = 0
    decisions_by_matchup: dict[str, int] = {}
    decision_counts: dict[str, dict[str, int]] = {
        "decision_kind": {}, "phase": {}, "action_type": {},
    }
    errors: list[str] = []

    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            while _should_start(config, attempted, decisions):
                game_index = attempted
                attempted += 1
                game_seed = _game_seed(config.seed, game_index)
                matchup = "v008_vs_v007" if game_index % 2 == 0 else "v008_vs_v004"
                try:
                    runner, teacher = _build_runner(
                        game_seed,
                        game_index,
                        matchup,
                        config,
                        teacher_profile,
                        opponent_profile,
                    )
                    records: list[dict[str, Any]] = []

                    def on_decision(observation, legal_actions, chosen, player_id) -> None:
                        if player_id != teacher.player_id:
                            return
                        if teacher.has_pending_macro_trace:
                            return
                        payload = teacher.pop_last_atomic_decision()
                        if payload is None:
                            return
                        records.append({
                            "dataset_schema_version": MACRO_DATASET_SCHEMA_VERSION,
                            "decision_kind": "atomic",
                            "game_id": f"game-{game_index:08d}",
                            "game_seed": game_seed,
                            "decision_index": len(records),
                            "turn_number": payload.observation.turn_number,
                            "acting_player": player_id.value,
                            "seat": player_id.value,
                            "teacher_profile_id": teacher_profile.profile_id,
                            "teacher_profile_path": str(config.teacher_profile_path),
                            "opponent_id": matchup.split("_vs_", 1)[1],
                            "phase": payload.observation.phase,
                            "observation": asdict(payload.observation),
                            "legal_actions": [asdict(action) for action in legal_actions],
                            "candidates": [asdict(candidate) for candidate in payload.candidate_representations],
                            "chosen_candidate_index": payload.chosen_candidate_index,
                            "chosen_action": asdict(chosen),
                            "legal_action_count": len(legal_actions),
                            "candidate_action_types": [candidate.action_type for candidate in payload.candidate_representations],
                        })

                    def on_macro(payload, player_id) -> None:
                        nonlocal macro_decisions
                        if player_id != teacher.player_id:
                            return
                        if any(
                            candidate.schema_version != MACRO_CANDIDATE_SCHEMA_VERSION
                            or candidate.root_action is None
                            for candidate in payload.candidate_representations
                        ):
                            raise ValueError("Unified dataset generation requires candidate schema V4")
                        records.append({
                            "dataset_schema_version": MACRO_DATASET_SCHEMA_VERSION,
                            "decision_kind": "macro_play",
                            "game_id": f"game-{game_index:08d}",
                            "game_seed": game_seed,
                            "decision_index": len(records),
                            "turn_number": payload.observation.turn_number,
                            "acting_player": player_id.value,
                            "seat": player_id.value,
                            "teacher_profile_id": teacher_profile.profile_id,
                            "teacher_profile_path": str(config.teacher_profile_path),
                            "opponent_id": matchup.split("_vs_", 1)[1],
                            "phase": payload.observation.phase,
                            "observation": asdict(payload.observation),
                            "candidates": [asdict(candidate) for candidate in payload.candidate_representations],
                            "chosen_candidate_index": payload.chosen_candidate_index,
                            "automatic_prefix_action_types": [type(action).__name__ for action in payload.automatic_prefix],
                            "selected_atomic_trace_action_types": [type(action).__name__ for action in payload.selected_atomic_trace],
                            "solver_expansions": payload.expansions,
                            "solver_memoized_states": payload.memoized_states,
                            "legal_action_count": len(payload.candidate_representations),
                            "candidate_action_types": [candidate.action_type for candidate in payload.candidate_representations],
                        })
                        macro_decisions += 1

                    final_state = runner.run(
                        decision_observer=on_decision,
                        macro_decision_observer=on_macro,
                    )
                    for record in records:
                        record["final_outcome"] = _outcome(final_state, teacher.player_id)
                        for dimension, value in (
                            ("decision_kind", record["decision_kind"]),
                            ("phase", record.get("phase", record["observation"].get("phase"))),
                            ("action_type", _chosen_action_type(record)),
                        ):
                            if value is not None:
                                decision_counts[dimension][value] = decision_counts[dimension].get(value, 0) + 1
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    decisions += len(records)
                    atomic_decisions += sum(record["decision_kind"] == "atomic" for record in records)
                    decisions_by_matchup[matchup] = decisions_by_matchup.get(matchup, 0) + len(records)
                    completed += 1
                except Exception as error:
                    excluded += 1
                    errors_count += 1
                    errors.append(f"game-{game_index:08d}: {error!r}")
                    if config.strict_errors:
                        raise
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    manifest = {
        "dataset_schema_version": MACRO_DATASET_SCHEMA_VERSION,
        "observation_schema_version": 3,
        "card_representation_schema_version": CARD_REPRESENTATION_SCHEMA_VERSION,
        "card_catalog_fingerprint": _card_catalog_fingerprint(),
        "candidate_schema_version": MACRO_CANDIDATE_SCHEMA_VERSION,
        "canonicalization_schema_version": CANONICALIZATION_SCHEMA_VERSION,
        "candidate_feature_set": "root_action_plus_known_consequence_plus_tactical_v1",
        "decision_kinds": ["macro_play", "atomic"],
        "decision_counts": decision_counts,
        "candidate_contract": "unified_macro_v4_atomic_v1",
        "solver_budgets": {
            "max_expansions": PlayTurnSolver().max_expansions,
            "max_memoized_states": PlayTurnSolver().max_memoized_states,
            "max_macro_candidates": PlayTurnSolver().max_macro_candidates,
            "max_atomic_actions_per_segment": PlayTurnSolver().max_atomic_actions_per_segment,
        },
        "seed": config.seed,
        "requested_games": config.games,
        "target_decisions": config.target_decisions,
        "attempted_games": attempted,
        "completed_games": completed,
        "excluded_games": excluded,
        "decision_count": decisions,
        "macro_decision_count": macro_decisions,
        "atomic_decision_count": atomic_decisions,
        "error_count": errors_count,
        "teacher": {"profile_id": teacher_profile.profile_id, "path": str(config.teacher_profile_path)},
        "opponents": [
            {"profile_id": opponent_profile.profile_id, "path": str(config.heuristic_opponent_profile_path)},
            {"profile_id": "v004", "path": str(config.neural_opponent_checkpoint_path)},
        ],
        "decisions_by_matchup": decisions_by_matchup,
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MacroDatasetGenerationResult(
        output_path, manifest_path, attempted, completed, excluded, decisions,
        macro_decisions, atomic_decisions, errors_count,
    )


def _build_runner(game_seed, game_index, matchup, config, teacher_profile, opponent_profile):
    root_rng = GameRandom(game_seed)
    game = Game.new(seed=game_seed, rng=root_rng.derive("engine"))
    teacher_id = PlayerId.PLAYER_2 if game_index % 2 else PlayerId.PLAYER_1
    opponent_id = teacher_id.opponent
    teacher_heuristic = HeuristicPlayer(
        teacher_id,
        teacher_profile.weights,
        teacher_profile.card_acquisition_weights,
        teacher_profile.constraint_weights,
    )
    teacher = MacroNeuralPlayer(
        teacher_id,
        game,
        solver=PlayTurnSolver(),
        candidate_scorer=heuristic_macro_selector(teacher_heuristic),
        candidate_schema_version=MACRO_CANDIDATE_SCHEMA_VERSION,
    )
    if matchup == "v008_vs_v007":
        opponent = HeuristicPlayer(
            opponent_id,
            opponent_profile.weights,
            opponent_profile.card_acquisition_weights,
            opponent_profile.constraint_weights,
        )
    else:
        opponent = NeuralPlayer(
            opponent_id,
            config.neural_opponent_checkpoint_path,
            root_rng.derive(f"player-{opponent_id.value}"),
        )
    return GameRunner(game, {teacher_id: teacher, opponent_id: opponent}, max_actions=config.max_actions, max_turns=config.max_turns), teacher


def _should_start(config, attempted, decisions):
    return (
        (config.games is None or attempted < config.games)
        and (config.target_decisions is None or decisions < config.target_decisions)
        and (config.max_games is None or attempted < config.max_games)
    )


def _outcome(state, player_id):
    if state.status is GameStatus.DRAW or state.winner is None:
        return "draw"
    return "win" if state.winner == player_id else "loss"


def _chosen_action_type(record: dict[str, Any]) -> str | None:
    chosen = record.get("chosen_action")
    if isinstance(chosen, dict):
        action_type = chosen.get("action_type")
        if action_type:
            return str(action_type)
    index = record.get("chosen_candidate_index")
    candidates = record.get("candidates", ())
    if isinstance(index, int) and 0 <= index < len(candidates):
        candidate = candidates[index]
        root = candidate.get("root_action")
        if isinstance(root, dict) and root.get("action_type"):
            return str(root["action_type"])
        return candidate.get("action_type")
    return None


def _game_seed(root_seed, game_index):
    payload = f"shards-ai-macro-imitation:{root_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _card_catalog_fingerprint():
    return hashlib.sha256(json.dumps(sorted(CARD_CATALOG), separators=(",", ":")).encode()).hexdigest()


__all__ = [
    "MACRO_DATASET_SCHEMA_VERSION",
    "MACRO_CANDIDATE_SCHEMA_VERSION",
    "MacroDatasetCampaignConfig",
    "MacroDatasetGenerationResult",
    "generate_macro_dataset",
    "heuristic_macro_selector",
]
