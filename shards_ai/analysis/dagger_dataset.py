"""Helpers for collecting and sampling targeted DAgGER imitation data."""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from shards_ai.ai.action_representation import representation_for_action
from shards_ai.ai.heuristic_player import HeuristicPlayer
from shards_ai.ai.neural_player import NeuralPlayer
from shards_ai.ai.neural_training import split_for_game_id
from shards_ai.game import Game, GameRandom, GameStatus, Phase, PlayerId

SCHEMA_VERSION = 1


def serialize_action(action, representation) -> dict[str, Any]:
    return {"action_type": representation.action_type, "parameters": asdict(action)}


def raw_ranks(scores: list[float]) -> list[int]:
    return [1 + sum(other > score for other in scores) for score in scores]


def state_signature(state, game: Game) -> tuple:
    """Canonical gameplay signature used for phase-level equivalence."""

    def card(card):
        return None if card is None else (card.instance_id, card.definition.card_id)

    def player(player):
        return (
            player.health, player.gems, player.mastery, player.power,
            tuple(card(item) for item in player.hand),
            tuple(card(item) for item in player.draw_pile),
            tuple(card(item) for item in player.discard_pile),
            tuple(card(item) for item in player.play_zone),
            tuple(card(item) for item in player.champions),
            tuple(sorted(player.activated_champion_ids)),
            player.pending_banishes, player.pending_free_recruit_cost,
            player.pending_free_recruit_to_hand,
            None if player.pending_decision is None else (
                player.pending_decision.kind, player.pending_decision.candidates,
            ),
        )

    legal = tuple(
        (type(action).__name__, tuple(asdict(action).values()))
        for action in game.legal_actions()
    ) if state.status is GameStatus.RUNNING else ()
    return (
        state.active_player, state.phase, state.status, state.winner,
        state.turn_number, tuple((player_id, player(state.players[player_id])) for player_id in PlayerId),
        tuple(card(item) for item in state.river),
        tuple(card(item) for item in state.central_deck),
        legal,
    )


def teacher_play_phase_end(start_state, start_rng, player_id: PlayerId, teacher: HeuristicPlayer):
    """Play v008 from a copied PLAY state until that phase ends."""
    branch = Game(copy.deepcopy(start_state), copy.deepcopy(start_rng))
    while (
        branch.state.status is GameStatus.RUNNING
        and branch.state.active_player is player_id
        and branch.state.phase is Phase.PLAY
    ):
        legal = branch.legal_actions()
        branch.apply(teacher.choose_action(branch.state, legal))
    return state_signature(branch.state, branch)


def priority_weight(record: dict[str, Any], action_weights: dict[str, float] | None = None) -> float:
    """Weight used by the deterministic weighted reservoir sampler."""
    weight = 1.0
    if record.get("phase") == "play" or record.get("observation", {}).get("phase") == "play":
        weight *= 1.5
    if record.get("strategic_divergence") is True:
        weight *= 4.0
    elif record.get("play_phase_equivalent") is False:
        weight *= 3.0
    if record.get("first_divergence") or record.get("decision_after_first_divergence"):
        weight *= 2.0
    if int(record.get("teacher_rank", 1)) > 3:
        weight *= 2.0
    weight *= min(1.0 + max(float(record.get("regret", 0.0)), 0.0), 4.0)
    if action_weights:
        weight *= action_weights.get(record.get("teacher_action_type"), 1.0)
    return weight


def _key(seed: int, category: str, game_id: str, decision_index: int) -> float:
    digest = hashlib.blake2b(
        f"dagger:{seed}:{category}:{game_id}:{decision_index}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def _reservoir_add(heap, capacity: int, key: float, reference: tuple, counter: int) -> None:
    item = (key, counter, reference)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, item)


def _iter_lines(path: Path):
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            yield offset, line


def _read_reference(reference: tuple) -> dict[str, Any]:
    path, offset = reference
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        return json.loads(stream.readline())


def sample_dataset(
    old_dataset: Path,
    dagger_dataset: Path,
    output: Path,
    validation_output: Path,
    *,
    target_records: int = 1_000_000,
    seed: int = 71,
    old_fraction: float = 0.45,
    play_fraction: float = 0.35,
    validation_seed: int = 0,
    action_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a weighted, deterministic DAgGER training set without loading lines in memory."""
    if target_records <= 0 or min(old_fraction, play_fraction) < 0 or old_fraction + play_fraction > 1:
        raise ValueError("invalid target or fractions")
    quotas = {
        "old": round(target_records * old_fraction),
        "dagger_play": round(target_records * play_fraction),
    }
    quotas["dagger_other"] = target_records - quotas["old"] - quotas["dagger_play"]
    heaps: dict[str, list] = {category: [] for category in quotas}
    old_validation: list[tuple[Path, int]] = []
    dagger_validation: list[tuple[Path, int]] = []
    counts = {category: 0 for category in quotas}
    validation_count = 0
    counter = 0
    for path in (Path(old_dataset), Path(dagger_dataset)):
        is_old = path == Path(old_dataset)
        for offset, line in _iter_lines(path):
            record = json.loads(line)
            game_id = str(record.get("game_id", record.get("seed", counter)))
            if is_old:
                if split_for_game_id(game_id, seed=validation_seed) == "validation":
                    old_validation.append((path, offset))
                    validation_count += 1
                    continue
                category = "old"
            else:
                if split_for_game_id(f"dagger-validation:{game_id}", seed=validation_seed) == "validation":
                    dagger_validation.append((path, offset))
                    validation_count += 1
                    continue
                category = "dagger_play" if record.get("observation", {}).get("phase") == "play" else "dagger_other"
            counts[category] += 1
            weight = priority_weight(record, action_weights) if category != "old" else 1.0
            uniform = max(_key(seed, category, game_id, int(record.get("decision_index", counter))), 1e-12)
            weighted_key = uniform ** (1.0 / weight)
            _reservoir_add(heaps[category], quotas[category], weighted_key, (path, offset), counter)
            counter += 1

    selected = [(item[0], item[2], category) for category, heap in heaps.items() for item in heap]
    selected.sort(key=lambda item: _key(seed + 1, item[2], str(item[1]), item[0]))
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as train_stream:
        for _key_value, reference, category in selected:
            record = _read_reference(reference)
            record["dagger_sample_bucket"] = category
            record["dagger_sample_weight"] = priority_weight(record, action_weights) if category != "old" else 1.0
            record["dagger_priority_weight"] = record["dagger_sample_weight"]
            train_stream.write(json.dumps(record, sort_keys=True) + "\n")
    with validation_output.open("w", encoding="utf-8") as validation_stream:
        for reference in [*old_validation, *dagger_validation]:
            validation_stream.write(json.dumps(_read_reference(reference), sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "old_dataset": str(old_dataset),
        "dagger_dataset": str(dagger_dataset),
        "output": str(output),
        "validation_output": str(validation_output),
        "target_records": target_records,
        "selected_records": len(selected),
        "quotas": quotas,
        "available_records": counts,
        "selected_by_bucket": {category: sum(item[2] == category for item in selected) for category in quotas},
        "historical_validation_records": validation_count,
        "dagger_validation_records": len(dagger_validation),
        "seed": seed,
        "action_weights": action_weights or {},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def sample_on_policy_dataset(
    dagger_dataset: Path,
    output: Path,
    validation_output: Path,
    *,
    target_records: int = 1_000_000,
    seed: int = 71,
    validation_seed: int = 0,
    action_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Sample only DAgGER records, preserving V8 labels and targeted weights."""
    if target_records <= 0:
        raise ValueError("target_records must be positive")
    heap: list = []
    validation: list[tuple[Path, int]] = []
    available = 0
    counter = 0
    dataset = Path(dagger_dataset)
    for offset, line in _iter_lines(dataset):
        record = json.loads(line)
        game_id = str(record.get("game_id", record.get("seed", counter)))
        if split_for_game_id(f"dagger-validation:{game_id}", seed=validation_seed) == "validation":
            validation.append((dataset, offset))
            continue
        available += 1
        weight = priority_weight(record, action_weights)
        uniform = max(_key(seed, "dagger_2", game_id, int(record.get("decision_index", counter))), 1e-12)
        _reservoir_add(heap, target_records, uniform ** (1.0 / weight), (dataset, offset), counter)
        counter += 1

    selected = [item[2] for item in heap]
    selected.sort(key=lambda reference: _key(seed + 1, "dagger_2", str(reference), 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for reference in selected:
            record = _read_reference(reference)
            record["dagger_sample_bucket"] = "dagger_2_weighted"
            record["dagger_sample_weight"] = priority_weight(record, action_weights)
            record["dagger_priority_weight"] = record["dagger_sample_weight"]
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with validation_output.open("w", encoding="utf-8") as stream:
        for reference in validation:
            stream.write(json.dumps(_read_reference(reference), sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dagger_dataset": str(dagger_dataset),
        "output": str(output),
        "validation_output": str(validation_output),
        "target_records": target_records,
        "selected_records": len(selected),
        "available_records": available,
        "validation_records": len(validation),
        "seed": seed,
        "action_weights": action_weights or {},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
