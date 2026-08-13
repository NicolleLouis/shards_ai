"""Features, datasets and classifiers for the active-player horizon forecast."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from shards_ai.game.cards import CardInstance
from shards_ai.game.enums import Faction
from shards_ai.game.state import GameState


HORIZON_DATASET_SCHEMA_VERSION = 2
HORIZON_CLASS_NAMES = ("T0", "T1", "T2", "T3", "T4", "T5", "T6+")
HORIZON_CLASS_COUNT = len(HORIZON_CLASS_NAMES)
HORIZON_FEATURE_SET_BASELINE = "turn_number_v1"
HORIZON_FEATURE_SET_V1 = "active_state_faction_counts_v1"
FACTIONS = (Faction.MAQUIS, Faction.SPECTRA, Faction.HOMODEUS, Faction.ORDER)
FACTION_COUNT_BOUND = 100.0


def horizon_class_for_remaining_turns(remaining_turns: int) -> int:
    if remaining_turns < 0:
        raise ValueError("remaining_turns must be non-negative")
    return min(remaining_turns, HORIZON_CLASS_COUNT - 1)


@dataclass(frozen=True, slots=True)
class HorizonFeatures:
    turn_number: float
    active_health: float
    opponent_health: float
    active_mastery: float
    opponent_mastery: float
    active_owned_card_count: float
    opponent_owned_card_count: float
    active_faction_counts: tuple[float, float, float, float]
    opponent_faction_counts: tuple[float, float, float, float]

    def baseline_vector(self) -> tuple[float, ...]:
        return (self.turn_number / 100.0,)

    def v1_vector(self) -> tuple[float, ...]:
        return (
            self.turn_number / 100.0,
            self.active_health / 50.0,
            self.opponent_health / 50.0,
            self.active_mastery / 30.0,
            self.opponent_mastery / 30.0,
            self.active_owned_card_count / FACTION_COUNT_BOUND,
            self.opponent_owned_card_count / FACTION_COUNT_BOUND,
            *(value / FACTION_COUNT_BOUND for value in self.active_faction_counts),
            *(value / FACTION_COUNT_BOUND for value in self.opponent_faction_counts),
        )


def _owned_cards(player) -> list[CardInstance]:
    return [*player.hand, *player.draw_pile, *player.discard_pile, *player.play_zone, *player.champions]


def _faction_counts(player) -> tuple[int, int, int, int]:
    counts = {faction: 0 for faction in FACTIONS}
    for card in _owned_cards(player):
        faction = card.definition.faction
        if faction in counts:
            counts[faction] += 1
    return tuple(counts[faction] for faction in FACTIONS)


def features_from_state(state: GameState) -> HorizonFeatures:
    active = state.players[state.active_player]
    opponent = state.players[state.active_player.opponent]
    return HorizonFeatures(
        turn_number=float(state.turn_number),
        active_health=float(active.health),
        opponent_health=float(opponent.health),
        active_mastery=float(active.mastery),
        opponent_mastery=float(opponent.mastery),
        active_owned_card_count=float(len(_owned_cards(active))),
        opponent_owned_card_count=float(len(_owned_cards(opponent))),
        active_faction_counts=tuple(float(value) for value in _faction_counts(active)),
        opponent_faction_counts=tuple(float(value) for value in _faction_counts(opponent)),
    )


def features_to_record(features: HorizonFeatures) -> dict[str, object]:
    return {
        "turn_number": int(features.turn_number),
        "active_health": int(features.active_health),
        "opponent_health": int(features.opponent_health),
        "active_mastery": int(features.active_mastery),
        "opponent_mastery": int(features.opponent_mastery),
        "active_owned_card_count": int(features.active_owned_card_count),
        "opponent_owned_card_count": int(features.opponent_owned_card_count),
        "active_faction_counts": list(map(int, features.active_faction_counts)),
        "opponent_faction_counts": list(map(int, features.opponent_faction_counts)),
    }


def features_from_record(record: dict[str, object]) -> HorizonFeatures:
    values = record["features"]
    if not isinstance(values, dict):
        raise ValueError("Horizon record features must be an object")
    return HorizonFeatures(
        turn_number=float(values["turn_number"]),
        active_health=float(values["active_health"]),
        opponent_health=float(values["opponent_health"]),
        active_mastery=float(values["active_mastery"]),
        opponent_mastery=float(values["opponent_mastery"]),
        active_owned_card_count=float(values["active_owned_card_count"]),
        opponent_owned_card_count=float(values["opponent_owned_card_count"]),
        active_faction_counts=tuple(float(x) for x in values["active_faction_counts"]),  # type: ignore[arg-type]
        opponent_faction_counts=tuple(float(x) for x in values["opponent_faction_counts"]),  # type: ignore[arg-type]
    )


def split_for_game_id(game_id: str, seed: int = 0) -> str:
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)
                record["_line_number"] = line_number
                yield record


def project_baseline_dataset(source: str | Path, destination: str | Path) -> int:
    count = 0
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for record in iter_jsonl(source):
            baseline = {
                "schema_version": HORIZON_DATASET_SCHEMA_VERSION,
                "feature_set": HORIZON_FEATURE_SET_BASELINE,
                "game_id": record["game_id"],
                "game_seed": record["game_seed"],
                "decision_index": record["decision_index"],
                "active_player": record["active_player"],
                "features": {"turn_number": record["features"]["turn_number"]},
                "target_horizon_class": record["target_horizon_class"],
            }
            output.write(json.dumps(baseline, sort_keys=True) + "\n")
            count += 1
    return count


class HorizonClassifier(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, HORIZON_CLASS_COUNT),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


def model_for_feature_set(feature_set: str) -> HorizonClassifier:
    if feature_set == HORIZON_FEATURE_SET_BASELINE:
        return HorizonClassifier(1)
    if feature_set == HORIZON_FEATURE_SET_V1:
        return HorizonClassifier(15)
    raise ValueError(f"Unsupported horizon feature set: {feature_set!r}")


def vector_for_record(record: dict, feature_set: str) -> tuple[float, ...]:
    if feature_set == HORIZON_FEATURE_SET_BASELINE:
        return (float(record["features"]["turn_number"]) / 100.0,)
    return features_from_record(record).v1_vector()


def _class_weights(records: Sequence[dict]) -> Tensor:
    counts = torch.bincount(
        torch.tensor([int(record["target_horizon_class"]) for record in records]),
        minlength=HORIZON_CLASS_COUNT,
    ).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].rsqrt()
    if present.any():
        weights[present] /= weights[present].mean()
    return weights


def evaluate_model(model: HorizonClassifier, records: Sequence[dict], feature_set: str) -> dict:
    if not records:
        return {"records": 0, "accuracy": 0.0, "balanced_accuracy": 0.0}
    model.eval()
    with torch.inference_mode():
        inputs = torch.tensor([vector_for_record(r, feature_set) for r in records], dtype=torch.float32)
        targets = torch.tensor([int(r["target_horizon_class"]) for r in records], dtype=torch.long)
        probabilities = model(inputs).softmax(dim=1)
        predicted = probabilities.argmax(dim=1)
    confusion = torch.zeros((HORIZON_CLASS_COUNT, HORIZON_CLASS_COUNT), dtype=torch.long)
    confusion.index_put_((targets, predicted), torch.ones_like(targets), accumulate=True)
    supports = confusion.sum(dim=1)
    recalls = torch.where(supports > 0, confusion.diag().float() / supports, torch.zeros_like(supports, dtype=torch.float))
    precisions = torch.where(confusion.sum(dim=0) > 0, confusion.diag().float() / confusion.sum(dim=0), torch.zeros(HORIZON_CLASS_COUNT))
    short_mask = targets <= 2
    late_mask = targets == 6
    within_one = (predicted - targets).abs() <= 1
    p_le_2 = probabilities[:, :3].sum(dim=1)
    p_le_5 = probabilities[:, :6].sum(dim=1)
    target_le_2 = short_mask.float()
    target_le_5 = (targets <= 5).float()
    return {
        "records": len(records),
        "accuracy": float((predicted == targets).float().mean()),
        "balanced_accuracy": float(recalls[supports > 0].mean()),
        "within_one_class_accuracy": float(within_one.float().mean()),
        "short_t0_t2_recall": float(recalls[:3].mean()),
        "short_t0_t2_precision": float(precisions[:3].mean()),
        "late_t6_plus_recall": float(recalls[6]),
        "brier_p_le_2": float(((p_le_2 - target_le_2) ** 2).mean()),
        "brier_p_le_5": float(((p_le_5 - target_le_5) ** 2).mean()),
        "class_metrics": {
            name: {"support": int(supports[index]), "precision": float(precisions[index]), "recall": float(recalls[index])}
            for index, name in enumerate(HORIZON_CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    records: Sequence[dict],
    feature_set: str,
    *,
    seed: int = 51200,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    patience: int = 15,
) -> tuple[HorizonClassifier, list[dict[str, float | int]]]:
    if not records:
        raise ValueError("Cannot train a horizon model on an empty dataset")
    random.seed(seed)
    torch.manual_seed(seed)
    model = model_for_feature_set(feature_set)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs = torch.tensor([vector_for_record(r, feature_set) for r in records], dtype=torch.float32)
    targets = torch.tensor([int(r["target_horizon_class"]) for r in records], dtype=torch.long)
    weights = _class_weights(records)
    best_state = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        history.append({"epoch": epoch, "train_loss": value})
        if value < best_loss:
            best_loss = value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_model(model: HorizonClassifier, path: str | Path, *, feature_set: str, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HORIZON_DATASET_SCHEMA_VERSION,
        "feature_set": feature_set,
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
