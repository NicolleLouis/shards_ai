"""Training utilities for the first action-conditioned imitation model."""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

from shards_ai.game.observation import (
    NeuralActivePlayerObservation, NeuralCardObservation, NeuralObservation,
    NeuralOpponentObservation, NeuralPendingObservation, NeuralRiverCardObservation,
)

from .action_representation import ActionRepresentation
from .neural_model import NeuralActionScorer


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    records: int
    mean_loss: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    records: int
    mean_loss: float
    top1_accuracy: float
    mean_chosen_rank: float
    mean_normalized_chosen_rank: float
    pairwise_accuracy: float
    pairwise_pairs: int


def iter_jsonl_records(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                record["_line_number"] = line_number
                yield record


def split_for_game_id(game_id: str, *, seed: int = 0) -> str:
    """Return a reproducible train/validation/test split for one complete game."""
    digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def is_targeted_mercenary_record(record: dict) -> bool:
    """Return whether one mercenary offers both immediate and long-term actions."""
    buy_cards = {
        value.get("card_definition_id")
        for value in record.get("action_representations", ())
        if value.get("action_type") == "buy_card" and value.get("card_definition_id") is not None
    }
    recruited_cards = {
        value.get("card_definition_id")
        for value in record.get("action_representations", ())
        if value.get("action_type") == "recruit_mercenary" and value.get("card_definition_id") is not None
    }
    return bool(buy_cards & recruited_cards)


def matches_imitation_slice(
    record: dict,
    *,
    action_types: frozenset[str] = frozenset(),
    min_legal_actions: int | None = None,
    max_legal_actions: int | None = None,
) -> bool:
    """Return whether a record belongs to an explicitly bounded training slice."""
    action_type_candidates = {
        record.get("chosen_action", {}).get("action_type"),
        record.get("teacher_action_type"),
        record.get("neural_action_type"),
    }
    legal_action_count = len(record.get("action_representations", ()))
    if action_types and not action_type_candidates.intersection(action_types):
        return False
    if min_legal_actions is not None and legal_action_count < min_legal_actions:
        return False
    if max_legal_actions is not None and legal_action_count > max_legal_actions:
        return False
    return True


def pairwise_ranking_loss(predicted: Tensor, teacher_scores: Tensor) -> Tensor:
    """Make higher-scored heuristic actions more likely than lower-scored ones."""
    if predicted.numel() < 2:
        return predicted.sum() * 0
    differences = predicted[:, None] - predicted[None, :]
    teacher_differences = teacher_scores[:, None] - teacher_scores[None, :]
    mask = teacher_differences > 0
    if not mask.any():
        return predicted.sum() * 0
    return F.softplus(-differences[mask]).mean()


def chosen_action_loss(predicted: Tensor, chosen_index: int | None) -> Tensor:
    if chosen_index is None or not 0 <= chosen_index < predicted.numel():
        return predicted.sum() * 0
    return F.cross_entropy(predicted.unsqueeze(0), torch.tensor([chosen_index], device=predicted.device))


def normalized_score_regression_loss(predicted: Tensor, teacher_scores: Tensor) -> Tensor:
    """Optional scale-invariant score target for experiments across profiles."""
    if predicted.numel() < 2:
        return predicted.sum() * 0
    target = (teacher_scores - teacher_scores.mean()) / teacher_scores.std(unbiased=False).clamp_min(1e-6)
    prediction = (predicted - predicted.mean()) / predicted.std(unbiased=False).clamp_min(1e-6)
    return F.mse_loss(prediction, target)


def combined_imitation_loss(
    predicted: Tensor,
    teacher_scores: Tensor,
    chosen_index: int | None,
    *,
    ranking_weight: float = 1.0,
    chosen_weight: float = 0.25,
    score_weight: float = 0.0,
) -> Tensor:
    loss = (
        ranking_weight * pairwise_ranking_loss(predicted, teacher_scores)
        + chosen_weight * chosen_action_loss(predicted, chosen_index)
    )
    if score_weight:
        loss = loss + score_weight * normalized_score_regression_loss(predicted, teacher_scores)
    return loss


def train_epoch(
    model: NeuralActionScorer,
    records: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    *,
    max_records: int | None = None,
    record_weight: Callable[[dict], float] | None = None,
) -> TrainingMetrics:
    model.train()
    total = 0.0
    count = 0
    for record in records:
        if max_records is not None and count >= max_records:
            break
        observation = observation_from_dict(record["observation"])
        actions = [ActionRepresentation(**value) for value in record["action_representations"]]
        predicted = model(observation, actions)
        teacher = _teacher_scores_tensor(record, len(actions), predicted.device)
        loss = combined_imitation_loss(predicted, teacher, record.get("chosen_action_index"))
        weight = record_weight(record) if record_weight is not None else 1.0
        if weight <= 0:
            raise ValueError("record_weight must return a positive value")
        loss = loss * weight
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return TrainingMetrics(records=count, mean_loss=total / count if count else 0.0)


@torch.no_grad()
def evaluate_epoch(
    model: NeuralActionScorer,
    records: Iterable[dict],
    *,
    max_records: int | None = None,
) -> EvaluationMetrics:
    """Evaluate ranking and chosen-action imitation without updating weights."""
    model.eval()
    total_loss = 0.0
    record_count = 0
    top1 = 0
    rank_total = 0.0
    normalized_rank_total = 0.0
    pairwise_correct = 0
    pairwise_total = 0
    for record in records:
        if max_records is not None and record_count >= max_records:
            break
        observation = observation_from_dict(record["observation"])
        actions = [ActionRepresentation(**value) for value in record["action_representations"]]
        predicted = model(observation, actions)
        teacher = _teacher_scores_tensor(record, len(actions), predicted.device)
        total_loss += float(combined_imitation_loss(predicted, teacher, record.get("chosen_action_index")))
        chosen_index = record.get("chosen_action_index")
        if chosen_index is not None and 0 <= chosen_index < len(actions):
            order = torch.argsort(predicted, descending=True).tolist()
            rank = order.index(chosen_index) + 1
            rank_total += rank
            normalized_rank_total += 1.0 - ((rank - 1) / max(1, len(actions) - 1))
            top1 += rank == 1
        predicted_difference = predicted[:, None] - predicted[None, :]
        teacher_difference = teacher[:, None] - teacher[None, :]
        strict_pairs = teacher_difference > 0
        pairwise_correct += int((predicted_difference[strict_pairs] > 0).sum())
        pairwise_total += int(strict_pairs.sum())
        record_count += 1
    return EvaluationMetrics(
        records=record_count,
        mean_loss=total_loss / record_count if record_count else 0.0,
        top1_accuracy=top1 / record_count if record_count else 0.0,
        mean_chosen_rank=rank_total / record_count if record_count else 0.0,
        mean_normalized_chosen_rank=normalized_rank_total / record_count if record_count else 0.0,
        pairwise_accuracy=pairwise_correct / pairwise_total if pairwise_total else 0.0,
        pairwise_pairs=pairwise_total,
    )


def train_jsonl(
    model: NeuralActionScorer,
    dataset_path: str | Path,
    *,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    max_records_per_epoch: int | None = None,
) -> list[TrainingMetrics]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, foreach=True)
    return [train_epoch(model, iter_jsonl_records(dataset_path), optimizer, max_records=max_records_per_epoch) for _ in range(epochs)]


def seed_training(seed: int, *, torch_threads: int | None = None) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch_threads is not None:
        if torch_threads < 1:
            raise ValueError("torch_threads must be at least 1")
        torch.set_num_threads(torch_threads)


def _teacher_scores_tensor(record: dict, action_count: int, device: torch.device) -> Tensor:
    """Return neutral ranking targets when a player has no score function (for example Random)."""
    values = record.get("teacher_scores", record.get("heuristic_scores"))
    if values is None:
        return torch.zeros(action_count, dtype=torch.float32, device=device)
    if len(values) != action_count:
        raise ValueError("heuristic_scores must have one value per action representation")
    return torch.tensor(values, dtype=torch.float32, device=device)


def observation_from_dict(value: dict) -> NeuralObservation:
    def card(item: dict) -> NeuralCardObservation:
        return NeuralCardObservation(**item)

    def counts(item: list | tuple) -> tuple[tuple[str, int], ...]:
        return tuple((str(card_id), int(count)) for card_id, count in item)

    active = value["active_player"]
    opponent = value["opponent"]
    pending = value.get("pending_decision")
    pending_value = NeuralPendingObservation(**pending) if pending is not None else None
    return NeuralObservation(
        phase=value["phase"], status=value["status"], winner=value.get("winner"),
        turn_number=int(value["turn_number"]),
        active_player=NeuralActivePlayerObservation(
            health=active["health"], mastery=active["mastery"], gems=active["gems"], power=active["power"],
            hand=tuple(card(item) for item in active["hand"]),
            draw_pile_counts=counts(active["draw_pile_counts"]), discard_counts=counts(active["discard_counts"]),
            play_zone=tuple(card(item) for item in active["play_zone"]), champions=tuple(card(item) for item in active["champions"]),
            owned_card_counts=counts(active["owned_card_counts"]),
            played_faction_mask=tuple(active["played_faction_mask"]),
            played_champion_faction_mask=tuple(active.get("played_champion_faction_mask", (False, False, False, False))),
            discard=tuple(card(item) for item in active.get("discard", ())),
        ),
        opponent=NeuralOpponentObservation(
            health=opponent["health"], mastery=opponent["mastery"], owned_card_counts=counts(opponent["owned_card_counts"]),
            discard_counts=counts(opponent["discard_counts"]), champions=tuple(card(item) for item in opponent["champions"]),
        ),
        central_deck_counts=counts(value["central_deck_counts"]),
        river=tuple(NeuralRiverCardObservation(slot=item["slot"], card=card(item["card"]) if item["card"] else None) for item in value["river"]),
        pending_decision=pending_value,
        schema_version=value.get("schema_version", 1),
    )
