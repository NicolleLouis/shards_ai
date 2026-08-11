"""Streaming imitation training utilities for macro candidate decisions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .macro_model import MacroActionScorer, macro_candidate_from_dict
from .neural_training import iter_jsonl_records, observation_from_dict


@dataclass(frozen=True, slots=True)
class MacroTrainingMetrics:
    records: int
    mean_loss: float
    all_records: int = 0
    non_trivial_records: int = 0
    by_decision_kind: dict[str, int] = field(default_factory=dict)
    by_phase: dict[str, int] = field(default_factory=dict)
    by_action_type: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MacroEvaluationMetrics:
    records: int
    mean_loss: float
    top1_accuracy: float
    mean_chosen_rank: float
    mean_normalized_chosen_rank: float
    all_records: int = 0
    non_trivial_records: int = 0
    by_decision_kind: dict[str, int] = field(default_factory=dict)
    by_phase: dict[str, int] = field(default_factory=dict)
    by_action_type: dict[str, int] = field(default_factory=dict)
    by_matchup: dict[str, int] = field(default_factory=dict)
    mean_candidate_count: float = 0.0
    collision_records: int = 0
    teacher_collision_records: int = 0


def macro_records(path: str) -> Iterable[dict]:
    """Yield historical or current non-trivial macro records."""

    for record in iter_jsonl_records(path):
        if record.get("decision_kind") != "macro_play":
            continue
        if record.get("dataset_schema_version") not in (2, 3):
            raise ValueError("Macro trainer requires dataset_schema_version=2 or 3")
        candidates = record.get("candidates", ())
        if any(
            candidate.get("schema_version") not in (2, 3, 4)
            or not isinstance(candidate.get("root_action"), dict)
            for candidate in candidates
        ):
            raise ValueError("Macro trainer requires candidate schema V2, V3 or V4 with root_action")
        if len(candidates) >= 2:
            yield record


def unified_records(path: str) -> Iterable[dict]:
    """Yield strict V4 macro and atomic records for the unified trainer."""

    for record in iter_jsonl_records(path):
        if record.get("decision_kind") not in {"macro_play", "atomic"}:
            continue
        if record.get("dataset_schema_version") != 3:
            raise ValueError("Unified trainer requires dataset_schema_version=3")
        candidates = record.get("candidates", ())
        if any(
            candidate.get("schema_version") != 4
            or candidate.get("decision_kind") != record["decision_kind"]
            or not isinstance(candidate.get("root_action"), dict)
            for candidate in candidates
        ):
            raise ValueError("Unified trainer requires candidate schema V4 with matching decision_kind")
        if len(candidates) >= 2:
            yield record


def unified_record_diagnostics(records: Iterable[dict]) -> dict[str, object]:
    """Report coverage and semantic collisions for both candidate spaces."""

    all_records = 0
    non_trivial_records = 0
    candidate_counts: Counter[int] = Counter()
    collisions = teacher_collisions = 0
    by_kind: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_action: Counter[str] = Counter()
    for record in records:
        if record.get("decision_kind") not in {"macro_play", "atomic"}:
            continue
        all_records += 1
        candidates = record.get("candidates", ())
        candidate_counts[len(candidates)] += 1
        by_kind[str(record["decision_kind"])] += 1
        by_phase[str(record.get("phase", record.get("observation", {}).get("phase", "unknown")))] += 1
        chosen = record.get("chosen_candidate_index")
        if isinstance(chosen, int) and 0 <= chosen < len(candidates):
            by_action[_candidate_action_type(candidates[chosen])] += 1
        if len(candidates) < 2:
            continue
        non_trivial_records += 1
        keys = [json.dumps(candidate, sort_keys=True, separators=(",", ":")) for candidate in candidates]
        counts = Counter(keys)
        if any(count > 1 for count in counts.values()):
            collisions += 1
        if isinstance(chosen, int) and 0 <= chosen < len(keys) and counts[keys[chosen]] > 1:
            teacher_collisions += 1
    return {
        "all_records": all_records,
        "non_trivial_records": non_trivial_records,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "collision_records": collisions,
        "teacher_collision_records": teacher_collisions,
        "by_decision_kind": dict(sorted(by_kind.items())),
        "by_phase": dict(sorted(by_phase.items())),
        "by_action_type": dict(sorted(by_action.items())),
    }


def macro_record_diagnostics(records: Iterable[dict]) -> dict[str, object]:
    """Summarize baseline quality without treating singleton records as choices."""

    all_records = 0
    non_trivial_records = 0
    candidate_counts: Counter[int] = Counter()
    teacher_collisions = 0
    alternative_collisions = 0
    for record in records:
        if record.get("decision_kind") != "macro_play":
            continue
        all_records += 1
        candidates = record.get("candidates", ())
        candidate_counts[len(candidates)] += 1
        if len(candidates) < 2:
            continue
        non_trivial_records += 1
        # JSON records may contain lists (for example ``trace_action_types``),
        # so use a canonical JSON key rather than a tuple of raw values.
        keys = [json.dumps(candidate, sort_keys=True, separators=(",", ":")) for candidate in candidates]
        counts = Counter(keys)
        chosen = record.get("chosen_candidate_index")
        if isinstance(chosen, int) and 0 <= chosen < len(keys) and counts[keys[chosen]] > 1:
            teacher_collisions += 1
        if any(count > 1 for count in counts.values()):
            alternative_collisions += 1
    return {
        "all_records": all_records,
        "non_trivial_records": non_trivial_records,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "teacher_collision_records": teacher_collisions,
        "alternative_collision_records": alternative_collisions,
    }


def macro_imitation_loss(predicted: Tensor, chosen_index: int | None) -> Tensor:
    """Train the chosen branch above every alternative and with cross-entropy."""

    if chosen_index is None or not 0 <= chosen_index < predicted.numel() or predicted.numel() < 2:
        return predicted.sum() * 0
    chosen = predicted[chosen_index]
    alternatives = torch.cat((predicted[:chosen_index], predicted[chosen_index + 1:]))
    ranking = F.softplus(-(chosen - alternatives)).mean()
    classification = F.cross_entropy(predicted.unsqueeze(0), torch.tensor([chosen_index], device=predicted.device))
    return ranking + 0.25 * classification


def _batch(record: dict, model: MacroActionScorer):
    observation = observation_from_dict(record["observation"])
    candidates = [macro_candidate_from_dict(value) for value in record["candidates"]]
    return observation, candidates, model(observation, candidates)


def train_macro_epoch(
    model: MacroActionScorer,
    records: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    *,
    max_records: int | None = None,
    record_weight=None,
) -> MacroTrainingMetrics:
    model.train()
    total = 0.0
    count = 0
    all_records = 0
    total_weight = 0.0
    by_decision_kind: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_action_type: Counter[str] = Counter()
    for record in records:
        all_records += 1
        if max_records is not None and count >= max_records:
            break
        if len(record.get("candidates", ())) < 2:
            continue
        _observation, _candidates, predicted = _batch(record, model)
        loss = macro_imitation_loss(predicted, record.get("chosen_candidate_index"))
        weight = float(record_weight(record)) if record_weight is not None else 1.0
        if weight <= 0:
            raise ValueError("record_weight must return a positive value")
        loss = loss * weight
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
        total_weight += weight
        _count_record(record, by_decision_kind, by_phase, by_action_type)
    return MacroTrainingMetrics(
        count, total / total_weight if total_weight else 0.0, all_records, count,
        dict(sorted(by_decision_kind.items())), dict(sorted(by_phase.items())),
        dict(sorted(by_action_type.items())),
    )


@torch.no_grad()
def evaluate_macro_epoch(
    model: MacroActionScorer,
    records: Iterable[dict],
    *,
    max_records: int | None = None,
    record_weight=None,
) -> MacroEvaluationMetrics:
    model.eval()
    total = 0.0
    count = top1 = 0
    rank_total = normalized_total = 0.0
    all_records = 0
    total_weight = 0.0
    by_decision_kind: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_action_type: Counter[str] = Counter()
    by_matchup: Counter[str] = Counter()
    candidate_total = 0
    collision_records = teacher_collision_records = 0
    for record in records:
        all_records += 1
        if max_records is not None and count >= max_records:
            break
        if len(record.get("candidates", ())) < 2:
            continue
        _observation, _candidates, predicted = _batch(record, model)
        chosen_index = record.get("chosen_candidate_index")
        weight = float(record_weight(record)) if record_weight is not None else 1.0
        if weight <= 0:
            raise ValueError("record_weight must return a positive value")
        total += float(macro_imitation_loss(predicted, chosen_index).cpu()) * weight
        total_weight += weight
        if chosen_index is not None and 0 <= chosen_index < predicted.numel():
            order = torch.argsort(predicted, descending=True).tolist()
            rank = order.index(chosen_index) + 1
            rank_total += rank
            normalized_total += 1.0 - ((rank - 1) / max(1, len(order) - 1))
            top1 += rank == 1
        count += 1
        _count_record(record, by_decision_kind, by_phase, by_action_type)
        by_matchup[str(record.get("opponent_id", "unknown"))] += 1
        collision, teacher_collision = _collision_flags(record)
        collision_records += collision
        teacher_collision_records += teacher_collision
        candidate_total += len(record.get("candidates", ()))
    return MacroEvaluationMetrics(
        records=count,
        mean_loss=total / total_weight if total_weight else 0.0,
        top1_accuracy=top1 / count if count else 0.0,
        mean_chosen_rank=rank_total / count if count else 0.0,
        mean_normalized_chosen_rank=normalized_total / count if count else 0.0,
        all_records=all_records,
        non_trivial_records=count,
        by_decision_kind=dict(sorted(by_decision_kind.items())),
        by_phase=dict(sorted(by_phase.items())),
        by_action_type=dict(sorted(by_action_type.items())),
        by_matchup=dict(sorted(by_matchup.items())),
        mean_candidate_count=candidate_total / count if count else 0.0,
        collision_records=collision_records,
        teacher_collision_records=teacher_collision_records,
    )


def _count_record(record, by_decision_kind, by_phase, by_action_type) -> None:
    by_decision_kind[str(record.get("decision_kind", "unknown"))] += 1
    by_phase[str(record.get("phase", record.get("observation", {}).get("phase", "unknown")))] += 1
    chosen = record.get("chosen_action", {})
    action_type = chosen.get("action_type") if isinstance(chosen, dict) else None
    if action_type is None:
        index = record.get("chosen_candidate_index")
        candidates = record.get("candidates", ())
        if isinstance(index, int) and 0 <= index < len(candidates):
            action_type = _candidate_action_type(candidates[index])
    by_action_type[str(action_type or "unknown")] += 1


def _collision_flags(record: dict) -> tuple[int, int]:
    candidates = record.get("candidates", ())
    keys = [json.dumps(candidate, sort_keys=True, separators=(",", ":")) for candidate in candidates]
    counts = Counter(keys)
    collision = int(any(count > 1 for count in counts.values()))
    chosen = record.get("chosen_candidate_index")
    teacher_collision = int(
        isinstance(chosen, int)
        and 0 <= chosen < len(keys)
        and counts[keys[chosen]] > 1
    )
    return collision, teacher_collision


def _candidate_action_type(candidate: dict) -> str:
    root = candidate.get("root_action")
    if isinstance(root, dict) and root.get("action_type"):
        return str(root["action_type"])
    return str(candidate.get("action_type", "unknown"))


__all__ = [
    "MacroEvaluationMetrics",
    "MacroTrainingMetrics",
    "evaluate_macro_epoch",
    "macro_imitation_loss",
    "macro_record_diagnostics",
    "macro_records",
    "unified_record_diagnostics",
    "unified_records",
    "train_macro_epoch",
]
