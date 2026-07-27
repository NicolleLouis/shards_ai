"""Load and save versioned heuristic profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .heuristic_evaluator import CardAcquisitionWeights, CardConstraintWeights, HeuristicWeights


@dataclass(frozen=True, slots=True)
class HeuristicProfile:
    profile_id: str
    weights: HeuristicWeights
    parent_profile_id: str | None = None
    metadata: Mapping[str, Any] | None = None
    card_acquisition_weights: CardAcquisitionWeights = CardAcquisitionWeights()
    constraint_weights: CardConstraintWeights = CardConstraintWeights()


def load_profile(path: str | Path) -> HeuristicProfile:
    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Profile must contain a mapping: {profile_path}")

    profile_id = document.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Profile must define a non-empty profile_id")
    weights = document.get("weights", {})
    if not isinstance(weights, Mapping):
        raise ValueError("Profile weights must be a mapping")
    parent_profile_id = document.get("parent_profile_id")
    if parent_profile_id is not None and not isinstance(parent_profile_id, str):
        raise ValueError("parent_profile_id must be a string or null")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Profile metadata must be a mapping")

    constraint_document = document.get("constraint_weights")
    if constraint_document is None:
        constraint_weights = CardConstraintWeights.legacy()
    elif not isinstance(constraint_document, Mapping):
        raise ValueError("Profile constraint_weights must be a mapping")
    else:
        constraint_weights = CardConstraintWeights.from_mapping(
            {str(key): float(value) for key, value in constraint_document.items()}
        )

    return HeuristicProfile(
        profile_id=profile_id,
        weights=HeuristicWeights.from_mapping({str(key): float(value) for key, value in weights.items()}),
        card_acquisition_weights=CardAcquisitionWeights.from_mapping(
            {
                str(key): float(value)
                for key, value in (document.get("card_acquisition_weights", {}) or {}).items()
            }
        ),
        parent_profile_id=parent_profile_id,
        metadata=dict(metadata),
        constraint_weights=constraint_weights,
    )


def save_profile(profile: HeuristicProfile, path: str | Path) -> Path:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "parent_profile_id": profile.parent_profile_id,
        "weights": asdict(profile.weights),
        "card_acquisition_weights": asdict(profile.card_acquisition_weights),
        "constraint_weights": asdict(profile.constraint_weights),
        "metadata": dict(profile.metadata or {}),
    }
    with profile_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
    return profile_path


__all__ = ["HeuristicProfile", "load_profile", "save_profile"]
