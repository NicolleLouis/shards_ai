"""Immutable, replayable profiles for composed HybridPlayer versions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_HYBRID_PROFILE_DIR = Path("configs/hybrid_profiles")


@dataclass(frozen=True, slots=True)
class HybridProfile:
    """A complete composition contract for one replayable hybrid version."""

    profile_id: str
    acquisition_policy_id: str
    acquisition_checkpoint: Path
    play_policy_id: str
    play_profile: Path
    banish_policy_id: str
    acquisition_policy_profile: Path | None = None
    play_policy_profile: Path | None = None
    banish_policy_profile: Path | None = None
    schema_version: int = 1
    parent_profile_id: str | None = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def resolved_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "parent_profile_id": self.parent_profile_id,
            "policies": {
                "acquisition": {
                    "policy_id": self.acquisition_policy_id,
                    "checkpoint": str(self.acquisition_checkpoint),
                    "profile": str(self.acquisition_policy_profile) if self.acquisition_policy_profile else None,
                },
                "play": {
                    "policy_id": self.play_policy_id,
                    "profile": str(self.play_profile),
                    "policy_profile": str(self.play_policy_profile) if self.play_policy_profile else None,
                },
                "banish": {
                    "policy_id": self.banish_policy_id,
                    "profile": str(self.banish_policy_profile) if self.banish_policy_profile else None,
                },
            },
            "metadata": dict(self.metadata or {}),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.resolved_document(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Hybrid profile field {key!r} must be a mapping")
    return value


def _path_from_profile(profile_path: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hybrid profile field {field_name!r} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_hybrid_profile(path_or_id: str | Path) -> HybridProfile:
    """Load one exact hybrid version; no active pointer is consulted."""

    requested = Path(path_or_id)
    if len(requested.parts) == 1 and requested.suffix == "":
        requested = DEFAULT_HYBRID_PROFILE_DIR / f"{requested.name}.yaml"
    profile_path = requested
    with profile_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"Hybrid profile must contain a mapping: {profile_path}")

    profile_id = document.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Hybrid profile must define a non-empty profile_id")
    schema_version = document.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"Unsupported hybrid profile schema: {schema_version!r}")
    policies = _required_mapping(document, "policies")
    acquisition = _required_mapping(policies, "acquisition")
    play = _required_mapping(policies, "play")
    banish = _required_mapping(policies, "banish")

    def policy_id(section: Mapping[str, Any], family: str) -> str:
        value = section.get("policy_id")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Hybrid {family} policy must define policy_id")
        return value

    parent = document.get("parent_profile_id")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("parent_profile_id must be a string or null")
    metadata = document.get("metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Hybrid profile metadata must be a mapping")

    return HybridProfile(
        profile_id=profile_id,
        acquisition_policy_id=policy_id(acquisition, "acquisition"),
        acquisition_checkpoint=_path_from_profile(
            profile_path, acquisition.get("checkpoint"), "policies.acquisition.checkpoint"
        ),
        play_policy_id=policy_id(play, "play"),
        play_profile=_path_from_profile(
            profile_path, play.get("profile"), "policies.play.profile"
        ),
        banish_policy_id=policy_id(banish, "banish"),
        acquisition_policy_profile=(
            _path_from_profile(profile_path, acquisition["profile"], "policies.acquisition.profile")
            if acquisition.get("profile") else None
        ),
        play_policy_profile=(
            _path_from_profile(profile_path, play["policy_profile"], "policies.play.policy_profile")
            if play.get("policy_profile") else None
        ),
        banish_policy_profile=(
            _path_from_profile(profile_path, banish["profile"], "policies.banish.profile")
            if banish.get("profile") else None
        ),
        schema_version=schema_version,
        parent_profile_id=parent,
        metadata=dict(metadata),
    )


__all__ = ["DEFAULT_HYBRID_PROFILE_DIR", "HybridProfile", "load_hybrid_profile"]
