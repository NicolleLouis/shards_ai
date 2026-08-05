"""Versioned configuration profiles for neural training runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .neural_model import NeuralModelConfig


DEFAULT_TRAINING_PROFILE_DIR = Path("configs/neural_training_profiles")
DEFAULT_ACTIVE_PROFILE_PATH = DEFAULT_TRAINING_PROFILE_DIR / "active.yaml"
DEFAULT_NEURAL_PROFILE_DIR = Path("configs/neural_profiles")
DEFAULT_ACTIVE_NEURAL_PROFILE_PATH = DEFAULT_NEURAL_PROFILE_DIR / "active.yaml"


@dataclass(frozen=True, slots=True)
class NeuralTrainingProfile:
    profile_id: str
    dataset: str | None
    output: str
    method: str = "imitation"
    parent_profile_id: str | None = None
    seed: int = 0
    split_seed: int = 0
    epochs: int = 1
    learning_rate: float = 1e-3
    torch_threads: int = 1
    max_records: int | None = None
    max_validation_records: int = 10000
    model: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    initial_checkpoint: str | None = None
    total_games: int = 100000
    games_per_update: int = 128
    optimization_epochs: int = 4
    minibatch_size: int = 2048
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    reference_kl_coefficient: float = 0.02
    evaluation_interval_games: int = 128
    evaluation_games: int = 64
    evaluation_seed: int = 91000
    max_transitions_per_update: int = 20000
    max_actions: int = 10000
    max_turns: int = 200
    opponents: Mapping[str, float] = field(default_factory=lambda: {
        "random": 1 / 3,
        "v007": 1 / 3,
        "v008": 1 / 3,
    })
    reward_shaping: Mapping[str, Any] = field(default_factory=dict)

    def resolved_model_config(self) -> NeuralModelConfig:
        return NeuralModelConfig(**dict(self.model or {}))

    def resolved_document(self) -> dict[str, Any]:
        document = {
            "profile_id": self.profile_id,
            "parent_profile_id": self.parent_profile_id,
            "method": self.method,
            "dataset": self.dataset,
            "output": self.output,
            "seed": self.seed,
            "split_seed": self.split_seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "torch_threads": self.torch_threads,
            "max_records": self.max_records,
            "max_validation_records": self.max_validation_records,
            "model": dict(self.model or {}),
            "metadata": dict(self.metadata or {}),
        }
        if self.method == "ppo":
            document.update({
                "initial_checkpoint": self.initial_checkpoint,
                "total_games": self.total_games,
                "games_per_update": self.games_per_update,
                "optimization_epochs": self.optimization_epochs,
                "minibatch_size": self.minibatch_size,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_loss_coefficient": self.value_loss_coefficient,
                "entropy_coefficient": self.entropy_coefficient,
                "reference_kl_coefficient": self.reference_kl_coefficient,
                "evaluation_interval_games": self.evaluation_interval_games,
                "evaluation_games": self.evaluation_games,
                "evaluation_seed": self.evaluation_seed,
                "max_transitions_per_update": self.max_transitions_per_update,
                "max_actions": self.max_actions,
                "max_turns": self.max_turns,
                "opponents": dict(self.opponents or {}),
                "reward_shaping": dict(self.reward_shaping or {}),
            })
        return document

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.resolved_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path.cwd() / path


@dataclass(frozen=True, slots=True)
class NeuralProfile:
    """Versioned inference checkpoint selected by the active neural pointer."""

    profile_id: str
    checkpoint_path: Path


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_training_profile(path: str | Path) -> NeuralTrainingProfile:
    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"Profile must contain a mapping: {profile_path}")

    profile_id = document.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Profile must define a non-empty profile_id")
    method = document.get("method", "imitation")
    if method not in {"imitation", "ppo"}:
        raise ValueError(f"Unsupported neural training method: {method!r}")
    parent = document.get("parent_profile_id")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("parent_profile_id must be a string or null")
    dataset = document.get("dataset")
    output = document.get("output")
    if method == "imitation" and (not isinstance(dataset, str) or not dataset):
        raise ValueError("Profile must define a non-empty dataset")
    if dataset is not None and (not isinstance(dataset, str) or not dataset):
        raise ValueError("dataset must be a non-empty string when provided")
    if not isinstance(output, str) or not output:
        raise ValueError("Profile must define a non-empty output")

    model = document.get("model", {}) or {}
    if not isinstance(model, Mapping):
        raise ValueError("Profile model must be a mapping")
    model_fields = {item.name for item in fields(NeuralModelConfig)}
    unknown_model_fields = set(model) - model_fields
    if unknown_model_fields:
        raise ValueError(f"Unknown NeuralModelConfig fields: {sorted(unknown_model_fields)}")
    try:
        NeuralModelConfig(**dict(model))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid model configuration: {error}") from error

    metadata = document.get("metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("Profile metadata must be a mapping")
    profile = NeuralTrainingProfile(
        profile_id=profile_id,
        parent_profile_id=parent,
        method=method,
        dataset=dataset,
        output=output,
        seed=document.get("seed", 0),
        split_seed=document.get("split_seed", 0),
        epochs=document.get("epochs", 1),
        learning_rate=document.get("learning_rate", 1e-3),
        torch_threads=document.get("torch_threads", 1),
        max_records=document.get("max_records"),
        max_validation_records=document.get("max_validation_records", 10000),
        model=dict(model),
        metadata=dict(metadata),
        initial_checkpoint=document.get("initial_checkpoint"),
        total_games=document.get("total_games", 100000),
        games_per_update=document.get("games_per_update", 128),
        optimization_epochs=document.get("optimization_epochs", 4),
        minibatch_size=document.get("minibatch_size", 2048),
        gamma=document.get("gamma", 0.995),
        gae_lambda=document.get("gae_lambda", 0.95),
        clip_epsilon=document.get("clip_epsilon", 0.2),
        value_loss_coefficient=document.get("value_loss_coefficient", 0.5),
        entropy_coefficient=document.get("entropy_coefficient", 0.01),
        reference_kl_coefficient=document.get("reference_kl_coefficient", 0.02),
        evaluation_interval_games=document.get("evaluation_interval_games", 128),
        evaluation_games=document.get("evaluation_games", 64),
        evaluation_seed=document.get("evaluation_seed", 91000),
        max_transitions_per_update=document.get("max_transitions_per_update", 20000),
        max_actions=document.get("max_actions", 10000),
        max_turns=document.get("max_turns", 200),
        opponents=dict(document.get("opponents", {
            "random": 1 / 3,
            "v007": 1 / 3,
            "v008": 1 / 3,
        }) or {}),
        reward_shaping=dict(document.get("reward_shaping", {}) or {}),
    )
    for name in ("seed", "split_seed"):
        _nonnegative_int(getattr(profile, name), name)
    _positive_int(profile.epochs, "epochs")
    _positive_int(profile.torch_threads, "torch_threads")
    _positive_int(profile.max_validation_records, "max_validation_records")
    _positive_int(profile.max_records, "max_records", allow_none=True)
    if profile.method == "ppo":
        for name in (
            "total_games", "games_per_update", "optimization_epochs", "minibatch_size",
            "evaluation_interval_games", "evaluation_games", "max_transitions_per_update",
            "max_actions", "max_turns",
        ):
            _positive_int(getattr(profile, name), name)
        _nonnegative_int(profile.evaluation_seed, "evaluation_seed")
        for name in ("gamma", "gae_lambda", "clip_epsilon"):
            value = getattr(profile, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if profile.gamma > 1 or profile.gae_lambda > 1 or profile.clip_epsilon >= 1:
            raise ValueError("gamma and gae_lambda must be <= 1 and clip_epsilon must be < 1")
        for name in ("value_loss_coefficient", "entropy_coefficient", "reference_kl_coefficient"):
            value = getattr(profile, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if not profile.opponents:
            raise ValueError("opponents must not be empty")
        if any(not isinstance(name, str) or not name for name in profile.opponents):
            raise ValueError("opponent names must be non-empty strings")
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0 for weight in profile.opponents.values()):
            raise ValueError("opponent weights must be non-negative numbers")
        if sum(profile.opponents.values()) <= 0:
            raise ValueError("opponent weights must have a positive sum")
    if isinstance(profile.learning_rate, bool) or not isinstance(profile.learning_rate, (int, float)) or profile.learning_rate <= 0:
        raise ValueError("learning_rate must be a positive number")
    return profile


def save_training_profile(profile: NeuralTrainingProfile, path: str | Path) -> Path:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, **profile.resolved_document()}
    with profile_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
    return profile_path


def load_active_training_profile(
    active_path: str | Path = DEFAULT_ACTIVE_PROFILE_PATH,
) -> NeuralTrainingProfile:
    pointer_path = Path(active_path)
    with pointer_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"Active profile must contain a mapping: {pointer_path}")
    profile_id = document.get("active_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Active profile must define a non-empty active_profile_id")
    profile_path = pointer_path.parent / f"{profile_id}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Active neural profile not found: {profile_path}")
    return load_training_profile(profile_path)


def load_active_neural_profile(
    active_path: str | Path = DEFAULT_ACTIVE_NEURAL_PROFILE_PATH,
) -> NeuralProfile:
    pointer_path = Path(active_path)
    with pointer_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"Active neural profile must contain a mapping: {pointer_path}")
    profile_id = document.get("active_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Active neural profile must define a non-empty active_profile_id")
    checkpoint_path = pointer_path.parent / f"{profile_id}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Active neural checkpoint not found: {checkpoint_path}")
    return NeuralProfile(profile_id=profile_id, checkpoint_path=checkpoint_path)


def versioned_training_profiles(
    directory: str | Path = DEFAULT_TRAINING_PROFILE_DIR,
) -> list[tuple[int, Path, NeuralTrainingProfile]]:
    result: list[tuple[int, Path, NeuralTrainingProfile]] = []
    for path in Path(directory).glob("v*.yaml"):
        profile_id = path.stem
        if len(profile_id) < 2 or not profile_id[1:].isdigit():
            continue
        result.append((int(profile_id[1:]), path, load_training_profile(path)))
    return sorted(result, key=lambda item: item[0])


def next_training_profile_id(
    directory: str | Path = DEFAULT_TRAINING_PROFILE_DIR,
) -> str:
    profiles = versioned_training_profiles(directory)
    return f"v{(profiles[-1][0] + 1 if profiles else 1):03d}"


__all__ = [
    "DEFAULT_ACTIVE_PROFILE_PATH",
    "DEFAULT_ACTIVE_NEURAL_PROFILE_PATH",
    "DEFAULT_NEURAL_PROFILE_DIR",
    "DEFAULT_TRAINING_PROFILE_DIR",
    "NeuralTrainingProfile",
    "NeuralProfile",
    "load_active_training_profile",
    "load_training_profile",
    "next_training_profile_id",
    "save_training_profile",
    "versioned_training_profiles",
]
