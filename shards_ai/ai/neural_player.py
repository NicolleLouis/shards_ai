"""Player adapter that uses a trained action-conditioned neural scorer."""

from __future__ import annotations

import math
import time
from pathlib import Path
from collections.abc import Sequence

import torch

from shards_ai.game.actions import Action
from shards_ai.game.cards import CARD_CATALOG
from shards_ai.game.enums import PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.observation import NeuralObservation
from shards_ai.game.random import GameRandom

from .action_representation import representation_for_neural_action
from .neural_model import NeuralActionScorer, NeuralModelConfig, build_neural_scorer
from .neural_training_profiles import load_active_neural_profile


class NeuralPlayer:
    """Select legal actions with a trained checkpoint from a masked observation."""

    observation_kind = "neural"
    observation_is_read_only = True

    def __init__(
        self,
        player_id: PlayerId,
        checkpoint_path: str | Path | None,
        rng: GameRandom,
        *,
        device: str = "cpu",
        tie_tolerance: float = 1e-6,
        scorer: NeuralActionScorer | None = None,
        mercenary_mode_bias: float = 0.0,
        deck_lean_bias: float = 0.0,
    ) -> None:
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")
        self.player_id = player_id
        self._rng = rng
        self.tie_tolerance = tie_tolerance
        self.mercenary_mode_bias = float(mercenary_mode_bias)
        self.deck_lean_bias = float(deck_lean_bias)
        self.decisions = 0
        self.total_inference_seconds = 0.0
        self.device = torch.device(device)
        self.model = scorer if scorer is not None else self._load_scorer(checkpoint_path, self.device)
        self.model.eval()

    @staticmethod
    def _load_scorer(
        checkpoint_path: str | Path | None,
        device: torch.device,
    ) -> NeuralActionScorer:
        if checkpoint_path is None:
            checkpoint_path = load_active_neural_profile().checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = NeuralModelConfig(**checkpoint["model_config"])
        model = build_neural_scorer(
            str(checkpoint.get("architecture", "independent_action")), model_config,
        ).to(device)
        if tuple(checkpoint.get("card_ids", ())) != model.card_ids:
            raise ValueError("Checkpoint card vocabulary does not match the current card catalog")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def load_scorer(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> NeuralActionScorer:
        """Load one read-only scorer for reuse across benchmark games."""
        return cls._load_scorer(checkpoint_path, torch.device(device))

    def choose_action(
        self,
        observation: NeuralObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise InvalidActionError("Cannot choose an action from an empty action list")
        if not isinstance(observation, NeuralObservation):
            raise TypeError("NeuralPlayer requires a NeuralObservation")
        representations = [representation_for_neural_action(action, observation) for action in actions]
        started = time.perf_counter()
        with torch.inference_mode():
            scores = self.model(observation, representations)
        if self.mercenary_mode_bias:
            scores = scores.clone()
            for index, representation in enumerate(representations):
                card_id = representation.card_definition_id
                is_mercenary = card_id is not None and CARD_CATALOG[card_id].is_mercenary
                if not is_mercenary:
                    continue
                if representation.action_type == "recruit_mercenary":
                    scores[index] += self.mercenary_mode_bias
                elif representation.action_type == "buy_card":
                    scores[index] -= self.mercenary_mode_bias
        if self.deck_lean_bias:
            scores = scores.clone()
            for index, representation in enumerate(representations):
                if representation.action_type == "buy_card":
                    scores[index] -= self.deck_lean_bias
        self.total_inference_seconds += time.perf_counter() - started
        self.decisions += 1
        if not torch.isfinite(scores).all():
            raise InvalidActionError("NeuralPlayer produced a non-finite action score")
        best_score = float(scores.max().item())
        best_indices = [
            index for index, score in enumerate(scores.tolist())
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=self.tie_tolerance)
        ]
        return actions[self._rng.choice(best_indices)]
