"""Neural players with targeted Heuristic V8 decision overrides."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from shards_ai.game import Game
from shards_ai.game.actions import (
    Action,
    BanishCard,
    BuyCard,
    RecruitFreeCard,
    RecruitMercenary,
    StopBuying,
)
from shards_ai.game.enums import Phase, PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.state import GameState

from .heuristic_evaluator import CardAcquisitionWeights, CardConstraintWeights, HeuristicWeights
from .heuristic_player import HeuristicPlayer
from .neural_model import NeuralActionScorer
from .neural_player import NeuralPlayer

HybridPolicy = Literal["purchase_recruitment", "play_phase", "banish"]


class HybridPlayer:
    """Delegate selected decision families to Heuristic V8, others to NeuralPlayer."""

    observation_kind = "game_state"
    observation_is_read_only = True
    POLICIES = frozenset(("purchase_recruitment", "play_phase", "banish"))

    def __init__(
        self,
        player_id: PlayerId,
        game: Game,
        rng,
        *,
        scorer: NeuralActionScorer,
        policy: HybridPolicy,
        weights: HeuristicWeights | None = None,
        acquisition_weights: CardAcquisitionWeights | None = None,
        constraint_weights: CardConstraintWeights | None = None,
    ) -> None:
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown hybrid policy: {policy!r}")
        self.player_id = player_id
        self.game = game
        self.policy = policy
        self.neural = NeuralPlayer(player_id, None, rng, scorer=scorer)
        self.heuristic = HeuristicPlayer(
            player_id, weights, acquisition_weights, constraint_weights
        )
        self.heuristic_decisions = 0

    @property
    def decisions(self) -> int:
        return self.neural.decisions + self.heuristic_decisions

    @property
    def total_inference_seconds(self) -> float:
        return self.neural.total_inference_seconds

    def _uses_heuristic(self, observation: GameState, legal_actions: Sequence[Action]) -> bool:
        if self.policy == "play_phase":
            return observation.phase is Phase.PLAY
        if self.policy == "banish":
            return any(isinstance(action, BanishCard) for action in legal_actions)
        return any(
            isinstance(action, (BuyCard, RecruitMercenary, RecruitFreeCard, StopBuying))
            for action in legal_actions
        )

    def choose_action(self, observation: GameState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise InvalidActionError("Cannot choose an action from an empty action list")
        if not isinstance(observation, GameState):
            raise TypeError("HybridPlayer requires a GameState")
        if self._uses_heuristic(observation, legal_actions):
            self.heuristic_decisions += 1
            return self.heuristic.choose_action(observation, legal_actions)
        return self.neural.choose_action(
            self.game.neural_observation_for(self.player_id), legal_actions
        )


__all__ = ["HybridPlayer", "HybridPolicy"]
