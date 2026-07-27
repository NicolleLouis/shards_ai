"""Optional transition-level reward shaping instrumentation."""

from __future__ import annotations

from dataclasses import dataclass

from shards_ai.ai.state_evaluator import StateRewardWeights, state_potential
from shards_ai.game.actions import Action
from shards_ai.game.enums import PlayerId
from shards_ai.game.state import GameState


@dataclass(frozen=True, slots=True)
class TransitionReward:
    turn_number: int
    player_id: PlayerId
    action_type: str
    reward: float
    potential_before: float
    potential_after: float


class RewardShapingTracker:
    """Collect shaping rewards for one player in one game."""

    def __init__(
        self,
        player_id: PlayerId,
        weights: StateRewardWeights | None = None,
        *,
        keep_transitions: bool = False,
    ) -> None:
        self.player_id = player_id
        self.weights = weights or StateRewardWeights()
        self.transitions: list[TransitionReward] = []
        self._keep_transitions = keep_transitions
        self._total_reward = 0.0
        self._transition_count = 0
        self._final_potential = 0.0

    def observe(
        self,
        before: GameState,
        action: Action,
        after: GameState,
        player_id: PlayerId,
    ) -> None:
        before_potential = state_potential(before, self.player_id, self.weights)
        after_potential = state_potential(after, self.player_id, self.weights)
        reward = self.weights.gamma * after_potential - before_potential
        self._total_reward += reward
        self._transition_count += 1
        self._final_potential = after_potential
        if self._keep_transitions:
            self.transitions.append(
                TransitionReward(
                    turn_number=before.turn_number,
                    player_id=player_id,
                    action_type=type(action).__name__,
                    reward=reward,
                    potential_before=before_potential,
                    potential_after=after_potential,
                )
            )

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def mean_reward(self) -> float:
        return self._total_reward / self._transition_count if self._transition_count else 0.0

    @property
    def final_potential(self) -> float:
        return self._final_potential


__all__ = ["RewardShapingTracker", "TransitionReward"]
