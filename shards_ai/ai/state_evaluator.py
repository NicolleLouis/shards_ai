"""Deterministic differential state evaluation for heuristic shaping."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

from shards_ai.game.enums import PlayerId
from shards_ai.game.state import GameState


def _clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class StateRewardWeights:
    """Fixed shaping weights; policy coefficients are kept in HeuristicWeights."""

    health_advantage_weight: float = 1.0 / 3.0
    mastery_advantage_weight: float = 1.0 / 3.0
    opponent_threat_weight: float = 1.0 / 3.0
    champion_presence_weight: float = 1.0
    champion_threat_scale: float = 4.0
    gamma: float = 1.0
    initial_alpha: float = 0.10

    def __post_init__(self) -> None:
        if self.champion_threat_scale <= 0 or not 0 <= self.initial_alpha <= 1:
            raise ValueError("Invalid state reward scale")
        if self.gamma != 1.0:
            raise ValueError("V1 state shaping requires gamma=1.0")
        for field in fields(self):
            if not isfinite(getattr(self, field.name)):
                raise ValueError(f"State reward weight {field.name} must be finite")


def health_advantage(state: GameState, player_id: PlayerId) -> float:
    player = state.players[player_id]
    opponent = state.players[player_id.opponent]
    return _clamp((player.health - opponent.health) / 50.0)


def mastery_advantage(state: GameState, player_id: PlayerId) -> float:
    player = state.players[player_id]
    opponent = state.players[player_id.opponent]
    return _clamp((player.mastery - opponent.mastery) / 30.0)


def champion_threat_advantage(
    state: GameState,
    player_id: PlayerId,
    weights: StateRewardWeights | None = None,
) -> float:
    weights = weights or StateRewardWeights()
    player = state.players[player_id]
    opponent = state.players[player_id.opponent]
    own_threat = weights.champion_presence_weight * len(player.champions)
    opponent_threat = weights.champion_presence_weight * len(opponent.champions)
    return _clamp((own_threat - opponent_threat) / weights.champion_threat_scale)


def state_potential(
    state: GameState,
    player_id: PlayerId,
    weights: StateRewardWeights | None = None,
) -> float:
    weights = weights or StateRewardWeights()
    player = state.players[player_id]
    opponent = state.players[player_id.opponent]
    health_delta = _clamp((player.health - opponent.health) / 50.0)
    mastery_delta = _clamp((player.mastery - opponent.mastery) / 30.0)
    threat_delta = _clamp(
        (
            weights.champion_presence_weight * len(player.champions)
            - weights.champion_presence_weight * len(opponent.champions)
        )
        / weights.champion_threat_scale
    )
    return (
        weights.health_advantage_weight * health_delta
        + weights.mastery_advantage_weight * mastery_delta
        + weights.opponent_threat_weight * threat_delta
    )


def shaping_reward(
    before: GameState,
    after: GameState,
    player_id: PlayerId,
    weights: StateRewardWeights | None = None,
) -> float:
    weights = weights or StateRewardWeights()
    return weights.gamma * state_potential(after, player_id, weights) - state_potential(
        before, player_id, weights
    )


__all__ = [
    "StateRewardWeights",
    "champion_threat_advantage",
    "health_advantage",
    "mastery_advantage",
    "shaping_reward",
    "state_potential",
]
