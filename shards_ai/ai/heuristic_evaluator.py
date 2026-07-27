"""Pure scoring primitives for the first weighted heuristic player."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ActionFeatures:
    """Non-negative action signals consumed by :class:`HeuristicWeights`."""

    cost_paid: float = 0.0
    gems_produced: float = 0.0
    power_produced: float = 0.0
    mastery_gained: float = 0.0
    health_gained: float = 0.0
    card_draw: float = 0.0
    shield_value: float = 0.0
    deck_thinning: float = 0.0
    card_acquisition_value: float = 0.0
    champion_value: float = 0.0
    target_denial: float = 0.0
    damage_value: float = 0.0
    constraint_penalty: float = 0.0
    phase_progress: float = 0.0
    action_penalty: float = 0.0
    lethal: float = 0.0
    terminal_win: float = 0.0
    health_advantage_delta: float = 0.0
    mastery_advantage_delta: float = 0.0
    opponent_threat_delta: float = 0.0
    self_threat_delta: float = 0.0
    purchase_opportunity_cost: float = 0.0
    mastery_threshold_value: float = 0.0
    projection_supported: bool = True

@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """Named coefficients for the active v008 heuristic profile."""

    cost_paid: float = 0.0
    gems_produced: float = 0.75
    power_produced: float = 0.5
    mastery_gained: float = 0.25
    health_gained: float = 2.75
    card_draw: float = 0.75
    shield_value: float = 0.5
    deck_thinning: float = 1.0
    card_acquisition_value: float = 1.0
    champion_value: float = 2.75
    target_denial: float = 1.5
    damage_value: float = 1.25
    constraint_penalty: float = -1.0
    phase_progress: float = 0.1
    action_penalty: float = -1.0
    lethal: float = 1000.0
    terminal_win: float = 1000.0
    health_advantage_delta: float = 0.0
    mastery_advantage_delta: float = 0.0
    opponent_threat_delta: float = 0.0
    self_threat_delta: float = 0.0
    purchase_opportunity_cost: float = -1.0
    mastery_threshold_value: float = 0.0
    buy_threshold: float = 0.625

    def __post_init__(self) -> None:
        for field in fields(self):
            if not isfinite(getattr(self, field.name)):
                raise ValueError(f"Weight {field.name} must be finite")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "HeuristicWeights":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown heuristic weights: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def zero(cls) -> "HeuristicWeights":
        """Return neutral weights, useful for isolated feature tests."""

        return cls(**{field.name: 0.0 for field in fields(cls)})

    def score(self, features: ActionFeatures) -> float:
        return (
            self.cost_paid * features.cost_paid
            + self.gems_produced * features.gems_produced
            + self.power_produced * features.power_produced
            + self.mastery_gained * features.mastery_gained
            + self.health_gained * features.health_gained
            + self.card_draw * features.card_draw
            + self.shield_value * features.shield_value
            + self.deck_thinning * features.deck_thinning
            + self.card_acquisition_value * features.card_acquisition_value
            + self.champion_value * features.champion_value
            + self.target_denial * features.target_denial
            + self.damage_value * features.damage_value
            + self.constraint_penalty * features.constraint_penalty
            + self.phase_progress * features.phase_progress
            + self.action_penalty * features.action_penalty
            + self.lethal * features.lethal
            + self.terminal_win * features.terminal_win
            + self.health_advantage_delta * features.health_advantage_delta
            + self.mastery_advantage_delta * features.mastery_advantage_delta
            + self.opponent_threat_delta * features.opponent_threat_delta
            + self.self_threat_delta * features.self_threat_delta
            + self.purchase_opportunity_cost * features.purchase_opportunity_cost
            + self.mastery_threshold_value * features.mastery_threshold_value
        )


@dataclass(frozen=True, slots=True)
class CardAcquisitionWeights:
    """Internal coefficients used to estimate the value of acquiring a card."""

    gems_produced: float = 0.0
    power_produced: float = 0.75
    mastery_gained: float = 0.75
    health_gained: float = 1.375
    card_draw: float = 1.75
    deck_thinning: float = 1.0
    target_denial: float = 1.0
    banish_threshold: float = 3.0
    durable_replay_factor: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"Acquisition weight {field.name} must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "CardAcquisitionWeights":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown card acquisition weights: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CardConstraintWeights:
    """Relative penalties for conditions attached to card effects."""

    mastery: float = 1.0
    health: float = 0.75
    inspiration: float = 0.5
    echo: float = 0.75
    union: float = 1.0
    domination: float = 1.5

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"Constraint weight {field.name} must be finite and non-negative")

    @classmethod
    def legacy(cls) -> "CardConstraintWeights":
        """Return the pre-v004 uniform-penalty behavior."""

        return cls(health=1.0, inspiration=1.0, echo=1.0, union=1.0, domination=1.0)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "CardConstraintWeights":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown card constraint weights: {sorted(unknown)}")
        return cls(**values)
