from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from shards_ai.game.cards import (
    CardDefinition,
    ChampionAbility,
    Effect,
    EffectStep,
    Operation,
)


CARD_REPRESENTATION_SCHEMA_VERSION = 1

SUPPORTED_OPERATION_KINDS = frozenset(
    {
        "gain_gems",
        "gain_power",
        "deal_damage",
        "gain_mastery",
        "gain_health",
        "draw_card",
        "copy_effect",
        "offer_banish",
        "gain_power_per_discard_faction",
        "recruit_free_card",
        "win",
        "inspiration",
        "lose_mastery",
        "destroy_champion",
        "destroy_all_champions",
        "recover_champion",
        "recover_mercenary",
    }
)

SUPPORTED_CHAMPION_ABILITY_KINDS = frozenset(
    {
        "gain_power",
        "gain_power_per_played_faction",
        "gain_mastery_then_draw",
        "draw_if_domination",
        "gain_mastery_if_domination",
        "draw_if_champion_faction_count",
        "gain_gem_and_arm_recruitment",
        "gain_health_per_champion",
        "gain_power_per_champion",
        "gain_power_threshold",
        "gain_power_then_recover_faction",
        "gain_gems_then_copy_faction",
    }
)


@dataclass(frozen=True, slots=True)
class OperationRepresentation:
    kind: str
    amount: int
    target: str
    mastery_at_least: int | None
    health_at_least: int | None
    faction: str | None
    requires_union: bool
    requires_echo: bool
    requires_domination: bool
    requires_inspiration: bool
    recruit_to_hand_at_mastery: int | None


@dataclass(frozen=True, slots=True)
class EffectStepRepresentation:
    mastery_at_least: int | None
    operations: tuple[OperationRepresentation, ...]


@dataclass(frozen=True, slots=True)
class EffectRepresentation:
    flat_gems: int
    flat_power: int
    steps: tuple[EffectStepRepresentation, ...]


@dataclass(frozen=True, slots=True)
class ChampionAbilityRepresentation:
    kind: str
    amount: int
    threshold: int | None
    faction: str | None
    secondary_amount: int
    draw_amount: int
    requires_domination: bool


@dataclass(frozen=True, slots=True)
class CardSemanticRepresentation:
    card_definition_id: str
    cost: int
    faction: str | None
    shield: int
    is_champion: bool
    champion_health: int | None
    is_mercenary: bool
    effect: EffectRepresentation
    on_play_effect: EffectRepresentation | None
    champion_ability: ChampionAbilityRepresentation | None
    passive_kind: str | None
    schema_version: int = CARD_REPRESENTATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic-schema-compatible representation for serialization."""
        return asdict(self)


def representation_for_definition(
    definition: CardDefinition,
) -> CardSemanticRepresentation:
    """Convert one immutable card definition into its semantic representation."""
    return _representation_for_definition(definition)


def clear_representation_cache() -> None:
    """Clear cached representations, primarily for catalog reloads and tests."""
    _representation_for_definition.cache_clear()


@lru_cache(maxsize=None)
def _representation_for_definition(
    definition: CardDefinition,
) -> CardSemanticRepresentation:
    return CardSemanticRepresentation(
        card_definition_id=definition.card_id,
        cost=definition.cost,
        faction=definition.faction.value if definition.faction is not None else None,
        shield=definition.shield,
        is_champion=definition.is_champion,
        champion_health=definition.champion_health,
        is_mercenary=definition.is_mercenary,
        effect=_effect_representation(definition.effect),
        on_play_effect=(
            _effect_representation(definition.on_play_effect)
            if definition.on_play_effect is not None
            else None
        ),
        champion_ability=(
            _champion_ability_representation(definition.champion_ability)
            if definition.champion_ability is not None
            else None
        ),
        passive_kind=definition.passive_kind,
    )


def _effect_representation(effect: Effect) -> EffectRepresentation:
    return EffectRepresentation(
        flat_gems=effect.gems,
        flat_power=effect.power,
        steps=tuple(_effect_step_representation(step) for step in effect.steps),
    )


def _effect_step_representation(step: EffectStep) -> EffectStepRepresentation:
    return EffectStepRepresentation(
        mastery_at_least=step.mastery_at_least,
        operations=tuple(_operation_representation(operation) for operation in step.operations),
    )


def _operation_representation(operation: Operation) -> OperationRepresentation:
    if operation.kind not in SUPPORTED_OPERATION_KINDS:
        raise ValueError(f"Unsupported operation kind for card representation: {operation.kind!r}")
    return OperationRepresentation(
        kind=operation.kind,
        amount=operation.amount,
        target=operation.target,
        mastery_at_least=operation.mastery_at_least,
        health_at_least=operation.health_at_least,
        faction=operation.faction.value if operation.faction is not None else None,
        requires_union=operation.requires_union,
        requires_echo=operation.requires_echo,
        requires_domination=operation.requires_domination,
        requires_inspiration=operation.requires_inspiration,
        recruit_to_hand_at_mastery=operation.recruit_to_hand_at_mastery,
    )


def _champion_ability_representation(
    ability: ChampionAbility,
) -> ChampionAbilityRepresentation:
    if ability.kind not in SUPPORTED_CHAMPION_ABILITY_KINDS:
        raise ValueError(
            "Unsupported champion ability kind for card representation: "
            f"{ability.kind!r}"
        )
    return ChampionAbilityRepresentation(
        kind=ability.kind,
        amount=ability.amount,
        threshold=ability.threshold,
        faction=ability.faction.value if ability.faction is not None else None,
        secondary_amount=ability.secondary_amount,
        draw_amount=ability.draw_amount,
        requires_domination=ability.requires_domination,
    )
