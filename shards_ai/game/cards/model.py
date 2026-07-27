from dataclasses import dataclass, field
from typing import Literal

from ..enums import Faction


OperationKind = Literal[
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
]


ChampionAbilityKind = Literal[
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
]


@dataclass(frozen=True, slots=True)
class ChampionAbility:
    """Declarative active ability for a champion."""

    kind: ChampionAbilityKind
    amount: int = 0
    threshold: int | None = None
    faction: Faction | None = None
    secondary_amount: int = 0
    draw_amount: int = 0
    requires_domination: bool = False

    def __post_init__(self) -> None:
        if self.amount < 0 or self.secondary_amount < 0 or self.draw_amount < 0:
            raise ValueError("Champion ability amounts cannot be negative")
        if self.threshold is not None and self.threshold < 0:
            raise ValueError("Champion ability thresholds cannot be negative")


@dataclass(frozen=True, slots=True)
class Operation:
    kind: OperationKind
    amount: int = 0
    target: str = "opponent"
    mastery_at_least: int | None = None
    requires_union: bool = False
    health_at_least: int | None = None
    requires_echo: bool = False
    faction: Faction | None = None
    requires_domination: bool = False
    requires_inspiration: bool = False
    recruit_to_hand_at_mastery: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "win":
            if self.amount != 0:
                raise ValueError("Win operation cannot have an amount")
        elif self.amount < 0:
            raise ValueError("Operation amounts cannot be negative")
        if self.mastery_at_least is not None and self.mastery_at_least < 0:
            raise ValueError("Mastery thresholds cannot be negative")
        if self.health_at_least is not None and self.health_at_least < 0:
            raise ValueError("Health thresholds cannot be negative")
        if self.recruit_to_hand_at_mastery is not None and self.recruit_to_hand_at_mastery < 0:
            raise ValueError("Recruitment mastery thresholds cannot be negative")


@dataclass(frozen=True, slots=True)
class EffectStep:
    """One mutually exclusive, mastery-conditional branch of a card effect."""

    operations: tuple[Operation, ...]
    mastery_at_least: int | None = None

    def __post_init__(self) -> None:
        if self.mastery_at_least is not None and self.mastery_at_least < 0:
            raise ValueError("Mastery thresholds cannot be negative")


@dataclass(frozen=True, slots=True, init=False)
class Effect:
    """Structured immediate effect produced when a card is played.

    ``damage`` is accepted as a compatibility constructor argument and exposed as
    an alias for ``power``. New definitions should use ``power`` or ``steps``.
    """

    gems: int
    power: int
    steps: tuple[EffectStep, ...]
    _flat_operations: tuple[Operation, ...] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        gems: int = 0,
        power: int = 0,
        *,
        damage: int | None = None,
        steps: tuple[EffectStep, ...] = (),
    ) -> None:
        if damage is not None:
            if power and power != damage:
                raise ValueError("Effect power and compatibility damage disagree")
            power = damage
        if gems < 0 or power < 0:
            raise ValueError("Card effect values cannot be negative")
        if steps and (gems or power):
            raise ValueError("Structured effects cannot also define flat values")
        object.__setattr__(self, "gems", gems)
        object.__setattr__(self, "power", power)
        normalized_steps = tuple(
            sorted(
                steps,
                key=lambda candidate: (
                    -1
                    if candidate.mastery_at_least is None
                    else candidate.mastery_at_least
                ),
                reverse=True,
            )
        )
        object.__setattr__(self, "steps", normalized_steps)
        flat_operations: list[Operation] = []
        if gems:
            flat_operations.append(Operation("gain_gems", gems))
        if power:
            flat_operations.append(Operation("gain_power", power))
        object.__setattr__(self, "_flat_operations", tuple(flat_operations))

    @property
    def damage(self) -> int:
        """Deprecated alias for the Power produced by a simple effect."""
        return self.power

    def operations_for_mastery(self, mastery: int) -> tuple[Operation, ...]:
        if self.steps:
            if len(self.steps) == 1 and self.steps[0].mastery_at_least is None:
                return self.steps[0].operations
            for step in self.steps:
                if (
                    step.mastery_at_least is None
                    or mastery >= step.mastery_at_least
                ):
                    return step.operations
            return ()

        return self._flat_operations


@dataclass(frozen=True, slots=True)
class CardDefinition:
    """Reusable, immutable description of a card type."""

    card_id: str
    name: str
    cost: int
    effect: Effect
    faction: Faction | None = None
    shield: int = 0
    is_champion: bool = False
    champion_health: int | None = None
    on_play_effect: Effect | None = None
    champion_ability: ChampionAbility | None = None
    passive_kind: str | None = None
    is_mercenary: bool = False

    def __post_init__(self) -> None:
        if not self.card_id:
            raise ValueError("Card ID cannot be empty")
        if not self.name:
            raise ValueError("Card name cannot be empty")
        if self.cost < 0:
            raise ValueError("Card cost cannot be negative")
        if self.shield < 0:
            raise ValueError("Card shield cannot be negative")
        if self.is_champion:
            if self.champion_health is None or self.champion_health <= 0:
                raise ValueError("Champions must have positive health")
        elif self.champion_health is not None:
            raise ValueError("Non-champions cannot have champion health")
        if self.is_mercenary and self.is_champion:
            raise ValueError("Champions cannot be mercenaries")

    @property
    def gems(self) -> int:
        """Compatibility/readability shortcut for the card's Gems effect."""
        return self.effect.gems

    @property
    def damage(self) -> int:
        """Deprecated compatibility shortcut for the card's Power effect."""
        return self.effect.damage

    @property
    def power(self) -> int:
        return self.effect.power


@dataclass(slots=True)
class CardInstance:
    """A concrete card that can move between player or central-deck zones."""

    instance_id: str
    definition: CardDefinition
