"""Card models, definitions and deck factories."""

from .catalog import CARD_CATALOG, card_definition
from .central_deck import CENTRAL_DECK_SIZE, VOID_ASSASSIN_COUNT, build_central_deck
from .model import CardDefinition, CardInstance, ChampionAbility, Effect, EffectStep, Operation
from .starter_deck import STARTER_DECK_SIZE, build_starter_deck

V0_CARD_DEFINITION = CardDefinition(
    card_id="v0_damage_card",
    name="V0 Damage Card",
    cost=0,
    effect=Effect(power=1),
)

__all__ = [
    "CARD_CATALOG",
    "CardDefinition",
    "CardInstance",
    "ChampionAbility",
    "CENTRAL_DECK_SIZE",
    "VOID_ASSASSIN_COUNT",
    "Effect",
    "EffectStep",
    "Operation",
    "STARTER_DECK_SIZE",
    "V0_CARD_DEFINITION",
    "build_central_deck",
    "build_starter_deck",
    "card_definition",
]
