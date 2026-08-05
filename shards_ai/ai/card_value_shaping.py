"""Fixed v008-derived card values used for PPO deckbuilding shaping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from shards_ai.game.actions import (
    Action,
    BanishCard,
    BuyCard,
    GainMastery,
    RecruitFreeCard,
    RecruitMercenary,
    SkipBanish,
    StopBuying,
)
from shards_ai.game.enums import PlayerId
from shards_ai.game.state import GameState


DEFAULT_CARD_VALUES_PATH = Path("configs/neural_training_profiles/card_values_v008.yaml")
DECKBUILDING_ACTIONS = (
    BanishCard,
    BuyCard,
    GainMastery,
    RecruitFreeCard,
    RecruitMercenary,
    SkipBanish,
    StopBuying,
)


def load_card_values(path: str | Path = DEFAULT_CARD_VALUES_PATH) -> dict[str, float]:
    """Load and validate one fixed card-value table."""
    with Path(path).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    values = document.get("values") if isinstance(document, Mapping) else None
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Card value table must contain a non-empty values mapping")
    result = {str(card_id): float(value) for card_id, value in values.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("Card values cannot be negative")
    return result


def is_deckbuilding_action(action: Action) -> bool:
    return isinstance(action, DECKBUILDING_ACTIONS)


def deck_card_ids(state: GameState, player_id: PlayerId) -> tuple[str, ...]:
    player = state.players[player_id]
    cards = (*player.draw_pile, *player.hand, *player.discard_pile, *player.play_zone, *player.champions)
    return tuple(card.definition.card_id for card in cards)


def deck_value_potential(
    state: GameState,
    player_id: PlayerId,
    card_values: Mapping[str, float],
) -> float:
    """Return the mean fixed value of the player's owned cards."""
    card_ids = deck_card_ids(state, player_id)
    if not card_ids:
        return 0.0
    unknown = sorted(set(card_ids) - set(card_values))
    if unknown:
        raise ValueError(f"Card value table is missing cards: {unknown}")
    return sum(card_values[card_id] for card_id in card_ids) / len(card_ids)


def deckbuilding_shaping_delta(
    before: GameState,
    after: GameState,
    action: Action,
    player_id: PlayerId,
    card_values: Mapping[str, float],
    *,
    clip: float = 1.0,
) -> float:
    """Return a bounded potential delta for one eligible deckbuilding action."""
    if not is_deckbuilding_action(action):
        return 0.0
    delta = deck_value_potential(after, player_id, card_values) - deck_value_potential(
        before, player_id, card_values
    )
    return max(-clip, min(clip, delta))


__all__ = [
    "DEFAULT_CARD_VALUES_PATH",
    "DECKBUILDING_ACTIONS",
    "deck_card_ids",
    "deck_value_potential",
    "deckbuilding_shaping_delta",
    "is_deckbuilding_action",
    "load_card_values",
]
