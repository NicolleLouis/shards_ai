"""Deterministic card-visibility estimates for horizon-training datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from shards_ai.game.actions import BuyCard
from shards_ai.game.cards import CardInstance, Effect, Operation
from shards_ai.game.state import GameState, PlayerState


VisibilityClass = Literal["zero", "once", "multiple"]
HORIZON_TAIL_CLASS = "T6+"
CARDS_PER_TURN = 5


@dataclass(frozen=True, slots=True)
class EffectiveDeckSnapshot:
    """Effective deck sizes after a candidate BuyCard, without mutating the game."""

    current_draw_pile_size: int
    remaining_draw_effective_size: int
    remaining_draw_cycle_turns: int
    future_deck_size: int
    total_draw_amount: int
    effective_deck_size: int
    full_deck_cycle_turns: int


@dataclass(frozen=True, slots=True)
class CardVisibilityForecast:
    selected_horizon_class: str
    visibility_class: VisibilityClass
    snapshot: EffectiveDeckSnapshot


def _future_deck_cards(player: PlayerState, candidate: CardInstance) -> list[CardInstance]:
    """Return cards that will participate in future deck cycles after this turn.

    Champions stay in play and are intentionally excluded. Cards in hand/play_zone are included
    because they return to the discard during cleanup; the candidate is added to the discard now.
    """

    return [*player.hand, *player.draw_pile, *player.discard_pile, *player.play_zone, candidate]


def _operation_is_unconditional(operation: Operation) -> bool:
    return not any(
        (
            operation.mastery_at_least is not None,
            operation.requires_union,
            operation.health_at_least is not None,
            operation.requires_echo,
            operation.requires_domination,
            operation.requires_inspiration,
            operation.recruit_to_hand_at_mastery is not None,
        )
    )


def _draw_amount_from_effect(effect: Effect | None, mastery: int) -> int:
    if effect is None:
        return 0
    amount = 0
    for operation in effect.operations_for_mastery(mastery):
        if operation.kind == "draw_card" and _operation_is_unconditional(operation):
            amount += max(1, operation.amount)
    return amount


def unconditional_draw_amount(card: CardInstance, mastery: int) -> int:
    """Return known unconditional draws produced when the card is played."""

    definition = card.definition
    return _draw_amount_from_effect(definition.effect, mastery) + _draw_amount_from_effect(
        definition.on_play_effect, mastery
    )


def effective_deck_snapshot(state: GameState, action: BuyCard) -> EffectiveDeckSnapshot:
    """Calculate effective current/full deck cycles for one BuyCard candidate."""

    if not 0 <= action.river_slot < len(state.river):
        raise ValueError(f"Invalid river slot: {action.river_slot}")
    candidate = state.river[action.river_slot]
    if candidate is None or candidate.instance_id != action.card_instance_id:
        raise ValueError("BuyCard candidate does not match the river")

    player = state.players[state.active_player]
    future_cards = _future_deck_cards(player, candidate)
    total_draw_amount = sum(unconditional_draw_amount(card, player.mastery) for card in future_cards)
    current_draw_amount = sum(
        unconditional_draw_amount(card, player.mastery) for card in player.draw_pile
    )
    remaining_draw_effective_size = max(0, len(player.draw_pile) - current_draw_amount)
    effective_deck_size = max(1, len(future_cards) - total_draw_amount)
    return EffectiveDeckSnapshot(
        current_draw_pile_size=len(player.draw_pile),
        remaining_draw_effective_size=remaining_draw_effective_size,
        remaining_draw_cycle_turns=math.ceil(remaining_draw_effective_size / CARDS_PER_TURN),
        future_deck_size=len(future_cards),
        total_draw_amount=total_draw_amount,
        effective_deck_size=effective_deck_size,
        full_deck_cycle_turns=math.ceil(effective_deck_size / CARDS_PER_TURN),
    )


def visibility_class_for_horizon(
    selected_horizon_class: str,
    snapshot: EffectiveDeckSnapshot,
) -> VisibilityClass:
    """Apply the deterministic 0/1/multiple cycle rule."""

    if selected_horizon_class == HORIZON_TAIL_CLASS:
        return "multiple"
    if selected_horizon_class not in {f"T{value}" for value in range(6)}:
        raise ValueError(f"Unsupported horizon class: {selected_horizon_class!r}")
    remaining_turns = int(selected_horizon_class[1:])
    if remaining_turns <= snapshot.remaining_draw_cycle_turns:
        return "zero"
    if remaining_turns > snapshot.remaining_draw_cycle_turns + 2 * snapshot.full_deck_cycle_turns:
        return "multiple"
    return "once"


def forecast_card_visibility(
    state: GameState,
    action: BuyCard,
    selected_horizon_class: str,
) -> CardVisibilityForecast:
    snapshot = effective_deck_snapshot(state, action)
    return CardVisibilityForecast(
        selected_horizon_class=selected_horizon_class,
        visibility_class=visibility_class_for_horizon(selected_horizon_class, snapshot),
        snapshot=snapshot,
    )
