from __future__ import annotations

from dataclasses import replace

import pytest

from shards_ai.ai.card_visibility import (
    effective_deck_snapshot,
    forecast_card_visibility,
    unconditional_draw_amount,
    visibility_class_for_horizon,
)
from shards_ai.game import BuyCard, CardDefinition, CardInstance, Effect, EffectStep, Game, Operation


def card(card_id: str, *, draw: int = 0) -> CardInstance:
    operations = (Operation("draw_card", draw),) if draw else ()
    definition = CardDefinition(
        card_id=card_id,
        name=card_id,
        cost=0,
        effect=Effect(steps=(EffectStep(operations=operations),)) if operations else Effect(),
    )
    return CardInstance(card_id, definition)


def visibility_game() -> tuple[Game, BuyCard]:
    game = Game.new(seed=91)
    active = game.active
    active.hand.clear()
    active.discard_pile.clear()
    active.play_zone.clear()
    active.champions.clear()
    active.draw_pile = [card(f"normal-{index}") for index in range(16)]
    game.state.river[0] = card("candidate")
    action = BuyCard(0, "candidate")
    return game, action


def test_draw_amount_uses_only_unconditional_draws() -> None:
    assert unconditional_draw_amount(card("draw-one", draw=1), 0) == 1
    assert unconditional_draw_amount(card("draw-two", draw=2), 0) == 2

    conditional = CardDefinition(
        card_id="conditional",
        name="conditional",
        cost=0,
        effect=Effect(
            steps=(EffectStep((Operation("draw_card", 2, mastery_at_least=10),)),)
        ),
    )
    assert unconditional_draw_amount(CardInstance("conditional", conditional), 0) == 0


def test_candidate_draw_affects_full_cycle_but_not_current_draw_cycle() -> None:
    game, action = visibility_game()
    game.state.river[0] = card("draw-two-candidate", draw=2)
    action = BuyCard(0, "draw-two-candidate")

    snapshot = effective_deck_snapshot(game.state, action)

    assert snapshot.future_deck_size == 17
    assert snapshot.current_draw_pile_size == 16
    assert snapshot.total_draw_amount == 2
    assert snapshot.effective_deck_size == 15
    assert snapshot.full_deck_cycle_turns == 3
    assert snapshot.remaining_draw_cycle_turns == 4


def test_two_draw_one_cards_make_seventeen_cards_effectively_fifteen() -> None:
    game, action = visibility_game()
    game.active.draw_pile[0] = card("draw-one-a", draw=1)
    game.active.draw_pile[1] = card("draw-one-b", draw=1)

    snapshot = effective_deck_snapshot(game.state, action)

    assert snapshot.future_deck_size == 17
    assert snapshot.total_draw_amount == 2
    assert snapshot.effective_deck_size == 15
    assert snapshot.full_deck_cycle_turns == 3


def test_visibility_boundaries_and_t6_plus() -> None:
    game, action = visibility_game()
    snapshot = replace(
        effective_deck_snapshot(game.state, action),
        remaining_draw_cycle_turns=1,
        full_deck_cycle_turns=3,
    )

    assert visibility_class_for_horizon("T1", snapshot) == "zero"
    assert visibility_class_for_horizon("T2", snapshot) == "once"
    assert visibility_class_for_horizon("T5", snapshot) == "once"
    assert visibility_class_for_horizon("T6+", snapshot) == "multiple"


def test_forecast_is_read_only_and_returns_snapshot() -> None:
    game, action = visibility_game()
    before = list(game.active.draw_pile)

    result = forecast_card_visibility(game.state, action, "T5")

    assert result.visibility_class == "once"
    assert game.active.draw_pile == before
    assert game.state.river[0].instance_id == "candidate"


def test_invalid_candidate_is_rejected() -> None:
    game, _action = visibility_game()
    with pytest.raises(ValueError):
        effective_deck_snapshot(game.state, BuyCard(0, "wrong"))
