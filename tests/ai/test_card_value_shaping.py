from dataclasses import replace

import pytest

from shards_ai.ai.card_value_shaping import (
    deck_value_potential,
    deckbuilding_shaping_delta,
    is_deckbuilding_action,
    load_card_values,
)
from shards_ai.game import CardInstance, Game, PlayerId, card_definition
from shards_ai.game.actions import BanishCard, PlayCard


def test_v008_card_value_table_is_complete_and_deterministic():
    first = load_card_values()
    second = load_card_values()

    assert first == second
    assert len(first) == 48
    assert all(value >= 0.0 for value in first.values())


def test_deck_potential_uses_the_mean_of_owned_cards():
    game = Game.new(seed=58009)
    player_id = PlayerId.PLAYER_1
    player = game.state.players[player_id]
    low = CardInstance("low", card_definition("crystal"))
    high = CardInstance("high", card_definition("blaster"))
    values = {low.definition.card_id: 2.0}
    state = replace(
        game.state,
        players={
            **game.state.players,
            player_id: replace(player, draw_pile=[low], hand=[high]),
        },
    )
    values[high.definition.card_id] = 6.0

    assert deck_value_potential(state, player_id, values) == pytest.approx(4.0)


def test_banish_below_average_produces_a_positive_delta():
    game = Game.new(seed=58010)
    player_id = PlayerId.PLAYER_1
    player = game.state.players[player_id]
    low = CardInstance("low", card_definition("crystal"))
    high = CardInstance("high", card_definition("blaster"))
    before = replace(
        game.state,
        players={player_id: replace(player, hand=[low], draw_pile=[high])},
    )
    after = replace(before, players={player_id: replace(before.players[player_id], hand=[])})
    values = {
        low.definition.card_id: 1.0,
        high.definition.card_id: 5.0,
    }

    assert deckbuilding_shaping_delta(
        before, after, BanishCard(low.instance_id), player_id, values
    ) == pytest.approx(1.0)


def test_playing_a_card_does_not_receive_deckbuilding_shaping():
    game = Game.new(seed=58011)
    player_id = PlayerId.PLAYER_1
    values = {card.definition.card_id: 1.0 for card in game.state.players[player_id].draw_pile}

    assert not is_deckbuilding_action(PlayCard("unknown"))
    assert deckbuilding_shaping_delta(
        game.state, game.state, PlayCard("unknown"), player_id, values
    ) == 0.0
