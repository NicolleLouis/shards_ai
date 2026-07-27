from __future__ import annotations

from shards_ai.game import (
    AssignPower,
    CardInstance,
    Game,
    PassPlayPhase,
    StopBuying,
    card_definition,
)


def test_neural_observation_exposes_active_zones_but_not_opponent_hidden_zones() -> None:
    game = Game.new(seed=301)

    observation = game.neural_observation_for(game.active_player)

    assert len(observation.active_player.hand) == 5
    assert observation.active_player.draw_pile_counts
    assert observation.opponent.owned_card_counts
    assert observation.opponent.discard_counts == ()
    assert not hasattr(observation.opponent, "hand")
    assert not hasattr(observation.opponent, "draw_pile_counts")
    assert all(
        card.instance_id not in repr(observation.opponent)
        for card in game.opponent.hand + game.opponent.draw_pile
    )


def test_neural_observation_aggregates_counts_in_stable_order() -> None:
    game = Game.new(seed=302)
    observation = game.neural_observation_for(game.active_player)

    for counts in (
        observation.active_player.draw_pile_counts,
        observation.active_player.discard_counts,
        observation.active_player.owned_card_counts,
        observation.opponent.owned_card_counts,
        observation.opponent.discard_counts,
        observation.central_deck_counts,
    ):
        assert tuple(card_id for card_id, _quantity in counts) == tuple(
            sorted(card_id for card_id, _quantity in counts)
        )


def test_hidden_zone_order_and_instance_ids_do_not_change_observation() -> None:
    first = Game.new(seed=303)
    second = first.clone()
    opponent = second.opponent

    opponent.hand.reverse()
    opponent.draw_pile.reverse()
    opponent.discard_pile.reverse()
    for card in opponent.hand + opponent.draw_pile + opponent.discard_pile:
        card.instance_id = f"hidden-{card.instance_id}"

    assert first.neural_observation_for(first.active_player) == second.neural_observation_for(
        second.active_player
    )


def test_neural_observation_keeps_visible_cards_and_river_slots() -> None:
    game = Game.new(seed=304)
    observation = game.neural_observation_for(game.active_player)

    assert [card.instance_id for card in observation.active_player.hand] == [
        card.instance_id for card in game.active.hand
    ]
    assert [river_card.slot for river_card in observation.river] == list(range(Game.RIVER_SIZE))
    assert all(river_card.card is not None for river_card in observation.river)
    assert observation.river[0].card.card_definition_id == game.state.river[0].definition.card_id


def test_played_faction_mask_has_four_playable_faction_positions() -> None:
    game = Game.new(seed=305)
    cards = [
        CardInstance("maquis", card_definition("aspirant_maquis")),
        CardInstance("spectra", card_definition("eclaireur_spectral")),
        CardInstance("homodeus", card_definition("drone_kiln")),
        CardInstance("order", card_definition("initie_de_l_ordre")),
    ]
    game.active.play_zone = cards
    game.active.played_card_ids_this_turn = {card.instance_id for card in cards}

    observation = game.neural_observation_for(game.active_player)

    assert observation.active_player.played_faction_mask == (True, True, True, True)
    assert len(observation.active_player.played_faction_mask) == 4


def test_played_faction_mask_is_cleared_at_cleanup() -> None:
    game = Game.new(seed=306)
    game.active.played_card_ids_this_turn = {game.active.hand[0].instance_id}
    game.apply(PassPlayPhase())
    game.apply(StopBuying())
    game.apply(AssignPower(0))

    assert not game.state.players[game.active_player.opponent].played_card_ids_this_turn


def test_neural_observation_exposes_pending_public_choices() -> None:
    game = Game.new(seed=307)
    game.active.pending_banishes = 1
    observation = game.neural_observation_for(game.active_player)

    assert observation.pending_decision is not None
    assert observation.pending_decision.kind == "banish"
    assert set(observation.pending_decision.candidates) == {
        *[card.instance_id for card in game.active.hand],
        *[card.instance_id for card in game.active.discard_pile],
    }


def test_neural_observation_is_read_only_and_does_not_share_game_state() -> None:
    game = Game.new(seed=308)
    observation = game.neural_observation_for(game.active_player)
    original_health = observation.active_player.health

    game.active.health = 1
    game.active.hand.clear()

    assert observation.active_player.health == original_health
    assert len(observation.active_player.hand) == 5
