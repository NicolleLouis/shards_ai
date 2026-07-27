from __future__ import annotations

import json

import pytest

from shards_ai.ai import (
    ACTION_REPRESENTATION_SCHEMA_VERSION,
    ActionRepresentation,
    representation_for_action,
    representation_for_neural_action,
)
from shards_ai.game import (
    ActivateChampion,
    AssignPower,
    BanishCard,
    BuyCard,
    CardInstance,
    ChoosePendingDecision,
    Game,
    GainMastery,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
    SkipBanish,
    StopBuying,
)
from shards_ai.game.cards import CARD_CATALOG
from shards_ai.game.state import PendingDecision


def test_actionless_decisions_have_stable_types_and_no_card() -> None:
    game = Game.new(seed=401)

    representations = [
        representation_for_action(PassPlayPhase(), game.state),
        representation_for_action(GainMastery(), game.state),
        representation_for_action(StopBuying(), game.state),
        representation_for_action(SkipBanish(), game.state),
    ]

    assert [representation.action_type for representation in representations] == [
        "pass_play_phase",
        "gain_mastery",
        "stop_buying",
        "skip_banish",
    ]
    assert all(representation.card_definition_id is None for representation in representations)
    assert all(representation.schema_version == ACTION_REPRESENTATION_SCHEMA_VERSION for representation in representations)


def test_card_actions_resolve_definition_and_instance_identity() -> None:
    game = Game.new(seed=402)
    card = game.active.hand[0]

    representation = representation_for_action(PlayCard(card.instance_id), game.state)

    assert isinstance(representation, ActionRepresentation)
    assert representation.action_type == "play_card"
    assert representation.card_definition_id == card.definition.card_id
    assert representation.card_instance_id == card.instance_id
    assert representation.phase == game.state.phase.value


def test_banish_resolves_a_card_from_the_active_discard() -> None:
    game = Game.new(seed=403)
    card = game.active.hand.pop()
    game.active.discard_pile.append(card)

    representation = representation_for_action(BanishCard(card.instance_id), game.state)

    assert representation.action_type == "banish_card"
    assert representation.card_definition_id == card.definition.card_id


def test_river_actions_keep_slot_and_resolve_card_definition() -> None:
    game = Game.new(seed=404)
    card = game.state.river[0]
    assert card is not None

    actions = [
        BuyCard(0, card.instance_id),
        RecruitMercenary(0, card.instance_id),
        RecruitFreeCard(0, card.instance_id),
    ]
    representations = [representation_for_action(action, game.state) for action in actions]

    assert all(representation.river_slot == 0 for representation in representations)
    assert all(representation.card_instance_id == card.instance_id for representation in representations)
    assert all(representation.card_definition_id == card.definition.card_id for representation in representations)


def test_assign_power_resolves_public_champion_target_but_not_player_target() -> None:
    game = Game.new(seed=405)
    champion = CardInstance("public-champion", CARD_CATALOG["additri_gaia_mancienne"])
    game.opponent.champions = [champion]

    player_target = representation_for_action(AssignPower(0), game.state)
    champion_target = representation_for_action(
        AssignPower(0, target=champion.instance_id),
        game.state,
    )

    assert player_target.target == "opponent"
    assert player_target.card_definition_id is None
    assert champion_target.target == champion.instance_id
    assert champion_target.card_definition_id == champion.definition.card_id
    assert champion_target.card_instance_id == champion.instance_id
    assert champion_target.amount == 0


def test_choose_pending_decision_keeps_choice_and_resolves_public_card_when_available() -> None:
    game = Game.new(seed=406)
    card = game.active.hand[0]
    game.active.pending_decision = PendingDecision("select_effect_copy", (card.instance_id,))

    representation = representation_for_action(
        ChoosePendingDecision(card.instance_id),
        game.state,
    )

    assert representation.action_type == "choose_pending_decision"
    assert representation.choice_id == card.instance_id
    assert representation.card_instance_id == card.instance_id
    assert representation.card_definition_id == card.definition.card_id


def test_pending_choice_that_is_not_a_card_does_not_invent_a_definition() -> None:
    game = Game.new(seed=407)
    game.active.pending_decision = PendingDecision("generic", ("choice-a",))

    representation = representation_for_action(
        ChoosePendingDecision("choice-a"),
        game.state,
    )

    assert representation.choice_id == "choice-a"
    assert representation.card_definition_id is None
    assert representation.card_instance_id is None


def test_hidden_opponent_card_cannot_be_resolved_as_a_banish_target() -> None:
    game = Game.new(seed=408)
    hidden_card = game.opponent.hand[0]

    with pytest.raises(ValueError, match="Cannot resolve public card"):
        representation_for_action(BanishCard(hidden_card.instance_id), game.state)


def test_legal_actions_and_representations_keep_positionally_aligned() -> None:
    game = Game.new(seed=409)
    actions = game.legal_actions()
    representations = [representation_for_action(action, game.state) for action in actions]

    assert len(representations) == len(actions)
    assert [representation.card_instance_id for representation in representations[: len(game.active.hand)]] == [
        card.instance_id for card in game.active.hand
    ]


def test_unknown_action_fails_explicitly() -> None:
    class UnknownAction:
        pass

    with pytest.raises(ValueError, match="Unsupported action"):
        representation_for_action(UnknownAction(), Game.new(seed=410).state)  # type: ignore[arg-type]


def test_action_representation_is_json_serializable() -> None:
    game = Game.new(seed=411)
    representation = representation_for_action(PlayCard(game.active.hand[0].instance_id), game.state)

    serialized = json.dumps(representation.to_dict(), sort_keys=True)

    assert "play_card" in serialized
    assert "schema_version" in serialized


def test_neural_action_representation_resolves_public_active_discard() -> None:
    game = Game.new(seed=412)
    card = game.active.hand.pop()
    game.active.discard_pile.append(card)
    observation = game.neural_observation_for(game.active_player)

    representation = representation_for_neural_action(BanishCard(card.instance_id), observation)

    assert representation.card_definition_id == card.definition.card_id
    assert representation.card_instance_id == card.instance_id


def test_neural_action_representation_does_not_resolve_hidden_opponent_card() -> None:
    game = Game.new(seed=413)
    hidden_card = game.opponent.hand[0]
    observation = game.neural_observation_for(game.active_player)

    with pytest.raises(ValueError, match="Cannot resolve public card"):
        representation_for_neural_action(BanishCard(hidden_card.instance_id), observation)
