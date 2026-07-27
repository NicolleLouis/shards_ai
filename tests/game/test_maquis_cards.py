from shards_ai.game import Faction, Game, PlayCard, PlayerId
from shards_ai.game.cards import CardInstance
from shards_ai.game.cards.definitions import (
    ASPIRANT_MAQUIS,
    CLERC_AUX_SPORES,
    CHEVALIER_LE_SHAI,
    GARDIEN_DE_LA_FORET,
    ELEMENTAL_DU_SILLON,
    ERMITE_FONGIQUE,
    OJAS,
    RACINE_DE_LA_FORET,
    SAULE_VENGEUR,
    ZELOTE_DES_EPINES,
)
from shards_ai.game.enums import Phase
from shards_ai.game.actions import AssignPower


def put_in_hand(game: Game, *definitions) -> None:
    game.active.hand = [CardInstance(f"test-{index}", definition) for index, definition in enumerate(definitions)]


def test_maquis_definitions_have_faction_and_expected_copies() -> None:
    game = Game.new(seed=101)
    cards = game.state.central_deck + [card for card in game.state.river if card]
    maquis = [card for card in cards if card.definition.faction is Faction.MAQUIS]

    assert len(maquis) == 22
    assert sum(card.definition is ASPIRANT_MAQUIS for card in maquis) == 3
    assert all(card.definition.faction is Faction.MAQUIS for card in maquis)


def test_union_requires_another_maquis_in_hand_or_play_zone() -> None:
    game = Game.new(seed=102)
    game.active.health = 40
    put_in_hand(game, ASPIRANT_MAQUIS)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.health == 43
    assert game.active.power == 0

    put_in_hand(game, ASPIRANT_MAQUIS, CLERC_AUX_SPORES)
    game.active.health = 40
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.health == 43
    assert game.active.power == 5


def test_fungal_hermit_checks_mastery_after_gaining_it() -> None:
    game = Game.new(seed=103)
    game.active.mastery = 9
    game.active.health = 40
    put_in_hand(game, ERMITE_FONGIQUE)
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.mastery == 10
    assert game.active.health == 45


def test_shields_reduce_attack_damage_without_leaving_the_hand() -> None:
    game = Game.new(seed=104)
    defender = game.opponent
    defender.hand = [CardInstance("shield", ZELOTE_DES_EPINES)]
    defender.health = 50
    game.active.power = 7
    game.state.phase = Phase.ATTACK

    game.apply(AssignPower(7))

    assert defender.health == 46
    assert defender.hand[0].definition is ZELOTE_DES_EPINES


def test_ojas_copies_the_last_non_champion_card() -> None:
    game = Game.new(seed=105)
    game.active.health = 40
    put_in_hand(game, CLERC_AUX_SPORES, OJAS)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.health == 48


def test_furrowing_elemental_gains_power_at_50_health() -> None:
    game = Game.new(seed=106)
    game.active.health = 50
    put_in_hand(game, ELEMENTAL_DU_SILLON)
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.health == 50
    assert game.active.power == 6


def test_remaining_maquis_card_effects_are_resolved() -> None:
    game = Game.new(seed=107)
    game.active.health = 40
    put_in_hand(game, CLERC_AUX_SPORES)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.health == 44

    game = Game.new(seed=108)
    put_in_hand(game, ZELOTE_DES_EPINES)
    initial_hand_size = len(game.active.hand)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert len(game.active.hand) == initial_hand_size

    game = Game.new(seed=109)
    put_in_hand(game, CHEVALIER_LE_SHAI, ASPIRANT_MAQUIS)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 6

    game = Game.new(seed=110)
    game.active.health = 40
    put_in_hand(game, GARDIEN_DE_LA_FORET, CLERC_AUX_SPORES)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 2
    assert game.active.health == 46

    game = Game.new(seed=111)
    put_in_hand(game, SAULE_VENGEUR)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 4

    game = Game.new(seed=112)
    game.active.health = 40
    put_in_hand(game, RACINE_DE_LA_FORET, ASPIRANT_MAQUIS)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.health == 50
    assert game.active.power == 10
