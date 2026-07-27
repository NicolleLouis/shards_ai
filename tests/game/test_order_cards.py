from shards_ai.game import Game, PlayCard, RecruitFreeCard
from shards_ai.game.cards import CardDefinition, CardInstance, Effect
from shards_ai.game.cards.definitions import (
    GARDE_MEMOIRE,
    INITIE_DE_L_ORDRE,
    LE_GRAND_ARCHITECTE,
    MOINE_CRYPTOPOING,
    MOINE_DU_PORTAIL,
    OMNIUS_L_ERUDIT,
    PIRATE_HERETIQUE,
    PROPHETE_DE_LECLAT,
    VOYANTE_DE_VOLONTE,
)
from shards_ai.game.cards.definitions import ASPIRANT_MAQUIS, ECLAIREUR_SPECTRAL
from shards_ai.game.enums import Faction


def put_in_hand(game: Game, *definitions) -> None:
    game.active.hand = [
        CardInstance(f"order-test-{index}", definition)
        for index, definition in enumerate(definitions)
    ]


def test_order_cards_have_expected_faction_and_copies() -> None:
    game = Game.new(seed=301)
    cards = game.state.central_deck + [card for card in game.state.river if card]
    order = [card for card in cards if card.definition.faction is Faction.ORDER]

    assert len(order) == 22
    assert sum(card.definition.card_id == "initie_de_l_ordre" for card in order) == 3


def test_domination_requires_homodeus_maquis_and_spectra() -> None:
    homodeus = CardDefinition(
        "test_homodeus", "Test Homodeus", 0, Effect(), faction=Faction.HOMODEUS
    )
    game = Game.new(seed=302)
    put_in_hand(game, INITIE_DE_L_ORDRE, homodeus)
    game.active.hand.append(CardInstance("maquis", ASPIRANT_MAQUIS))
    game.active.hand.append(CardInstance("spectra", ECLAIREUR_SPECTRAL))
    initial_mastery = game.active.mastery
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.gems == 2
    assert game.active.mastery == initial_mastery + 2


def test_domination_does_not_trigger_without_all_three_factions() -> None:
    game = Game.new(seed=303)
    put_in_hand(game, INITIE_DE_L_ORDRE)
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.gems == 2
    assert game.active.mastery == 0


def test_memory_guardian_checks_mastery_after_gaining_it() -> None:
    game = Game.new(seed=304)
    game.active.mastery = 9
    put_in_hand(game, GARDE_MEMOIRE)
    initial_draw_size = len(game.active.draw_pile)
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.mastery == 10
    assert len(game.active.draw_pile) == initial_draw_size - 1


def test_order_draw_cards_draw_their_declared_amount() -> None:
    game = Game.new(seed=305)
    put_in_hand(game, PIRATE_HERETIQUE)
    initial_hand_size = len(game.active.hand)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert len(game.active.hand) == initial_hand_size + 1

    game = Game.new(seed=306)
    put_in_hand(game, MOINE_CRYPTOPOING)
    initial_hand_size = len(game.active.hand)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert len(game.active.hand) == initial_hand_size


def test_portal_monk_recruits_to_discard_or_hand_at_mastery_fifteen() -> None:
    game = Game.new(seed=307)
    recruited = CardInstance("free-card", PROPHETE_DE_LECLAT)
    game.state.river = [recruited, None, None, None, None, None]
    put_in_hand(game, MOINE_DU_PORTAIL)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    action = game.legal_actions()[0]
    assert isinstance(action, RecruitFreeCard)
    game.apply(action)
    assert recruited in game.active.discard_pile

    game = Game.new(seed=308)
    recruited = CardInstance("free-card-in-hand", PROPHETE_DE_LECLAT)
    game.state.river = [recruited, None, None, None, None, None]
    game.active.mastery = 15
    put_in_hand(game, MOINE_DU_PORTAIL)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    game.apply(game.legal_actions()[0])
    assert recruited in game.active.hand


def test_order_shields_and_mastery_cards_resolve() -> None:
    assert VOYANTE_DE_VOLONTE.shield == 5
    assert MOINE_CRYPTOPOING.shield == 8

    game = Game.new(seed=309)
    put_in_hand(game, PROPHETE_DE_LECLAT)
    initial_mastery = game.active.mastery
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.mastery == initial_mastery + 2

    game = Game.new(seed=310)
    put_in_hand(game, LE_GRAND_ARCHITECTE)
    initial_mastery = game.active.mastery
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.mastery == initial_mastery + 5

    game = Game.new(seed=311)
    game.active.mastery = 10
    put_in_hand(game, OMNIUS_L_ERUDIT)
    initial_hand_size = len(game.active.hand)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert len(game.active.hand) == initial_hand_size + 1
