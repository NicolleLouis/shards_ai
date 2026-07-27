import pytest

from shards_ai.game import Faction, Game, PlayCard
from shards_ai.game.cards import CardInstance
from shards_ai.game.cards.definitions import (
    DRONE_KILN,
    DRONE_REACTEUR,
    DRONES_MINIERS,
    LEGIONNAIRE_KORVUS,
    VALKYRIE_DES_LANDES,
)


@pytest.mark.parametrize(
    ("definition", "cost", "shield", "operations"),
    [
        (DRONE_KILN, 1, 0, (("gain_gems", 2),)),
        (DRONES_MINIERS, 2, 0, (("gain_gems", 1), ("draw_card", 0))),
        (LEGIONNAIRE_KORVUS, 3, 2, (("gain_power", 2), ("recover_champion", 0))),
        (DRONE_REACTEUR, 3, 0, (("gain_gems", 3),)),
        (VALKYRIE_DES_LANDES, 4, 0, (("gain_power", 4), ("lose_mastery", 2))),
    ],
)
def test_homodeus_definitions_match_the_card_list(
    definition, cost, shield, operations
) -> None:
    assert definition.faction is Faction.HOMODEUS
    assert definition.cost == cost
    assert definition.shield == shield
    assert tuple(
        (operation.kind, operation.amount)
        for operation in definition.effect.operations_for_mastery(0)
    ) == operations


def put_in_hand(game: Game, *definitions) -> None:
    game.active.hand = [
        CardInstance(f"homodeus-test-{index}", definition)
        for index, definition in enumerate(definitions)
    ]


def test_homodeus_cards_have_expected_faction_and_copies() -> None:
    game = Game.new(seed=401)
    cards = game.state.central_deck + [card for card in game.state.river if card]
    homodeus = [card for card in cards if card.definition.faction is Faction.HOMODEUS]

    assert len(homodeus) == 22
    assert {
        card.definition.card_id: sum(
            candidate.definition.card_id == card.definition.card_id
            for candidate in homodeus
        )
        for card in homodeus
    } == {
        "drone_kiln": 3,
        "drones_miniers": 3,
        "legionnaire_korvus": 3,
        "drone_reacteur": 3,
        "valkyrie_des_landes": 1,
        "primus_pilus": 1,
        "drones_numeri": 2,
        "evokatus": 2,
        "broyeu_optio": 2,
        "drakonarius": 1,
        "general_decurion": 1,
    }


def test_homodeus_resource_and_attack_cards_resolve() -> None:
    game = Game.new(seed=402)
    put_in_hand(game, DRONE_KILN)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.gems == 2

    game = Game.new(seed=403)
    put_in_hand(game, DRONE_REACTEUR)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.gems == 3

    game = Game.new(seed=404)
    put_in_hand(game, VALKYRIE_DES_LANDES)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 4

    game = Game.new(seed=406)
    put_in_hand(game, LEGIONNAIRE_KORVUS)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 2


def test_miners_drone_draws_and_korvus_has_a_shield() -> None:
    game = Game.new(seed=405)
    initial_hand_size = 1
    put_in_hand(game, DRONES_MINIERS)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert len(game.active.hand) == initial_hand_size
    assert game.active.gems == 1

    assert LEGIONNAIRE_KORVUS.shield == 2
