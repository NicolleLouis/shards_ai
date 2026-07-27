from shards_ai.ai import RandomPlayer
import pytest

from shards_ai.game import BanishCard, Faction, Game, GameRandom, InvalidActionError, PlayCard, SkipBanish
from shards_ai.game.actions import AssignPower
from shards_ai.game.cards import CardInstance
from shards_ai.game.cards.definitions import (
    APOTRE_DES_OMBRES,
    BRISE_ETHER,
    ECLAIREUR_SPECTRAL,
    HERITIER_DU_NEANT,
    SENTINELLE_DES_TENEBRES,
    FLEAU_DES_OMBRES,
    ZARA_RA,
)
from shards_ai.game.enums import Phase


def put_in_hand(game: Game, *definitions) -> None:
    game.active.hand = [
        CardInstance(f"spectra-test-{index}", definition)
        for index, definition in enumerate(definitions)
    ]


def test_spectra_cards_have_expected_faction_and_copies() -> None:
    game = Game.new(seed=201)
    cards = game.state.central_deck + [card for card in game.state.river if card]
    spectra = [card for card in cards if card.definition.faction is Faction.SPECTRA]

    assert len(spectra) == 21
    assert sum(card.definition.card_id == "void_assassin" for card in spectra) == 3


def test_echo_requires_spectra_in_discard_not_play_zone() -> None:
    game = Game.new(seed=202)
    put_in_hand(game, ECLAIREUR_SPECTRAL)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 2

    game.active.hand.clear()
    game.active.play_zone.clear()
    game.active.power = 0
    game.active.discard_pile.append(CardInstance("discarded-spectra", ECLAIREUR_SPECTRAL))
    put_in_hand(game, ECLAIREUR_SPECTRAL)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 6


def test_heritier_echo_scales_with_spectra_cards_in_discard() -> None:
    game = Game.new(seed=203)
    game.active.discard_pile.extend(
        [
            CardInstance("discarded-spectra-1", ECLAIREUR_SPECTRAL),
            CardInstance("discarded-spectra-2", BRISE_ETHER),
        ]
    )
    put_in_hand(game, HERITIER_DU_NEANT)
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.power == 7


def test_brise_ether_has_a_mastery_threshold() -> None:
    game = Game.new(seed=204)
    game.active.mastery = 9
    put_in_hand(game, BRISE_ETHER)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 4

    game.active.hand.clear()
    game.active.play_zone.clear()
    game.active.power = 0
    game.active.mastery = 10
    put_in_hand(game, BRISE_ETHER)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 8


def test_banish_is_optional_and_cannot_target_a_played_card() -> None:
    game = Game.new(seed=205)
    put_in_hand(game, APOTRE_DES_OMBRES, ECLAIREUR_SPECTRAL)
    discarded = CardInstance("discarded-card", ECLAIREUR_SPECTRAL)
    game.active.discard_pile.append(discarded)
    apotre_id = game.active.hand[0].instance_id
    game.apply(PlayCard(apotre_id))

    actions = game.legal_actions()
    assert SkipBanish() in actions
    assert BanishCard(apotre_id) not in actions
    assert BanishCard(discarded.instance_id) in actions

    game.apply(BanishCard(discarded.instance_id))
    assert game.active.pending_banishes == 0
    assert discarded not in game.active.discard_pile


def test_zara_can_banish_two_cards_after_reaching_ten_mastery() -> None:
    game = Game.new(seed=206)
    game.active.mastery = 9
    first = CardInstance("banishable-1", ECLAIREUR_SPECTRAL)
    second = CardInstance("banishable-2", BRISE_ETHER)
    game.active.discard_pile.extend([first, second])
    put_in_hand(game, ZARA_RA)
    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.active.mastery == 10
    assert game.active.pending_banishes == 2
    game.apply(BanishCard(first.instance_id))
    game.apply(BanishCard(second.instance_id))
    assert game.active.pending_banishes == 0


def test_banish_rejects_a_played_card_without_mutating_state() -> None:
    game = Game.new(seed=208)
    put_in_hand(game, APOTRE_DES_OMBRES)
    played_id = game.active.hand[0].instance_id
    game.apply(PlayCard(played_id))
    before_play_zone = list(game.active.play_zone)
    before_pending = game.active.pending_banishes

    with pytest.raises(InvalidActionError):
        game.apply(BanishCard(played_id))

    assert game.active.play_zone == before_play_zone
    assert game.active.pending_banishes == before_pending


def test_remaining_spectra_card_effects_are_resolved() -> None:
    game = Game.new(seed=209)
    put_in_hand(game, SENTINELLE_DES_TENEBRES)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.power == 3

    game = Game.new(seed=210)
    initial_mastery = game.active.mastery
    put_in_hand(game, FLEAU_DES_OMBRES)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    assert game.active.mastery == initial_mastery + 1
    assert game.active.pending_banishes == 1


def test_random_player_chooses_whether_to_banish_with_fifty_percent_threshold() -> None:
    class FixedRandom(GameRandom):
        def __init__(self, value: float) -> None:
            super().__init__(seed=0)
            self.value = value

        def random(self) -> float:
            return self.value

    game = Game.new(seed=207)
    put_in_hand(game, APOTRE_DES_OMBRES, ECLAIREUR_SPECTRAL)
    game.apply(PlayCard(game.active.hand[0].instance_id))
    actions = game.legal_actions()

    assert isinstance(
        RandomPlayer(game.active_player, FixedRandom(0.49)).choose_action(game.state, actions),
        SkipBanish,
    )
    assert isinstance(
        RandomPlayer(game.active_player, FixedRandom(0.50)).choose_action(game.state, actions),
        BanishCard,
    )
