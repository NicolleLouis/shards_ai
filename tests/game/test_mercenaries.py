import pytest

from shards_ai.ai import RandomPlayer
from shards_ai.game import (
    AssignPower,
    BuyCard,
    CARD_CATALOG,
    CardDefinition,
    CardInstance,
    ChoosePendingDecision,
    Effect,
    Game,
    GameRandom,
    InvalidActionError,
    PassPlayPhase,
    PlayCard,
    RecruitMercenary,
    StopBuying,
)
from shards_ai.game.cards.definitions import (
    APOTRE_DES_OMBRES,
    SENTINELLE_DES_TENEBRES,
    VOID_ASSASSIN,
)
from shards_ai.game.actions import SkipBanish


MERCENARY_IDS = {
    "valkyrie_des_landes",
    "saule_vengeur",
    "racine_de_la_foret",
    "chevalier_le_shai",
    "ermite_fongique",
    "clerc_aux_spores",
    "void_assassin",
    "apotre_des_ombres",
    "fleau_des_ombres",
    "brise_ether",
    "heritier_du_neant",
    "zara_ra",
    "le_grand_architecte",
    "omnius_l_erudit",
    "pirate_heretique",
    "prophete_de_leclat",
    "garde_memoire",
}


def mercenary_game(*, central_deck=None) -> Game:
    game = Game.new(seed=301)
    game.state.river = [
        CardInstance("mercenary-in-river", VOID_ASSASSIN),
        None,
        None,
        None,
        None,
        None,
    ]
    game.state.central_deck = list(central_deck or [])
    game.active.gems = VOID_ASSASSIN.cost
    game.apply(PassPlayPhase())
    return game


def test_catalog_marks_exactly_the_declared_mercenaries() -> None:
    actual = {
        card_id for card_id, definition in CARD_CATALOG.items() if definition.is_mercenary
    }

    assert actual == MERCENARY_IDS
    assert all(not CARD_CATALOG[card_id].is_champion for card_id in actual)


def test_champion_cannot_be_declared_as_mercenary() -> None:
    with pytest.raises(ValueError, match="cannot be mercenaries"):
        CardDefinition(
            "invalid_mercenary_champion",
            "Invalid",
            1,
            Effect(power=1),
            is_champion=True,
            champion_health=1,
            is_mercenary=True,
        )


def test_recruiting_a_mercenary_plays_it_and_returns_it_to_the_bottom() -> None:
    replacement = CardInstance("replacement", SENTINELLE_DES_TENEBRES)
    game = mercenary_game(central_deck=[replacement])
    action = RecruitMercenary(0, "mercenary-in-river")

    assert action in game.legal_actions()
    game.apply(action)

    assert game.active.gems == 0
    action_card = game.active.play_zone[0]
    assert game.active.play_zone == [action_card]
    assert action_card.instance_id == "mercenary-in-river"
    assert game.active.power == 5
    assert game.state.river[0] is replacement
    assert game.state.central_deck == []

    game.apply(StopBuying())
    game.apply(AssignPower(5))

    assert game.state.central_deck == [action_card]
    assert action_card not in game.state.players[game.active_player].discard_pile


def test_normal_purchase_of_a_mercenary_stays_in_the_player_deck() -> None:
    game = mercenary_game()
    original_player = game.active_player
    game.apply(BuyCard(0, "mercenary-in-river"))
    purchased = game.active.discard_pile[-1]
    game.apply(StopBuying())
    game.apply(AssignPower(0))

    assert purchased in game.state.players[original_player].discard_pile


def test_recruitment_is_not_legal_for_a_non_mercenary() -> None:
    game = Game.new(seed=302)
    non_mercenary = CardInstance("normal-card", SENTINELLE_DES_TENEBRES)
    game.state.river[0] = non_mercenary
    game.active.gems = non_mercenary.definition.cost
    game.apply(PassPlayPhase())
    before = (list(game.state.river), list(game.active.discard_pile), game.active.gems)

    with pytest.raises(InvalidActionError):
        game.apply(RecruitMercenary(0, non_mercenary.instance_id))

    assert (game.state.river, game.active.discard_pile, game.active.gems) == before


def test_multiple_recruited_mercenaries_keep_their_bottom_stack_order() -> None:
    game = Game.new(seed=307)
    first = CardInstance("first-recruited", VOID_ASSASSIN)
    second = CardInstance("second-recruited", VOID_ASSASSIN)
    game.state.river = [first, second, None, None, None, None]
    game.state.central_deck = []
    game.active.gems = first.definition.cost + second.definition.cost
    game.apply(PassPlayPhase())

    game.apply(RecruitMercenary(0, first.instance_id))
    game.apply(RecruitMercenary(1, second.instance_id))
    game.apply(StopBuying())
    game.apply(AssignPower(10))

    assert game.state.central_deck == [second, first]


def test_recruited_mercenary_can_resolve_a_decision_during_buy() -> None:
    game = Game.new(seed=306)
    recruited = CardInstance("recruited-apotre", APOTRE_DES_OMBRES)
    game.state.river[0] = recruited
    game.state.central_deck = []
    game.active.gems = recruited.definition.cost
    game.apply(PassPlayPhase())

    game.apply(RecruitMercenary(0, recruited.instance_id))

    assert game.active.pending_banishes == 1
    assert SkipBanish() in game.legal_actions()
    assert all(not isinstance(action, (RecruitMercenary, StopBuying)) for action in game.legal_actions())
    game.apply(SkipBanish())
    game.apply(StopBuying())


def test_sentinelle_recovers_one_mercenary_from_discard() -> None:
    game = Game.new(seed=303)
    first = CardInstance("discarded-mercenary-1", VOID_ASSASSIN)
    second = CardInstance("discarded-mercenary-2", VOID_ASSASSIN)
    game.active.discard_pile.extend([first, second])
    sentinel = CardInstance("sentinel", SENTINELLE_DES_TENEBRES)
    game.active.hand = [sentinel]

    game.apply(PlayCard(sentinel.instance_id))

    assert game.active.power == 3
    assert game.active.pending_decision is not None
    assert game.legal_actions() == [
        ChoosePendingDecision(first.instance_id),
        ChoosePendingDecision(second.instance_id),
    ]

    game.apply(ChoosePendingDecision(second.instance_id))

    assert second in game.active.hand
    assert first in game.active.discard_pile
    assert second not in game.active.discard_pile


def test_sentinelle_has_no_recovery_decision_without_a_mercenary() -> None:
    game = Game.new(seed=304)
    sentinel = CardInstance("sentinel", SENTINELLE_DES_TENEBRES)
    game.active.hand = [sentinel]

    game.apply(PlayCard(sentinel.instance_id))

    assert game.active.power == 3
    assert game.active.pending_decision is None


def test_random_player_chooses_a_mercenary_recovery_candidate() -> None:
    game = Game.new(seed=305)
    discarded = CardInstance("discarded-mercenary", VOID_ASSASSIN)
    other = CardInstance("other-discarded-mercenary", VOID_ASSASSIN)
    game.active.discard_pile.extend([discarded, other])
    sentinel = CardInstance("sentinel", SENTINELLE_DES_TENEBRES)
    game.active.hand = [sentinel]
    game.apply(PlayCard(sentinel.instance_id))

    action = RandomPlayer(game.active_player, GameRandom(1)).choose_action(
        game.state, game.legal_actions()
    )

    assert action in {
        ChoosePendingDecision(discarded.instance_id),
        ChoosePendingDecision(other.instance_id),
    }
