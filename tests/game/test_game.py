import pytest

from shards_ai.game import (
    AssignDamage,
    AssignPower,
    BuyCard,
    CARD_CATALOG,
    Game,
    GameStatus,
    Faction,
    GainMastery,
    InvalidActionError,
    PassPlayPhase,
    Phase,
    PlayerId,
    PlayCard,
    RecruitMercenary,
    StopBuying,
    card_definition,
)


def test_game_starts_with_two_shuffled_decks_and_five_card_hands() -> None:
    game = Game.new(seed=7)

    assert all(player.health == 50 for player in game.state.players.values())
    assert all(len(player.hand) == 5 for player in game.state.players.values())
    assert all(len(player.draw_pile) == 5 for player in game.state.players.values())
    assert all(len(player.discard_pile) == 0 for player in game.state.players.values())
    assert game.state.phase is Phase.PLAY


def test_observation_is_detached_from_mutable_game_state() -> None:
    game = Game.new(seed=7)
    observation = game.observation_for(game.active_player)
    observed_card = observation.players[game.active_player].hand[0]
    actual_card = game.active.hand[0]

    observation.players[game.active_player].hand.clear()
    observation.players[game.active_player].health = 1
    observation.central_deck.clear()
    observation.river[0] = None

    assert len(game.active.hand) == 5
    assert game.active.health == 50
    assert game.state.central_deck
    assert game.state.river[0] is not None
    assert observation.players[game.active_player].hand is not game.active.hand
    assert observed_card is not actual_card
    assert observed_card.definition is actual_card.definition


def test_shaping_observation_keeps_only_state_potential_data_detached() -> None:
    game = Game.new(seed=107)
    player_id = game.active_player
    snapshot = game.shaping_observation_for(player_id)

    assert snapshot is not game.state
    assert snapshot.players[player_id].health == game.state.players[player_id].health
    assert snapshot.players[player_id].mastery == game.state.players[player_id].mastery
    assert len(snapshot.players[player_id].champions) == len(game.state.players[player_id].champions)
    assert snapshot.players[player_id].hand == []
    assert snapshot.central_deck == []
    assert snapshot.river == []


def test_game_initializes_mastery_from_the_starting_player() -> None:
    game = Game.new(seed=6)

    assert game.state.starting_player == game.active_player
    assert game.state.players[game.state.starting_player].mastery == 0
    assert game.state.players[game.state.starting_player.opponent].mastery == 1


def test_game_starts_with_starter_catalog_and_central_river() -> None:
    game = Game.new(seed=7)

    for player in game.state.players.values():
        assert [card.definition.card_id for card in player.hand + player.draw_pile].count(
            "crystal"
        ) == 7
        assert [card.definition.card_id for card in player.hand + player.draw_pile].count(
            "blaster"
        ) == 1
        assert [card.definition.card_id for card in player.hand + player.draw_pile].count(
            "shard_reactor"
        ) == 1
        assert [card.definition.card_id for card in player.hand + player.draw_pile].count(
            "infinity_shard"
        ) == 1
    assert len(game.state.central_deck) == 81
    assert len(game.state.river) == 6
    assert all(card is not None for card in game.state.river)
    all_central_cards = game.state.central_deck + [card for card in game.state.river if card]
    assert len(all_central_cards) == 87


def test_buy_card_moves_card_discards_cost_and_refills_river() -> None:
    game = Game.new(seed=8)
    player = game.active
    game.apply(PassPlayPhase())
    slot = next(index for index, card in enumerate(game.state.river) if card is not None)
    card = game.state.river[slot]
    assert card is not None
    player.gems = card.definition.cost
    central_size = len(game.state.central_deck)

    game.apply(BuyCard(slot, card.instance_id))

    assert player.gems == 0
    assert player.discard_pile[-1] is card
    assert game.state.river[slot] is not card
    assert len(game.state.central_deck) == central_size - 1


def test_buying_stops_and_clears_unspent_gems_before_attack() -> None:
    game = Game.new(seed=9)
    game.active.gems = 1
    game.apply(PassPlayPhase())
    game.apply(StopBuying())

    assert game.active.gems == 0
    assert game.state.phase is Phase.ATTACK


def test_buy_legal_actions_only_include_affordable_river_cards() -> None:
    game = Game.new(seed=10)
    game.apply(PassPlayPhase())

    assert game.legal_actions() == [StopBuying()]

    game.active.gems = 2
    actions = game.legal_actions()
    affordable_cards = sum(
        card is not None and card.definition.cost <= 2
        for card in game.state.river
    )
    affordable_mercenaries = sum(
        card is not None
        and card.definition.cost <= 2
        and card.definition.is_mercenary
        for card in game.state.river
    )
    assert len(actions) == affordable_cards + affordable_mercenaries + 1
    assert all(isinstance(action, (BuyCard, RecruitMercenary, StopBuying)) for action in actions)


def test_invalid_buy_does_not_mutate_state() -> None:
    game = Game.new(seed=12)
    game.active.gems = 2
    game.apply(PassPlayPhase())
    before_river = list(game.state.river)
    before_discard = list(game.active.discard_pile)
    before_gems = game.active.gems

    with pytest.raises(InvalidActionError):
        game.apply(BuyCard(0, "stale-instance-id"))

    assert game.state.river == before_river
    assert game.active.discard_pile == before_discard
    assert game.active.gems == before_gems


def test_river_becomes_unbuyable_after_central_deck_is_exhausted() -> None:
    game = Game.new(seed=14)
    game.state.central_deck.clear()
    game.active.gems = 100
    game.apply(PassPlayPhase())

    for slot, card in enumerate(list(game.state.river)):
        assert card is not None
        game.apply(BuyCard(slot, card.instance_id))

    assert game.state.river == [None] * game.RIVER_SIZE
    assert game.legal_actions() == [StopBuying()]


def test_card_catalog_is_indexed_by_stable_id() -> None:
    assert set(CARD_CATALOG) == {
        "crystal",
        "blaster",
        "shard_reactor",
        "infinity_shard",
        "void_assassin",
        "aspirant_maquis", "clerc_aux_spores", "ermite_fongique", "zelote_des_epines",
        "chevalier_le_shai", "gardien_de_la_foret", "saule_vengeur", "ojas",
        "elemental_du_sillon", "racine_de_la_foret",
        "eclaireur_spectral", "apotre_des_ombres", "sentinelle_des_tenebres",
        "fleau_des_ombres", "brise_ether", "heritier_du_neant", "zara_ra",
        "initie_de_l_ordre", "garde_memoire", "prophete_de_leclat", "pirate_heretique",
        "moine_du_portail", "voyante_de_volonte", "moine_cryptopoing",
        "omnius_l_erudit", "le_grand_architecte",
        "drone_kiln", "drones_miniers", "legionnaire_korvus", "drone_reacteur",
        "valkyrie_des_landes",
        "additri_gaia_mancienne", "li_hin_la_brisee", "zen_chi_set",
        "ia_systema", "giga_adepte_de_la_source", "zetta_l_encodeuse",
        "primus_pilus", "drones_numeri", "evokatus", "broyeu_optio",
        "drakonarius", "general_decurion",
    }
    assert card_definition("crystal").name == "Cristal"
    assert card_definition("void_assassin").power == 5
    assert card_definition("void_assassin").damage == 5
    assert AssignDamage is AssignPower


def test_starting_cards_are_neutral() -> None:
    assert all(
        card_definition(card_id).faction is Faction.NEUTRAL
        for card_id in ("blaster", "crystal", "infinity_shard", "shard_reactor")
    )


def test_same_seed_produces_same_initial_state() -> None:
    first = Game.new(seed=11)
    second = Game.new(seed=11)

    assert first.active_player == second.active_player
    for player_id in PlayerId:
        first_player = first.state.players[player_id]
        second_player = second.state.players[player_id]
        assert [card.instance_id for card in first_player.hand] == [
            card.instance_id for card in second_player.hand
        ]
        assert [card.instance_id for card in first_player.draw_pile] == [
            card.instance_id for card in second_player.draw_pile
        ]


def test_playing_cards_accumulates_their_effects() -> None:
    game = Game.new(seed=1)
    cards = list(game.active.hand)

    for card in cards:
        game.apply(PlayCard(card.instance_id))

    expected_operations = [
        operation
        for card in cards
        for operation in card.definition.effect.operations_for_mastery(
            game.active.mastery
        )
    ]
    assert game.active.power == sum(
        operation.amount
        for operation in expected_operations
        if operation.kind == "gain_power"
    )
    assert game.active.gems == sum(card.definition.gems for card in cards)
    assert len(game.active.hand) == 0
    assert len(game.active.play_zone) == 5


def test_full_turn_deals_damage_discards_and_draws_five() -> None:
    game = Game.new(seed=2)
    active_player = game.active_player
    opponent = game.opponent
    initial_opponent_health = opponent.health

    cards = list(game.active.hand)
    expected_power = sum(
        operation.amount
        for card in cards
        for operation in card.definition.effect.operations_for_mastery(
            game.active.mastery
        )
        if operation.kind == "gain_power"
    )
    for card in cards:
        game.apply(PlayCard(card.instance_id))
    game.apply(PassPlayPhase())
    game.apply(StopBuying())

    assert game.state.phase is Phase.ATTACK
    game.apply(AssignPower(expected_power))

    assert opponent.health == initial_opponent_health - expected_power
    assert game.active_player == active_player.opponent
    assert game.state.turn_number == 2
    assert game.state.phase is Phase.PLAY
    assert len(game.state.players[active_player].hand) == 5
    assert len(game.state.players[active_player].discard_pile) == 5


def test_discard_is_reshuffled_mid_draw() -> None:
    game = Game.new(seed=3)
    player = game.active
    initial_hand = list(player.hand)
    player.hand.clear()
    player.draw_pile.clear()
    player.discard_pile.clear()
    player.discard_pile.extend(initial_hand)
    game.draw_many(player.player_id, 5)

    assert len(player.hand) == 5
    assert not player.draw_pile
    assert not player.discard_pile


def test_draw_many_preserves_one_by_one_draw_order_across_reshuffle() -> None:
    bulk_game = Game.new(seed=13)
    single_game = Game.new(seed=13)
    bulk_player = bulk_game.active
    single_player = single_game.active

    for player in (bulk_player, single_player):
        initial_hand = list(player.hand)
        player.hand.clear()
        player.draw_pile = player.draw_pile[:2]
        player.discard_pile = initial_hand[:4]

    bulk_drawn = bulk_game.draw_many(bulk_player.player_id, 5)
    single_drawn = [single_game.draw_one(single_player.player_id) for _ in range(5)]

    assert [card.instance_id for card in bulk_drawn] == [
        card.instance_id for card in single_drawn
    ]


def test_draw_many_draws_only_cards_that_are_available() -> None:
    game = Game.new(seed=14)
    player = game.active
    player.hand.clear()
    player.draw_pile = player.draw_pile[:1]
    player.discard_pile.clear()

    drawn = game.draw_many(player.player_id, 5)

    assert len(drawn) == 1
    assert len(player.hand) == 1
    assert not player.draw_pile
    assert not player.discard_pile


def test_draw_many_is_empty_when_no_cards_are_available() -> None:
    game = Game.new(seed=15)
    player = game.active
    player.hand.clear()
    player.draw_pile.clear()
    player.discard_pile.clear()

    assert game.draw_many(player.player_id, 5) == []


def test_draw_one_returns_none_when_no_cards_are_available() -> None:
    game = Game.new(seed=16)
    player = game.active
    player.hand.clear()
    player.draw_pile.clear()
    player.discard_pile.clear()

    assert game.draw_one(player.player_id) is None


def test_invalid_actions_do_not_mutate_state() -> None:
    game = Game.new(seed=4)
    card_id = game.active.hand[0].instance_id
    before_hand = list(game.active.hand)

    with pytest.raises(InvalidActionError):
        game.apply(AssignPower(0))

    assert game.active.hand == before_hand
    assert game.active.hand[0].instance_id == card_id


def test_gain_mastery_costs_one_gem_and_is_once_per_turn() -> None:
    game = Game.new(seed=41)
    game.active.gems = 2
    initial_mastery = game.active.mastery

    assert GainMastery() in game.legal_actions()
    game.apply(GainMastery())

    assert game.active.gems == 1
    assert game.active.mastery == initial_mastery + 1
    assert game.active.mastery_action_used is True
    assert GainMastery() not in game.legal_actions()

    with pytest.raises(InvalidActionError):
        game.apply(GainMastery())


def test_gain_mastery_is_unavailable_without_gems_or_at_the_cap() -> None:
    game = Game.new(seed=45)
    game.active.mastery = 30
    game.active.gems = 10

    assert GainMastery() not in game.legal_actions()

    game.active.mastery = 0
    game.active.gems = 0
    assert GainMastery() not in game.legal_actions()
    with pytest.raises(InvalidActionError):
        game.apply(GainMastery())


def test_mastery_action_resets_at_cleanup_but_mastery_persists() -> None:
    game = Game.new(seed=42)
    player = game.active
    player.gems = 1
    game.apply(GainMastery())
    mastery = player.mastery
    game.apply(PassPlayPhase())
    game.apply(StopBuying())
    game.apply(AssignPower(0))

    assert game.state.players[player.player_id].mastery == mastery
    assert game.state.players[player.player_id].mastery_action_used is False


@pytest.mark.parametrize(
    ("card_id", "mastery", "expected_gems", "expected_power"),
    [
        ("shard_reactor", 0, 2, 0),
        ("shard_reactor", 5, 3, 0),
        ("shard_reactor", 15, 4, 0),
        ("infinity_shard", 0, 0, 2),
        ("infinity_shard", 10, 0, 3),
        ("infinity_shard", 20, 0, 5),
    ],
)
def test_mastery_threshold_cards_resolve_the_highest_applicable_branch(
    card_id: str,
    mastery: int,
    expected_gems: int,
    expected_power: int,
) -> None:
    definition = CARD_CATALOG[card_id]
    game = Game.new(seed=43, card_definition=definition)
    game.active.mastery = mastery
    card = game.active.hand[0]

    game.apply(PlayCard(card.instance_id))

    assert game.active.gems == expected_gems
    assert game.active.power == expected_power


def test_infinity_shard_wins_at_30_mastery() -> None:
    game = Game.new(seed=44, card_definition=CARD_CATALOG["infinity_shard"])
    game.active.mastery = 30

    game.apply(PlayCard(game.active.hand[0].instance_id))

    assert game.state.status is GameStatus.FINISHED
    assert game.state.winner == game.active_player


def test_game_ends_when_health_reaches_zero() -> None:
    game = Game.new(seed=5)
    game.opponent.health = 1
    game.active.power = 1
    game.apply(PassPlayPhase())
    game.apply(StopBuying())
    game.apply(AssignPower(1))

    assert game.state.status is GameStatus.FINISHED
    assert game.state.winner == game.active_player
    assert game.opponent.health == 0

    with pytest.raises(InvalidActionError):
        game.apply(PassPlayPhase())
