from shards_ai.game import (
    ActivateChampion,
    AssignPower,
    BuyCard,
    ChoosePendingDecision,
    Faction,
    Game,
    Phase,
    PlayCard,
    PassPlayPhase,
)
from shards_ai.game.cards import CardInstance
from shards_ai.game.cards.definitions import (
    ADDITRI_GAIAMENCIENNE,
    ASPIRANT_MAQUIS,
    DRONES_NUMERI,
    BROYEU_OPTIO,
    DRAKONARIUS,
    DRONE_KILN,
    EVOKATUS,
    GENERAL_DECURION,
    GIGA_ADEPTE_DE_LA_SOURCE,
    IA_SYSTEMA,
    LI_HIN_LA_BRISEE,
    LEGIONNAIRE_KORVUS,
    PRIMUS_PILUS,
    SAULE_VENGEUR,
    ZEN_CHI_SET,
    ZETTA_L_ENCODEUSE,
    VALKYRIE_DES_LANDES,
    ZELOTE_DES_EPINES,
)


def test_champion_is_persistent_and_additri_counts_itself() -> None:
    game = Game.new(seed=601)
    game.active.hand = [CardInstance("additri", ADDITRI_GAIAMENCIENNE)]

    game.apply(PlayCard("additri"))
    assert [card.instance_id for card in game.active.champions] == ["additri"]
    assert game.active.play_zone == []

    game.apply(ActivateChampion("additri"))
    assert game.active.power == 4


def test_attack_destroys_a_lethal_champion_and_keeps_remaining_power() -> None:
    game = Game.new(seed=602)
    champion = CardInstance("enemy-champion", ADDITRI_GAIAMENCIENNE)
    game.opponent.champions = [champion]
    game.active.power = 7
    game.state.phase = Phase.ATTACK

    game.apply(AssignPower(7, target="enemy-champion"))

    assert champion in game.opponent.discard_pile
    assert game.opponent.champions == []
    assert game.active.power == 2
    assert game.state.phase is Phase.ATTACK


def test_attack_cannot_target_a_champion_with_insufficient_power() -> None:
    game = Game.new(seed=603)
    champion = CardInstance("enemy-champion", ADDITRI_GAIAMENCIENNE)
    game.opponent.champions = [champion]
    game.active.power = 4
    game.state.phase = Phase.ATTACK

    assert game.legal_actions() == [AssignPower(4)]


def test_li_hin_is_not_a_power_target() -> None:
    game = Game.new(seed=604)
    champion = CardInstance("li-hin", LI_HIN_LA_BRISEE)
    game.opponent.champions = [champion]
    game.active.power = 10
    game.state.phase = Phase.ATTACK

    assert game.legal_actions() == [AssignPower(10)]


def test_zelote_directly_destroys_the_only_union_target() -> None:
    game = Game.new(seed=605)
    enemy = CardInstance("enemy-champion", LI_HIN_LA_BRISEE)
    game.opponent.champions = [enemy]
    game.active.hand = [
        CardInstance("zelote", ZELOTE_DES_EPINES),
        CardInstance("maquis", ASPIRANT_MAQUIS),
    ]

    game.apply(PlayCard("zelote"))

    assert enemy in game.opponent.discard_pile
    assert game.active.pending_decision is None


def test_saule_requires_a_choice_for_multiple_champions() -> None:
    game = Game.new(seed=606)
    first = CardInstance("first", ADDITRI_GAIAMENCIENNE)
    second = CardInstance("second", ADDITRI_GAIAMENCIENNE)
    game.opponent.champions = [first, second]
    game.active.mastery = 15
    game.active.hand = [CardInstance("saule", SAULE_VENGEUR)]

    game.apply(PlayCard("saule"))

    choices = game.legal_actions()
    assert set(action.choice_id for action in choices) == {"first", "second"}
    game.apply(ChoosePendingDecision("second"))
    assert game.opponent.champions == [first]
    assert {action.choice_id for action in game.legal_actions()} == {"first"}


def test_giga_draws_at_pose() -> None:
    game = Game.new(seed=607)
    game.active.hand = [CardInstance("giga", GIGA_ADEPTE_DE_LA_SOURCE)]
    initial_draw_size = len(game.active.draw_pile)

    game.apply(PlayCard("giga"))

    assert len(game.active.draw_pile) == initial_draw_size - 1
    assert len(game.active.champions) == 1


def test_evokatus_gains_power_per_homodeus_champion() -> None:
    game = Game.new(seed=609)
    game.active.champions = [
        CardInstance("first", EVOKATUS),
        CardInstance("second", EVOKATUS),
    ]
    game.active.hand = [CardInstance("third", EVOKATUS)]

    game.apply(PlayCard("third"))
    game.apply(ActivateChampion("first"))

    assert game.active.power == 3
    assert game.active.health == game.STARTING_HEALTH


def test_drones_numeri_plays_and_activates_next_homodeus_recruitment() -> None:
    game = Game.new(seed=608)
    drones = CardInstance("drones", DRONES_NUMERI)
    recruited = CardInstance("recruited", BROYEU_OPTIO)
    game.active.hand = [drones]
    game.apply(PlayCard("drones"))
    game.apply(ActivateChampion("drones"))
    game.apply(PassPlayPhase())
    game.active.gems = recruited.definition.cost
    game.state.river = [recruited, None, None, None, None, None]

    game.apply(BuyCard(0, "recruited"))

    assert recruited in game.active.champions
    assert recruited.instance_id in game.active.activated_champion_ids
    assert game.active.power == 3


def test_zen_chi_set_recovers_any_spectra_card_after_power() -> None:
    game = Game.new(seed=609)
    recovered = CardInstance("spectra-discard", LI_HIN_LA_BRISEE)
    game.active.discard_pile = [recovered]
    game.active.hand = [CardInstance("zen", ZEN_CHI_SET)]

    game.apply(PlayCard("zen"))
    game.apply(ActivateChampion("zen"))

    assert game.active.power == 3
    assert recovered in game.active.hand


def test_systema_checks_mastery_after_its_gain() -> None:
    game = Game.new(seed=610)
    game.active.mastery = 19
    game.active.hand = [CardInstance("systema", IA_SYSTEMA)]
    initial_draw = len(game.active.draw_pile)

    game.apply(PlayCard("systema"))
    game.apply(ActivateChampion("systema"))

    assert game.active.mastery == 20
    assert len(game.active.draw_pile) == initial_draw - 2


def test_primus_counts_itself_and_other_homodeus_champions() -> None:
    game = Game.new(seed=611)
    game.active.champions = [
        CardInstance("other-1", DRONES_NUMERI),
        CardInstance("other-2", BROYEU_OPTIO),
        CardInstance("primus", PRIMUS_PILUS),
    ]
    initial_draw = len(game.active.draw_pile)

    game.apply(ActivateChampion("primus"))

    assert len(game.active.draw_pile) == initial_draw - 2


def test_inspiration_only_works_with_a_champion_and_targets_opponent() -> None:
    game = Game.new(seed=612)
    game.active.champions = [CardInstance("champion", ADDITRI_GAIAMENCIENNE)]
    game.active.hand = [CardInstance("valk", VALKYRIE_DES_LANDES)]
    game.opponent.mastery = 5

    game.apply(PlayCard("valk"))

    assert game.opponent.mastery == 3


def test_korvus_recovers_a_champion_from_discard() -> None:
    game = Game.new(seed=613)
    recovered = CardInstance("discarded-champion", ADDITRI_GAIAMENCIENNE)
    game.active.discard_pile = [recovered]
    game.active.hand = [CardInstance("korvus", LEGIONNAIRE_KORVUS)]

    game.apply(PlayCard("korvus"))

    assert recovered in game.active.hand
    assert game.active.power == 2


def test_zetta_makes_itself_the_only_champion_power_target() -> None:
    game = Game.new(seed=614)
    zetta = CardInstance("zetta", ZETTA_L_ENCODEUSE)
    other = CardInstance("other", ADDITRI_GAIAMENCIENNE)
    game.opponent.champions = [zetta, other]
    game.active.power = 10
    game.state.phase = Phase.ATTACK

    assert game.legal_actions() == [
        AssignPower(10), AssignPower(10, target="zetta")
    ]


def test_general_copies_selected_homodeus_effect_with_current_state() -> None:
    game = Game.new(seed=615)
    game.active.mastery = 20
    game.active.hand = [
        CardInstance("drone", DRONE_KILN),
        CardInstance("general", GENERAL_DECURION),
    ]

    game.apply(PlayCard("drone"))
    game.apply(PlayCard("general"))
    game.apply(ActivateChampion("general"))

    assert game.active.gems == 5
    assert game.active.pending_decision is not None
    game.apply(ChoosePendingDecision("drone"))
    assert game.active.gems == 7


def test_drakonarius_is_protected_by_general_but_directly_destroyable() -> None:
    game = Game.new(seed=616)
    drakonarius = CardInstance("drakonarius", DRAKONARIUS)
    general = CardInstance("general", GENERAL_DECURION)
    game.opponent.champions = [drakonarius, general]
    game.active.power = 10
    game.state.phase = Phase.ATTACK

    assert AssignPower(10, target="drakonarius") not in game.legal_actions()
