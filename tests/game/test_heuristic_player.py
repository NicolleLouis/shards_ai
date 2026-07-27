from dataclasses import replace

import pytest

from shards_ai.ai import CardAcquisitionWeights, HeuristicPlayer, HeuristicWeights
from shards_ai.game import (
    AssignPower,
    BanishCard,
    BuyCard,
    CardDefinition,
    CardInstance,
    Effect,
    EffectStep,
    Game,
    GameRandom,
    GameRunner,
    GameStatus,
    GainMastery,
    Phase,
    PlayerId,
    PassPlayPhase,
    PlayCard,
    RecruitMercenary,
    StopBuying,
    SkipBanish,
)
from shards_ai.game.cards import Operation
from shards_ai.game.cards.definitions import EVOKATUS
from shards_ai.game.errors import InvalidActionError


def card(
    card_id: str,
    *,
    cost: int = 0,
    effect: Effect | None = None,
    is_mercenary: bool = False,
    is_champion: bool = False,
    champion_health: int | None = None,
    on_play_effect: Effect | None = None,
) -> CardInstance:
    return CardInstance(
        instance_id=f"instance-{card_id}",
        definition=CardDefinition(
            card_id=card_id,
            name=card_id,
            cost=cost,
            effect=effect or Effect(),
            is_mercenary=is_mercenary,
            is_champion=is_champion,
            champion_health=champion_health,
            on_play_effect=on_play_effect,
        ),
    )


def test_heuristic_player_rejects_empty_actions() -> None:
    game = Game.new(seed=1)
    player = HeuristicPlayer(game.active_player)

    with pytest.raises(InvalidActionError, match="empty action list"):
        player.choose_action(game.observation_for(game.active_player), [])


def test_heuristic_player_prefers_card_draw_when_weighted_alone() -> None:
    game = Game.new(seed=2)
    draw_card = card("draw", effect=Effect(steps=(EffectStep((Operation("draw_card", 2),)),)))
    power_card = card("power", effect=Effect(power=3))
    game.active.hand = [draw_card, power_card]
    weights = replace(HeuristicWeights.zero(), card_draw=1.0)
    player = HeuristicPlayer(game.active_player, weights)

    action = player.choose_action(game.observation_for(game.active_player), game.legal_actions())

    assert action == PlayCard(draw_card.instance_id)


def test_heuristic_player_compares_normal_purchase_and_mercenary_recruitment() -> None:
    game = Game.new(seed=3)
    mercenary = card(
        "mercenary",
        cost=2,
        effect=Effect(power=4),
        is_mercenary=True,
    )
    game.state.phase = Phase.BUY
    game.active.gems = 2
    game.state.river = [mercenary]
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(game.observation_for(game.active_player), game.legal_actions())

    assert action == BuyCard(0, mercenary.instance_id)
    assert player.features_for_action(
        game.state, RecruitMercenary(0, mercenary.instance_id)
    ).card_acquisition_value == 0.0
    assert player.score_action(game.state, action) > player.score_action(
        game.state, RecruitMercenary(0, mercenary.instance_id)
    )


@pytest.mark.parametrize(
    ("health", "expected_health"),
    ((50, 0.0), (48, 2.0), (40, 4.0)),
)
def test_mercenary_immediate_health_value_is_capped_by_missing_health(
    health: int, expected_health: float
) -> None:
    game = Game.new(seed=33)
    mercenary = card(
        "healer",
        cost=1,
        is_mercenary=True,
        effect=Effect(steps=(EffectStep((Operation("gain_health", 4),)),)),
    )
    game.state.phase = Phase.BUY
    game.active.gems = 1
    game.active.health = health
    game.state.river = [mercenary]
    player = HeuristicPlayer(game.active_player)

    features = player.features_for_action(
        game.state, RecruitMercenary(0, mercenary.instance_id)
    )

    assert features.health_gained == expected_health
    assert features.health_advantage_delta == pytest.approx(expected_health / 50.0)


def test_durable_health_purchase_keeps_prospective_nominal_value() -> None:
    game = Game.new(seed=34)
    durable = card(
        "durable-healer",
        cost=1,
        effect=Effect(steps=(EffectStep((Operation("gain_health", 4),)),)),
    )
    game.state.phase = Phase.BUY
    game.active.gems = 1
    game.active.health = 48
    game.state.river = [durable]
    player = HeuristicPlayer(
        game.active_player,
        acquisition_weights=CardAcquisitionWeights(
            health_gained=1.0,
            durable_replay_factor=0.0,
        ),
    )

    features = player.features_for_action(game.state, BuyCard(0, durable.instance_id))

    assert features.card_acquisition_value == 4.0


def test_buy_threshold_rejects_zero_value_durable_purchase() -> None:
    game = Game.new(seed=30)
    durable = card("zero-value", cost=1)
    game.state.phase = Phase.BUY
    game.active.gems = 1
    game.state.river = [durable]
    player = HeuristicPlayer(game.active_player, weights=replace(HeuristicWeights(), buy_threshold=0.0))

    assert player.choose_action(game.observation_for(game.active_player), game.legal_actions()) == StopBuying()


def test_buy_threshold_keeps_positive_durable_purchase() -> None:
    game = Game.new(seed=31)
    durable = card(
        "gem-producer",
        cost=1,
        effect=Effect(steps=(EffectStep((Operation("gain_gems", 1),)),)),
    )
    game.state.phase = Phase.BUY
    game.active.gems = 1
    game.state.river = [durable]
    player = HeuristicPlayer(
        game.active_player,
        weights=replace(HeuristicWeights(), buy_threshold=0.0),
        acquisition_weights=CardAcquisitionWeights(gems_produced=1.0),
    )

    assert player.choose_action(game.observation_for(game.active_player), game.legal_actions()) == BuyCard(
        0, durable.instance_id
    )


def test_buy_threshold_does_not_filter_mercenary_recruitment() -> None:
    game = Game.new(seed=32)
    mercenary = card("zero-value-mercenary", cost=1, is_mercenary=True)
    game.state.phase = Phase.BUY
    game.active.gems = 1
    game.state.river = [mercenary]
    player = HeuristicPlayer(game.active_player, weights=replace(HeuristicWeights(), buy_threshold=2.0))

    assert player.choose_action(game.observation_for(game.active_player), game.legal_actions()) == RecruitMercenary(
        0, mercenary.instance_id
    )


def test_durable_purchase_value_is_discounted_as_game_progresses() -> None:
    game = Game.new(seed=9)
    durable = card("durable", effect=Effect(power=4))
    game.state.river = [durable]
    player = HeuristicPlayer(
        game.active_player,
        acquisition_weights=CardAcquisitionWeights(
            mastery_gained=0.0,
            power_produced=1.0,
            health_gained=0.0,
            card_draw=0.0,
            deck_thinning=0.0,
            target_denial=0.0,
            durable_replay_factor=1.0,
        ),
    )

    early = player.features_for_action(game.state, BuyCard(0, durable.instance_id))
    game.active.mastery = 15
    late = player.features_for_action(game.state, BuyCard(0, durable.instance_id))

    assert early.card_acquisition_value > late.card_acquisition_value


def test_gain_mastery_exposes_the_signed_purchase_opportunity_delta() -> None:
    game = Game.new(seed=13)
    game.active.gems = 2
    game.state.river = [
        card(
            "expensive",
            cost=2,
            effect=Effect(steps=(EffectStep((Operation("gain_power", 4),)),)),
        )
    ]
    acquisition_weights = CardAcquisitionWeights(
        power_produced=1.0,
        mastery_gained=0.0,
        health_gained=0.0,
        card_draw=0.0,
        deck_thinning=0.0,
        target_denial=0.0,
        durable_replay_factor=0.0,
    )
    player = HeuristicPlayer(
        game.active_player,
        weights=replace(
            HeuristicWeights.zero(),
            card_acquisition_value=1.0,
            purchase_opportunity_cost=-1.0,
        ),
        acquisition_weights=acquisition_weights,
    )

    features = player.features_for_action(game.state, GainMastery())

    assert features.purchase_opportunity_cost == 4.0
    assert game.active.gems == 2
    assert game.active.mastery == 0


def test_gain_mastery_purchase_projection_already_sees_future_card_potential() -> None:
    game = Game.new(seed=14)
    game.active.gems = 2
    game.active.mastery = 9
    threshold_card = card(
        "threshold-purchase",
        cost=1,
        effect=Effect(
            steps=(EffectStep((Operation("gain_power", 4),), mastery_at_least=10),)
        ),
    )
    game.state.river = [threshold_card]
    acquisition_weights = CardAcquisitionWeights(
        power_produced=1.0,
        mastery_gained=0.0,
        health_gained=0.0,
        card_draw=0.0,
        deck_thinning=0.0,
        target_denial=0.0,
        durable_replay_factor=0.0,
    )
    player = HeuristicPlayer(
        game.active_player,
        weights=replace(
            HeuristicWeights.zero(),
            card_acquisition_value=1.0,
            purchase_opportunity_cost=-1.0,
        ),
        acquisition_weights=acquisition_weights,
    )

    features = player.features_for_action(game.state, GainMastery())

    assert features.purchase_opportunity_cost == 0.0


def test_play_card_ignores_future_threshold_penalties() -> None:
    game = Game.new(seed=16)
    shard = card(
        "threshold-shard",
        effect=Effect(
            steps=(
                EffectStep((Operation("win"),), mastery_at_least=30),
                EffectStep((Operation("gain_power", 5),), mastery_at_least=20),
                EffectStep((Operation("gain_power", 3),), mastery_at_least=10),
                EffectStep((Operation("gain_power", 2),)),
            )
        ),
    )
    game.active.hand = [shard]
    player = HeuristicPlayer(game.active_player)
    play = PlayCard(shard.instance_id)

    features = player.features_for_action(game.state, play)

    assert features.power_produced == 2.0
    assert features.constraint_penalty == 0.0
    assert player.score_action(game.state, play) > player.score_action(
        game.state, PassPlayPhase()
    )
    assert player.choose_action(game.observation_for(game.active_player), game.legal_actions()) == play


def test_durable_purchase_keeps_future_effect_potential_with_constraint_malus() -> None:
    game = Game.new(seed=17)
    threshold_card = card(
        "threshold-durable",
        cost=1,
        effect=Effect(
            steps=(EffectStep((Operation("gain_power", 4),), mastery_at_least=10),)
        ),
    )
    game.state.phase = Phase.BUY
    game.active.gems = 1
    game.state.river = [threshold_card]
    player = HeuristicPlayer(
        game.active_player,
        acquisition_weights=CardAcquisitionWeights(
            power_produced=1.0,
            mastery_gained=0.0,
            health_gained=0.0,
            card_draw=0.0,
            deck_thinning=0.0,
            target_denial=0.0,
            durable_replay_factor=0.0,
        ),
    )

    early = player.features_for_action(
        game.state, BuyCard(0, threshold_card.instance_id)
    )
    game.active.mastery = 10
    late = player.features_for_action(
        game.state, BuyCard(0, threshold_card.instance_id)
    )

    assert early.card_acquisition_value == late.card_acquisition_value == 4.0
    assert early.constraint_penalty > 0.0
    assert late.constraint_penalty == 0.0


def test_gain_mastery_threshold_value_adds_all_playable_hand_cards() -> None:
    game = Game.new(seed=15)
    game.active.gems = 1
    game.active.mastery = 9
    first = card(
        "threshold-first",
        effect=Effect(
            steps=(EffectStep((Operation("gain_power", 3),), mastery_at_least=10),)
        ),
    )
    second = card(
        "threshold-second",
        effect=Effect(
            steps=(EffectStep((Operation("gain_power", 2),), mastery_at_least=10),)
        ),
    )
    game.active.hand = [first, second]
    player = HeuristicPlayer(
        game.active_player,
        weights=replace(
            HeuristicWeights.zero(),
            power_produced=1.0,
            mastery_threshold_value=1.0,
        ),
    )

    features = player.features_for_action(game.state, GainMastery())

    assert features.mastery_threshold_value == 5.0


def test_constraint_penalty_makes_higher_mastery_threshold_less_valuable() -> None:
    game = Game.new(seed=4)
    low_threshold = card(
        "threshold-10",
        effect=Effect(steps=(EffectStep((Operation("gain_power", 5, mastery_at_least=10),)),)),
    )
    high_threshold = card(
        "threshold-20",
        effect=Effect(steps=(EffectStep((Operation("gain_power", 5, mastery_at_least=20),)),)),
    )
    game.state.phase = Phase.BUY
    game.active.gems = 0
    game.state.river = [low_threshold, high_threshold]
    weights = replace(HeuristicWeights.zero(), constraint_penalty=-1.0)
    player = HeuristicPlayer(
        game.active_player,
        weights,
        acquisition_weights=CardAcquisitionWeights(power_produced=1.0),
    )

    low_features = player.features_for_action(
        game.state, BuyCard(0, low_threshold.instance_id)
    )
    high_features = player.features_for_action(
        game.state, BuyCard(1, high_threshold.instance_id)
    )

    assert high_features.constraint_penalty > low_features.constraint_penalty


def test_omnius_plays_unconditional_draw_without_penalizing_locked_mastery() -> None:
    game = Game.new(seed=18)
    omnius = card(
        "omnius",
        effect=Effect(
            steps=(
                EffectStep(
                    (Operation("draw_card", 2), Operation("gain_mastery", 5, requires_domination=True)),
                ),
            )
        ),
    )
    game.active.hand = [omnius]
    player = HeuristicPlayer(game.active_player)
    play = PlayCard(omnius.instance_id)

    features = player.features_for_action(game.state, play)

    assert features.card_draw == 2.0
    assert features.constraint_penalty == 0.0
    assert player.choose_action(game.observation_for(game.active_player), game.legal_actions()) == play


@pytest.mark.parametrize(
    "constraint",
    ["requires_union", "requires_echo", "requires_domination"],
)
def test_play_card_ignores_inactive_union_echo_and_domination_operations(
    constraint: str,
) -> None:
    game = Game.new(seed=19)
    constrained = card(
        f"{constraint}-card",
        effect=Effect(
            steps=(EffectStep((
                Operation("gain_power", 2),
                Operation("gain_power", 5, **{constraint: True}),
            )),),
        ),
    )
    game.active.hand = [constrained]
    player = HeuristicPlayer(game.active_player)

    features = player.features_for_action(
        game.state, PlayCard(constrained.instance_id)
    )

    assert features.power_produced == 2.0
    assert features.constraint_penalty == 0.0


def test_inactive_conditional_effect_does_not_change_immediate_card_value() -> None:
    game = Game.new(seed=20)
    baseline = card("baseline", effect=Effect(power=5))
    conditional = card(
        "conditional",
        effect=Effect(
            steps=(EffectStep((
                Operation("gain_power", 5),
                Operation("gain_mastery", 1, requires_domination=True),
            )),),
        ),
    )
    game.active.hand = [baseline, conditional]
    player = HeuristicPlayer(game.active_player)

    baseline_features = player.features_for_action(
        game.state, PlayCard(baseline.instance_id)
    )
    conditional_features = player.features_for_action(
        game.state, PlayCard(conditional.instance_id)
    )

    assert conditional_features == baseline_features


def test_banish_tiebreak_prefers_lower_immediate_value_over_action_order() -> None:
    game = Game.new(seed=21)
    spectral = card("spectral", cost=1, effect=Effect(power=2))
    blaster = card("blaster", cost=0, effect=Effect(power=1))
    game.active.hand = [spectral, blaster]
    game.active.pending_banishes = 1
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player), game.legal_actions()
    )

    assert action == BanishCard(blaster.instance_id)


def test_heuristic_skips_banish_for_a_high_value_card() -> None:
    game = Game.new(seed=22)
    valuable = card(
        "valuable",
        effect=Effect(steps=(EffectStep((Operation("draw_card", 3),)),)),
    )
    game.active.hand = [valuable]
    game.active.pending_banishes = 1
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player), game.legal_actions()
    )

    assert action == SkipBanish()


@pytest.mark.parametrize("card_effect", [
    Effect(steps=(EffectStep((Operation("draw_card"),)),)),
    Effect(steps=(EffectStep((Operation("gain_gems", 1), Operation("draw_card"))),)),
])
def test_heuristic_protects_cards_that_replace_themselves(card_effect: Effect) -> None:
    game = Game.new(seed=25)
    replacement = card("replacement", effect=card_effect)
    game.active.hand = [replacement]
    game.active.pending_banishes = 1
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player), game.legal_actions()
    )

    assert action == SkipBanish()


def test_heuristic_protects_champion_with_draw_on_play() -> None:
    game = Game.new(seed=26)
    champion = card(
        "drawing-champion",
        is_champion=True,
        champion_health=4,
        on_play_effect=Effect(steps=(EffectStep((Operation("draw_card"),)),)),
    )
    game.active.hand = [champion]
    game.active.pending_banishes = 1
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player), game.legal_actions()
    )

    assert action == SkipBanish()


def test_heuristic_does_not_protect_inactive_conditional_draw() -> None:
    game = Game.new(seed=27)
    conditional = card(
        "conditional-draw",
        effect=Effect(steps=(EffectStep((Operation("draw_card", mastery_at_least=10),)),)),
    )
    game.active.hand = [conditional]
    game.active.pending_banishes = 1
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player), game.legal_actions()
    )

    assert action == BanishCard(conditional.instance_id)


def test_heuristic_banishes_a_low_value_card_instead_of_skipping() -> None:
    game = Game.new(seed=23)
    weak = card("weak")
    game.active.hand = [weak]
    game.active.pending_banishes = 1
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player), game.legal_actions()
    )

    assert action == BanishCard(weak.instance_id)


def test_banish_value_includes_evokatus_active_power() -> None:
    game = Game.new(seed=24)
    game.active.champions = [CardInstance("existing-evokatus", EVOKATUS)]
    candidate = CardInstance("candidate-evokatus", EVOKATUS)
    game.active.hand = [candidate]
    player = HeuristicPlayer(
        game.active_player,
        acquisition_weights=CardAcquisitionWeights(
            power_produced=1.0,
            card_draw=0.0,
        ),
    )

    features = player.features_for_action(
        game.state, BanishCard(candidate.instance_id)
    )

    assert features.deck_thinning == -2.0


def test_heuristic_player_targets_a_valuable_champion() -> None:
    game = Game.new(seed=5)
    target = card("champion", is_champion=True, champion_health=4)
    game.opponent.champions = [target]
    game.active.power = 4
    game.state.phase = Phase.ATTACK
    weights = replace(HeuristicWeights.zero(), champion_value=1.0)
    player = HeuristicPlayer(game.active_player, weights)

    action = player.choose_action(game.observation_for(game.active_player), game.legal_actions())

    assert action == AssignPower(4, target=target.instance_id)


def test_heuristic_duel_finishes_and_is_reproducible() -> None:
    def run(seed: int):
        root = GameRandom(seed)
        game = Game.new(seed=seed, rng=root.derive("engine"))
        players = {
            player_id: HeuristicPlayer(player_id)
            for player_id in PlayerId
        }
        return GameRunner(game, players).run()

    first = run(6)
    second = run(6)

    assert first.status in {GameStatus.FINISHED, GameStatus.DRAW}
    assert first == second


def test_heuristic_player_stops_buying_when_no_purchase_has_value() -> None:
    game = Game.new(seed=7)
    game.state.phase = Phase.BUY
    game.active.gems = 0
    player = HeuristicPlayer(game.active_player)

    assert player.choose_action(game.observation_for(game.active_player), game.legal_actions()) == StopBuying()


def test_heuristic_player_never_banishes_a_card_with_a_win_branch() -> None:
    game = Game.new(seed=8)
    win_card = card(
        "conditional-win",
        effect=Effect(steps=(EffectStep((Operation("win"),), mastery_at_least=30),)),
    )
    game.active.hand = [win_card]
    player = HeuristicPlayer(game.active_player)

    action = player.choose_action(
        game.observation_for(game.active_player),
        [BanishCard(win_card.instance_id), SkipBanish()],
    )

    assert action == SkipBanish()
