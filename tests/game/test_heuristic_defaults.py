from shards_ai.ai import CardAcquisitionWeights, CardConstraintWeights, HeuristicPlayer, HeuristicWeights
from shards_ai.game import Game


def test_heuristic_player_uses_v008_weights_by_default() -> None:
    player = HeuristicPlayer(Game.new(seed=1).active_player)

    assert player.weights == HeuristicWeights(
        cost_paid=0.0,
        gems_produced=0.75,
        power_produced=0.5,
        mastery_gained=0.25,
        health_gained=2.75,
        card_draw=0.75,
        shield_value=0.5,
        deck_thinning=1.0,
        card_acquisition_value=1.0,
        champion_value=2.75,
        target_denial=1.5,
        damage_value=1.25,
        constraint_penalty=-1.0,
        phase_progress=0.1,
        action_penalty=-1.0,
        lethal=1000.0,
        terminal_win=1000.0,
        buy_threshold=0.625,
    )
    assert player.acquisition_weights == CardAcquisitionWeights(
        gems_produced=0.0,
        power_produced=0.75,
        mastery_gained=0.75,
        health_gained=1.375,
        card_draw=1.75,
        deck_thinning=1.0,
        target_denial=1.0,
        banish_threshold=3.0,
    )
    assert player.constraint_weights == CardConstraintWeights(
        mastery=1.0,
        health=0.75,
        inspiration=0.5,
        echo=0.75,
        union=1.0,
        domination=1.5,
    )
