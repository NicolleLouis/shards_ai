from shards_ai.ai import CardConstraintWeights
from shards_ai.ai.heuristic_features import _operation_constraint_penalty, features_for_action
from shards_ai.game.cards import Operation
from shards_ai.game import Game, GainMastery, Phase


def test_gain_mastery_exposes_normalized_mastery_delta() -> None:
    game = Game.new(seed=11)
    game.state.phase = Phase.PLAY
    game.active.gems = 1

    features = features_for_action(game.state, GainMastery(), game.active_player)

    assert features.mastery_advantage_delta == 1 / 30
    assert features.projection_supported


def test_domination_penalty_is_heavier_than_union_and_inspiration() -> None:
    game = Game.new(seed=12)
    player = game.active
    operation = Operation(
        kind="gain_power",
        amount=1,
        requires_domination=True,
        requires_union=True,
        requires_inspiration=True,
    )

    penalty = _operation_constraint_penalty(
        game.state,
        player,
        player.hand[0].definition,
        operation,
        (False, False, False, False),
        CardConstraintWeights(),
    )

    assert penalty == 1.5 + 1.0 + 0.5
