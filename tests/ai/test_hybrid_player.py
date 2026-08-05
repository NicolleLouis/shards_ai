from unittest.mock import Mock

import pytest

from shards_ai.ai import HybridPlayer
from shards_ai.game import Game, GameRandom, PlayerId
from shards_ai.game.actions import BanishCard, BuyCard, PassPlayPhase
from shards_ai.game.enums import Phase


@pytest.mark.parametrize(
    ("policy", "phase", "actions", "heuristic"),
    [
        ("purchase_recruitment", Phase.BUY, [PassPlayPhase()], False),
        ("purchase_recruitment", Phase.BUY, [BuyCard(0, "x")], True),
        ("play_phase", Phase.PLAY, [PassPlayPhase()], True),
        ("play_phase", Phase.BUY, [PassPlayPhase()], False),
        ("banish", Phase.PLAY, [BanishCard("x")], True),
        ("banish", Phase.PLAY, [PassPlayPhase()], False),
    ],
)
def test_hybrid_routes_decisions_by_policy(policy, phase, actions, heuristic) -> None:
    game = Game.new(seed=123)
    game.state.active_player = PlayerId.PLAYER_1
    game.state.phase = phase
    player = HybridPlayer(
        PlayerId.PLAYER_1,
        game,
        GameRandom(7),
        scorer=Mock(),
        policy=policy,
    )
    player.neural.choose_action = Mock(return_value=actions[0])
    player.heuristic.choose_action = Mock(return_value=actions[0])

    player.choose_action(game.state, actions)

    assert player.heuristic.choose_action.called is heuristic
    assert player.neural.choose_action.called is not heuristic


def test_hybrid_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unknown hybrid policy"):
        HybridPlayer(
            PlayerId.PLAYER_1,
            Game.new(seed=124),
            GameRandom(8),
            scorer=Mock(),
            policy="unknown",
        )
