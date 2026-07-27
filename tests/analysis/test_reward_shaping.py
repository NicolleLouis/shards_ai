from copy import deepcopy

import pytest

from shards_ai.ai.state_evaluator import (
    StateRewardWeights,
    champion_threat_advantage,
    mastery_advantage,
    shaping_reward,
    state_potential,
)
from shards_ai.analysis import RewardShapingTracker
from shards_ai.game import Game, PlayerId, PlayCard


def test_mastery_advantage_uses_the_thirty_point_scale() -> None:
    state = Game.new(seed=1).state
    state.players[PlayerId.PLAYER_1].mastery = 30
    state.players[PlayerId.PLAYER_2].mastery = 0

    assert mastery_advantage(state, PlayerId.PLAYER_1) == pytest.approx(1.0)


def test_champion_threat_uses_count_and_fixed_scale() -> None:
    game = Game.new(seed=2)
    game.active.champions = [game.active.hand[0], game.active.hand[1]]
    game.opponent.champions = [game.opponent.hand[0]]

    assert champion_threat_advantage(game.state, game.active_player) == pytest.approx(0.25)


def test_gamma_one_shaping_is_the_potential_difference() -> None:
    before = Game.new(seed=3).state
    after = deepcopy(before)
    after.players[PlayerId.PLAYER_1].health += 10

    expected = state_potential(after, PlayerId.PLAYER_1) - state_potential(
        before, PlayerId.PLAYER_1
    )
    assert shaping_reward(before, after, PlayerId.PLAYER_1) == pytest.approx(expected)


def test_tracker_records_every_transition_from_the_candidate_perspective() -> None:
    game = Game.new(seed=4)
    tracker = RewardShapingTracker(game.active_player, keep_transitions=True)
    before = game.observation_for(game.active_player)
    action = game.legal_actions()[0]
    game.apply(action)
    after = game.observation_for(before.active_player)

    tracker.observe(before, action, after, before.active_player.opponent)
    assert len(tracker.transitions) == 1
    tracker.observe(before, action, after, before.active_player)
    assert len(tracker.transitions) == 2
