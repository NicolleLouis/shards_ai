from __future__ import annotations

from shards_ai.ai.player_middleware import LegacyActionMiddleware
from shards_ai.game import (
    BuyCard,
    EndMainPhase,
    GainMastery,
    Game,
    PassPlayPhase,
    Phase,
    RecruitMercenary,
    StopBuying,
)
from shards_ai.game.cards import CardInstance
from shards_ai.game.cards.definitions import VOID_ASSASSIN


class StubLegacyPlayer:
    player_id = None
    observation_kind = "game_state"

    def choose_action(self, _observation, legal_actions):
        return legal_actions[-1]


class BoundaryGainMasteryPlayer(StubLegacyPlayer):
    legacy_capability_profile_id = "boundary_gain_mastery_v1"


def test_legacy_middleware_consumes_phase_switch_without_engine_transition() -> None:
    game = Game.new(seed=9001)
    game.enable_modern_mode()
    player = StubLegacyPlayer()
    player.player_id = game.active_player
    middleware = LegacyActionMiddleware(game, player)

    observation, actions, action = middleware.choose_visible_action()
    assert observation.phase is Phase.PLAY
    assert PassPlayPhase() in actions
    assert action == PassPlayPhase()
    assert middleware.translate(action) is None
    assert game.state.phase is Phase.PLAY

    game.active.gems = VOID_ASSASSIN.cost
    game.state.river[0] = CardInstance("mercenary", VOID_ASSASSIN)
    observation, actions, _ = middleware.choose_visible_action()
    assert observation.phase is Phase.BUY
    assert BuyCard(0, "mercenary") in actions
    assert StopBuying() in actions


def test_modern_main_phase_allows_buy_then_mastery_then_end() -> None:
    game = Game.new(seed=9002)
    game.enable_modern_mode()
    game.active.gems = 3
    game.state.river[0] = CardInstance("mercenary", VOID_ASSASSIN)

    recruit = RecruitMercenary(0, "mercenary")
    assert recruit in game.legal_actions()
    game.apply(recruit)

    assert GainMastery() in game.legal_actions()
    game.apply(GainMastery())
    assert EndMainPhase() in game.legal_actions()
    game.apply(EndMainPhase())
    assert game.state.phase is Phase.ATTACK


def test_boundary_gain_mastery_converts_stop_buy_once_and_keeps_buy_view() -> None:
    game = Game.new(seed=9003)
    game.enable_modern_mode()
    game.active.gems = 1
    player = BoundaryGainMasteryPlayer()
    player.player_id = game.active_player
    middleware = LegacyActionMiddleware(game, player)

    middleware.observation_and_actions()
    assert middleware.translate(PassPlayPhase()) is None
    middleware.observation_and_actions()
    assert middleware.view_mode is Phase.BUY

    translated = middleware.translate(StopBuying())

    assert translated == GainMastery()
    assert middleware.boundary_gain_mastery_conversions == 1
    assert middleware.view_mode is Phase.BUY
    assert game.state.phase is Phase.PLAY

    game.apply(translated)
    middleware.observation_and_actions()
    assert GainMastery() not in game.legal_actions()
    assert middleware.translate(StopBuying()) == EndMainPhase()
    assert middleware.view_mode is Phase.PLAY


def test_legacy_boundary_does_not_convert_stop_buy() -> None:
    game = Game.new(seed=9004)
    game.enable_modern_mode()
    game.active.gems = 1
    player = StubLegacyPlayer()
    player.player_id = game.active_player
    middleware = LegacyActionMiddleware(game, player)

    middleware.observation_and_actions()
    assert middleware.translate(PassPlayPhase()) is None
    middleware.observation_and_actions()
    assert middleware.translate(StopBuying()) == EndMainPhase()
    assert middleware.view_mode is Phase.PLAY
    assert middleware.boundary_gain_mastery_conversions == 0
