import pytest

from shards_ai.ai import RandomPlayer
from shards_ai.game import (
    AssignPower,
    BuyCard,
    Game,
    GameRandom,
    GameRunner,
    GameStatus,
    GainMastery,
    PassPlayPhase,
    Phase,
    PlayerId,
    PlayCard,
    RecruitMercenary,
    StopBuying,
)
from shards_ai.game.errors import InvalidActionError, InvalidGameStateError


class StubRandom(GameRandom):
    def __init__(self, value: float) -> None:
        super().__init__(seed=0)
        self.value = value

    def random(self) -> float:
        return self.value


class PreferMasteryRandom(GameRandom):
    def choice(self, values):
        for value in values:
            if isinstance(value, GainMastery):
                return value
        return super().choice(values)


def test_random_player_plays_every_card_before_passing() -> None:
    game = Game.new(seed=17)
    player = RandomPlayer(game.active_player, GameRandom(17).derive("player"))
    played_ids: list[str] = []

    while game.state.phase is Phase.PLAY:
        legal_actions = game.legal_actions()
        action = player.choose_action(
            game.observation_for(game.active_player),
            legal_actions,
        )
        if isinstance(action, PlayCard):
            played_ids.append(action.card_id)
            game.apply(action)
        elif isinstance(action, GainMastery):
            game.apply(action)
        else:
            assert isinstance(action, PassPlayPhase)
            assert len(played_ids) == 5
            game.apply(action)

    assert len(set(played_ids)) == 5
    assert game.state.phase is Phase.BUY


def test_random_player_can_choose_gain_mastery_among_play_actions() -> None:
    game = Game.new(seed=27)
    game.active.gems = 1
    player = RandomPlayer(game.active_player, PreferMasteryRandom(seed=27))

    action = player.choose_action(
        game.observation_for(game.active_player),
        game.legal_actions(),
    )

    assert action == GainMastery()


def test_random_player_assigns_all_available_power() -> None:
    game = Game.new(seed=18)
    player = RandomPlayer(game.active_player, GameRandom(18).derive("player"))

    for card in list(game.active.hand):
        game.apply(PlayCard(card.instance_id))
    game.apply(PassPlayPhase())
    game.apply(StopBuying())

    action = player.choose_action(
        game.observation_for(game.active_player),
        game.legal_actions(),
    )

    assert action == AssignPower(game.active.power)


def test_random_player_stops_buying_at_the_ten_percent_threshold() -> None:
    game = Game.new(seed=25)
    game.apply(PassPlayPhase())
    game.active.gems = 2
    player = RandomPlayer(game.active_player, StubRandom(0.09))

    assert isinstance(player.choose_action(game.state, game.legal_actions()), StopBuying)


def test_random_player_buys_when_it_does_not_stop() -> None:
    game = Game.new(seed=26)
    game.apply(PassPlayPhase())
    game.active.gems = 2
    player = RandomPlayer(game.active_player, StubRandom(0.10))

    action = player.choose_action(game.state, game.legal_actions())

    assert isinstance(action, (BuyCard, RecruitMercenary))


def test_random_duel_runner_finishes_a_game() -> None:
    runner = GameRunner.random_duel(seed=21)

    state = runner.run()

    assert state.status is GameStatus.FINISHED
    assert state.winner in PlayerId
    assert runner.actions_played > 0


def test_random_duel_runner_is_reproducible() -> None:
    first = GameRunner.random_duel(seed=22).run()
    second = GameRunner.random_duel(seed=22).run()

    assert first == second


def test_random_duel_runner_rejects_a_player_action_outside_legal_actions() -> None:
    class InvalidPlayer:
        def choose_action(self, observation, legal_actions):
            return AssignPower(0)

    game = Game.new(seed=23)
    players = {player_id: InvalidPlayer() for player_id in PlayerId}
    runner = GameRunner(game, players)

    with pytest.raises(InvalidActionError):
        runner.run()


def test_random_duel_runner_has_an_action_limit() -> None:
    runner = GameRunner.random_duel(seed=24, max_actions=1)

    with pytest.raises(InvalidGameStateError, match="max_actions=1"):
        runner.run()


def test_runner_declares_a_draw_after_the_turn_limit() -> None:
    game = Game.new(seed=25)
    game.state.turn_number = 2
    players = {player_id: RandomPlayer(player_id, GameRandom(25)) for player_id in PlayerId}
    runner = GameRunner(game, players, max_turns=1)

    state = runner.run()

    assert state.status is GameStatus.DRAW
    assert state.winner is None
