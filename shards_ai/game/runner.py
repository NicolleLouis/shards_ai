from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .enums import GameStatus, PlayerId
from .actions import Action
from .errors import InvalidActionError, InvalidGameStateError
from .game import Game
from .players import Player
from .random import GameRandom
from .state import GameState


class GameRunner:
    """Run one game by coordinating a game engine and independent players."""

    DEFAULT_MAX_ACTIONS = 10_000
    MAX_TURNS_PER_PLAYER = 100

    def __init__(
        self,
        game: Game,
        players: Mapping[PlayerId, Player],
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_turns: int | None = None,
    ) -> None:
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.game = game
        self.players = dict(players)
        self.max_actions = max_actions
        self.max_turns = max_turns if max_turns is not None else self.MAX_TURNS_PER_PLAYER * len(self.players)
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.actions_played = 0

    @classmethod
    def random_duel(
        cls,
        seed: int | None = None,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_turns: int | None = None,
    ) -> "GameRunner":
        root_rng = GameRandom(seed)
        game = Game.new(seed=seed, rng=root_rng.derive("engine"))

        from shards_ai.ai import RandomPlayer

        players = {
            player_id: RandomPlayer(
                player_id,
                root_rng.derive(f"player-{player_id.value}"),
            )
            for player_id in PlayerId
        }
        return cls(game=game, players=players, max_actions=max_actions, max_turns=max_turns)

    def run(
        self,
        transition_observer: Callable[[GameState, Action, GameState, PlayerId], None] | None = None,
        decision_observer: Callable[[GameState, Sequence[Action], Action, PlayerId], None] | None = None,
        *,
        observer_receives_detached_state: bool = True,
        players_receive_detached_observation: bool | None = None,
        observer_before_state_factory: Callable[[PlayerId], GameState] | None = None,
    ) -> GameState:
        while self.game.state.status is GameStatus.RUNNING:
            if self.game.state.turn_number > self.max_turns:
                self.game.state.status = GameStatus.DRAW
                self.game.state.winner = None
                break
            if self.actions_played >= self.max_actions:
                raise InvalidGameStateError(
                    f"Game exceeded max_actions={self.max_actions} "
                    f"at seed={self.game.state.seed}"
                )

            player_id = self.game.active_player
            player = self.players.get(player_id)
            if player is None:
                raise InvalidGameStateError(f"No player configured for {player_id}")

            observation_kind = getattr(player, "observation_kind", "game_state")
            if observation_kind == "neural":
                observation = self.game.neural_observation_for(player_id)
            elif observation_kind == "game_state":
                receive_detached = (
                    players_receive_detached_observation
                    if players_receive_detached_observation is not None
                    else transition_observer is not None or decision_observer is not None
                    or not getattr(player, "observation_is_read_only", False)
                )
                observation = self.game.observation_for(player_id) if receive_detached else self.game.state
            else:
                raise InvalidGameStateError(f"Unsupported observation kind: {observation_kind!r}")
            legal_actions = self.game.legal_actions()
            if not legal_actions:
                raise InvalidGameStateError(
                    f"No legal action for running game at phase={self.game.state.phase.value}"
                )

            action = player.choose_action(observation, legal_actions)
            if action not in legal_actions:
                raise InvalidActionError(
                    f"Player {player_id} returned an illegal action: {action!r}"
                )
            if decision_observer is not None:
                decision_observer(observation, legal_actions, action, player_id)
            if transition_observer is not None:
                before = (
                    observer_before_state_factory(player_id)
                    if observer_before_state_factory is not None
                    else observation
                )
                self.game.apply(action)
                after = (
                    self.game.observation_for(player_id)
                    if observer_receives_detached_state
                    else self.game.state
                )
                transition_observer(before, action, after, player_id)
            else:
                self.game.apply(action)
            self.actions_played += 1

        return self.game.state
