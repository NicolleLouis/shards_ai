from collections.abc import Sequence
from typing import Protocol

from .actions import Action
from .state import GameState


class Player(Protocol):
    """Interface implemented by human, random, heuristic or neural players."""

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> Action:
        ...
