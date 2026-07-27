from collections.abc import Sequence

from shards_ai.game.actions import (
    Action,
    ActivateChampion,
    AssignPower,
    BanishCard,
    BuyCard,
    ChoosePendingDecision,
    GainMastery,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
    StopBuying,
    SkipBanish,
)
from shards_ai.game.enums import Phase, PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.random import GameRandom
from shards_ai.game.state import GameState


class RandomPlayer:
    """Random but valid policy for the duel V0 rules."""

    # The policy only reads the observation.  GameRunner can therefore avoid
    # cloning the state when no transition observer is active.
    observation_is_read_only = True

    def __init__(self, player_id: PlayerId, rng: GameRandom) -> None:
        self.player_id = player_id
        self._rng = rng

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise InvalidActionError("Cannot choose an action from an empty action list")

        pending_choices = []
        free_recruit_actions = []
        banish_actions = []
        play_actions = []
        power_actions = []
        purchase_actions = []
        has_skip_banish = False
        has_pass_play = False
        has_stop_buying = False
        for action in actions:
            if isinstance(action, ChoosePendingDecision):
                pending_choices.append(action)
            if isinstance(action, RecruitFreeCard):
                free_recruit_actions.append(action)
            if isinstance(action, BanishCard):
                banish_actions.append(action)
            elif isinstance(action, SkipBanish):
                has_skip_banish = True
            if isinstance(action, (PlayCard, GainMastery, ActivateChampion)):
                play_actions.append(action)
            elif isinstance(action, PassPlayPhase):
                has_pass_play = True
            if isinstance(action, AssignPower):
                power_actions.append(action)
            if isinstance(action, (BuyCard, RecruitMercenary)):
                purchase_actions.append(action)
            elif isinstance(action, StopBuying):
                has_stop_buying = True

        if pending_choices:
            return self._rng.choice(pending_choices)

        if free_recruit_actions:
            return self._rng.choice(free_recruit_actions)

        if banish_actions:
            if self._rng.random() < 0.50:
                return SkipBanish()
            return self._rng.choice(banish_actions)
        if has_skip_banish:
            return SkipBanish()

        if observation.phase is Phase.PLAY:
            if play_actions:
                return self._rng.choice(play_actions)
            if has_pass_play:
                return PassPlayPhase()
        elif observation.phase is Phase.ATTACK:
            if power_actions:
                return self._rng.choice(power_actions)
        elif observation.phase is Phase.BUY:
            if not purchase_actions:
                return StopBuying()
            if self._rng.random() < 0.10:
                return StopBuying()
            return self._rng.choice(purchase_actions)

        raise InvalidActionError(
            f"No supported random policy for phase {observation.phase.value} "
            f"and actions {actions!r}"
        )
