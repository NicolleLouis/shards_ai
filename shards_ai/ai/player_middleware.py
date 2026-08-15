"""Per-player capability views over the complete game action space."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from shards_ai.game.actions import (
    Action,
    BanishCard,
    ChoosePendingDecision,
    EndMainPhase,
    GainMastery,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
    BuyCard,
    StopBuying,
    SkipBanish,
    ActivateChampion,
)
from shards_ai.game.enums import Phase
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.game import Game
from shards_ai.game.observation import NeuralObservation


LEGACY_PLAY_ACTIONS = (PlayCard, ActivateChampion, GainMastery, PassPlayPhase)
LEGACY_BUY_ACTIONS = (BuyCard, RecruitMercenary, StopBuying)
PENDING_ACTIONS = (BanishCard, SkipBanish, ChoosePendingDecision, RecruitFreeCard)
BOUNDARY_GAIN_MASTERY_CAPABILITY = "boundary_gain_mastery_v1"


class LegacyActionMiddleware:
    """Expose the historical PLAY/BUY decision contract over modern ``MAIN``."""

    capability_profile_id = "legacy_play_buy_v1"
    observation_is_read_only = True

    def __init__(self, game: Game, player: object, player_id=None) -> None:
        self.game = game
        self.player = player
        self.player_id = getattr(player, "player_id", player_id)
        self.observation_kind = getattr(player, "observation_kind", "game_state")
        self.view_mode = Phase.PLAY
        self.virtual_decisions = 0
        self.force_detached_observation = False
        self.capability_profile_id = getattr(
            player, "legacy_capability_profile_id", "legacy_play_buy_v1"
        )
        self.boundary_gain_mastery_conversions = 0
        self._pending_engine_actions: tuple[Action, ...] | None = None
        self._pending_view_mode: Phase | None = None
        if hasattr(player, "legacy_view_mode"):
            player.legacy_view_mode = self.view_mode
        if hasattr(player, "legacy_decision_mode"):
            player.legacy_decision_mode = True
        acquisition_policy = getattr(player, "acquisition_policy", None)
        if acquisition_policy is not None and hasattr(acquisition_policy, "legacy_view_mode"):
            acquisition_policy.legacy_view_mode = self.view_mode
        acquisition_player = getattr(acquisition_policy, "_player", None)
        if acquisition_player is not None and hasattr(acquisition_player, "legacy_decision_mode"):
            acquisition_player.legacy_decision_mode = True

    @property
    def decisions(self):
        return getattr(self.player, "decisions", 0)

    @property
    def total_inference_seconds(self):
        return getattr(self.player, "total_inference_seconds", 0.0)

    def reset_for_turn(self) -> None:
        self.view_mode = Phase.PLAY
        self._pending_engine_actions = None
        self._pending_view_mode = None

    def __getattr__(self, name: str):
        return getattr(self.player, name)

    def observation_and_actions(self) -> tuple[object, list[Action]]:
        if hasattr(self.player, "legacy_view_mode"):
            self.player.legacy_view_mode = self.view_mode
        acquisition_policy = getattr(self.player, "acquisition_policy", None)
        if acquisition_policy is not None and hasattr(acquisition_policy, "legacy_view_mode"):
            acquisition_policy.legacy_view_mode = self.view_mode
        complete = self.game.legal_actions()
        self._pending_engine_actions = tuple(complete)
        self._pending_view_mode = self.view_mode
        visible_phase = self.view_mode if self.game.state.phase is Phase.PLAY else self.game.state.phase
        if self.observation_kind == "neural":
            observation = self.game.neural_observation_for(self.player_id)
            observation = replace(observation, phase=visible_phase.value)
        else:
            if (
                not self.force_detached_observation
                and visible_phase is self.game.state.phase
                and self.view_mode is Phase.PLAY
            ):
                observation = self.game.state
            else:
                observation = self.game.observation_for(self.player_id)
                observation = replace(observation, phase=visible_phase)
        return observation, self._visible_actions(complete)

    def _visible_actions(self, actions: Sequence[Action]) -> list[Action]:
        pending = [action for action in actions if isinstance(action, PENDING_ACTIONS)]
        if pending:
            return pending
        if self.game.state.phase is Phase.ATTACK:
            return list(actions)
        if self.view_mode is Phase.PLAY:
            visible = [action for action in actions if isinstance(action, LEGACY_PLAY_ACTIONS)]
            if any(isinstance(action, EndMainPhase) for action in actions):
                visible.append(PassPlayPhase())
            return visible
        visible = [action for action in actions if isinstance(action, LEGACY_BUY_ACTIONS)]
        if any(isinstance(action, EndMainPhase) for action in actions):
            visible.append(StopBuying())
        return visible

    def choose_visible_action(self) -> tuple[object, list[Action], Action]:
        observation, actions = self.observation_and_actions()
        if not actions:
            raise InvalidActionError(
                f"Legacy middleware hid every action in view={self.view_mode.value}, "
                f"engine_actions={self.game.legal_actions()!r}"
            )
        action = self.choose_action(observation, actions)
        if action not in actions:
            raise InvalidActionError(f"Legacy player returned an invisible action: {action!r}")
        return observation, actions, action

    def choose_action(self, observation, legal_actions: Sequence[Action]) -> Action:
        """Compatibility seam for callers that instrument player.choose_action."""
        return self.player.choose_action(observation, legal_actions)

    def translate(self, action: Action) -> Action | None:
        pending_actions = self._pending_engine_actions
        pending_view_mode = self._pending_view_mode
        self._pending_engine_actions = None
        self._pending_view_mode = None

        if isinstance(action, PassPlayPhase):
            if self.view_mode is not Phase.PLAY:
                raise InvalidActionError("PassPlayPhase is only available in legacy PLAY view")
            if pending_view_mode is not self.view_mode or not any(
                isinstance(candidate, EndMainPhase) for candidate in (pending_actions or ())
            ):
                raise InvalidActionError("PassPlayPhase is not available in the current engine state")
            self.view_mode = Phase.BUY
            self.virtual_decisions += 1
            return None
        if isinstance(action, StopBuying):
            if self.view_mode is not Phase.BUY:
                raise InvalidActionError("StopBuying is only available in legacy BUY view")
            if pending_view_mode is not self.view_mode or not any(
                isinstance(candidate, EndMainPhase) for candidate in (pending_actions or ())
            ):
                raise InvalidActionError("StopBuying is not available in the current engine state")
            if (
                self.capability_profile_id == BOUNDARY_GAIN_MASTERY_CAPABILITY
                and self.game.active.gems >= 1
                and any(isinstance(candidate, GainMastery) for candidate in (pending_actions or ()))
            ):
                self.boundary_gain_mastery_conversions += 1
                return GainMastery()
            self.view_mode = Phase.PLAY
            return EndMainPhase()
        complete = pending_actions if pending_view_mode is self.view_mode else self.game.legal_actions()
        if action not in complete and not (
            isinstance(action, (PassPlayPhase, StopBuying))
            and any(isinstance(candidate, EndMainPhase) for candidate in complete)
        ):
            raise InvalidActionError(f"Middleware action is not legal in the engine: {action!r}")
        return action


def is_modern_player(player: object) -> bool:
    return getattr(player, "capability_profile_id", None) == "full_main_v1"
