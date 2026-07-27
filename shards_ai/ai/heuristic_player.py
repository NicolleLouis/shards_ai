"""Deterministic weighted heuristic player."""

from __future__ import annotations

from collections.abc import Sequence

from shards_ai.game.actions import (
    Action,
    ActivateChampion,
    BanishCard,
    BuyCard,
    GainMastery,
    PassPlayPhase,
    RecruitFreeCard,
    RecruitMercenary,
    StopBuying,
)
from shards_ai.game.enums import Phase, PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.state import GameState

from .heuristic_evaluator import (
    ActionFeatures,
    CardAcquisitionWeights,
    CardConstraintWeights,
    HeuristicWeights,
)
from .heuristic_features import (
    _common_condition_flags,
    _durable_replay_multiplier,
    _find_card,
    is_replacement_card,
    _play_card_features,
    features_for_action,
    is_win_card,
)


class HeuristicPlayer:
    """Select the legal action with the highest weighted heuristic score."""

    observation_is_read_only = True

    def __init__(
        self,
        player_id: PlayerId,
        weights: HeuristicWeights | None = None,
        acquisition_weights: CardAcquisitionWeights | None = None,
        constraint_weights: CardConstraintWeights | None = None,
    ) -> None:
        self.player_id = player_id
        self.weights = weights or HeuristicWeights()
        self.acquisition_weights = acquisition_weights or CardAcquisitionWeights()
        self.constraint_weights = constraint_weights or CardConstraintWeights(
            mastery=1.0,
            health=0.75,
            inspiration=0.5,
            echo=0.75,
            union=1.0,
            domination=1.5,
        )

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise InvalidActionError("Cannot choose an action from an empty action list")

        common_condition_flags = _common_condition_flags(observation.players[self.player_id])
        # Keep every card with a conditional win branch available as a future
        # win condition. The engine still allows the action; this policy must
        # never select it as a banish target.
        protected_banishes = {
            action
            for action in actions
            if isinstance(action, BanishCard)
            and (card := _find_card(
                observation.players[self.player_id],
                action.card_id,
                zones=("hand", "discard_pile"),
            )) is not None
            and (
                is_win_card(card)
                or is_replacement_card(
                    observation,
                    observation.players[self.player_id],
                    card,
                    self.constraint_weights,
                    common_condition_flags,
                )
            )
        }
        if protected_banishes:
            actions = [action for action in actions if action not in protected_banishes]
            if not actions:
                raise InvalidActionError("No legal non-protected action remains")

        banish_actions = [action for action in actions if isinstance(action, BanishCard)]
        banish_tiebreak = {}
        if len(banish_actions) > 1:
            player = observation.players[self.player_id]
            banish_cards = {
                action.card_id: _find_card(
                    player, action.card_id, zones=("hand", "discard_pile")
                )
                for action in banish_actions
            }
            ordered_banish_ids = sorted(
                (
                    self.weights.score(
                        _play_card_features(
                            observation,
                            player,
                            card,
                            self.constraint_weights,
                            common_condition_flags,
                        )
                    )
                    if card is not None
                    else float("inf"),
                    card.definition.cost if card is not None else 0,
                    card.definition.card_id if card is not None else action.card_id,
                    action.card_id,
                )
                for action, card in ((action, banish_cards[action.card_id]) for action in banish_actions)
            )
            banish_tiebreak = {
                card_instance_id: -rank
                for rank, (_, _, _, card_instance_id) in enumerate(ordered_banish_ids)
            }

        ranked: list[tuple[tuple[float, float, float, int, int, int], Action]] = []
        precomputed_features = None
        replay_multiplier = None
        if any(isinstance(action, (BuyCard, RecruitFreeCard)) for action in actions):
            replay_multiplier = _durable_replay_multiplier(
                observation,
                observation.players[self.player_id],
                self.acquisition_weights.durable_replay_factor,
            )
        if (
            observation.phase is Phase.BUY
            and any(isinstance(action, StopBuying) for action in actions)
            and any(isinstance(action, BuyCard) for action in actions)
        ):
            precomputed_features = {}
            admissible_actions = []
            for action in actions:
                if not isinstance(action, BuyCard):
                    admissible_actions.append(action)
                    continue
                features = features_for_action(
                    observation,
                    action,
                    self.player_id,
                    self.acquisition_weights,
                    self.constraint_weights,
                    common_condition_flags,
                    replay_multiplier,
                    self.weights,
                )
                precomputed_features[action] = features
                if self.weights.score(features) > self.weights.buy_threshold:
                    admissible_actions.append(action)
            actions = admissible_actions
        for index, action in enumerate(actions):
            features = (
                precomputed_features.get(action)
                if precomputed_features is not None
                else None
            )
            if features is None:
                features = features_for_action(
                    observation,
                    action,
                    self.player_id,
                    self.acquisition_weights,
                    self.constraint_weights,
                    common_condition_flags,
                    replay_multiplier,
                    self.weights,
                )
            if observation.phase is Phase.PLAY:
                phase_priority = 1 if isinstance(action, PassPlayPhase) else 0
            elif observation.phase is Phase.BUY:
                phase_priority = 1 if isinstance(action, StopBuying) else 0
            else:
                phase_priority = 0
            ranked.append(
                (
                    (
                        features.terminal_win,
                        features.lethal,
                        self.weights.score(features),
                        -phase_priority,
                        banish_tiebreak.get(action.card_id, 0)
                        if isinstance(action, BanishCard)
                        else 0,
                        -index,
                    ),
                    action,
                )
            )
        return max(ranked, key=lambda item: item[0])[1]

    def score_action(
        self,
        observation: GameState,
        action: Action,
    ) -> float:
        """Return an action score for tests and opt-in debug tooling."""

        return self.weights.score(
            features_for_action(
                observation,
                action,
                self.player_id,
                self.acquisition_weights,
                self.constraint_weights,
                heuristic_weights=self.weights,
            )
        )

    def features_for_action(
        self,
        observation: GameState,
        action: Action,
    ) -> ActionFeatures:
        """Expose features without adding logging or persistence by default."""

        return features_for_action(
            observation,
            action,
            self.player_id,
            self.acquisition_weights,
            self.constraint_weights,
            heuristic_weights=self.weights,
        )

    def _rank(
        self,
        observation: GameState,
        action: Action,
        features: ActionFeatures,
        index: int,
    ) -> tuple[float, float, float, int, int]:
        return (
            features.terminal_win,
            features.lethal,
            self.weights.score(features),
            -self._phase_priority(observation.phase, action),
            -index,
        )

    @staticmethod
    def _phase_priority(phase: Phase, action: Action) -> int:
        if phase is Phase.PLAY:
            return 1 if isinstance(action, PassPlayPhase) else 0
        if phase is Phase.BUY:
            return 1 if isinstance(action, StopBuying) else 0
        if phase is Phase.ATTACK:
            return 0
        return 0


__all__ = ["HeuristicPlayer"]
