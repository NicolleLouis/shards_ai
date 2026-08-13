"""Versioned deterministic PLAY policy based on card-effect priorities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shards_ai.game.actions import Action, ActivateChampion, GainMastery, PassPlayPhase, PlayCard
from shards_ai.game.cards import CardInstance, Operation
from shards_ai.game.enums import Faction, PlayerId
from shards_ai.game.errors import InvalidActionError, InvalidGameStateError
from shards_ai.game.state import GameState, PlayerState


@dataclass(frozen=True, slots=True)
class PlayCardClassification:
    """Observable classification used by the deterministic ordering."""

    priority: int
    card_id: str
    instance_id: str
    draw_amount: int
    reshuffles: bool
    offers_banish: bool
    spectra_echo: bool
    has_constraints: bool
    constraints_valid: bool
    newly_validated: bool
    constraint_labels: tuple[str, ...]


class AlgorithmicPlayPolicy:
    """Play cards according to the first deterministic PLAY ordering.

    The policy never applies an action itself. It only ranks the actions supplied
    by ``Game.legal_actions()``; the game engine remains authoritative.
    """

    policy_id = "algorithmic_play_v001"
    version = "v001"

    def __init__(self, player_id: PlayerId) -> None:
        self.player_id = player_id
        self._turn_number: int | None = None
        self._initial_validity: dict[str, bool] = {}
        self.last_classification: PlayCardClassification | None = None

    def choose_action(
        self,
        observation: GameState,
        legal_actions: Sequence[Action],
    ) -> tuple[Action, str]:
        actions = list(legal_actions)
        if not actions:
            raise InvalidActionError("Play policy received no legal actions")
        if observation.active_player is not self.player_id:
            raise InvalidGameStateError("Play policy received a non-active player state")
        self._reset_turn_if_needed(observation)
        player = observation.players[self.player_id]
        cards = {card.instance_id: card for card in player.hand}
        classifications = [
            self._classify(observation, player, cards[action.card_id])
            for action in actions
            if isinstance(action, PlayCard) and action.card_id in cards
        ]
        if classifications:
            chosen = min(
                classifications,
                key=lambda item: (item.priority, item.card_id, item.instance_id),
            )
            self.last_classification = chosen
            return PlayCard(chosen.instance_id), self._reason(chosen)

        self.last_classification = None
        # Champion activation, mastery and passing are deliberately conservative
        # fallbacks until they receive their own versioned action ordering.
        for action in actions:
            if isinstance(action, ActivateChampion):
                return action, "fallback_activate_champion"
        for action in actions:
            if isinstance(action, GainMastery):
                return action, "fallback_gain_mastery"
        for action in actions:
            if isinstance(action, PassPlayPhase):
                return action, "fallback_pass_play_phase"
        return actions[0], "fallback_first_legal_action"

    def _reset_turn_if_needed(self, observation: GameState) -> None:
        if self._turn_number == observation.turn_number:
            return
        self._turn_number = observation.turn_number
        player = observation.players[self.player_id]
        self._initial_validity = {
            card.instance_id: self._constraints_valid(observation, player, card)
            for card in player.hand
        }

    def _classify(
        self,
        observation: GameState,
        player: PlayerState,
        card: CardInstance,
    ) -> PlayCardClassification:
        operations = tuple(_all_operations(card))
        draw_amount = sum(
            operation.amount
            for operation in operations
            if operation.kind == "draw_card"
        )
        reshuffles = draw_amount > len(player.draw_pile)
        offers_banish = any(operation.kind == "offer_banish" for operation in operations)
        constraint_labels = _constraint_labels(operations)
        has_constraints = bool(constraint_labels)
        constraints_valid = self._constraints_valid(observation, player, card)
        initially_valid = self._initial_validity.get(card.instance_id, constraints_valid)
        newly_validated = has_constraints and not initially_valid and constraints_valid
        spectra_echo = (
            card.definition.faction is Faction.SPECTRA
            and any(operation.requires_echo for operation in operations)
        )

        if draw_amount and not reshuffles:
            priority = 1
        elif offers_banish:
            priority = 2
        elif spectra_echo:
            priority = 3
        elif not draw_amount and (not has_constraints or (constraints_valid and not newly_validated)):
            priority = 4
        elif draw_amount and (not has_constraints or (constraints_valid and not newly_validated)):
            priority = 5
        elif not draw_amount and newly_validated:
            priority = 6
        elif draw_amount and not constraints_valid:
            priority = 7
        else:
            priority = 8
        return PlayCardClassification(
            priority=priority,
            card_id=card.definition.card_id,
            instance_id=card.instance_id,
            draw_amount=draw_amount,
            reshuffles=reshuffles,
            offers_banish=offers_banish,
            spectra_echo=spectra_echo,
            has_constraints=has_constraints,
            constraints_valid=constraints_valid,
            newly_validated=newly_validated,
            constraint_labels=constraint_labels,
        )

    @staticmethod
    def _reason(classification: PlayCardClassification) -> str:
        return f"priority_{classification.priority}"

    @staticmethod
    def _constraints_valid(
        observation: GameState,
        player: PlayerState,
        card: CardInstance,
    ) -> bool:
        operations = tuple(_all_operations(card))
        return all(_operation_valid(observation, player, card, operation) for operation in operations)


def _all_operations(card: CardInstance) -> tuple[Operation, ...]:
    effects = [card.definition.effect]
    if card.definition.on_play_effect is not None:
        effects.append(card.definition.on_play_effect)
    operations: list[Operation] = []
    for effect in effects:
        if effect.steps:
            for step in effect.steps:
                operations.extend(step.operations)
        else:
            operations.extend(effect.operations_for_mastery(0))
    return tuple(operations)


def _constraint_labels(operations: Sequence[Operation]) -> tuple[str, ...]:
    labels: set[str] = set()
    for operation in operations:
        if operation.requires_echo:
            labels.add("spectra_echo")
        if operation.health_at_least is not None:
            labels.add("health")
        if operation.mastery_at_least is not None:
            labels.add("mastery")
        if operation.requires_domination:
            labels.add("domination")
        if operation.requires_union:
            labels.add("union")
        if operation.requires_inspiration:
            labels.add("inspiration")
    return tuple(sorted(labels))


def _operation_valid(
    observation: GameState,
    player: PlayerState,
    card: CardInstance,
    operation: Operation,
) -> bool:
    if operation.mastery_at_least is not None and player.mastery < operation.mastery_at_least:
        return False
    if operation.health_at_least is not None and player.health < operation.health_at_least:
        return False
    if operation.requires_echo and not _has_echo(player):
        return False
    if operation.requires_domination and not _has_domination(player):
        return False
    if operation.requires_inspiration and not player.champions:
        return False
    if operation.requires_union and not _has_union(player, card.definition.faction):
        return False
    return True


def _has_echo(player: PlayerState) -> bool:
    return any(card.definition.faction is Faction.SPECTRA for card in player.discard_pile)


def _has_domination(player: PlayerState) -> bool:
    factions = {
        card.definition.faction
        for zone in (player.hand, player.play_zone, player.champions)
        for card in zone
    }
    return {Faction.HOMODEUS, Faction.MAQUIS, Faction.SPECTRA} <= factions


def _has_union(player: PlayerState, faction: Faction | None) -> bool:
    if faction is None:
        return False
    return sum(
        card.definition.faction is faction
        for zone in (player.hand, player.play_zone, player.champions)
        for card in zone
    ) >= 2


__all__ = ["AlgorithmicPlayPolicy", "PlayCardClassification"]
