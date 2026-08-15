"""Bounded, conservative abstraction of PLAY-phase atomic decisions.

The game engine remains the authority on legality and effects.  This module only
explores detached game clones and returns atomic traces that can be replayed through
``Game.apply``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from shards_ai.game import Game
from shards_ai.game.actions import (
    Action,
    ActivateChampion,
    BanishCard,
    ChoosePendingDecision,
    GainMastery,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    SkipBanish,
)
from shards_ai.game.enums import GameStatus, Phase
from shards_ai.game.errors import InvalidActionError

from .action_representation import ActionRepresentation, representation_for_neural_action

MAX_EXPANSIONS = 256
MAX_MEMOIZED_STATES = 128
MAX_MACRO_CANDIDATES = 16
MAX_ATOMIC_ACTIONS_PER_SEGMENT = 32


@dataclass(frozen=True, slots=True)
class DependencyDescriptor:
    """Conservative read/write summary used to identify safe fixed effects."""

    reads: frozenset[str]
    writes: frozenset[str]
    strategic: bool
    random: bool = False

    @property
    def canonicalizable(self) -> bool:
        return not self.strategic and not self.random


@dataclass(frozen=True, slots=True)
class PlayTurnOutcomeSummary:
    """Small, serializable summary exposed to a future macro scorer."""

    phase: str
    terminal_kind: str
    gems: int
    mastery: int
    power: int
    hand_size: int
    discard_size: int
    play_zone_size: int
    pending_kind: str | None
    atomic_action_count: int


@dataclass(frozen=True, slots=True)
class MacroActionRepresentation:
    """Action-conditioned representation of one solver branch.

    This is deliberately separate from ``ActionRepresentation``: a macro candidate is
    a consequence and not one engine action.  The future macro model can add learned
    fields without changing the atomic action schema.
    """

    schema_version: int
    action_type: str
    trace_action_types: tuple[str, ...]
    terminal_kind: str
    phase: str
    gems: int
    mastery: int
    power: int
    hand_size: int
    discard_size: int
    play_zone_size: int
    atomic_action_count: int
    physical_variant_count: int = 1
    # V2 only.  The instance and free-form choice identifiers are removed
    # before this representation reaches the model.
    root_action: ActionRepresentation | None = None
    decision_kind: str = "macro_play"
    # V3 known-consequence fields. Defaults preserve loading historical V1/V2
    # records; newly generated candidates always populate these fields.
    delta_gems: int = 0
    delta_mastery: int = 0
    delta_power: int = 0
    delta_active_health: int = 0
    delta_opponent_health: int = 0
    delta_hand_size: int = 0
    delta_discard_size: int = 0
    delta_play_zone_size: int = 0
    delta_active_champion_count: int = 0
    delta_opponent_champion_count: int = 0
    pending_kind: str | None = None
    pending_choice_count: int = 0
    known_card_definition_ids: tuple[str, ...] = ()
    played_faction_mask: tuple[bool, ...] = ()
    played_champion_faction_mask: tuple[bool, ...] = ()
    immediate_victory: bool = False
    requires_union: bool = False
    union_active: bool = False
    requires_echo: bool = False
    echo_active: bool = False
    requires_domination: bool = False
    domination_active: bool = False
    domination_missing_count: float = 0.0


@dataclass(frozen=True, slots=True)
class PlayTurnCandidate:
    """One complete atomic branch from a solver decision point."""

    atomic_trace: tuple[Action, ...]
    summary: PlayTurnOutcomeSummary
    representation: MacroActionRepresentation
    canonical_key: tuple[object, ...]
    physical_variant_count: int = 1


@dataclass(frozen=True, slots=True)
class PlayTurnResolution:
    """Result of resolving the safe prefix of one current game state."""

    automatic_prefix: tuple[Action, ...]
    candidates: tuple[PlayTurnCandidate, ...]
    observation_game: Game
    expansions: int
    memoized_states: int
    budget_boundary_reason: str | None = None


def _action_key(action: Action) -> tuple[object, ...]:
    values = tuple(getattr(action, field) for field in getattr(action, "__dataclass_fields__", {}))
    return (type(action).__name__, *values)


def _card_location(game: Game, instance_id: str) -> tuple[str, object] | None:
    """Return the logical zone and card for a visible card instance."""

    player = game.active
    for zone, cards in (
        ("hand", player.hand),
        ("discard_pile", player.discard_pile),
        ("play_zone", player.play_zone),
        ("champions", player.champions),
    ):
        for card in cards:
            if card.instance_id == instance_id:
                return zone, card
    for slot, card in enumerate(game.state.river):
        if card is not None and card.instance_id == instance_id:
            return f"river:{slot}", card
    return None


def equivalence_key_for_action(game: Game, action: Action) -> tuple[object, ...]:
    """Return a conservative semantic key for interchangeable legal actions.

    Instance IDs remain in the action trace. This key is only used to choose one
    deterministic representative before scoring equivalent physical actions.
    """

    if isinstance(action, (PlayCard, BanishCard, ActivateChampion)):
        instance_id = (
            action.card_id
            if isinstance(action, (PlayCard, BanishCard))
            else action.champion_id
        )
        location = _card_location(game, instance_id)
        if location is None:
            return (type(action).__name__, "unresolved", instance_id)
        zone, card = location
        flags: tuple[object, ...] = ()
        if zone == "champions":
            flags = (instance_id in game.active.activated_champion_ids,)
        return (type(action).__name__, card.definition.card_id, zone, *flags)

    if hasattr(action, "river_slot") and hasattr(action, "card_instance_id"):
        location = _card_location(game, action.card_instance_id)
        if location is not None:
            _zone, card = location
            return (type(action).__name__, card.definition.card_id, action.river_slot)

    if isinstance(action, ChoosePendingDecision):
        location = _card_location(game, action.choice_id)
        if location is not None:
            zone, card = location
            return (type(action).__name__, card.definition.card_id, zone)

    return _action_key(action)


def _representative_actions(
    game: Game,
    actions: Sequence[Action],
) -> tuple[tuple[Action, int, tuple[object, ...]], ...]:
    """Group equivalent actions and retain a deterministic physical representative."""

    groups: dict[tuple[object, ...], tuple[Action, int]] = {}
    for action in sorted(actions, key=_action_key):
        key = equivalence_key_for_action(game, action)
        if key in groups:
            representative, count = groups[key]
            groups[key] = (representative, count + 1)
        else:
            groups[key] = (action, 1)
    return tuple(
        (representative, count, key)
        for key, (representative, count) in groups.items()
    )


def _atomic_candidate_for_action(
    game: Game,
    action: Action,
    *,
    physical_variant_count: int = 1,
    canonical_key: tuple[object, ...] | None = None,
) -> PlayTurnCandidate:
    """Represent one legal action as a V4-compatible atomic candidate."""

    observation = game.neural_observation_for(game.active_player)
    root = representation_for_neural_action(action, observation)
    root_without_identity = ActionRepresentation(
        action_type=root.action_type,
        phase=root.phase,
        card_definition_id=root.card_definition_id,
        river_slot=root.river_slot,
        target=root.target,
        amount=root.amount,
    )
    representation = MacroActionRepresentation(
        schema_version=4,
        action_type=root.action_type,
        trace_action_types=(type(action).__name__,),
        terminal_kind="strategic_choice",
        phase=observation.phase,
        gems=0,
        mastery=0,
        power=0,
        hand_size=0,
        discard_size=0,
        play_zone_size=0,
        atomic_action_count=1,
        physical_variant_count=physical_variant_count,
        root_action=root_without_identity,
        decision_kind="atomic",
    )
    return PlayTurnCandidate(
        atomic_trace=(action,),
        summary=PlayTurnOutcomeSummary(
            phase=observation.phase,
            terminal_kind="strategic_choice",
            gems=0,
            mastery=0,
            power=0,
            hand_size=0,
            discard_size=0,
            play_zone_size=0,
            pending_kind=None,
            atomic_action_count=1,
        ),
        representation=representation,
        canonical_key=canonical_key or _action_key(action),
        physical_variant_count=physical_variant_count,
    )


def _atomic_candidates_for_state(game: Game) -> tuple[PlayTurnCandidate, ...]:
    return tuple(
        _atomic_candidate_for_action(game, action, physical_variant_count=count, canonical_key=key)
        for action, count, key in _representative_actions(game, game.legal_actions())
    )


def _card_operations(card) -> tuple[object, ...]:
    definition = card.definition
    operations = []
    for effect in (definition.effect, definition.on_play_effect):
        if effect is not None:
            for operation in effect.operations_for_mastery(0):
                operations.append(operation)
    return tuple(operations)


def _condition_is_active(game: Game, card, operation) -> bool:
    """Return whether one currently selected operation is already enabled."""

    player = game.active
    if operation.mastery_at_least is not None and player.mastery < operation.mastery_at_least:
        return False
    if operation.health_at_least is not None and player.health < operation.health_at_least:
        return False
    if operation.requires_union and not game._has_union_card(player, card):
        return False
    if operation.requires_echo and not game._has_echo_card(player):
        return False
    if operation.requires_domination and not game._has_domination(player, card):
        return False
    if operation.requires_inspiration and not player.champions:
        return False
    return True


def _highest_mastery_threshold(card) -> int | None:
    thresholds = [
        step.mastery_at_least
        for effect in (card.definition.effect, card.definition.on_play_effect)
        if effect is not None
        for step in effect.steps
        if step.mastery_at_least is not None
    ]
    return max(thresholds) if thresholds else None


def dependency_for_action(game: Game, action: Action) -> DependencyDescriptor:
    """Return a conservative dependency descriptor for a legal action.

    Unknown or choice-producing actions are strategic by default.  The function is
    intentionally conservative: false negatives reduce abstraction, while false
    positives could change the policy by hiding a real decision.
    """

    if isinstance(action, PlayCard):
        card = next((candidate for candidate in game.active.hand if candidate.instance_id == action.card_id), None)
        if card is None:
            raise InvalidActionError(f"Cannot describe missing card {action.card_id!r}")
        definition = card.definition
        operations = _card_operations(card)
        selected_operations = tuple(
            operation
            for effect in (definition.effect, definition.on_play_effect)
            if effect is not None
            for operation in effect.operations_for_mastery(game.active.mastery)
        )
        highest_mastery_threshold = _highest_mastery_threshold(card)
        strategic = (
            definition.is_champion
            or definition.champion_ability is not None
            or (
                highest_mastery_threshold is not None
                and game.active.mastery < highest_mastery_threshold
            )
            or any(not _condition_is_active(game, card, operation) for operation in selected_operations)
            or any(
                getattr(operation, "kind", "") in {
                    "draw_card",
                    "offer_banish",
                    "copy_effect",
                    "recruit_free_card",
                    "destroy_champion",
                    "destroy_all_champions",
                    "recover_champion",
                    "recover_mercenary",
                    "gain_power_per_discard_faction",
                    "gain_mastery",
                    "lose_mastery",
                    "deal_damage",
                    "win",
                }
                for operation in operations
            )
        )
        reads = {"hand", "play_zone"}
        writes = {"hand", "play_zone", "played_cards"}
        if definition.faction is not None:
            writes.add("played_factions")
        for operation in operations:
            kind = getattr(operation, "kind", "")
            if getattr(operation, "requires_union", False):
                reads.add("union")
            if getattr(operation, "requires_echo", False):
                reads.add("echo")
            if getattr(operation, "requires_domination", False):
                reads.add("domination")
            if getattr(operation, "requires_inspiration", False):
                reads.add("inspiration")
            if kind in {"gain_gems", "gain_mastery", "lose_mastery"}:
                writes.add("gems_mastery")
            if kind in {"gain_mastery", "lose_mastery"}:
                writes.add("mastery")
            elif kind in {"gain_power", "deal_damage"}:
                writes.add("power_health")
            elif kind == "gain_health":
                writes.add("health")
            elif kind in {
                "draw_card",
                "offer_banish",
                "recover_champion",
                "recover_mercenary",
            }:
                writes.update({"draw_discard", "echo", "union", "domination"})
        return DependencyDescriptor(
            reads=frozenset(reads),
            writes=frozenset(writes),
            strategic=strategic,
        )

    if isinstance(action, (BanishCard, SkipBanish)):
        return DependencyDescriptor(
            reads=frozenset({"hand", "discard", "pending_banishes"}),
            writes=frozenset(
                {
                    "hand", "discard", "deck", "pending_banishes",
                    "echo", "union", "domination",
                }
            ),
            strategic=True,
        )

    if isinstance(action, ActivateChampion):
        return DependencyDescriptor(
            reads=frozenset({"champions", "played_cards", "mastery", "conditions"}),
            writes=frozenset(
                {
                    "champions", "resources", "mastery", "pending",
                    "inspiration", "domination", "union", "echo",
                }
            ),
            strategic=True,
        )

    if isinstance(action, GainMastery):
        return DependencyDescriptor(
            reads=frozenset({"gems", "mastery"}),
            writes=frozenset({"gems", "mastery", "conditions"}),
            strategic=True,
        )

    if isinstance(action, (ChoosePendingDecision, RecruitFreeCard)):
        return DependencyDescriptor(
            reads=frozenset({"pending", "river", "discard", "hand"}),
            writes=frozenset(
                {
                    "pending", "river", "discard", "hand", "play_zone",
                    "draw_discard", "echo", "union", "domination",
                }
            ),
            strategic=True,
        )

    if isinstance(action, PassPlayPhase):
        return DependencyDescriptor(
            reads=frozenset({"phase"}), writes=frozenset({"phase"}), strategic=True
        )

    return DependencyDescriptor(reads=frozenset(), writes=frozenset(), strategic=True)


def _descriptors_commute(first: DependencyDescriptor, second: DependencyDescriptor) -> bool:
    if not first.canonicalizable or not second.canonicalizable:
        return False
    # All currently canonicalizable actions are fixed, additive card effects.  They
    # necessarily consume one card from the same hand and write the same play zone,
    # but those shared containers do not make their effects order-sensitive.  Any
    # operation that could inspect or alter those zones conditionally is marked
    # strategic above and never reaches this branch.
    return True


def _strategic_action_can_invalidate(
    safe: DependencyDescriptor,
    other: DependencyDescriptor,
) -> bool:
    """Prevent canonicalization across a branch that can disable its condition."""

    return other.strategic and bool(safe.reads & other.writes)


class PlayTurnSolver:
    """Conservative bounded solver for one PLAY segment."""

    def __init__(
        self,
        *,
        max_expansions: int = MAX_EXPANSIONS,
        max_memoized_states: int = MAX_MEMOIZED_STATES,
        max_macro_candidates: int = MAX_MACRO_CANDIDATES,
        max_atomic_actions_per_segment: int = MAX_ATOMIC_ACTIONS_PER_SEGMENT,
    ) -> None:
        if (
            max_expansions != MAX_EXPANSIONS
            or max_memoized_states != MAX_MEMOIZED_STATES
            or max_macro_candidates != MAX_MACRO_CANDIDATES
            or max_atomic_actions_per_segment != MAX_ATOMIC_ACTIONS_PER_SEGMENT
        ):
            raise ValueError("Solver budgets are fixed architecture constants")
        self.max_expansions = MAX_EXPANSIONS
        self.max_memoized_states = MAX_MEMOIZED_STATES
        self.max_macro_candidates = MAX_MACRO_CANDIDATES
        self.max_atomic_actions_per_segment = MAX_ATOMIC_ACTIONS_PER_SEGMENT

    def resolve(self, game: Game) -> PlayTurnResolution:
        working = game.clone()
        prefix: list[Action] = []
        expansions = 0
        memo: set[tuple[object, ...]] = set()
        budget_boundary_reason: str | None = None

        while working.state.status is GameStatus.RUNNING and working.state.phase is Phase.PLAY:
            if len(prefix) >= self.max_atomic_actions_per_segment:
                budget_boundary_reason = "max_atomic_actions_per_segment"
                break
            legal = list(working.legal_actions())
            descriptors = {
                action: dependency_for_action(working, action)
                for action in legal
            }
            safe = [action for action in legal if descriptors[action].canonicalizable]
            if not safe:
                break
            safe.sort(key=_action_key)
            action = safe[0]
            descriptor = descriptors[action]
            if any(
                _strategic_action_can_invalidate(descriptor, descriptors[other])
                for other in legal
                if other != action
            ):
                break
            if any(
                not _descriptors_commute(descriptor, descriptors[other])
                for other in safe[1:]
            ):
                break
            key = _state_key(working)
            if key in memo:
                budget_boundary_reason = "memoization_cycle"
                break
            if len(memo) >= self.max_memoized_states:
                budget_boundary_reason = "max_memoized_states"
                break
            memo.add(key)
            working.apply(action)
            prefix.append(action)

        if budget_boundary_reason is not None:
            return PlayTurnResolution(
                tuple(prefix),
                _atomic_candidates_for_state(working),
                working,
                expansions,
                len(memo),
                budget_boundary_reason,
            )

        legal = list(working.legal_actions()) if working.state.status is GameStatus.RUNNING else []
        if not legal:
            return PlayTurnResolution(tuple(prefix), (), working, expansions, len(memo))

        candidates: list[PlayTurnCandidate] = []
        for action, physical_variant_count, canonical_key in _representative_actions(working, legal):
            if expansions >= self.max_expansions or len(candidates) >= self.max_macro_candidates:
                budget_boundary_reason = "search_budget"
                break
            expansions += 1
            branch = working.clone()
            branch.apply(action)
            trace = [action]
            suffix, branch, suffix_reason = self._consume_safe_prefix(branch, trace, memo, expansions)
            expansions += suffix
            if suffix_reason is not None:
                budget_boundary_reason = suffix_reason
                break
            candidates.append(
                self._candidate(
                    branch,
                    tuple(trace),
                    working,
                    canonical_key=canonical_key,
                    physical_variant_count=physical_variant_count,
                )
            )

        if budget_boundary_reason is not None:
            return PlayTurnResolution(
                tuple(prefix),
                _atomic_candidates_for_state(working),
                working,
                expansions,
                len(memo),
                budget_boundary_reason,
            )
        return PlayTurnResolution(tuple(prefix), tuple(candidates), working, expansions, len(memo))

    def _consume_safe_prefix(
        self,
        working: Game,
        trace: list[Action],
        memo: set[tuple[object, ...]],
        expansions: int,
    ) -> tuple[int, Game, str | None]:
        consumed = 0
        while working.state.status is GameStatus.RUNNING and working.state.phase is Phase.PLAY:
            if len(trace) >= self.max_atomic_actions_per_segment:
                return consumed, working, "max_atomic_actions_per_segment"
            legal = list(working.legal_actions())
            descriptors = {
                action: dependency_for_action(working, action)
                for action in legal
            }
            safe = [action for action in legal if descriptors[action].canonicalizable]
            if not safe:
                return consumed, working, None
            safe.sort(key=_action_key)
            action = safe[0]
            descriptor = descriptors[action]
            if any(
                _strategic_action_can_invalidate(descriptor, descriptors[other])
                for other in legal
                if other != action
            ):
                return consumed, working, None
            if any(
                not _descriptors_commute(descriptor, descriptors[other])
                for other in safe[1:]
            ):
                return consumed, working, None
            key = _state_key(working)
            if key in memo:
                return consumed, working, "memoization_cycle"
            if len(memo) >= self.max_memoized_states:
                return consumed, working, "max_memoized_states"
            memo.add(key)
            working.apply(action)
            trace.append(action)
            consumed += 1
            if expansions + consumed >= self.max_expansions:
                return consumed, working, "max_expansions"
        return consumed, working, None

    @staticmethod
    def _candidate(
        game: Game,
        trace: tuple[Action, ...],
        baseline: Game,
        *,
        canonical_key: tuple[object, ...],
        physical_variant_count: int = 1,
    ) -> PlayTurnCandidate:
        player = game.active
        pending = player.pending_decision
        if game.state.status is not GameStatus.RUNNING:
            terminal_kind = "game_end"
        elif game.state.phase is not Phase.PLAY:
            terminal_kind = "phase_end"
        else:
            terminal_kind = "strategic_choice"
        summary = PlayTurnOutcomeSummary(
            phase=game.state.phase.value,
            terminal_kind=terminal_kind,
            gems=player.gems,
            mastery=player.mastery,
            power=player.power,
            hand_size=len(player.hand),
            discard_size=len(player.discard_pile),
            play_zone_size=len(player.play_zone),
            pending_kind=pending.kind if pending is not None else None,
            atomic_action_count=len(trace),
        )
        consequence_pending_kind = summary.pending_kind
        consequence_pending_count = len(pending.candidates) if pending is not None else 0
        if consequence_pending_kind is None and player.pending_free_recruit_cost is not None:
            consequence_pending_kind = "recruit_free_card"
            consequence_pending_count = len(game.legal_actions())
        elif consequence_pending_kind is None and player.pending_banishes:
            consequence_pending_kind = "banish"
            consequence_pending_count = len(game.legal_actions())
        representation = MacroActionRepresentation(
            schema_version=3,
            action_type="play_turn_branch",
            trace_action_types=tuple(type(action).__name__ for action in trace),
            terminal_kind=terminal_kind,
            phase=summary.phase,
            gems=summary.gems,
            mastery=summary.mastery,
            power=summary.power,
            hand_size=summary.hand_size,
            discard_size=summary.discard_size,
            play_zone_size=summary.play_zone_size,
            atomic_action_count=summary.atomic_action_count,
            physical_variant_count=physical_variant_count,
            delta_gems=player.gems - baseline.active.gems,
            delta_mastery=player.mastery - baseline.active.mastery,
            delta_power=player.power - baseline.active.power,
            delta_active_health=player.health - baseline.active.health,
            delta_opponent_health=game.opponent.health - baseline.opponent.health,
            delta_hand_size=len(player.hand) - len(baseline.active.hand),
            delta_discard_size=len(player.discard_pile) - len(baseline.active.discard_pile),
            delta_play_zone_size=len(player.play_zone) - len(baseline.active.play_zone),
            delta_active_champion_count=len(player.champions) - len(baseline.active.champions),
            delta_opponent_champion_count=len(game.opponent.champions) - len(baseline.opponent.champions),
            pending_kind=consequence_pending_kind,
            pending_choice_count=consequence_pending_count,
            known_card_definition_ids=_known_card_definition_ids(baseline, trace),
            played_faction_mask=tuple(
                game.neural_observation_for(game.active_player).active_player.played_faction_mask
            ),
            played_champion_faction_mask=tuple(
                game.neural_observation_for(game.active_player).active_player.played_champion_faction_mask
            ),
            immediate_victory=(
                game.state.status is GameStatus.FINISHED
                and game.state.winner == game.active_player
            ),
        )
        return PlayTurnCandidate(
            trace,
            summary,
            representation,
            canonical_key,
            physical_variant_count,
        )


def _known_card_definition_ids(game: Game, trace: Sequence[Action]) -> tuple[str, ...]:
    """Return only card definitions resolvable from the pre-decision view.

    A branch clone knows the real draw pile, but that knowledge is deliberately not
    consulted here.  Action representations resolve cards against the masked
    observation; actions targeting a card revealed only by a draw are therefore
    ignored rather than leaking its definition into the candidate tensor.
    """

    observation = game.neural_observation_for(game.active_player)
    definitions: set[str] = set()
    for action in trace:
        try:
            representation = representation_for_neural_action(action, observation)
        except (ValueError, KeyError):
            continue
        if representation.card_definition_id is not None:
            definitions.add(representation.card_definition_id)
    return tuple(sorted(definitions))


def _state_key(game: Game) -> tuple[object, ...]:
    """Stable state key for safe local memoization.

    The key intentionally includes ordered zones and instance IDs.  It therefore does
    not claim that two physically different shuffled decks are equivalent.
    """

    def card_key(card) -> tuple[str, str]:
        return card.instance_id, card.definition.card_id

    player = game.active
    opponent = game.opponent
    return (
        game.state.active_player.value,
        game.state.phase.value,
        game.state.status.value,
        game.state.turn_number,
        tuple(
            (
                current.player_id.value,
                current.health,
                current.gems,
                current.mastery,
                current.mastery_action_used,
                current.power,
                tuple(card_key(card) for card in current.hand),
                tuple(card_key(card) for card in current.draw_pile),
                tuple(card_key(card) for card in current.discard_pile),
                tuple(card_key(card) for card in current.play_zone),
                tuple(card_key(card) for card in current.champions),
                tuple(sorted(current.activated_champion_ids)),
                tuple(sorted(current.played_card_ids_this_turn)),
                current.pending_decision,
                current.pending_banishes,
                current.pending_free_recruit_cost,
                current.pending_free_recruit_to_hand,
            )
            for current in (player, opponent)
        ),
        tuple(card_key(card) if card is not None else None for card in game.state.river),
        tuple(card_key(card) for card in game.state.central_deck),
    )


def macro_representations(
    observation,
    candidates: Sequence[PlayTurnCandidate],
) -> tuple[MacroActionRepresentation, ...]:
    """Return candidate representations while keeping a stable positional mapping."""

    del observation
    return tuple(candidate.representation for candidate in candidates)


def macro_representations_v2(
    observation,
    candidates: Sequence[PlayTurnCandidate],
) -> tuple[MacroActionRepresentation, ...]:
    """Build V2 candidates with a stable, information-masked root action."""

    representations = []
    for candidate in candidates:
        if not candidate.atomic_trace:
            raise InvalidActionError("A macro candidate must contain a root action")
        root = representation_for_neural_action(
            candidate.atomic_trace[0],
            observation,
        )
        representations.append(
            MacroActionRepresentation(
                schema_version=2,
                action_type=candidate.representation.action_type,
                trace_action_types=candidate.representation.trace_action_types,
                terminal_kind=candidate.representation.terminal_kind,
                phase=candidate.representation.phase,
                gems=candidate.representation.gems,
                mastery=candidate.representation.mastery,
                power=candidate.representation.power,
                hand_size=candidate.representation.hand_size,
                discard_size=candidate.representation.discard_size,
            play_zone_size=candidate.representation.play_zone_size,
            atomic_action_count=candidate.representation.atomic_action_count,
            physical_variant_count=candidate.physical_variant_count,
            root_action=ActionRepresentation(
                    action_type=root.action_type,
                    phase=root.phase,
                    card_definition_id=root.card_definition_id,
                    river_slot=root.river_slot,
                    target=root.target,
                amount=root.amount,
            ),
            decision_kind=candidate.representation.decision_kind,
                delta_gems=candidate.representation.delta_gems,
                delta_mastery=candidate.representation.delta_mastery,
                delta_power=candidate.representation.delta_power,
                delta_active_health=candidate.representation.delta_active_health,
                delta_opponent_health=candidate.representation.delta_opponent_health,
                delta_hand_size=candidate.representation.delta_hand_size,
                delta_discard_size=candidate.representation.delta_discard_size,
                delta_play_zone_size=candidate.representation.delta_play_zone_size,
                delta_active_champion_count=candidate.representation.delta_active_champion_count,
                delta_opponent_champion_count=candidate.representation.delta_opponent_champion_count,
                pending_kind=candidate.representation.pending_kind,
                pending_choice_count=candidate.representation.pending_choice_count,
                known_card_definition_ids=candidate.representation.known_card_definition_ids,
                played_faction_mask=candidate.representation.played_faction_mask,
                played_champion_faction_mask=candidate.representation.played_champion_faction_mask,
                immediate_victory=candidate.representation.immediate_victory,
            )
        )
    return tuple(representations)


def macro_representations_v3(
    observation,
    candidates: Sequence[PlayTurnCandidate],
) -> tuple[MacroActionRepresentation, ...]:
    """Build the default V3 candidate contract with known consequences."""

    representations = macro_representations_v2(observation, candidates)
    return tuple(
        MacroActionRepresentation(
            schema_version=3,
            action_type=item.action_type,
            trace_action_types=item.trace_action_types,
            terminal_kind=item.terminal_kind,
            phase=item.phase,
            gems=item.gems,
            mastery=item.mastery,
            power=item.power,
            hand_size=item.hand_size,
            discard_size=item.discard_size,
            play_zone_size=item.play_zone_size,
            atomic_action_count=item.atomic_action_count,
            physical_variant_count=item.physical_variant_count,
            root_action=item.root_action,
            decision_kind=item.decision_kind,
            delta_gems=item.delta_gems,
            delta_mastery=item.delta_mastery,
            delta_power=item.delta_power,
            delta_active_health=item.delta_active_health,
            delta_opponent_health=item.delta_opponent_health,
            delta_hand_size=item.delta_hand_size,
            delta_discard_size=item.delta_discard_size,
            delta_play_zone_size=item.delta_play_zone_size,
            delta_active_champion_count=item.delta_active_champion_count,
            delta_opponent_champion_count=item.delta_opponent_champion_count,
            pending_kind=item.pending_kind,
            pending_choice_count=item.pending_choice_count,
            known_card_definition_ids=item.known_card_definition_ids,
            played_faction_mask=item.played_faction_mask,
            played_champion_faction_mask=item.played_champion_faction_mask,
            immediate_victory=item.immediate_victory,
            requires_union=item.requires_union,
            union_active=item.union_active,
            requires_echo=item.requires_echo,
            echo_active=item.echo_active,
            requires_domination=item.requires_domination,
            domination_active=item.domination_active,
            domination_missing_count=item.domination_missing_count,
        )
        for item in representations
    )


def macro_representations_v4(
    observation,
    candidates: Sequence[PlayTurnCandidate],
) -> tuple[MacroActionRepresentation, ...]:
    """Build V3 candidates with action-conditioned V6 tactical features."""

    from shards_ai.ai.card_representation import representation_for_definition
    from shards_ai.ai.structured_v006 import tactical_features_for_representation
    from shards_ai.game import CARD_CATALOG

    base = macro_representations_v3(observation, candidates)
    enriched = []
    for item, candidate in zip(base, candidates):
        tactical = (0.0,) * 7
        if candidate.atomic_trace:
            root = representation_for_neural_action(candidate.atomic_trace[0], observation)
            if root.action_type == "play_card" and root.card_definition_id is not None:
                tactical = tactical_features_for_representation(
                    observation,
                    root,
                    representation_for_definition(CARD_CATALOG[root.card_definition_id]),
                )
        enriched.append(MacroActionRepresentation(
            schema_version=4,
            action_type=item.action_type,
            trace_action_types=item.trace_action_types,
            terminal_kind=item.terminal_kind,
            phase=item.phase,
            gems=item.gems,
            mastery=item.mastery,
            power=item.power,
            hand_size=item.hand_size,
            discard_size=item.discard_size,
            play_zone_size=item.play_zone_size,
            atomic_action_count=item.atomic_action_count,
            physical_variant_count=item.physical_variant_count,
            root_action=item.root_action,
            decision_kind=item.decision_kind,
            delta_gems=item.delta_gems,
            delta_mastery=item.delta_mastery,
            delta_power=item.delta_power,
            delta_active_health=item.delta_active_health,
            delta_opponent_health=item.delta_opponent_health,
            delta_hand_size=item.delta_hand_size,
            delta_discard_size=item.delta_discard_size,
            delta_play_zone_size=item.delta_play_zone_size,
            delta_active_champion_count=item.delta_active_champion_count,
            delta_opponent_champion_count=item.delta_opponent_champion_count,
            pending_kind=item.pending_kind,
            pending_choice_count=item.pending_choice_count,
            known_card_definition_ids=item.known_card_definition_ids,
            played_faction_mask=item.played_faction_mask,
            played_champion_faction_mask=item.played_champion_faction_mask,
            immediate_victory=item.immediate_victory,
            requires_union=bool(tactical[0]),
            union_active=bool(tactical[1]),
            requires_echo=bool(tactical[2]),
            echo_active=bool(tactical[3]),
            requires_domination=bool(tactical[4]),
            domination_active=bool(tactical[5]),
            domination_missing_count=tactical[6],
        ))
    return tuple(enriched)


def atomic_candidates_for_actions(
    game: Game,
    actions: Sequence[Action],
) -> tuple[PlayTurnCandidate, ...]:
    """Build canonical atomic candidates for a caller-owned legal action subset."""

    return tuple(
        _atomic_candidate_for_action(game, action, physical_variant_count=count, canonical_key=key)
        for action, count, key in _representative_actions(game, actions)
    )


__all__ = [
    "DependencyDescriptor",
    "MacroActionRepresentation",
    "MAX_ATOMIC_ACTIONS_PER_SEGMENT",
    "MAX_EXPANSIONS",
    "MAX_MACRO_CANDIDATES",
    "MAX_MEMOIZED_STATES",
    "PlayTurnCandidate",
    "PlayTurnOutcomeSummary",
    "PlayTurnResolution",
    "PlayTurnSolver",
    "atomic_candidates_for_actions",
    "dependency_for_action",
    "macro_representations",
    "macro_representations_v2",
    "macro_representations_v3",
    "macro_representations_v4",
]
