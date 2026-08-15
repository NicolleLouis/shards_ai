"""Player adapter that replays bounded solver branches through the atomic engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import time

from shards_ai.game import Game
from shards_ai.game.actions import Action
from shards_ai.game.enums import Phase, PlayerId
from shards_ai.game.errors import InvalidActionError
from shards_ai.game.observation import NeuralObservation

from .action_representation import ActionRepresentation, representation_for_neural_action
from .play_turn_solver import (
    MacroActionRepresentation,
    PlayTurnSolver,
    macro_representations,
    macro_representations_v2,
    macro_representations_v3,
    macro_representations_v4,
    PlayTurnCandidate,
    PlayTurnOutcomeSummary,
    PlayTurnResolution,
)


CandidateScorer = Callable[
    [Game, NeuralObservation, Sequence[PlayTurnCandidate]], int
]


@dataclass(frozen=True, slots=True)
class MacroDecisionPayload:
    """Metadata for one neural branch choice, before its atomic replay."""

    observation: NeuralObservation
    candidate_representations: tuple[MacroActionRepresentation, ...]
    chosen_candidate_index: int
    automatic_prefix: tuple[Action, ...]
    selected_atomic_trace: tuple[Action, ...]
    expansions: int
    memoized_states: int
    physical_variant_count: int


@dataclass(frozen=True, slots=True)
class AtomicDecisionPayload:
    """Metadata for one non-macro decision scored by the unified candidate model."""

    observation: NeuralObservation
    candidate_representations: tuple[MacroActionRepresentation, ...]
    chosen_candidate_index: int
    selected_action: Action


class MacroNeuralPlayer:
    """Replay solver traces and delegate strategic branch selection to a scorer.

    ``candidate_scorer`` is intentionally a small protocol boundary.  A macro neural
    model can be added later without making the engine or atomic action representation
    understand macro-actions.
    """

    observation_kind = "neural"
    observation_is_read_only = True

    def __init__(
        self,
        player_id: PlayerId,
        game: Game,
        *,
        solver: PlayTurnSolver | None = None,
        candidate_scorer: CandidateScorer | None = None,
        candidate_schema_version: int = 4,
    ) -> None:
        self.player_id = player_id
        self.game = game
        self.solver = solver or PlayTurnSolver()
        self.candidate_scorer = candidate_scorer
        if candidate_schema_version not in (1, 2, 3, 4):
            raise ValueError("candidate_schema_version must be 1, 2, 3 or 4")
        self.candidate_schema_version = candidate_schema_version
        self._pending_trace: list[Action] = []
        self.macro_decisions = 0
        self.atomic_decisions = 0
        self.atomic_replays = 0
        self.macro_inference_seconds = 0.0
        self._last_action_kind: str | None = None
        self._last_macro_decision: MacroDecisionPayload | None = None
        self._last_atomic_decision: AtomicDecisionPayload | None = None
        self._last_scored_candidates: tuple[PlayTurnCandidate, ...] = ()
        self.legacy_decision_mode = False
        self._last_candidate_scores: tuple[float, ...] = ()

    @property
    def decisions(self) -> int:
        return self.macro_decisions + self.atomic_decisions

    @property
    def total_inference_seconds(self) -> float:
        return self.macro_inference_seconds

    @property
    def last_action_kind(self) -> str | None:
        """Classify the last returned atomic action for diagnostics."""

        return self._last_action_kind

    def choose_action(
        self,
        observation: NeuralObservation,
        legal_actions: Sequence[Action],
    ) -> Action:
        actions = list(legal_actions)
        if not actions:
            raise InvalidActionError("Cannot choose an action from an empty action list")
        if not isinstance(observation, NeuralObservation):
            raise TypeError("MacroNeuralPlayer requires a NeuralObservation")
        self._last_atomic_decision = None

        if self._pending_trace:
            action = self._pending_trace.pop(0)
            if action not in actions:
                self._pending_trace.clear()
                raise InvalidActionError(f"Solver trace action is no longer legal: {action!r}")
            self.atomic_replays += 1
            self._last_action_kind = "macro_replay"
            return action

        if observation.phase != Phase.PLAY.value:
            return self._choose_unified_atomic(observation, actions)

        self._last_macro_decision = None
        self._last_atomic_decision = None
        solver_game = self.game
        if self.legacy_decision_mode:
            solver_game = self.game.clone()
            solver_game.modern_mode = False
        resolution = self.solver.resolve(solver_game)
        self._pending_trace.extend(resolution.automatic_prefix)
        if resolution.budget_boundary_reason is not None:
            if self._pending_trace:
                self._last_action_kind = "macro_replay"
                return self._pop_trace_action(actions)
            candidates = resolution.candidates or tuple(
                _atomic_candidate(observation, action, self.candidate_schema_version)
                for action in actions
            )
            return self._choose_unified_atomic(
                observation,
                actions,
                candidates=candidates,
                observation_game=resolution.observation_game,
            )

        if not resolution.candidates:
            if self._pending_trace:
                self._last_action_kind = "macro_replay"
                return self._pop_trace_action(actions)
            return self._choose_unified_atomic(observation, actions)

        # A singleton resolution is not a neural decision.  Replaying it keeps
        # the solver's deterministic contract while preventing fake training
        # records and scorer calls with no alternative to compare.
        if len(resolution.candidates) == 1:
            self._pending_trace.extend(resolution.candidates[0].atomic_trace)
            self._last_action_kind = "macro_replay"
            return self._pop_trace_action(actions)

        scored_observation = resolution.observation_game.neural_observation_for(self.player_id)
        representation_builder = {
            1: macro_representations,
            2: macro_representations_v2,
            3: macro_representations_v3,
            4: macro_representations_v4,
        }[self.candidate_schema_version]
        representations = representation_builder(scored_observation, resolution.candidates)
        index = self._choose_candidate(
            scored_observation,
            resolution,
            representations=representations,
        )
        if not 0 <= index < len(resolution.candidates):
            raise InvalidActionError(f"Macro scorer returned invalid candidate index {index}")
        self._pending_trace.extend(resolution.candidates[index].atomic_trace)
        self.macro_decisions += 1
        self._last_macro_decision = MacroDecisionPayload(
            observation=scored_observation,
            candidate_representations=tuple(representations),
            chosen_candidate_index=index,
            automatic_prefix=resolution.automatic_prefix,
            selected_atomic_trace=resolution.candidates[index].atomic_trace,
            expansions=resolution.expansions,
            memoized_states=resolution.memoized_states,
            physical_variant_count=resolution.candidates[index].physical_variant_count,
        )
        self._last_action_kind = "macro_choice"
        return self._pop_trace_action(actions)

    def pop_last_macro_decision(self) -> MacroDecisionPayload | None:
        payload = self._last_macro_decision
        self._last_macro_decision = None
        return payload

    def pop_last_atomic_decision(self) -> AtomicDecisionPayload | None:
        payload = self._last_atomic_decision
        self._last_atomic_decision = None
        return payload

    @property
    def last_scored_candidates(self) -> tuple[PlayTurnCandidate, ...]:
        return self._last_scored_candidates

    @property
    def last_candidate_scores(self) -> tuple[float, ...]:
        return self._last_candidate_scores

    @property
    def has_pending_macro_trace(self) -> bool:
        """Whether the next actions are deterministic replay, not new decisions."""

        return bool(self._pending_trace)

    def _choose_candidate(self, observation, resolution, *, representations=None) -> int:
        if self.candidate_scorer is None:
            return 0
        started = time.perf_counter()
        try:
            if representations is None:
                builder = {
                    1: macro_representations,
                    2: macro_representations_v2,
                    3: macro_representations_v3,
                    4: macro_representations_v4,
                }[self.candidate_schema_version]
                representations = builder(observation, resolution.candidates)
            candidates = tuple(
                PlayTurnCandidate(
                    candidate.atomic_trace,
                    candidate.summary,
                    representation,
                    candidate.canonical_key,
                    candidate.physical_variant_count,
                )
                for candidate, representation in zip(resolution.candidates, representations)
            )
            self._last_scored_candidates = candidates
            return int(self.candidate_scorer(resolution.observation_game, observation, candidates))
        finally:
            self.macro_inference_seconds += time.perf_counter() - started

    def _choose_unified_atomic(
        self,
        observation: NeuralObservation,
        actions: Sequence[Action],
        *,
        candidates: Sequence[PlayTurnCandidate] | None = None,
        observation_game: Game | None = None,
    ) -> Action:
        if len(actions) == 1:
            self._last_action_kind = "atomic_replay"
            return actions[0]
        if self.candidate_scorer is None:
            self._last_action_kind = "atomic_default"
            return actions[0]
        candidates = tuple(candidates or (
            _atomic_candidate(observation, action, self.candidate_schema_version)
            for action in actions
        ))
        resolution = PlayTurnResolution((), candidates, observation_game or self.game, 0, 0)
        builder = {
            1: macro_representations,
            2: macro_representations_v2,
            3: macro_representations_v3,
            4: macro_representations_v4,
        }[self.candidate_schema_version]
        representations = builder(observation, candidates)
        index = self._choose_candidate(observation, resolution, representations=representations)
        candidate_actions = tuple(candidate.atomic_trace[0] for candidate in candidates)
        if not 0 <= index < len(candidate_actions):
            raise InvalidActionError(f"Atomic scorer returned invalid candidate index {index}")
        self.atomic_decisions += 1
        self._last_atomic_decision = AtomicDecisionPayload(
            observation=observation,
            candidate_representations=tuple(
                candidate.representation for candidate in self._last_scored_candidates
            ),
            chosen_candidate_index=index,
            selected_action=candidate_actions[index],
        )
        self._last_action_kind = "atomic_choice"
        return candidate_actions[index]

    def _pop_trace_action(self, legal_actions: Sequence[Action]) -> Action:
        if not self._pending_trace:
            raise InvalidActionError("Solver returned an empty replay trace")
        action = self._pending_trace.pop(0)
        if action not in legal_actions:
            self._pending_trace.clear()
            raise InvalidActionError(f"Solver trace action is not legal: {action!r}")
        self.atomic_replays += 1
        return action


__all__ = ["AtomicDecisionPayload", "CandidateScorer", "MacroDecisionPayload", "MacroNeuralPlayer"]


def _atomic_candidate(
    observation: NeuralObservation,
    action: Action,
    schema_version: int,
) -> PlayTurnCandidate:
    """Build a length-one candidate without exposing macro consequences."""

    root = representation_for_neural_action(action, observation)
    root_without_identity = ActionRepresentation(
        action_type=root.action_type,
        phase=root.phase,
        card_definition_id=root.card_definition_id,
        river_slot=root.river_slot,
        target=root.target,
        amount=root.amount,
    )
    tactical = (0.0,) * 7
    if schema_version >= 4 and root.card_definition_id is not None:
        from .card_representation import representation_for_definition
        from .structured_v006 import tactical_features_for_representation
        from shards_ai.game import CARD_CATALOG

        tactical = tactical_features_for_representation(
            observation,
            root,
            representation_for_definition(CARD_CATALOG[root.card_definition_id]),
        )
    representation = MacroActionRepresentation(
        schema_version=schema_version,
        decision_kind="atomic",
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
        root_action=root_without_identity,
        known_card_definition_ids=(),
        played_faction_mask=(),
        played_champion_faction_mask=(),
        requires_union=bool(tactical[0]),
        union_active=bool(tactical[1]),
        requires_echo=bool(tactical[2]),
        echo_active=bool(tactical[3]),
        requires_domination=bool(tactical[4]),
        domination_active=bool(tactical[5]),
        domination_missing_count=tactical[6],
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
        canonical_key=(type(action).__name__, repr(action)),
    )
