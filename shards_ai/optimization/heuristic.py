"""Reproducible coordinate search for HeuristicPlayer weights."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from statistics import NormalDist

from shards_ai.ai import (
    CardAcquisitionWeights,
    CardConstraintWeights,
    HeuristicPlayer,
    HeuristicWeights,
    RandomPlayer,
    StateRewardWeights,
)
from shards_ai.analysis.reward_shaping import RewardShapingTracker
from shards_ai.ai.heuristic_profiles import HeuristicProfile, save_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


OPTIMIZABLE_FIELDS = tuple(
    field.name
    for field in fields(HeuristicWeights)
    if field.name not in {"lethal", "terminal_win"}
)

DEFAULT_ACTIVE_FIELDS = (
    "gems_produced",
    "power_produced",
    "damage_value",
    "card_draw",
    "health_gained",
    "mastery_gained",
    "champion_value",
)
DEFAULT_JOINT_PAIRS = (
    ("gems_produced", "power_produced"),
    ("power_produced", "damage_value"),
    ("gems_produced", "card_draw"),
    ("health_gained", "mastery_gained"),
    ("champion_value", "damage_value"),
)


@dataclass(frozen=True, slots=True)
class CoefficientBound:
    minimum: float
    maximum: float
    step: float

    def __post_init__(self) -> None:
        if self.minimum > self.maximum or self.step <= 0:
            raise ValueError("Invalid coefficient bound")


ACQUISITION_OPTIMIZABLE_FIELDS = tuple(field.name for field in fields(CardAcquisitionWeights))
DEFAULT_ACQUISITION_ACTIVE_FIELDS = ACQUISITION_OPTIMIZABLE_FIELDS
ACQUISITION_BOUNDS: Mapping[str, CoefficientBound] = {
    "gems_produced": CoefficientBound(0.0, 4.0, 0.5),
    "power_produced": CoefficientBound(0.0, 5.0, 0.5),
    "mastery_gained": CoefficientBound(0.0, 4.0, 0.5),
    "health_gained": CoefficientBound(0.0, 3.0, 0.5),
    "card_draw": CoefficientBound(0.0, 5.0, 0.5),
    "deck_thinning": CoefficientBound(0.0, 3.0, 0.5),
    "target_denial": CoefficientBound(0.0, 3.0, 0.5),
    "banish_threshold": CoefficientBound(0.0, 8.0, 0.5),
    "durable_replay_factor": CoefficientBound(0.0, 3.0, 0.5),
}

CONSTRAINT_OPTIMIZABLE_FIELDS = tuple(field.name for field in fields(CardConstraintWeights))
CONSTRAINT_BOUNDS: Mapping[str, CoefficientBound] = {
    "mastery": CoefficientBound(0.0, 3.0, 0.25),
    "health": CoefficientBound(0.0, 3.0, 0.25),
    "inspiration": CoefficientBound(0.0, 3.0, 0.25),
    "echo": CoefficientBound(0.0, 3.0, 0.25),
    "union": CoefficientBound(0.0, 3.0, 0.25),
    "domination": CoefficientBound(0.0, 3.0, 0.25),
}


DEFAULT_BOUNDS: Mapping[str, CoefficientBound] = {
    "cost_paid": CoefficientBound(-4.0, 0.0, 0.75),
    "gems_produced": CoefficientBound(0.0, 5.0, 1.0),
    "power_produced": CoefficientBound(0.0, 6.0, 1.0),
    "mastery_gained": CoefficientBound(0.0, 5.0, 1.0),
    "health_gained": CoefficientBound(0.0, 3.0, 0.5),
    "card_draw": CoefficientBound(0.0, 6.0, 1.0),
    "shield_value": CoefficientBound(0.0, 3.0, 0.5),
    "deck_thinning": CoefficientBound(0.0, 4.0, 0.75),
    "card_acquisition_value": CoefficientBound(0.0, 5.0, 1.0),
    "champion_value": CoefficientBound(0.0, 6.0, 1.0),
    "target_denial": CoefficientBound(0.0, 5.0, 1.0),
    "damage_value": CoefficientBound(0.0, 6.0, 1.0),
    "constraint_penalty": CoefficientBound(-4.0, 0.0, 0.75),
    "phase_progress": CoefficientBound(0.0, 1.0, 0.2),
    "action_penalty": CoefficientBound(-4.0, 0.0, 0.75),
    "health_advantage_delta": CoefficientBound(-6.0, 6.0, 1.0),
    "mastery_advantage_delta": CoefficientBound(-6.0, 6.0, 1.0),
    "opponent_threat_delta": CoefficientBound(-6.0, 6.0, 1.0),
    "self_threat_delta": CoefficientBound(-6.0, 6.0, 1.0),
    "purchase_opportunity_cost": CoefficientBound(-6.0, 6.0, 1.0),
    "mastery_threshold_value": CoefficientBound(0.0, 6.0, 1.0),
    "buy_threshold": CoefficientBound(0.0, 2.0, 0.25),
}


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    # Historical field name kept for API compatibility; this is a CPU-time budget.
    duration_seconds: float = 60.0
    batch_seconds: float = 20.0
    games_per_candidate: int = 100
    seed: int = 1
    promotion_threshold: float = 0.90
    minimum_games_for_promotion: int = 500
    mix_random_ratio: float = 0.50
    max_actions: int = GameRunner.DEFAULT_MAX_ACTIONS
    max_turns: int | None = None
    frozen_fields: frozenset[str] = frozenset()
    state_reward_weights: StateRewardWeights = field(default_factory=StateRewardWeights)
    initial_games: int | None = None
    racing_games: int | None = None
    validation_games: int | None = None
    test_games: int | None = None
    active_fields: tuple[str, ...] = DEFAULT_ACTIVE_FIELDS
    confidence_level: float = 0.95
    minimum_gain: float = 0.01
    racing_top_k: int = 5
    racing_finalists: int = 2
    track_zero_alpha_shaping: bool = False
    shaping_observer_detached: bool = False
    trusted_live_observations: bool = True
    active_acquisition_fields: tuple[str, ...] = DEFAULT_ACQUISITION_ACTIVE_FIELDS
    acquisition_only: bool = False
    active_constraint_fields: tuple[str, ...] = ()
    constraints_only: bool = False
    combined: bool = False

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0 or self.batch_seconds <= 0:
            raise ValueError("Optimization durations must be positive")
        if self.games_per_candidate <= 0 or self.minimum_games_for_promotion <= 0:
            raise ValueError("Optimization game counts must be positive")
        if not 0 < self.mix_random_ratio <= 1:
            raise ValueError("mix_random_ratio must be in (0, 1]")
        if self.max_actions <= 0 or (self.max_turns is not None and self.max_turns <= 0):
            raise ValueError("Game limits must be positive")
        for value in (self.initial_games, self.racing_games, self.validation_games, self.test_games):
            if value is not None and value <= 0:
                raise ValueError("Racing game counts must be positive")
        if not 0 < self.confidence_level < 1 or self.minimum_gain < 0:
            raise ValueError("Invalid validation thresholds")
        if self.combined and (self.acquisition_only or self.constraints_only):
            raise ValueError("combined cannot be used with a specialized optimization mode")
        if self.racing_top_k <= 0 or self.racing_finalists <= 0:
            raise ValueError("Racing population sizes must be positive")
        unknown = set(self.active_fields) - set(OPTIMIZABLE_FIELDS)
        if unknown:
            raise ValueError(f"Unknown active fields: {sorted(unknown)}")
        unknown_acquisition = set(self.active_acquisition_fields) - set(ACQUISITION_OPTIMIZABLE_FIELDS)
        if unknown_acquisition:
            raise ValueError(f"Unknown acquisition fields: {sorted(unknown_acquisition)}")
        unknown_constraints = set(self.active_constraint_fields) - set(CONSTRAINT_OPTIMIZABLE_FIELDS)
        if unknown_constraints:
            raise ValueError(f"Unknown constraint fields: {sorted(unknown_constraints)}")

    @property
    def initial_game_count(self) -> int:
        return self.initial_games or self.games_per_candidate

    @property
    def racing_game_count(self) -> int:
        return self.racing_games or max(self.games_per_candidate, self.games_per_candidate * 5)

    @property
    def validation_game_count(self) -> int:
        return self.validation_games or max(self.minimum_games_for_promotion, self.games_per_candidate)

    @property
    def test_game_count(self) -> int:
        return self.test_games or self.validation_game_count


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    errors: int
    utility: float
    objective: float
    shaping_alpha: float
    mean_shaping: float
    final_potential: float
    complete: bool
    outcomes: tuple[float, ...] = field(default=(), repr=False)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass(frozen=True, slots=True)
class CandidateResult:
    weights: Mapping[str, float]
    evaluations: tuple[EvaluationResult, ...]
    aggregate_utility: float
    aggregate_objective: float
    accepted: bool
    stage: str = "initial"
    acquisition_weights: Mapping[str, float] = field(default_factory=dict)
    constraint_weights: Mapping[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all(evaluation.complete for evaluation in self.evaluations)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    root_seed: int
    elapsed_seconds: float
    initial_profile_id: str
    final_profile_id: str
    mixed_phase_started: bool
    accepted_profile: HeuristicProfile
    history: tuple[CandidateResult, ...]
    validation: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "root_seed": self.root_seed,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "initial_profile_id": self.initial_profile_id,
            "final_profile_id": self.final_profile_id,
            "mixed_phase_started": self.mixed_phase_started,
            "validation": dict(self.validation),
            "accepted_profile": {
                "profile_id": self.accepted_profile.profile_id,
                "parent_profile_id": self.accepted_profile.parent_profile_id,
                "weights": asdict(self.accepted_profile.weights),
                "card_acquisition_weights": asdict(self.accepted_profile.card_acquisition_weights),
                "constraint_weights": asdict(self.accepted_profile.constraint_weights),
                "metadata": dict(self.accepted_profile.metadata or {}),
            },
            "history": [
                {
                    "weights": dict(candidate.weights),
                    "acquisition_weights": dict(candidate.acquisition_weights),
                    "constraint_weights": dict(candidate.constraint_weights),
                    "aggregate_utility": candidate.aggregate_utility,
                    "aggregate_objective": candidate.aggregate_objective,
                    "accepted": candidate.accepted,
                    "stage": candidate.stage,
                    "evaluations": [
                        {
                            key: value
                            for key, value in asdict(evaluation).items()
                            if key != "outcomes"
                        }
                        for evaluation in candidate.evaluations
                    ],
                }
                for candidate in self.history
            ],
        }


def _checkpoint_profile(profile: HeuristicProfile) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "parent_profile_id": profile.parent_profile_id,
        "weights": asdict(profile.weights),
        "card_acquisition_weights": asdict(profile.card_acquisition_weights),
        "constraint_weights": asdict(profile.constraint_weights),
        "metadata": dict(profile.metadata or {}),
    }


def load_optimization_checkpoint(path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(path)
    with checkpoint_path.open(encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if checkpoint.get("schema_version") != 1:
        raise ValueError(f"Unsupported optimization checkpoint: {checkpoint_path}")
    if checkpoint.get("mode") != "combined":
        raise ValueError("Only combined optimization checkpoints are currently supported")
    return checkpoint


def _write_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_payload(
    *,
    initial_profile: HeuristicProfile,
    reference_profile: HeuristicProfile,
    config: OptimizationConfig,
    current: tuple[HeuristicWeights, CardAcquisitionWeights, CardConstraintWeights],
    step_scales: Mapping[str, float],
    batch: int,
    mixed: bool,
    compute_seconds_consumed: float,
    consecutive_failed_batches: int,
    stop_reason: str | None = None,
    phase: str = "search",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "combined",
        "root_seed": config.seed,
        "initial_profile": _checkpoint_profile(initial_profile),
        "reference_profile": _checkpoint_profile(reference_profile),
        "config": {
            "initial_games": config.initial_games,
            "racing_games": config.racing_games,
            "validation_games": config.validation_games,
            "test_games": config.test_games,
            "active_fields": list(config.active_fields),
            "active_acquisition_fields": list(config.active_acquisition_fields),
            "active_constraint_fields": list(config.active_constraint_fields),
            "racing_top_k": config.racing_top_k,
            "racing_finalists": config.racing_finalists,
        },
        "current": {
            "weights": asdict(current[0]),
            "card_acquisition_weights": asdict(current[1]),
            "constraint_weights": asdict(current[2]),
        },
        "step_scales": dict(step_scales),
        "next_batch": batch,
        "mixed": mixed,
        "compute_seconds_consumed": round(compute_seconds_consumed, 6),
        "compute_seconds_target": config.duration_seconds,
        "consecutive_failed_batches": consecutive_failed_batches,
        "stop_reason": stop_reason,
        "phase": phase,
    }


def _seed(root_seed: int, batch: int, game_index: int, opponent: str) -> int:
    payload = f"shards-ai-heuristic-optimization:{root_seed}:{batch}:{game_index}:{opponent}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _candidate_weights(weights: HeuristicWeights, name: str, value: float) -> HeuristicWeights:
    return replace(weights, **{name: value})


def _neighbours(
    weights: HeuristicWeights,
    bounds: Mapping[str, CoefficientBound],
    frozen_fields: frozenset[str],
    step_scales: Mapping[str, float],
    fields_to_search: Sequence[str] = OPTIMIZABLE_FIELDS,
) -> Iterable[HeuristicWeights]:
    seen: set[HeuristicWeights] = set()
    for name in fields_to_search:
        if name in frozen_fields:
            continue
        bound = bounds[name]
        current = getattr(weights, name)
        step = bound.step * step_scales.get(name, 1.0)
        for value in (current - step, current + step):
            candidate = _candidate_weights(weights, name, max(bound.minimum, min(bound.maximum, value)))
            if candidate != weights and candidate not in seen:
                seen.add(candidate)
                yield candidate


def _candidate_pool(
    weights: HeuristicWeights,
    bounds: Mapping[str, CoefficientBound],
    frozen_fields: frozenset[str],
    fields_to_search: Sequence[str],
    step_scales: Mapping[str, float],
    joint_pairs: Sequence[tuple[str, str]] = DEFAULT_JOINT_PAIRS,
) -> tuple[HeuristicWeights, ...]:
    candidates = list(_neighbours(weights, bounds, frozen_fields, step_scales, fields_to_search))
    available_fields = set(fields_to_search) - frozen_fields
    selected_pairs = [
        pair for pair in joint_pairs if pair[0] in available_fields and pair[1] in available_fields
    ]
    if not selected_pairs and len(fields_to_search) >= 2:
        selected_pairs = list(zip(fields_to_search, fields_to_search[1:], strict=False))
    for left_name, right_name in selected_pairs:
        if left_name not in available_fields or right_name not in available_fields:
            continue
        left_bound = bounds[left_name]
        right_bound = bounds[right_name]
        left_step = left_bound.step * step_scales.get(left_name, 1.0)
        right_step = right_bound.step * step_scales.get(right_name, 1.0)
        for left_sign, right_sign in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            left_value = max(left_bound.minimum, min(left_bound.maximum, getattr(weights, left_name) + left_sign * left_step))
            right_value = max(right_bound.minimum, min(right_bound.maximum, getattr(weights, right_name) + right_sign * right_step))
            candidate = replace(weights, **{left_name: left_value, right_name: right_value})
            if candidate != weights and candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _run_game(
    candidate: HeuristicWeights,
    opponent: str,
    previous: HeuristicWeights | None,
    seed: int,
    config: OptimizationConfig,
    game_index: int,
    state_reward_weights: StateRewardWeights,
    track_shaping: bool = True,
    acquisition_weights: CardAcquisitionWeights | None = None,
    previous_acquisition_weights: CardAcquisitionWeights | None = None,
    constraint_weights: CardConstraintWeights | None = None,
    previous_constraint_weights: CardConstraintWeights | None = None,
) -> tuple[GameStatus, PlayerId | None, RewardShapingTracker | None]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    candidate_id = PlayerId.PLAYER_1 if game_index % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = candidate_id.opponent
    players: dict[PlayerId, object] = {}
    players[candidate_id] = HeuristicPlayer(
        candidate_id, candidate, acquisition_weights, constraint_weights
    )
    tracker = RewardShapingTracker(candidate_id, state_reward_weights) if track_shaping else None
    if opponent == "random":
        players[opponent_id] = RandomPlayer(opponent_id, root_rng.derive(f"player-{opponent_id.value}"))
    elif opponent == "previous" and previous is not None:
        players[opponent_id] = HeuristicPlayer(
            opponent_id,
            previous,
            previous_acquisition_weights,
            previous_constraint_weights,
        )
    else:
        raise ValueError(f"Unsupported opponent: {opponent}")
    runner = GameRunner(
        game,
        players,  # type: ignore[arg-type]
        max_actions=config.max_actions,
        max_turns=config.max_turns,
    )
    state = runner.run(
        transition_observer=tracker.observe if tracker is not None else None,
        observer_receives_detached_state=config.shaping_observer_detached,
        players_receive_detached_observation=not config.trusted_live_observations,
        observer_before_state_factory=game.shaping_observation_for if track_shaping else None,
    )
    return state.status, state.winner, tracker


def _evaluate(
    weights: HeuristicWeights,
    opponent: str,
    previous: HeuristicWeights | None,
    config: OptimizationConfig,
    batch: int,
    deadline: float,
    clock: Callable[[], float],
    shaping_alpha: float,
    game_count: int | None = None,
    acquisition_weights: CardAcquisitionWeights | None = None,
    previous_acquisition_weights: CardAcquisitionWeights | None = None,
    constraint_weights: CardConstraintWeights | None = None,
    previous_constraint_weights: CardConstraintWeights | None = None,
) -> EvaluationResult:
    games = wins = draws = losses = errors = 0
    total_shaping = total_final_potential = 0.0
    outcomes: list[float] = []
    target_games = game_count or config.games_per_candidate
    completed_target = True
    for game_index in range(target_games):
        if clock() >= deadline:
            completed_target = False
            break
        games += 1
        try:
            status, winner, tracker = _run_game(
                weights,
                opponent,
                previous,
                _seed(config.seed, batch, game_index, opponent),
                config,
                game_index,
                config.state_reward_weights,
                shaping_alpha != 0.0 or config.track_zero_alpha_shaping,
                acquisition_weights,
                previous_acquisition_weights,
                constraint_weights,
                previous_constraint_weights,
            )
        except Exception:
            errors += 1
            outcomes.append(0.0)
            continue
        if tracker is not None:
            total_shaping += tracker.total_reward
            total_final_potential += tracker.final_potential
        if status is GameStatus.DRAW or winner is None:
            draws += 1
            outcomes.append(0.5)
        elif winner is (PlayerId.PLAYER_1 if game_index % 2 == 0 else PlayerId.PLAYER_2):
            wins += 1
            outcomes.append(1.0)
        else:
            losses += 1
            outcomes.append(0.0)
    utility = (wins + 0.5 * draws) / games if games else 0.0
    mean_shaping = total_shaping / games if games else 0.0
    return EvaluationResult(
        opponent,
        games,
        wins,
        draws,
        losses,
        errors,
        utility,
        utility + shaping_alpha * mean_shaping,
        shaping_alpha,
        mean_shaping,
        total_final_potential / games if games else 0.0,
        completed_target and games == target_games and errors == 0,
        tuple(outcomes),
    )


def _evaluate_candidate(
    weights: HeuristicWeights,
    previous: HeuristicWeights | None,
    mixed: bool,
    config: OptimizationConfig,
    batch: int,
    deadline: float,
    clock: Callable[[], float],
    shaping_alpha: float,
    game_count: int | None = None,
    stage: str = "initial",
    acquisition_weights: CardAcquisitionWeights | None = None,
    previous_acquisition_weights: CardAcquisitionWeights | None = None,
    constraint_weights: CardConstraintWeights | None = None,
    previous_constraint_weights: CardConstraintWeights | None = None,
) -> CandidateResult:
    evaluations = [
        _evaluate(
            weights,
            "random",
            previous,
            config,
            batch,
            deadline,
            clock,
            shaping_alpha,
            game_count,
            acquisition_weights,
            previous_acquisition_weights,
            constraint_weights,
            previous_constraint_weights,
        )
    ]
    if mixed and clock() < deadline:
        evaluations.append(
            _evaluate(
                weights,
                "previous",
                previous,
                config,
                batch,
                deadline,
                clock,
                shaping_alpha,
                game_count,
                acquisition_weights,
                previous_acquisition_weights,
                constraint_weights,
                previous_constraint_weights,
            )
        )
    if mixed and len(evaluations) == 2:
        aggregate_utility = (
            config.mix_random_ratio * evaluations[0].utility
            + (1.0 - config.mix_random_ratio) * evaluations[1].utility
        )
        aggregate_objective = (
            config.mix_random_ratio * evaluations[0].objective
            + (1.0 - config.mix_random_ratio) * evaluations[1].objective
        )
    else:
        aggregate_utility = evaluations[0].utility
        aggregate_objective = evaluations[0].objective
    return CandidateResult(
        asdict(weights),
        tuple(evaluations),
        aggregate_utility,
        aggregate_objective,
        False,
        stage,
        asdict(acquisition_weights or CardAcquisitionWeights()),
        asdict(constraint_weights or CardConstraintWeights()),
    )


def _paired_confidence_interval(
    candidate: EvaluationResult,
    reference: EvaluationResult,
    confidence_level: float,
) -> tuple[float, float, float]:
    differences = [
        candidate_outcome - reference_outcome
        for candidate_outcome, reference_outcome in zip(
            candidate.outcomes, reference.outcomes, strict=False
        )
    ]
    if not differences:
        return 0.0, float("-inf"), float("inf")
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        return mean, mean, mean
    variance = sum((difference - mean) ** 2 for difference in differences) / (len(differences) - 1)
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    margin = z_score * math.sqrt(variance / len(differences))
    return mean, mean - margin, mean + margin


def _validate_candidate(
    candidate: HeuristicWeights,
    reference: HeuristicWeights,
    mixed: bool,
    config: OptimizationConfig,
    candidate_acquisition: CardAcquisitionWeights | None = None,
    reference_acquisition: CardAcquisitionWeights | None = None,
    candidate_constraints: CardConstraintWeights | None = None,
    reference_constraints: CardConstraintWeights | None = None,
) -> dict[str, object]:
    validation: dict[str, object] = {
        "games_per_opponent": config.validation_game_count,
        "confidence_level": config.confidence_level,
        "minimum_gain": config.minimum_gain,
        "validation_rule": "positive_confidence_lower_bound_vs_previous",
        "adversaries": {},
    }
    adversaries = ("random", "previous") if mixed else ("random",)
    accepted = True
    for index, opponent in enumerate(adversaries):
        candidate_result = _evaluate(
            candidate,
            opponent,
            reference,
            config,
            2_000_000 + index,
            float("inf"),
            lambda: 0.0,
            0.0,
            config.validation_game_count,
            candidate_acquisition,
            reference_acquisition,
            candidate_constraints,
            reference_constraints,
        )
        reference_result = _evaluate(
            reference,
            opponent,
            reference,
            config,
            2_000_000 + index,
            float("inf"),
            lambda: 0.0,
            0.0,
            config.validation_game_count,
            reference_acquisition,
            reference_acquisition,
            reference_constraints,
            reference_constraints,
        )
        mean, lower, upper = _paired_confidence_interval(
            candidate_result, reference_result, config.confidence_level
        )
        required_lower = -config.minimum_gain if opponent == "random" else 0.0
        passes = candidate_result.games >= config.validation_game_count and lower > required_lower
        accepted = accepted and passes
        validation["adversaries"][opponent] = {
            "candidate_games": candidate_result.games,
            "candidate_utility": candidate_result.utility,
            "reference_utility": reference_result.utility,
            "mean_difference": mean,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "required_lower": required_lower,
            "passed": passes,
        }
    validation["passed"] = accepted
    return validation


def _combined_candidate_pool(
    action_weights: HeuristicWeights,
    acquisition_weights: CardAcquisitionWeights,
    constraint_weights: CardConstraintWeights,
    config: OptimizationConfig,
    step_scales: Mapping[str, float],
) -> tuple[tuple[HeuristicWeights, CardAcquisitionWeights, CardConstraintWeights], ...]:
    """Build a deduplicated neighborhood of complete weight triplets."""

    candidates: list[tuple[HeuristicWeights, CardAcquisitionWeights, CardConstraintWeights]] = [
        (action_weights, acquisition_weights, constraint_weights)
    ]

    def add(candidate: tuple[HeuristicWeights, CardAcquisitionWeights, CardConstraintWeights]) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in _candidate_pool(
        action_weights,
        DEFAULT_BOUNDS,
        config.frozen_fields,
        config.active_fields,
        step_scales,
    ):
        add((candidate, acquisition_weights, constraint_weights))
    for candidate in _candidate_pool(
        acquisition_weights,
        ACQUISITION_BOUNDS,
        frozenset(),
        config.active_acquisition_fields,
        step_scales,
    ):
        add((action_weights, candidate, constraint_weights))
    for candidate in _candidate_pool(
        constraint_weights,
        CONSTRAINT_BOUNDS,
        frozenset(),
        config.active_constraint_fields,
        step_scales,
    ):
        add((action_weights, acquisition_weights, candidate))
    return tuple(candidates)


def optimize_combined_weights(
    initial_profile: HeuristicProfile,
    config: OptimizationConfig,
    *,
    start_mixed: bool = False,
    reference_profile: HeuristicProfile | None = None,
    clock: Callable[[], float] = time.process_time,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: Mapping[str, object] | None = None,
) -> OptimizationResult:
    """Optimize action, acquisition, and constraint weights as one profile state."""

    started = clock()
    previous_compute_seconds = float(
        resume_checkpoint.get("compute_seconds_consumed", 0.0)
        if resume_checkpoint
        else 0.0
    )
    remaining_compute_seconds = max(0.0, config.duration_seconds - previous_compute_seconds)
    deadline = started + remaining_compute_seconds
    reference_profile = reference_profile or initial_profile
    if resume_checkpoint:
        current_document = resume_checkpoint["current"]
        current = (
            HeuristicWeights.from_mapping(current_document["weights"]),
            CardAcquisitionWeights.from_mapping(current_document["card_acquisition_weights"]),
            CardConstraintWeights.from_mapping(current_document["constraint_weights"]),
        )
    else:
        current = (
            initial_profile.weights,
            initial_profile.card_acquisition_weights,
            initial_profile.constraint_weights,
        )
    reference = (
        reference_profile.weights,
        reference_profile.card_acquisition_weights,
        reference_profile.constraint_weights,
    )
    active_fields = (
        OPTIMIZABLE_FIELDS
        if config.active_fields == DEFAULT_ACTIVE_FIELDS
        else config.active_fields
    )
    active_constraints = config.active_constraint_fields or CONSTRAINT_OPTIMIZABLE_FIELDS
    combined_config = replace(
        config,
        active_fields=tuple(active_fields),
        active_constraint_fields=tuple(active_constraints),
    )
    history: list[CandidateResult] = []
    mixed = bool(resume_checkpoint.get("mixed", start_mixed)) if resume_checkpoint else start_mixed
    batch = int(resume_checkpoint.get("next_batch", 0)) if resume_checkpoint else 0
    consecutive_failed_batches = (
        int(resume_checkpoint.get("consecutive_failed_batches", 0))
        if resume_checkpoint
        else 0
    )
    stop_reason: str | None = None
    step_scales = (
        {str(name): float(value) for name, value in resume_checkpoint["step_scales"].items()}
        if resume_checkpoint
        else {
            **{name: 1.0 for name in OPTIMIZABLE_FIELDS},
            **{name: 1.0 for name in ACQUISITION_OPTIMIZABLE_FIELDS},
            **{name: 1.0 for name in CONSTRAINT_OPTIMIZABLE_FIELDS},
        }
    )

    def save_checkpoint(phase: str = "search") -> None:
        if checkpoint_path is None:
            return
        consumed = previous_compute_seconds + max(0.0, clock() - started)
        _write_checkpoint(
            Path(checkpoint_path),
            _checkpoint_payload(
                initial_profile=initial_profile,
                reference_profile=reference_profile,
                config=config,
                current=current,
                step_scales=step_scales,
                batch=batch,
                mixed=mixed,
                compute_seconds_consumed=consumed,
                consecutive_failed_batches=consecutive_failed_batches,
                stop_reason=stop_reason,
                phase=phase,
            ),
        )

    save_checkpoint()

    while clock() < deadline and consecutive_failed_batches < 2:
        progress = max(0.0, min(1.0, (clock() - started) / config.duration_seconds))
        shaping_alpha = config.state_reward_weights.initial_alpha * (1.0 - progress)
        batch_deadline = deadline
        candidates = _combined_candidate_pool(
            *current,
            combined_config,
            step_scales,
        )
        initial_results: list[CandidateResult] = []
        for candidate in candidates:
            if clock() >= batch_deadline:
                break
            result = _evaluate_candidate(
                candidate[0],
                reference[0],
                mixed,
                config,
                batch,
                batch_deadline,
                clock,
                shaping_alpha,
                config.initial_game_count,
                "initial",
                candidate[1],
                reference[1],
                candidate[2],
                reference[2],
            )
            initial_results.append(result)
            history.append(result)

        ranked = sorted(
            (result for result in initial_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        if not ranked:
            break

        racing_results: list[CandidateResult] = []
        for result in ranked[: config.racing_top_k]:
            if clock() >= batch_deadline:
                break
            racing_result = _evaluate_candidate(
                HeuristicWeights.from_mapping(result.weights),
                reference[0],
                mixed,
                config,
                batch + 10_000,
                batch_deadline,
                clock,
                shaping_alpha,
                config.racing_game_count,
                "racing",
                CardAcquisitionWeights.from_mapping(result.acquisition_weights),
                reference[1],
                CardConstraintWeights.from_mapping(result.constraint_weights),
                reference[2],
            )
            racing_results.append(racing_result)
            history.append(racing_result)

        ranked_racing = sorted(
            (result for result in racing_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        finalist_results: list[CandidateResult] = []
        for result in ranked_racing[: config.racing_finalists]:
            if clock() >= batch_deadline:
                break
            finalist = _evaluate_candidate(
                HeuristicWeights.from_mapping(result.weights),
                reference[0],
                mixed,
                config,
                batch + 20_000,
                batch_deadline,
                clock,
                shaping_alpha,
                config.test_game_count,
                "finalist",
                CardAcquisitionWeights.from_mapping(result.acquisition_weights),
                reference[1],
                CardConstraintWeights.from_mapping(result.constraint_weights),
                reference[2],
            )
            finalist_results.append(finalist)
            history.append(finalist)

        best_results = finalist_results or ranked_racing or ranked
        best = max(best_results, key=lambda result: result.aggregate_objective, default=None)
        baseline_result = next(
            (
                result
                for result in initial_results
                if result.complete
                and result.weights == asdict(current[0])
                and result.acquisition_weights == asdict(current[1])
                and result.constraint_weights == asdict(current[2])
            ),
            None,
        )
        improved = (
            best is not None
            and best.complete
            and baseline_result is not None
            and best.aggregate_objective > baseline_result.aggregate_objective
        )
        if improved and best is not None:
            consecutive_failed_batches = 0
            current = (
                HeuristicWeights.from_mapping(best.weights),
                CardAcquisitionWeights.from_mapping(best.acquisition_weights),
                CardConstraintWeights.from_mapping(best.constraint_weights),
            )
            for history_index in range(len(history) - 1, -1, -1):
                if history[history_index] is best:
                    history[history_index] = replace(history[history_index], accepted=True)
                    break
        else:
            consecutive_failed_batches += 1
            step_scales = {
                name: max(0.25, scale * 0.5)
                for name, scale in step_scales.items()
            }

        fresh = _evaluate_candidate(
            current[0],
            reference[0],
            mixed,
            config,
            batch + 100_000,
            batch_deadline,
            clock,
            0.0,
            config.initial_game_count,
            "validation-sample",
            current[1],
            reference[1],
            current[2],
            reference[2],
        )
        history.append(fresh)
        random_eval = next(result for result in fresh.evaluations if result.opponent == "random")
        if (
            not mixed
            and random_eval.games >= config.minimum_games_for_promotion
            and random_eval.win_rate >= config.promotion_threshold
        ):
            mixed = True
        batch += 1
        save_checkpoint()
        if consecutive_failed_batches >= 2:
            stop_reason = "two_consecutive_failed_batches"
            save_checkpoint()
            break

    save_checkpoint("validation_pending")
    validation = (
        _validate_candidate(
            current[0],
            reference[0],
            mixed,
            config,
            current[1],
            reference[1],
            current[2],
            reference[2],
        )
        if config.validation_games is not None
        else {"passed": False, "skipped": True}
    )
    save_checkpoint("completed")
    validated = bool(validation.get("passed", False))
    accepted = current if validated else (
        initial_profile.weights,
        initial_profile.card_acquisition_weights,
        initial_profile.constraint_weights,
    )
    profile_id = initial_profile.profile_id + "-combined-optimized" if validated else initial_profile.profile_id
    profile = HeuristicProfile(
        profile_id=profile_id,
        parent_profile_id=initial_profile.profile_id,
        weights=accepted[0],
        card_acquisition_weights=accepted[1],
        constraint_weights=accepted[2],
        metadata={
            "optimizer": "hybrid_racing_combined",
            "combined": True,
            "mixed_phase_started": mixed,
            "reference_profile_id": reference_profile.profile_id,
            "root_seed": config.seed,
            "batches": batch,
            "active_fields": list(config.active_fields),
            "active_acquisition_fields": list(config.active_acquisition_fields),
            "active_constraint_fields": list(active_constraints),
            "validation": validation,
            "state_reward_weights": asdict(config.state_reward_weights),
            "consecutive_failed_batches": consecutive_failed_batches,
            "stop_reason": stop_reason,
        },
    )
    return OptimizationResult(
        root_seed=config.seed,
        elapsed_seconds=previous_compute_seconds + clock() - started,
        initial_profile_id=initial_profile.profile_id,
        final_profile_id=profile_id,
        mixed_phase_started=mixed,
        accepted_profile=profile,
        history=tuple(history),
        validation=validation,
    )


def optimize_heuristic(
    initial_profile: HeuristicProfile,
    config: OptimizationConfig = OptimizationConfig(),
    *,
    bounds: Mapping[str, CoefficientBound] = DEFAULT_BOUNDS,
    start_mixed: bool = False,
    reference_profile: HeuristicProfile | None = None,
    clock: Callable[[], float] = time.process_time,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: Mapping[str, object] | None = None,
) -> OptimizationResult:
    reference_profile = reference_profile or initial_profile
    if config.combined:
        return optimize_combined_weights(
            initial_profile,
            config,
            start_mixed=start_mixed,
            reference_profile=reference_profile,
            clock=clock,
            checkpoint_path=checkpoint_path,
            resume_checkpoint=resume_checkpoint,
        )
    if config.constraints_only:
        return optimize_constraint_weights(
            initial_profile,
            config,
            start_mixed=start_mixed,
            reference_profile=reference_profile,
            clock=clock,
        )
    if config.acquisition_only:
        return optimize_acquisition_weights(
            initial_profile,
            config,
            bounds=ACQUISITION_BOUNDS,
            start_mixed=start_mixed,
            reference_profile=reference_profile,
            clock=clock,
        )
    started = clock()
    deadline = started + config.duration_seconds
    current = initial_profile.weights
    reference = reference_profile.weights
    acquisition_weights = initial_profile.card_acquisition_weights
    reference_acquisition_weights = reference_profile.card_acquisition_weights
    constraint_weights = initial_profile.constraint_weights
    reference_constraint_weights = reference_profile.constraint_weights
    mixed = start_mixed
    history: list[CandidateResult] = []
    batch = 0
    step_scales = {name: 1.0 for name in OPTIMIZABLE_FIELDS}

    while clock() < deadline:
        progress = max(0.0, min(1.0, (clock() - started) / config.duration_seconds))
        shaping_alpha = config.state_reward_weights.initial_alpha * (1.0 - progress)
        # A candidate comparison is atomic: do not cut a candidate at a short batch deadline.
        # The global deadline may still interrupt the final in-flight candidate, which is then
        # marked incomplete and cannot influence the search.
        batch_deadline = deadline
        candidates = (current,) + _candidate_pool(
            current,
            bounds,
            config.frozen_fields,
            config.active_fields,
            step_scales,
        )
        initial_results: list[CandidateResult] = []
        for candidate in candidates:
            if clock() >= batch_deadline:
                break
            result = _evaluate_candidate(
                candidate,
                reference,
                mixed,
                config,
                batch,
                batch_deadline,
                clock,
                shaping_alpha,
                config.initial_game_count,
                "initial",
                acquisition_weights,
                reference_acquisition_weights,
                constraint_weights,
                reference_constraint_weights,
            )
            initial_results.append(result)
            history.append(result)

        ranked = sorted(
            (result for result in initial_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        if not ranked:
            break
        racing_results: list[CandidateResult] = []
        for result in ranked[: config.racing_top_k]:
            if clock() >= batch_deadline:
                break
            racing_result = _evaluate_candidate(
                HeuristicWeights.from_mapping(result.weights),
                reference,
                mixed,
                config,
                batch + 10_000,
                batch_deadline,
                clock,
                shaping_alpha,
                config.racing_game_count,
                "racing",
                acquisition_weights,
                reference_acquisition_weights,
                constraint_weights,
                reference_constraint_weights,
            )
            racing_results.append(racing_result)
            history.append(racing_result)

        ranked_racing = sorted(
            (result for result in racing_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        finalist_results: list[CandidateResult] = []
        for result in ranked_racing[: config.racing_finalists]:
            if clock() >= batch_deadline:
                break
            finalist = _evaluate_candidate(
                HeuristicWeights.from_mapping(result.weights),
                reference,
                mixed,
                config,
                batch + 20_000,
                batch_deadline,
                clock,
                shaping_alpha,
                config.test_game_count,
                "finalist",
                acquisition_weights,
                reference_acquisition_weights,
                constraint_weights,
                reference_constraint_weights,
            )
            finalist_results.append(finalist)
            history.append(finalist)

        best_results = finalist_results or ranked_racing or ranked
        best = max(best_results, key=lambda result: result.aggregate_objective, default=None)
        baseline_result = next(
            (result for result in initial_results if result.weights == asdict(current) and result.complete),
            None,
        )
        improved = (
            best is not None
            and best.complete
            and baseline_result is not None
            and best.aggregate_objective > baseline_result.aggregate_objective
        )
        if improved and best is not None:
            current = HeuristicWeights.from_mapping(best.weights)
            for history_index in range(len(history) - 1, -1, -1):
                if history[history_index] is best:
                    history[history_index] = replace(history[history_index], accepted=True)
                    break
        else:
            step_scales = {
                name: max(0.25, scale * 0.5)
                for name, scale in step_scales.items()
            }

        fresh = _evaluate_candidate(
            current,
            reference,
            mixed,
            config,
            batch + 100_000,
            deadline,
            clock,
            0.0,
            config.initial_game_count,
            "validation-sample",
            acquisition_weights,
            reference_acquisition_weights,
            constraint_weights,
            reference_constraint_weights,
        )
        history.append(fresh)
        random_eval = next(result for result in fresh.evaluations if result.opponent == "random")
        if (
            not mixed
            and random_eval.games >= config.minimum_games_for_promotion
            and random_eval.win_rate >= config.promotion_threshold
        ):
            mixed = True
        batch += 1

    validation = (
        _validate_candidate(
            current,
            reference,
            mixed,
            config,
            acquisition_weights,
            reference_acquisition_weights,
            constraint_weights,
            reference_constraint_weights,
        )
        if config.validation_games is not None
        else {"passed": False, "skipped": True}
    )
    validated = bool(validation.get("passed", False))
    accepted_weights = current if validated else initial_profile.weights
    profile_id = initial_profile.profile_id + "-optimized" if validated else initial_profile.profile_id
    profile = HeuristicProfile(
        profile_id=profile_id,
        parent_profile_id=initial_profile.profile_id,
        weights=accepted_weights,
        card_acquisition_weights=acquisition_weights,
        metadata={
            "optimizer": "hybrid_racing",
            "mixed_phase_started": mixed,
            "reference_profile_id": reference_profile.profile_id,
            "root_seed": config.seed,
            "batches": batch,
            "active_fields": list(config.active_fields),
            "active_acquisition_fields": [],
            "constraint_weights": asdict(constraint_weights),
            "track_zero_alpha_shaping": config.track_zero_alpha_shaping,
            "shaping_observer_detached": config.shaping_observer_detached,
            "trusted_live_observations": config.trusted_live_observations,
            "validation": validation,
            "state_reward_weights": asdict(config.state_reward_weights),
        },
    )
    return OptimizationResult(
        root_seed=config.seed,
        elapsed_seconds=clock() - started,
        initial_profile_id=initial_profile.profile_id,
        final_profile_id=profile.profile_id,
        mixed_phase_started=mixed,
        accepted_profile=profile,
        history=tuple(history),
        validation=validation,
    )


def optimize_acquisition_weights(
    initial_profile: HeuristicProfile,
    config: OptimizationConfig,
    *,
    bounds: Mapping[str, CoefficientBound] = ACQUISITION_BOUNDS,
    start_mixed: bool = False,
    reference_profile: HeuristicProfile | None = None,
    clock: Callable[[], float] = time.process_time,
) -> OptimizationResult:
    """Optimize only the internal card acquisition coefficients."""

    started = clock()
    deadline = started + config.duration_seconds
    reference_profile = reference_profile or initial_profile
    current = initial_profile.card_acquisition_weights
    reference = reference_profile.card_acquisition_weights
    fixed_weights = initial_profile.weights
    constraint_weights = initial_profile.constraint_weights
    reference_weights = reference_profile.weights
    reference_constraint_weights = reference_profile.constraint_weights
    mixed = start_mixed
    history: list[CandidateResult] = []
    batch = 0
    step_scales = {name: 1.0 for name in ACQUISITION_OPTIMIZABLE_FIELDS}

    while clock() < deadline:
        progress = max(0.0, min(1.0, (clock() - started) / config.duration_seconds))
        shaping_alpha = config.state_reward_weights.initial_alpha * (1.0 - progress)
        candidates = (current,) + _candidate_pool(
            current,
            bounds,
            frozenset(),
            config.active_acquisition_fields,
            step_scales,
        )
        initial_results: list[CandidateResult] = []
        for candidate in candidates:
            if clock() >= deadline:
                break
            result = _evaluate_candidate(
                fixed_weights,
                reference_weights,
                mixed,
                config,
                batch,
                deadline,
                clock,
                shaping_alpha,
                config.initial_game_count,
                "initial",
                candidate,
                reference,
                constraint_weights,
                reference_constraint_weights,
            )
            initial_results.append(result)
            history.append(result)

        ranked = sorted(
            (result for result in initial_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        if not ranked:
            break
        racing_results: list[CandidateResult] = []
        for result in ranked[: config.racing_top_k]:
            if clock() >= deadline:
                break
            racing_result = _evaluate_candidate(
                fixed_weights,
                reference_weights,
                mixed,
                config,
                batch + 10_000,
                deadline,
                clock,
                shaping_alpha,
                config.racing_game_count,
                "racing",
                CardAcquisitionWeights.from_mapping(result.acquisition_weights),
                reference,
                constraint_weights,
                reference_constraint_weights,
            )
            racing_results.append(racing_result)
            history.append(racing_result)

        ranked_racing = sorted(
            (result for result in racing_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        finalist_results: list[CandidateResult] = []
        for result in ranked_racing[: config.racing_finalists]:
            if clock() >= deadline:
                break
            finalist = _evaluate_candidate(
                fixed_weights,
                reference_weights,
                mixed,
                config,
                batch + 20_000,
                deadline,
                clock,
                shaping_alpha,
                config.test_game_count,
                "finalist",
                CardAcquisitionWeights.from_mapping(result.acquisition_weights),
                reference,
                constraint_weights,
                reference_constraint_weights,
            )
            finalist_results.append(finalist)
            history.append(finalist)

        best_results = [result for result in finalist_results if result.complete]
        best_results = best_results or ranked_racing or ranked
        best = max(best_results, key=lambda result: result.aggregate_objective, default=None)
        baseline_result = next(
            (
                result
                for result in initial_results
                if result.acquisition_weights == asdict(current) and result.complete
            ),
            None,
        )
        improved = (
            best is not None
            and best.complete
            and baseline_result is not None
            and best.aggregate_objective > baseline_result.aggregate_objective
        )
        if improved and best is not None:
            current = CardAcquisitionWeights.from_mapping(best.acquisition_weights)
            for history_index in range(len(history) - 1, -1, -1):
                if history[history_index] is best:
                    history[history_index] = replace(history[history_index], accepted=True)
                    break
        else:
            step_scales = {name: max(0.25, scale * 0.5) for name, scale in step_scales.items()}

        fresh = _evaluate_candidate(
            fixed_weights,
            reference_weights,
            mixed,
            config,
            batch + 100_000,
            deadline,
            clock,
            0.0,
            config.initial_game_count,
            "validation-sample",
            current,
            reference,
            constraint_weights,
            reference_constraint_weights,
        )
        history.append(fresh)
        random_eval = next(result for result in fresh.evaluations if result.opponent == "random")
        if (
            not mixed
            and random_eval.games >= config.minimum_games_for_promotion
            and random_eval.win_rate >= config.promotion_threshold
        ):
            mixed = True
        batch += 1

    validation = (
        _validate_candidate(
            fixed_weights,
            reference_weights,
            mixed,
            config,
            current,
            reference,
            constraint_weights,
            reference_constraint_weights,
        )
        if config.validation_games is not None
        else {"passed": False, "skipped": True}
    )
    validated = bool(validation.get("passed", False))
    accepted_acquisition = current if validated else initial_profile.card_acquisition_weights
    profile_id = initial_profile.profile_id + "-acquisition-optimized" if validated else initial_profile.profile_id
    profile = HeuristicProfile(
        profile_id=profile_id,
        parent_profile_id=initial_profile.profile_id,
        weights=fixed_weights,
        card_acquisition_weights=accepted_acquisition,
        constraint_weights=constraint_weights,
        metadata={
            "optimizer": "hybrid_racing_acquisition_only",
            "mixed_phase_started": mixed,
            "reference_profile_id": reference_profile.profile_id,
            "root_seed": config.seed,
            "batches": batch,
            "active_fields": [],
            "active_acquisition_fields": list(config.active_acquisition_fields),
            "constraint_weights": asdict(constraint_weights),
            "validation": validation,
            "state_reward_weights": asdict(config.state_reward_weights),
        },
    )
    return OptimizationResult(
        root_seed=config.seed,
        elapsed_seconds=clock() - started,
        initial_profile_id=initial_profile.profile_id,
        final_profile_id=profile.profile_id,
        mixed_phase_started=mixed,
        accepted_profile=profile,
        history=tuple(history),
        validation=validation,
    )


def optimize_constraint_weights(
    initial_profile: HeuristicProfile,
    config: OptimizationConfig,
    *,
    bounds: Mapping[str, CoefficientBound] = CONSTRAINT_BOUNDS,
    start_mixed: bool = False,
    reference_profile: HeuristicProfile | None = None,
    clock: Callable[[], float] = time.process_time,
) -> OptimizationResult:
    """Optimize only the condition-penalty coefficients."""

    started = clock()
    deadline = started + config.duration_seconds
    reference_profile = reference_profile or initial_profile
    current = initial_profile.constraint_weights
    reference = reference_profile.constraint_weights
    fixed_weights = initial_profile.weights
    reference_weights = reference_profile.weights
    acquisition_weights = initial_profile.card_acquisition_weights
    reference_acquisition_weights = reference_profile.card_acquisition_weights
    mixed = start_mixed
    history: list[CandidateResult] = []
    batch = 0
    step_scales = {name: 1.0 for name in CONSTRAINT_OPTIMIZABLE_FIELDS}

    while clock() < deadline:
        progress = max(0.0, min(1.0, (clock() - started) / config.duration_seconds))
        shaping_alpha = config.state_reward_weights.initial_alpha * (1.0 - progress)
        candidates = (current,) + _candidate_pool(
            current,
            bounds,
            frozenset(),
            config.active_constraint_fields,
            step_scales,
        )
        initial_results: list[CandidateResult] = []
        for candidate in candidates:
            if clock() >= deadline:
                break
            result = _evaluate_candidate(
                fixed_weights,
                reference_weights,
                mixed,
                config,
                batch,
                deadline,
                clock,
                shaping_alpha,
                config.initial_game_count,
                "initial",
                acquisition_weights,
                reference_acquisition_weights,
                candidate,
                reference,
            )
            initial_results.append(result)
            history.append(result)

        ranked = sorted(
            (result for result in initial_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        if not ranked:
            break
        racing_results: list[CandidateResult] = []
        for result in ranked[: config.racing_top_k]:
            if clock() >= deadline:
                break
            racing_result = _evaluate_candidate(
                fixed_weights,
                reference_weights,
                mixed,
                config,
                batch + 10_000,
                deadline,
                clock,
                shaping_alpha,
                config.racing_game_count,
                "racing",
                acquisition_weights,
                reference_acquisition_weights,
                CardConstraintWeights.from_mapping(result.constraint_weights),
                reference,
            )
            racing_results.append(racing_result)
            history.append(racing_result)

        ranked_racing = sorted(
            (result for result in racing_results if result.complete),
            key=lambda result: result.aggregate_objective,
            reverse=True,
        )
        finalist_results: list[CandidateResult] = []
        for result in ranked_racing[: config.racing_finalists]:
            if clock() >= deadline:
                break
            finalist = _evaluate_candidate(
                fixed_weights,
                reference_weights,
                mixed,
                config,
                batch + 20_000,
                deadline,
                clock,
                shaping_alpha,
                config.test_game_count,
                "finalist",
                acquisition_weights,
                reference_acquisition_weights,
                CardConstraintWeights.from_mapping(result.constraint_weights),
                reference,
            )
            finalist_results.append(finalist)
            history.append(finalist)

        best_results = [result for result in finalist_results if result.complete]
        best_results = best_results or ranked_racing or ranked
        best = max(best_results, key=lambda result: result.aggregate_objective, default=None)
        baseline_result = next(
            (
                result
                for result in initial_results
                if result.constraint_weights == asdict(current) and result.complete
            ),
            None,
        )
        improved = (
            best is not None
            and best.complete
            and baseline_result is not None
            and best.aggregate_objective > baseline_result.aggregate_objective
        )
        if improved and best is not None:
            current = CardConstraintWeights.from_mapping(best.constraint_weights)
            for history_index in range(len(history) - 1, -1, -1):
                if history[history_index] is best:
                    history[history_index] = replace(history[history_index], accepted=True)
                    break
        else:
            step_scales = {name: max(0.25, scale * 0.5) for name, scale in step_scales.items()}

        fresh = _evaluate_candidate(
            fixed_weights,
            reference_weights,
            mixed,
            config,
            batch + 100_000,
            deadline,
            clock,
            0.0,
            config.initial_game_count,
            "validation-sample",
            acquisition_weights,
            reference_acquisition_weights,
            current,
            reference,
        )
        history.append(fresh)
        random_eval = next(result for result in fresh.evaluations if result.opponent == "random")
        if (
            not mixed
            and random_eval.games >= config.minimum_games_for_promotion
            and random_eval.win_rate >= config.promotion_threshold
        ):
            mixed = True
        batch += 1

    validation = (
        _validate_candidate(
            fixed_weights,
            reference_weights,
            mixed,
            config,
            acquisition_weights,
            reference_acquisition_weights,
            current,
            reference,
        )
        if config.validation_games is not None
        else {"passed": False, "skipped": True}
    )
    validated = bool(validation.get("passed", False))
    accepted_constraints = current if validated else initial_profile.constraint_weights
    profile_id = initial_profile.profile_id + "-constraints-optimized" if validated else initial_profile.profile_id
    profile = HeuristicProfile(
        profile_id=profile_id,
        parent_profile_id=initial_profile.profile_id,
        weights=fixed_weights,
        card_acquisition_weights=acquisition_weights,
        constraint_weights=accepted_constraints,
        metadata={
            "optimizer": "hybrid_racing_constraint_only",
            "mixed_phase_started": mixed,
            "reference_profile_id": reference_profile.profile_id,
            "root_seed": config.seed,
            "batches": batch,
            "active_fields": [],
            "active_acquisition_fields": [],
            "active_constraint_fields": list(config.active_constraint_fields),
            "validation": validation,
            "state_reward_weights": asdict(config.state_reward_weights),
        },
    )
    return OptimizationResult(
        root_seed=config.seed,
        elapsed_seconds=clock() - started,
        initial_profile_id=initial_profile.profile_id,
        final_profile_id=profile_id,
        mixed_phase_started=mixed,
        accepted_profile=profile,
        history=tuple(history),
        validation=validation,
    )


def write_optimization_result(result: OptimizationResult, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    yaml_path = save_profile(result.accepted_profile, output / f"{result.final_profile_id}.yaml")
    return json_path, yaml_path


__all__ = [
    "CoefficientBound",
    "CONSTRAINT_OPTIMIZABLE_FIELDS",
    "DEFAULT_ACTIVE_FIELDS",
    "DEFAULT_JOINT_PAIRS",
    "EvaluationResult",
    "OptimizationConfig",
    "OptimizationResult",
    "optimize_acquisition_weights",
    "optimize_combined_weights",
    "optimize_constraint_weights",
    "optimize_heuristic",
    "write_optimization_result",
]
