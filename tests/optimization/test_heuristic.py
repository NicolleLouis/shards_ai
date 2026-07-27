from dataclasses import asdict, replace
from pathlib import Path

from shards_ai.ai import CardAcquisitionWeights, CardConstraintWeights, HeuristicWeights
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile, save_profile
import shards_ai.optimization.heuristic as heuristic_optimization
from shards_ai.optimization.heuristic import (
    CoefficientBound,
    EvaluationResult,
    OptimizationConfig,
    _candidate_pool,
    _evaluate,
    load_optimization_checkpoint,
    optimize_heuristic,
)


def test_validation_accepts_positive_lower_bound_without_minimum_gain(monkeypatch) -> None:
    candidate = HeuristicWeights(power_produced=1.0)
    reference = HeuristicWeights()
    config = OptimizationConfig(
        validation_games=10,
        minimum_gain=0.01,
        confidence_level=0.95,
    )

    def fake_evaluate(*args, **kwargs):
        return EvaluationResult(
            opponent=args[1],
            games=10,
            wins=5,
            draws=0,
            losses=5,
            errors=0,
            utility=0.5,
            objective=0.5,
            shaping_alpha=0.0,
            mean_shaping=0.0,
            final_potential=0.0,
            complete=True,
        )

    bounds = iter(((-0.001, -0.002, 0.001), (0.01, 0.001, 0.02)))
    monkeypatch.setattr(heuristic_optimization, "_evaluate", fake_evaluate)
    monkeypatch.setattr(
        heuristic_optimization,
        "_paired_confidence_interval",
        lambda *args, **kwargs: next(bounds),
    )

    validation = heuristic_optimization._validate_candidate(
        candidate,
        reference,
        True,
        config,
    )

    assert validation["passed"] is True
    assert validation["adversaries"]["previous"]["required_lower"] == 0.0
    assert validation["adversaries"]["random"]["required_lower"] == -0.01


def test_profile_round_trip_preserves_weights_and_metadata(tmp_path: Path) -> None:
    profile = HeuristicProfile(
        profile_id="test-profile",
        parent_profile_id="parent",
        weights=replace(HeuristicWeights.zero(), power_produced=2.5),
        metadata={"source": "test"},
    )

    loaded = load_profile(save_profile(profile, tmp_path / "profile.yaml"))

    assert loaded.profile_id == profile.profile_id
    assert loaded.parent_profile_id == profile.parent_profile_id
    assert loaded.weights == profile.weights
    assert loaded.metadata == profile.metadata


def test_profile_round_trip_preserves_card_acquisition_weights(tmp_path: Path) -> None:
    profile = HeuristicProfile(
        profile_id="acquisition-profile",
        weights=HeuristicWeights(),
        card_acquisition_weights=CardAcquisitionWeights(
            card_draw=4.0, power_produced=3.0, banish_threshold=4.5
        ),
    )

    loaded = load_profile(save_profile(profile, tmp_path / "profile.yaml"))

    assert loaded.card_acquisition_weights == profile.card_acquisition_weights


def test_profile_round_trip_preserves_constraint_weights(tmp_path: Path) -> None:
    profile = HeuristicProfile(
        profile_id="constraint-profile",
        weights=HeuristicWeights(),
        constraint_weights=CardConstraintWeights(domination=2.0, inspiration=0.25),
    )

    loaded = load_profile(save_profile(profile, tmp_path / "profile.yaml"))

    assert loaded.constraint_weights == profile.constraint_weights


def test_legacy_profile_uses_uniform_constraint_weights(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text("profile_id: legacy\nweights: {}\n", encoding="utf-8")

    loaded = load_profile(path)

    assert loaded.constraint_weights == CardConstraintWeights.legacy()


def test_acquisition_only_optimization_freezes_heuristic_weights() -> None:
    profile = HeuristicProfile("baseline", HeuristicWeights())
    result = optimize_heuristic(
        profile,
        OptimizationConfig(
            duration_seconds=0.2,
            batch_seconds=0.1,
            initial_games=1,
            racing_games=1,
            validation_games=None,
            test_games=1,
            minimum_games_for_promotion=1,
            active_acquisition_fields=("card_draw",),
            acquisition_only=True,
            seed=13,
        ),
        start_mixed=True,
    )

    assert result.accepted_profile.weights == profile.weights
    assert all(candidate.weights == asdict(profile.weights) for candidate in result.history)
    assert all(
        candidate.acquisition_weights["gems_produced"]
        == profile.card_acquisition_weights.gems_produced
        for candidate in result.history
    )


def test_combined_optimization_tracks_all_weight_families() -> None:
    profile = HeuristicProfile("baseline", HeuristicWeights())
    result = optimize_heuristic(
        profile,
        OptimizationConfig(
            duration_seconds=0.2,
            batch_seconds=0.1,
            initial_games=1,
            racing_games=1,
            validation_games=None,
            test_games=1,
            active_fields=("power_produced",),
            active_acquisition_fields=("banish_threshold",),
            active_constraint_fields=("domination",),
            combined=True,
            seed=14,
        ),
        start_mixed=True,
    )

    assert result.accepted_profile.metadata["combined"] is True
    assert result.history
    assert all(candidate.acquisition_weights for candidate in result.history)
    assert all(candidate.constraint_weights for candidate in result.history)


def test_combined_optimization_writes_a_resume_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    result = optimize_heuristic(
        HeuristicProfile("baseline", HeuristicWeights()),
        OptimizationConfig(
            duration_seconds=0.05,
            initial_games=1,
            racing_games=1,
            validation_games=None,
            test_games=1,
            active_fields=("power_produced",),
            active_acquisition_fields=("card_draw",),
            active_constraint_fields=("domination",),
            combined=True,
            seed=15,
        ),
        checkpoint_path=checkpoint,
    )

    saved = load_optimization_checkpoint(checkpoint)

    assert result.accepted_profile.metadata["combined"] is True
    assert saved["mode"] == "combined"
    assert saved["phase"] == "completed"
    assert saved["next_batch"] >= 0


def test_constraint_only_optimization_records_complete_reference_profile() -> None:
    profile = HeuristicProfile(
        "candidate",
        HeuristicWeights(),
        constraint_weights=CardConstraintWeights(domination=1.5),
    )
    reference = HeuristicProfile(
        "previous",
        HeuristicWeights(),
        constraint_weights=CardConstraintWeights.legacy(),
    )
    result = optimize_heuristic(
        profile,
        OptimizationConfig(
            duration_seconds=0.2,
            batch_seconds=0.1,
            initial_games=1,
            racing_games=1,
            validation_games=None,
            test_games=1,
            minimum_games_for_promotion=1,
            active_constraint_fields=("domination",),
            constraints_only=True,
            seed=14,
        ),
        start_mixed=True,
        reference_profile=reference,
    )

    assert result.accepted_profile.metadata["reference_profile_id"] == "previous"
    assert all(candidate.constraint_weights for candidate in result.history)


def test_optimization_respects_frozen_terminal_weights() -> None:
    profile = HeuristicProfile("baseline", HeuristicWeights())
    result = optimize_heuristic(
        profile,
        OptimizationConfig(
            duration_seconds=0.2,
            batch_seconds=0.1,
            games_per_candidate=1,
            minimum_games_for_promotion=1,
            seed=7,
            frozen_fields=frozenset({"power_produced"}),
        ),
    )

    assert result.accepted_profile.weights.lethal == profile.weights.lethal
    assert result.accepted_profile.weights.terminal_win == profile.weights.terminal_win
    assert all(candidate.weights["power_produced"] == profile.weights.power_produced for candidate in result.history)
    assert result.history


def test_mixed_phase_can_be_started_explicitly() -> None:
    result = optimize_heuristic(
        HeuristicProfile("baseline", HeuristicWeights()),
        OptimizationConfig(
            duration_seconds=0.2,
            batch_seconds=0.1,
            games_per_candidate=1,
            minimum_games_for_promotion=1,
            seed=8,
        ),
        start_mixed=True,
    )

    assert result.mixed_phase_started
    assert any(
        evaluation.opponent == "previous"
        for candidate in result.history
        for evaluation in candidate.evaluations
    )


def test_coefficient_bound_rejects_invalid_step() -> None:
    try:
        CoefficientBound(0.0, 1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid coefficient bound was accepted")


def test_candidate_pool_contains_joint_mutations_without_duplicates() -> None:
    weights = HeuristicWeights()
    bounds = {
        "gems_produced": CoefficientBound(0.0, 5.0, 1.0),
        "power_produced": CoefficientBound(0.0, 6.0, 1.0),
    }

    candidates = _candidate_pool(
        weights,
        bounds,
        frozenset(),
        ("gems_produced", "power_produced"),
        {"gems_produced": 1.0, "power_produced": 1.0},
    )

    assert len(candidates) == len(set(candidates))
    assert any(
        candidate.gems_produced != weights.gems_produced
        and candidate.power_produced != weights.power_produced
        for candidate in candidates
    )


def test_hybrid_search_respects_active_fields_and_records_stages() -> None:
    profile = HeuristicProfile("baseline", HeuristicWeights())
    result = optimize_heuristic(
        profile,
        OptimizationConfig(
            duration_seconds=0.25,
            batch_seconds=0.1,
            games_per_candidate=1,
            initial_games=1,
            racing_games=1,
            test_games=1,
            validation_games=None,
            minimum_games_for_promotion=1,
            active_fields=("power_produced",),
            seed=11,
        ),
    )

    assert result.history
    assert {candidate.stage for candidate in result.history} <= {
        "initial",
        "racing",
        "finalist",
        "validation-sample",
    }
    assert all(
        candidate.weights["gems_produced"] == profile.weights.gems_produced
        for candidate in result.history
    )


def test_deadline_marks_evaluation_incomplete() -> None:
    result = _evaluate(
        HeuristicWeights(),
        "random",
        None,
        OptimizationConfig(games_per_candidate=2, validation_games=None),
        batch=1,
        deadline=0.0,
        clock=lambda: 1.0,
        shaping_alpha=0.0,
        game_count=2,
    )

    assert result.games == 0
    assert not result.complete
