#!/usr/bin/env python3
"""Optimize a versioned HeuristicPlayer profile with reproducible game batches."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from shards_ai.ai import CardAcquisitionWeights, CardConstraintWeights, HeuristicWeights
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile, save_profile
from shards_ai.optimization.heuristic import (
    DEFAULT_ACQUISITION_ACTIVE_FIELDS,
    DEFAULT_ACTIVE_FIELDS,
    CONSTRAINT_OPTIMIZABLE_FIELDS,
    OPTIMIZABLE_FIELDS,
    OptimizationConfig,
    load_optimization_checkpoint,
    optimize_heuristic,
    write_optimization_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = PROJECT_ROOT / "configs/heuristic_profiles/v008.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/heuristic_optimization"


def _profile_from_checkpoint(document: dict[str, object]) -> HeuristicProfile:
    return HeuristicProfile(
        profile_id=str(document["profile_id"]),
        parent_profile_id=document.get("parent_profile_id"),
        weights=HeuristicWeights.from_mapping(document["weights"]),
        card_acquisition_weights=CardAcquisitionWeights.from_mapping(
            document["card_acquisition_weights"]
        ),
        constraint_weights=CardConstraintWeights.from_mapping(document["constraint_weights"]),
        metadata=document.get("metadata", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--reference-profile",
        type=Path,
        default=None,
        help="Complete profile used for the previous opponent during validation.",
    )
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument(
        "--compute-seconds",
        type=float,
        default=None,
        help="CPU-time budget for the optimizer; preferred spelling of duration-seconds.",
    )
    parser.add_argument(
        "--batch-seconds",
        type=float,
        default=20.0,
        help="Legacy compatibility option; candidate stages now use the global deadline.",
    )
    parser.add_argument("--games-per-candidate", type=int, default=100)
    parser.add_argument("--minimum-games-for-promotion", type=int, default=500)
    parser.add_argument("--promotion-threshold", type=float, default=0.90)
    parser.add_argument("--initial-games", type=int, default=200)
    parser.add_argument("--racing-games", type=int, default=500)
    parser.add_argument("--validation-games", type=int, default=1000)
    parser.add_argument("--test-games", type=int, default=3000)
    parser.add_argument(
        "--active-fields",
        default=",".join(DEFAULT_ACTIVE_FIELDS),
        help="Comma-separated coefficients to mutate during the first search pass.",
    )
    parser.add_argument(
        "--active-acquisition-fields",
        default=",".join(DEFAULT_ACQUISITION_ACTIVE_FIELDS),
        help="Comma-separated internal card acquisition coefficients to mutate.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument(
        "--track-zero-alpha-shaping",
        action="store_true",
        help="Keep transition shaping metrics during alpha=0 validation at extra cost.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-mixed", action="store_true", help="Start directly with the 50/50 opponent mix.")
    parser.add_argument(
        "--acquisition-only",
        action="store_true",
        help="Freeze HeuristicWeights and optimize only card acquisition coefficients.",
    )
    parser.add_argument(
        "--constraints-only",
        action="store_true",
        help="Freeze action and acquisition weights and optimize only constraint coefficients.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Optimize action, acquisition, and constraint weights as one complete candidate.",
    )
    parser.add_argument(
        "--active-constraint-fields",
        default="",
        help="Comma-separated constraint coefficients to mutate in constraints-only mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a combined optimization from a checkpoint JSON file.",
    )
    parser.add_argument("--publish-profile", type=Path, default=None)
    args = parser.parse_args()
    resume_checkpoint = (
        load_optimization_checkpoint(args.resume) if args.resume is not None else None
    )
    if resume_checkpoint is not None and resume_checkpoint.get("phase") == "completed":
        parser.error("checkpoint already completed; start a new run to fork it")
    if resume_checkpoint is not None:
        profile = _profile_from_checkpoint(resume_checkpoint["initial_profile"])
        reference_profile = _profile_from_checkpoint(resume_checkpoint["reference_profile"])
        saved_config = resume_checkpoint.get("config", {})
        args.combined = True
        args.seed = int(resume_checkpoint["root_seed"])
        if args.compute_seconds is None:
            args.compute_seconds = float(resume_checkpoint["compute_seconds_target"])
        for name in (
            "initial_games", "racing_games", "validation_games", "test_games",
            "racing_top_k", "racing_finalists",
        ):
            if name in saved_config:
                setattr(args, name, saved_config[name])
        if saved_config.get("active_fields"):
            args.active_fields = ",".join(saved_config["active_fields"])
        if saved_config.get("active_acquisition_fields"):
            args.active_acquisition_fields = ",".join(saved_config["active_acquisition_fields"])
        if saved_config.get("active_constraint_fields"):
            args.active_constraint_fields = ",".join(saved_config["active_constraint_fields"])
    else:
        compute_seconds = (
            args.compute_seconds if args.compute_seconds is not None else args.duration_seconds
        )
        profile = load_profile(args.profile)
        reference_profile = load_profile(args.reference_profile) if args.reference_profile else profile
    compute_seconds = (
        args.compute_seconds if args.compute_seconds is not None else args.duration_seconds
    )
    if args.acquisition_only and args.constraints_only:
        parser.error("--acquisition-only and --constraints-only are mutually exclusive")
    active_fields = tuple(field.strip() for field in args.active_fields.split(",") if field.strip())
    if args.combined and args.active_fields == ",".join(DEFAULT_ACTIVE_FIELDS):
        active_fields = OPTIMIZABLE_FIELDS
    active_acquisition_fields = tuple(
        field.strip() for field in args.active_acquisition_fields.split(",") if field.strip()
    )
    active_constraint_fields = tuple(
        field.strip() for field in args.active_constraint_fields.split(",") if field.strip()
    )
    if args.combined and not active_constraint_fields:
        active_constraint_fields = CONSTRAINT_OPTIMIZABLE_FIELDS
    started = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or DEFAULT_OUTPUT / started
    result = optimize_heuristic(
        profile,
        OptimizationConfig(
            duration_seconds=compute_seconds,
            batch_seconds=args.batch_seconds,
            games_per_candidate=args.games_per_candidate,
            minimum_games_for_promotion=args.minimum_games_for_promotion,
            promotion_threshold=args.promotion_threshold,
            seed=args.seed,
            initial_games=args.initial_games,
            racing_games=args.racing_games,
            validation_games=args.validation_games,
            test_games=args.test_games,
            active_fields=active_fields,
            active_acquisition_fields=active_acquisition_fields,
            acquisition_only=args.acquisition_only,
            active_constraint_fields=active_constraint_fields,
            constraints_only=args.constraints_only,
            combined=args.combined,
            confidence_level=args.confidence_level,
            minimum_gain=args.minimum_gain,
            track_zero_alpha_shaping=args.track_zero_alpha_shaping,
        ),
        start_mixed=args.start_mixed,
        reference_profile=reference_profile,
        checkpoint_path=args.checkpoint or args.resume,
        resume_checkpoint=resume_checkpoint,
    )
    result_path, profile_path = write_optimization_result(result, output_dir)
    published = False
    if args.publish_profile is not None and result.validation.get("passed", False):
        published_id = args.publish_profile.stem
        published_profile = replace(result.accepted_profile, profile_id=published_id)
        save_profile(published_profile, args.publish_profile)
        profile_path = args.publish_profile
        published = True

    print(f"initial_profile={result.initial_profile_id}")
    print(f"final_profile={result.final_profile_id}")
    print(f"elapsed_seconds={result.elapsed_seconds:.2f}")
    print(f"compute_seconds_budget={compute_seconds:.2f}")
    print(f"batches={result.accepted_profile.metadata.get('batches', 0)}")
    print(f"mixed_phase_started={result.mixed_phase_started}")
    print(f"acquisition_only={args.acquisition_only}")
    print(f"constraints_only={args.constraints_only}")
    print(f"combined={args.combined}")
    print(f"validation_passed={result.validation.get('passed', False)}")
    print(f"profile_published={published}")
    print(f"results={result_path}")
    print(f"profile={profile_path}")


if __name__ == "__main__":
    main()
