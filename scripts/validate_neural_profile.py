#!/usr/bin/env python3
"""Validate and optionally promote a neural checkpoint against a fixed opponent panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from shards_ai.ai import (
    HeuristicPlayer,
    RandomPlayer,
    build_neural_player,
    build_hybrid_player,
    load_active_training_profile,
    load_hybrid_profile,
    load_training_profile,
    next_training_profile_id,
    save_training_profile,
    versioned_training_profiles,
)
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


QUALITY_OPPONENT_WEIGHTS = {
    "hybrid:v006": 1.0,
    "hybrid:v004": 1.0,
    "hybrid:v005": 1.0,
    "v008": 1.0,
    "hybrid:v001": 0.75,
    "hybrid:v003": 0.75,
}
QUALITY_GATE_EPSILON = 1e-12
NEURAL_REFERENCE_PROFILE_IDS = ()
NEURAL_REFERENCE_COUNT = len(NEURAL_REFERENCE_PROFILE_IDS)
HYBRID_REFERENCE_PROFILE_IDS = ("v006", "v004", "v005", "v001", "v003")


def _play(
    seed: int,
    candidate_scorer,
    opponent: str,
    heuristic_profiles: dict[str, object],
    neural_scorers: dict[str, object],
    hybrid_profiles: dict[str, object],
    max_actions: int,
    max_turns: int | None,
    hybrid_scorers: dict[str, object] | None = None,
) -> dict[str, object]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    candidate_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = candidate_id.opponent
    candidate = build_neural_player(
        candidate_id, game, root_rng.derive("candidate"), scorer=candidate_scorer,
    )
    if opponent == "random":
        other = RandomPlayer(opponent_id, root_rng.derive("opponent"))
    elif opponent.startswith("neural:"):
        other = build_neural_player(
            opponent_id,
            game,
            root_rng.derive("opponent"),
            scorer=neural_scorers[opponent.removeprefix("neural:")],
        )
    elif opponent.startswith("hybrid:"):
        other = build_hybrid_player(
            opponent_id,
            game,
            root_rng.derive("opponent"),
            profile=hybrid_profiles[opponent.removeprefix("hybrid:")],
            acquisition_scorer=(hybrid_scorers or {}).get(opponent.removeprefix("hybrid:")),
        )
    else:
        profile = heuristic_profiles[opponent]
        other = HeuristicPlayer(
            opponent_id,
            profile.weights,
            profile.card_acquisition_weights,
            profile.constraint_weights,
        )
    state = GameRunner(
        game,
        {candidate_id: candidate, opponent_id: other},
        max_actions=max_actions,
        max_turns=max_turns,
    ).run()
    return {
        "seed": seed,
        "candidate_won": state.winner is candidate_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "status": state.status.value,
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    games = len(records)
    wins = sum(bool(record["candidate_won"]) for record in records)
    losses = sum(bool(record["opponent_won"]) for record in records)
    draws = sum(bool(record["draw"]) for record in records)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "records": records,
    }


def checkpoint_weights_digest(checkpoint_path: Path) -> str:
    """Return a stable digest of the model architecture and learned weights.

    Promotion updates checkpoint metadata (profile id and fingerprint), so a
    byte-for-byte file hash would not describe whether the learned model was
    preserved.  This digest deliberately covers the model config, architecture,
    card vocabulary, and every tensor in the state dict while ignoring mutable
    provenance metadata.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()

    def add(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    add(str(checkpoint.get("architecture", "independent_action")).encode("utf-8"))
    add(json.dumps(checkpoint["model_config"], sort_keys=True).encode("utf-8"))
    add(json.dumps(checkpoint.get("card_ids", ())).encode("utf-8"))
    for name, tensor in sorted(checkpoint["model_state_dict"].items()):
        tensor = tensor.detach().cpu().contiguous()
        add(name.encode("utf-8"))
        add(str(tensor.dtype).encode("utf-8"))
        add(json.dumps(list(tensor.shape)).encode("utf-8"))
        add(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def acceptance_metrics(
    results: dict[str, dict[str, object]],
    category_results: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Apply the quality gate once per opponent, never once per game.

    The only quality condition is a strictly positive weighted mean: hybrid v006,
    v004, v005 and v008 have weight 1, while hybrid v001 and v003 have weight
    0.75. Every opponent is a weighted quality signal rather than a hard
    non-regression guard. Random and retired profiles are outside this gate.
    """
    deltas = {opponent: float(item["delta_win_rate"]) for opponent, item in results.items()}
    if "v008" not in deltas:
        return {
            "accepted": False,
            "reason": "missing_v008_result",
            "mean_delta_win_rate": None,
            "deltas": deltas,
        }
    weighted_sum = 0.0
    total_weight = 0.0
    applied_weights = {}
    for opponent, delta in deltas.items():
        weight = QUALITY_OPPONENT_WEIGHTS.get(opponent)
        if weight is None:
            continue
        weighted_sum += weight * delta
        total_weight += weight
        applied_weights[opponent] = weight
    missing_opponents = sorted(set(QUALITY_OPPONENT_WEIGHTS) - set(applied_weights))
    if missing_opponents:
        return {
            "accepted": False,
            "reason": "missing_neural_references",
            "mean_delta_win_rate": None,
            "opponent_count": len(deltas),
            "neural_reference_count": sum(key.startswith("neural:") for key in applied_weights),
            "required_neural_reference_count": NEURAL_REFERENCE_COUNT,
            "deltas": deltas,
            "opponent_weights": applied_weights,
            "missing_opponents": missing_opponents,
            "minimum_weighted_mean_delta": 0.0,
        }
    mean_delta = weighted_sum / total_weight
    accepted = mean_delta > QUALITY_GATE_EPSILON
    return {
        "accepted": accepted,
        "reason": "weighted_mean_gain" if accepted else "weighted_mean_gain_failed",
        "mean_delta_win_rate": mean_delta,
        "opponent_count": len(deltas),
        "deltas": deltas,
        "opponent_weights": applied_weights,
        "minimum_weighted_mean_delta": 0.0,
    }


def acceptance_decision(results: dict[str, dict[str, object]]) -> bool:
    """Apply the weighted opponent-level gain rule."""
    return bool(acceptance_metrics(results)["accepted"])


def format_validation_line(
    opponent: str,
    result: dict[str, object],
    candidate_label: str,
    reference_label: str,
) -> str:
    """Format one human-readable candidate/reference comparison."""
    candidate = result["candidate"]
    reference = result["reference"]
    status = "PROGRES" if result["improved"] else "OK" if result["not_regressed"] else "BAISSE"
    return (
        f"{opponent} : {candidate_label} {candidate['win_rate']:.2%} "
        f"({candidate['wins']}/{candidate['games']}) | "
        f"{reference_label} {reference['win_rate']:.2%} "
        f"({reference['wins']}/{reference['games']}) | "
        f"delta {result['delta_win_rate']:+.2%} | {status}"
    )


def _panel(
    args: argparse.Namespace, candidate_profile_id: str,
) -> tuple[list[str], dict[str, object], dict[str, object], dict[str, object]]:
    heuristic_profiles = {"v008": load_profile(args.profile_v008)}
    neural_profiles = {}
    for _number, path, profile in reversed(versioned_training_profiles(args.profile_dir)):
        if profile.profile_id not in NEURAL_REFERENCE_PROFILE_IDS or profile.profile_id == candidate_profile_id:
            continue
        checkpoint = profile.resolve_path(profile.output)
        if checkpoint.exists():
            neural_profiles[profile.profile_id] = (path, profile, checkpoint)
        if len(neural_profiles) == NEURAL_REFERENCE_COUNT:
            break
    if len(neural_profiles) != NEURAL_REFERENCE_COUNT:
        raise ValueError(
            f"quality validation requires {NEURAL_REFERENCE_COUNT} promoted neural references; "
            f"found {len(neural_profiles)}"
        )
    hybrid_paths = {
        "v006": args.profile_hybrid_v006,
        "v004": args.profile_hybrid_v004,
        "v005": args.profile_hybrid_v005,
        "v001": args.profile_hybrid_v001,
        "v003": args.profile_hybrid_v003,
    }
    hybrid_profiles = {
        profile_id: load_hybrid_profile(hybrid_paths[profile_id])
        for profile_id in HYBRID_REFERENCE_PROFILE_IDS
    }
    opponents = [
        "hybrid:v006",
        "hybrid:v004",
        "hybrid:v005",
        "v008",
        "hybrid:v001",
        "hybrid:v003",
    ]
    return opponents, heuristic_profiles, neural_profiles, hybrid_profiles


def promote_candidate(args: argparse.Namespace, candidate_profile, active_profile, candidate_checkpoint: Path) -> dict[str, object]:
    """Promote a validated candidate without rerunning the games."""
    promoted_id = next_training_profile_id(args.profile_dir)
    promoted_checkpoint = args.checkpoint_dir / f"{promoted_id}.pt"
    if promoted_checkpoint.exists():
        raise FileExistsError(f"Promoted checkpoint already exists: {promoted_checkpoint}")
    promoted = replace(
        candidate_profile,
        profile_id=promoted_id,
        parent_profile_id=active_profile.profile_id,
        output=str(promoted_checkpoint),
    )
    promoted_path = args.profile_dir / f"{promoted_id}.yaml"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_weights_digest = checkpoint_weights_digest(candidate_checkpoint)
    checkpoint = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    checkpoint["profile_id"] = promoted.profile_id
    checkpoint["profile_fingerprint"] = promoted.fingerprint
    training_config = dict(checkpoint.get("training_config", {}))
    training_config.update({
        "profile_id": promoted.profile_id,
        "profile_fingerprint": promoted.fingerprint,
        "output": str(promoted_checkpoint),
    })
    checkpoint["training_config"] = training_config
    temporary_checkpoint = promoted_checkpoint.with_suffix(".pt.tmp")
    try:
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, promoted_checkpoint)
        promoted_weights_digest = checkpoint_weights_digest(promoted_checkpoint)
        if promoted_weights_digest != candidate_weights_digest:
            raise ValueError("Promoted checkpoint weights differ from the validated candidate")
        validation_game = Game.new(seed=0, rng=GameRandom(0).derive("promotion-check"))
        build_neural_player(
            PlayerId.PLAYER_1,
            validation_game,
            GameRandom(0).derive("promotion-player"),
            checkpoint_path=promoted_checkpoint,
        )
    except Exception:
        temporary_checkpoint.unlink(missing_ok=True)
        promoted_checkpoint.unlink(missing_ok=True)
        raise
    save_training_profile(promoted, promoted_path)
    args.active_profile.parent.mkdir(parents=True, exist_ok=True)
    args.active_profile.write_text(
        yaml.safe_dump({"schema_version": 1, "active_profile_id": promoted_id}, sort_keys=False),
        encoding="utf-8",
    )
    args.active_neural_profile.parent.mkdir(parents=True, exist_ok=True)
    args.active_neural_profile.write_text(
        yaml.safe_dump({"schema_version": 1, "active_profile_id": promoted_id}, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "profile_id": promoted_id,
        "profile_path": str(promoted_path),
        "checkpoint_path": str(promoted_checkpoint),
        "candidate_weights_digest": candidate_weights_digest,
        "promoted_weights_digest": promoted_weights_digest,
        "post_promotion_checkpoint_validation": "passed",
    }


def validate(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.torch_threads)
    candidate_profile = load_training_profile(args.candidate_profile)
    candidate_checkpoint = args.candidate_checkpoint or candidate_profile.resolve_path(candidate_profile.output)
    if not candidate_checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate_checkpoint}")
    active_profile = load_active_training_profile(args.active_profile)
    reference_checkpoint = active_profile.resolve_path(active_profile.output)
    if not reference_checkpoint.exists():
        raise FileNotFoundError(f"Active reference checkpoint not found: {reference_checkpoint}")

    opponents, heuristic_profiles, neural_profiles, hybrid_profiles = _panel(args, candidate_profile.profile_id)
    candidate_scorer = NeuralPlayer.load_scorer(candidate_checkpoint)
    reference_scorer = NeuralPlayer.load_scorer(reference_checkpoint)
    neural_scorers = {
        profile_id: NeuralPlayer.load_scorer(checkpoint)
        for profile_id, (_path, _profile, checkpoint) in neural_profiles.items()
    }
    by_opponent = {}
    for opponent in opponents:
        candidate_records = [
            _play(args.seed + index, candidate_scorer, opponent, heuristic_profiles, neural_scorers, hybrid_profiles,
                  args.max_actions, args.max_turns)
            for index in range(args.games)
        ]
        reference_records = [
            _play(args.seed + index, reference_scorer, opponent, heuristic_profiles, neural_scorers, hybrid_profiles,
                  args.max_actions, args.max_turns)
            for index in range(args.games)
        ]
        candidate_summary = _aggregate(candidate_records)
        reference_summary = _aggregate(reference_records)
        delta = candidate_summary["win_rate"] - reference_summary["win_rate"]
        by_opponent[opponent] = {
            "candidate": candidate_summary,
            "reference": reference_summary,
            "delta_win_rate": delta,
            "improved": delta > 0,
            "not_regressed": delta >= 0,
        }

    decision_metrics = acceptance_metrics(by_opponent)
    accepted = bool(decision_metrics["accepted"])
    report = {
        "candidate_profile": candidate_profile.profile_id,
        "candidate_checkpoint": str(candidate_checkpoint),
        "reference_profile": active_profile.profile_id,
        "reference_checkpoint": str(reference_checkpoint),
        "games_per_opponent": args.games,
        "seed": args.seed,
        "opponents": opponents,
        "results": by_opponent,
        "decision_metrics": decision_metrics,
        "decision": "accepted" if accepted else "rejected",
        "promotion": None,
    }
    if accepted and not args.no_promote:
        report["promotion"] = promote_candidate(args, candidate_profile, active_profile, candidate_checkpoint)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--profile-dir", type=Path, default=Path("configs/neural_training_profiles"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("configs/neural_profiles"))
    parser.add_argument("--active-profile", type=Path, default=Path("configs/neural_training_profiles/active.yaml"))
    parser.add_argument("--active-neural-profile", type=Path, default=Path("configs/neural_profiles/active.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--profile-hybrid-v001", type=Path, default=Path("configs/hybrid_profiles/hybrid-v001.yaml"))
    parser.add_argument("--profile-hybrid-v003", type=Path, default=Path("configs/hybrid_profiles/hybrid-v003.yaml"))
    parser.add_argument("--profile-hybrid-v004", type=Path, default=Path("configs/hybrid_profiles/hybrid-v004.yaml"))
    parser.add_argument("--profile-hybrid-v005", type=Path, default=Path("configs/hybrid_profiles/hybrid-v005.yaml"))
    parser.add_argument("--profile-hybrid-v006", type=Path, default=Path("configs/hybrid_profiles/hybrid-v006.yaml"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_validation/latest.json"))
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Candidat : {report['candidate_profile']}")
    print(f"Référence : {report['reference_profile']}")
    candidate_label = report["candidate_profile"]
    reference_label = report["reference_profile"]
    for opponent, result in report["results"].items():
        print(format_validation_line(opponent, result, candidate_label, reference_label))
    print(f"Décision : {report['decision'].upper()}")
    if report["promotion"]:
        print(f"Profil actif : {report['promotion']['profile_id']}")
    return 0 if report["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
