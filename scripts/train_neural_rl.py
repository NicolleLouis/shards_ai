#!/usr/bin/env python3
"""Train the neural player online with terminal win/loss rewards and PPO."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

import torch

from shards_ai.ai.neural_model import NeuralModelConfig
from shards_ai.ai.neural_reporting import write_training_report
from shards_ai.ai.neural_training import seed_training
from shards_ai.ai.neural_training_profiles import load_training_profile
from shards_ai.ai.rl_training import (
    NeuralActorCritic,
    collect_rollout,
    evaluate_greedy_model,
    is_monotonic_evaluation_improvement,
    ppo_update,
)


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_checkpoint(
    path: Path,
    model: NeuralActorCritic,
    optimizer: torch.optim.Optimizer,
    profile,
    *,
    updates_seen: int,
    games_seen: int,
    transitions_seen: int,
    metrics: list[dict],
    best_evaluation: dict,
    best_evaluation_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        # Keeps the promoted checkpoint readable by NeuralPlayer v001+.
        "model_state_dict": model.inference_state_dict(),
        "actor_critic_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "architecture": model.architecture,
        "card_ids": model.card_ids,
        "optimizer_state_dict": optimizer.state_dict(),
        "profile_id": profile.profile_id,
        "parent_profile_id": profile.parent_profile_id,
        "profile_fingerprint": profile.fingerprint,
        "method": profile.method,
        "update_index": updates_seen,
        "games_seen": games_seen,
        "transitions_seen": transitions_seen,
        "seed": profile.seed,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "training_metrics": metrics,
        "best_evaluation_score": best_evaluation_score,
        "best_evaluation": best_evaluation,
        "training_config": profile.resolved_document(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _write_metrics(metrics: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = (
        "update", "games_seen", "transitions_seen", "games", "policy_loss", "value_loss",
        "entropy", "approx_kl", "clip_fraction", "reference_kl", "evaluation_score",
        "best_evaluation_score", "wins", "losses", "draws",
    )
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in metrics:
            outcomes = item["outcomes"]
            writer.writerow({
                "update": item["update"],
                "games_seen": item["games_seen"],
                "transitions_seen": item["transitions_seen"],
                "games": item["games"],
                **{key: item["ppo"][key] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")},
                "reference_kl": item["ppo"]["reference_kl"],
                "evaluation_score": item.get("evaluation", {}).get("score"),
                "best_evaluation_score": item.get("best_evaluation_score"),
                "wins": sum(value.get("win", 0) for value in outcomes.values()),
                "losses": sum(value.get("loss", 0) for value in outcomes.values()),
                "draws": sum(value.get("draw", 0) for value in outcomes.values()),
            })
    write_training_report(
        [
            {
                "epoch": item["update"],
                "train": {"records": item["transitions_seen"], "mean_loss": item["ppo"]["policy_loss"]},
                "validation": {
                    "records": item["games"],
                    "mean_loss": item["ppo"]["value_loss"],
                    "top1_accuracy": item["win_rate"],
                    "mean_chosen_rank": 0.0,
                    "mean_normalized_chosen_rank": item["ppo"]["entropy"],
                    "pairwise_accuracy": 0.0,
                    "pairwise_pairs": item["transitions_seen"],
                },
            }
            for item in metrics
        ],
        output.with_suffix(".html"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--total-games", type=int)
    parser.add_argument("--games-per-update", type=int)
    parser.add_argument("--optimization-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--gamma", type=float,
                        help="Override the PPO discount factor for this reproducible run.")
    parser.add_argument("--gae-lambda", type=float,
                        help="Override the GAE lambda for this reproducible run.")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent game-collection workers (default: 1).")
    args = parser.parse_args()

    profile = load_training_profile(args.profile)
    if profile.method != "ppo":
        parser.error(f"Profile method must be ppo, got {profile.method!r}")
    gamma = args.gamma if args.gamma is not None else profile.gamma
    gae_lambda = args.gae_lambda if args.gae_lambda is not None else profile.gae_lambda
    for name, value in (("gamma", gamma), ("gae-lambda", gae_lambda)):
        if not 0 < value <= 1:
            parser.error(f"--{name} must be greater than 0 and at most 1")
    profile = replace(profile, gamma=gamma, gae_lambda=gae_lambda)
    output = args.output or Path(profile.output)
    total_games = args.total_games or profile.total_games
    games_per_update = args.games_per_update or profile.games_per_update
    optimization_epochs = args.optimization_epochs or profile.optimization_epochs
    minibatch_size = args.minibatch_size or profile.minibatch_size
    for name, value in (
        ("total-games", total_games), ("games-per-update", games_per_update),
        ("optimization-epochs", optimization_epochs), ("minibatch-size", minibatch_size),
    ):
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    seed_training(profile.seed, torch_threads=args.torch_threads or 1)

    resume_path = args.resume_from
    if resume_path is not None:
        checkpoint = _load(resume_path)
        model = NeuralActorCritic.from_checkpoint(checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        updates_seen = int(checkpoint.get("update_index", 0))
        games_seen = int(checkpoint.get("games_seen", 0))
        transitions_seen = int(checkpoint.get("transitions_seen", 0))
        metrics = list(checkpoint.get("training_metrics", []))
        best_evaluation = checkpoint.get("best_evaluation")
        best_evaluation_score = float(checkpoint.get("best_evaluation_score", "-inf"))
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
    else:
        initial_path = profile.initial_checkpoint
        if not initial_path:
            parser.error("PPO profile must define initial_checkpoint when starting a new run")
        checkpoint = _load(Path(initial_path))
        model = NeuralActorCritic.from_checkpoint(checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate)
        updates_seen = games_seen = transitions_seen = 0
        metrics = []
        best_evaluation = None
        best_evaluation_score = float("-inf")

    if not profile.initial_checkpoint:
        parser.error("PPO profile must define initial_checkpoint for the frozen reference policy")
    reference_checkpoint = _load(profile.resolve_path(profile.initial_checkpoint))
    reference_model = NeuralActorCritic.from_checkpoint(reference_checkpoint)
    reference_model.eval()
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    if not isinstance(best_evaluation, dict) or "by_opponent" not in best_evaluation:
        best_evaluation = evaluate_greedy_model(model, profile)
    best_evaluation_score = float(best_evaluation["score"])
    next_evaluation_games = profile.evaluation_interval_games

    if games_seen >= total_games:
        parser.error(f"checkpoint already contains games_seen={games_seen} >= total_games={total_games}")

    while games_seen < total_games:
        rollout_games = min(games_per_update, total_games - games_seen)
        rollout = collect_rollout(
            model,
            profile,
            start_game_index=games_seen,
            games=rollout_games,
            max_transitions=profile.max_transitions_per_update,
            workers=args.workers,
        )
        if rollout.games <= 0:
            raise RuntimeError("Rollout collected no games")
        update = ppo_update(
            model,
            optimizer,
            rollout.transitions,
            optimization_epochs=optimization_epochs,
            minibatch_size=minibatch_size,
            gamma=profile.gamma,
            gae_lambda=profile.gae_lambda,
            clip_epsilon=profile.clip_epsilon,
            value_loss_coefficient=profile.value_loss_coefficient,
            entropy_coefficient=profile.entropy_coefficient,
            reference_model=reference_model,
            reference_kl_coefficient=profile.reference_kl_coefficient,
        )
        updates_seen += 1
        games_seen += rollout.games
        transitions_seen += len(rollout.transitions)
        item = {
            "update": updates_seen,
            "games_seen": games_seen,
            "transitions_seen": transitions_seen,
            "games": rollout.games,
            "games_by_opponent": dict(rollout.games_by_opponent),
            "outcomes": {key: dict(value) for key, value in rollout.outcomes_by_opponent.items()},
            "win_rate": sum(value["win"] for value in rollout.outcomes_by_opponent.values()) / rollout.games,
            "ppo": asdict(update),
        }
        if games_seen >= next_evaluation_games:
            evaluation = evaluate_greedy_model(model, profile)
            item["evaluation"] = evaluation
            next_evaluation_games += profile.evaluation_interval_games
            if is_monotonic_evaluation_improvement(
                evaluation,
                best_evaluation,
                profile.opponents,
                tolerated_opponents=("random", "v007"),
                tolerance_rate=1.0 / profile.evaluation_games,
            ):
                best_evaluation_score = float(evaluation["score"])
                best_evaluation = evaluation
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                item["accepted_best"] = True
            else:
                model.load_state_dict(best_model_state)
                optimizer.load_state_dict(best_optimizer_state)
                item["restored_best"] = True
        item["best_evaluation_score"] = best_evaluation_score
        metrics.append(item)
        _save_checkpoint(
            output,
            model,
            optimizer,
            profile,
            updates_seen=updates_seen,
            games_seen=games_seen,
            transitions_seen=transitions_seen,
            metrics=metrics,
            best_evaluation=best_evaluation,
            best_evaluation_score=best_evaluation_score,
        )
        _write_metrics(metrics, output.with_suffix(".metrics.json"))
        print(json.dumps(item, sort_keys=True))
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    _save_checkpoint(
        output,
        model,
        optimizer,
        profile,
        updates_seen=updates_seen,
        games_seen=games_seen,
        transitions_seen=transitions_seen,
        metrics=metrics,
        best_evaluation=best_evaluation,
        best_evaluation_score=best_evaluation_score,
    )
    _write_metrics(metrics, output.with_suffix(".metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
