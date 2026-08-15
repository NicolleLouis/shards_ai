#!/usr/bin/env python3
"""Promote the best state of a hybrid deckbuilding PPO checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from shards_ai.ai.rl_training import NeuralActorCritic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-acquisition-profile", type=Path, required=True)
    parser.add_argument("--output-composition-profile", type=Path, required=True)
    parser.add_argument("--stable-id", required=True)
    parser.add_argument("--hybrid-id", required=True)
    parser.add_argument("--parent-stable-id", default="v007")
    parser.add_argument("--parent-hybrid-id", default="hybrid-v004")
    parser.add_argument("--parent-acquisition-policy", default="neural_v007")
    args = parser.parse_args()

    if args.output_checkpoint.exists():
        parser.error(f"Refusing to overwrite existing checkpoint: {args.output_checkpoint}")
    for path in (args.output_profile, args.output_acquisition_profile, args.output_composition_profile):
        if path.exists():
            parser.error(f"Refusing to overwrite existing profile: {path}")

    payload = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=False)
    best_state = payload.get("best_actor_critic_state_dict")
    if best_state is None:
        parser.error("Candidate checkpoint does not contain best_actor_critic_state_dict")

    best_model = NeuralActorCritic.from_checkpoint({**payload, "actor_critic_state_dict": best_state})
    promoted = copy.deepcopy(payload)
    promoted["model_state_dict"] = best_model.inference_state_dict()
    promoted["actor_critic_state_dict"] = copy.deepcopy(best_state)
    promoted["latest_actor_critic_state_dict"] = copy.deepcopy(best_state)
    if "best_optimizer_state_dict" in payload:
        promoted["optimizer_state_dict"] = copy.deepcopy(payload["best_optimizer_state_dict"])
        promoted["latest_optimizer_state_dict"] = copy.deepcopy(payload["best_optimizer_state_dict"])
    promoted["profile_id"] = args.stable_id
    promoted["parent_profile_id"] = args.parent_stable_id
    promoted["promoted_from_profile_id"] = payload.get("profile_id")
    promoted["promoted_from_checkpoint"] = str(args.candidate_checkpoint)
    promoted["promotion_source"] = "best_actor_critic_state_dict"

    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    profile["profile_id"] = args.stable_id
    profile["parent_profile_id"] = args.parent_stable_id
    profile["output"] = str(args.output_checkpoint)
    profile["initial_checkpoint"] = f"configs/neural_profiles/{args.parent_stable_id}.pt"
    profile["composition_profile"] = str(args.output_composition_profile)
    profile.setdefault("metadata", {})["promotion_source"] = (
        f"{payload.get('profile_id', 'candidate')}-best-update{payload.get('best_update_index')}"
    )
    profile["metadata"]["parent_stable_profile"] = args.parent_hybrid_id

    acquisition = {
        "schema_version": 1,
        "policy_id": f"neural_{args.stable_id}",
        "family": "acquisition",
        "version": args.stable_id,
        "checkpoint": str(args.output_checkpoint),
        "contract": "macro_candidate_scorer",
        "parent_policy": args.parent_acquisition_policy,
    }
    composition = {
        "schema_version": 1,
        "profile_id": args.hybrid_id,
        "parent_profile_id": args.parent_hybrid_id,
        "policies": {
            "acquisition": {
                "policy_id": f"neural_{args.stable_id}",
                "checkpoint": str(args.output_checkpoint),
                "profile": str(args.output_acquisition_profile),
            },
            "play": {
                "policy_id": "heuristic_v008",
                "profile": "configs/heuristic_profiles/v008.yaml",
            },
            "banish": {
                "policy_id": "deterministic_blaster_crystal",
                "profile": "configs/player_policies/banish/v001.yaml",
            },
        },
        "metadata": {
            "objective": "promoted_ppo_deckbuilding_with_fixed_play_banish",
            "capability_profile_id": "boundary_gain_mastery_v1",
            "composition": f"acquisition_{args.stable_id}_play_v008_banish_v001",
            "promotion_source": profile["metadata"]["promotion_source"],
            "gate": "official_quality_gate_200_games_per_opponent",
        },
    }

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_checkpoint.with_suffix(args.output_checkpoint.suffix + ".tmp")
    torch.save(promoted, temporary)
    temporary.replace(args.output_checkpoint)
    for path, document in (
        (args.output_profile, profile),
        (args.output_acquisition_profile, acquisition),
        (args.output_composition_profile, composition),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    print(json.dumps({
        "stable_profile": args.stable_id,
        "hybrid_profile": args.hybrid_id,
        "checkpoint": str(args.output_checkpoint),
        "source": str(args.candidate_checkpoint),
        "source_best_update": payload.get("best_update_index"),
        "source_best_games": payload.get("best_games_seen"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
