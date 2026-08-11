#!/usr/bin/env python3
"""Train the strategic macro scorer from the generated Heuristic V8 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from shards_ai.ai import (
    build_neural_scorer,
    load_training_profile,
    migrate_v004_checkpoint_to_deck_state,
    seed_training,
)
from shards_ai.ai.macro_model import (
    MACRO_V2_ARCHITECTURE,
    MACRO_V3_ARCHITECTURE,
    MACRO_V4_ARCHITECTURE,
    MacroActionScorerV2,
    MacroActionScorerV3,
    MacroActionScorerV4,
)
from shards_ai.ai.macro_training import evaluate_macro_epoch, macro_records, train_macro_epoch, unified_records
from shards_ai.ai.neural_training import split_for_game_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/neural_training_profiles/candidates/exp00112-macro-v4-tactical-action.yaml"),
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume-from", type=Path)
    initialization.add_argument("--initialize-from", type=Path)
    initialization.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not initialize from V004; train the macro model with fresh weights.",
    )
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--output", type=Path, help="Override the profile output; use NEURAL_CHECKPOINT in Make targets.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = load_training_profile(args.profile)
    dataset = profile.resolve_path(profile.dataset)
    output = args.output or profile.resolve_path(profile.output)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_training(profile.seed, torch_threads=profile.torch_threads)

    architecture = str(profile.metadata.get("architecture", MACRO_V4_ARCHITECTURE))
    if architecture not in {MACRO_V2_ARCHITECTURE, MACRO_V3_ARCHITECTURE, MACRO_V4_ARCHITECTURE}:
        raise ValueError(f"Macro trainer requires a supported macro architecture, got {architecture!r}")
    model = build_neural_scorer(architecture, profile.resolved_model_config())
    if architecture == MACRO_V3_ARCHITECTURE and not isinstance(model, MacroActionScorerV3):
        raise TypeError("V3 macro architecture did not build MacroActionScorerV3")
    if architecture == MACRO_V4_ARCHITECTURE and not isinstance(model, MacroActionScorerV4):
        raise TypeError("V4 macro architecture did not build MacroActionScorerV4")
    if architecture == MACRO_V2_ARCHITECTURE and not isinstance(model, MacroActionScorerV2):
        raise TypeError("V2 macro architecture did not build MacroActionScorerV2")
    if architecture == MACRO_V4_ARCHITECTURE:
        if manifest.get("dataset_schema_version") != 3:
            raise ValueError("V4 macro training requires dataset_schema_version=3")
        if manifest.get("candidate_schema_version") != 4:
            raise ValueError("V4 macro training requires candidate_schema_version=4")
        if manifest.get("candidate_feature_set") != "root_action_plus_known_consequence_plus_tactical_v1":
            raise ValueError("Dataset candidate feature set does not match the V4 macro architecture")

    checkpoint_path = args.resume_from
    initialize_path = args.initialize_from
    if not args.from_scratch and initialize_path is None and checkpoint_path is None:
        configured_initial = profile.metadata.get("initial_checkpoint")
        initialize_path = profile.resolve_path(configured_initial) if configured_initial else None

    checkpoint = None
    first_epoch = 0
    if initialize_path is not None:
        checkpoint = torch.load(initialize_path, map_location="cpu", weights_only=False)
        migrated = migrate_v004_checkpoint_to_deck_state(checkpoint)
        model.load_state_dict(migrated["model_state_dict"], strict=False)
    elif checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != architecture:
            raise ValueError("Resume checkpoint architecture does not match the macro profile")
        model.load_state_dict(checkpoint["model_state_dict"])
        first_epoch = int(checkpoint.get("epoch", 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate, foreach=True)
    if checkpoint_path is not None and checkpoint and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = profile.learning_rate

    records_reader = unified_records if architecture == MACRO_V4_ARCHITECTURE else macro_records

    configured_weights = profile.metadata.get("decision_kind_weights", {"macro_play": 1.0, "atomic": 1.0})
    if not isinstance(configured_weights, dict) or set(configured_weights) != {"macro_play", "atomic"}:
        raise ValueError("decision_kind_weights must define exactly macro_play and atomic")
    decision_kind_weights = {key: float(value) for key, value in configured_weights.items()}
    if any(value <= 0 for value in decision_kind_weights.values()):
        raise ValueError("decision_kind_weights must be positive")

    def record_weight(record):
        return decision_kind_weights[str(record.get("decision_kind"))]

    def training_records():
        for record in records_reader(dataset):
            if split_for_game_id(str(record["game_id"]), seed=profile.split_seed) == "train":
                yield record

    def validation_records():
        for record in records_reader(dataset):
            if split_for_game_id(str(record["game_id"]), seed=profile.split_seed) == "validation":
                yield record

    metrics = list(checkpoint.get("metrics", [])) if checkpoint and checkpoint_path is not None else []
    for offset in range(profile.epochs):
        train_result = train_macro_epoch(
            model, training_records(), optimizer, max_records=profile.max_records,
            record_weight=record_weight,
        )
        validation_result = evaluate_macro_epoch(
            model, validation_records(), max_records=profile.max_validation_records,
            record_weight=record_weight,
        )
        item = {
            "epoch": first_epoch + offset + 1,
            "train": asdict(train_result),
            "validation": asdict(validation_result),
        }
        metrics.append(item)
        print(json.dumps(item, sort_keys=True))

    metrics_output = args.metrics_output or output.with_suffix(".metrics.json")
    metadata = {
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "dataset": str(dataset),
        "output": str(output),
        "architecture": architecture,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_fingerprint": manifest.get("card_catalog_fingerprint"),
        "settings": profile.resolved_document(),
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps({"metadata": metadata, "epochs": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(metrics, metrics_output.with_suffix(".csv"))

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "card_ids": model.card_ids,
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": metrics[-1]["epoch"] if metrics else first_epoch,
        "seed": profile.seed,
        "split": "game_id",
        "split_seed": profile.split_seed,
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "architecture": architecture,
        "training_config": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def _write_csv(metrics, path: Path) -> None:
    fields = (
        "epoch", "train_records", "train_mean_loss", "validation_records",
        "validation_mean_loss", "validation_top1_accuracy", "validation_mean_chosen_rank",
        "validation_mean_normalized_chosen_rank",
        "validation_mean_candidate_count", "validation_collision_records",
        "validation_teacher_collision_records", "validation_by_decision_kind",
        "validation_by_phase", "validation_by_action_type", "validation_by_matchup",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                "epoch": item["epoch"],
                "train_records": item["train"]["records"],
                "train_mean_loss": item["train"]["mean_loss"],
                "validation_records": item["validation"]["records"],
                "validation_mean_loss": item["validation"]["mean_loss"],
                "validation_top1_accuracy": item["validation"]["top1_accuracy"],
                "validation_mean_chosen_rank": item["validation"]["mean_chosen_rank"],
                "validation_mean_normalized_chosen_rank": item["validation"]["mean_normalized_chosen_rank"],
                "validation_mean_candidate_count": item["validation"].get("mean_candidate_count", 0.0),
                "validation_collision_records": item["validation"].get("collision_records", 0),
                "validation_teacher_collision_records": item["validation"].get("teacher_collision_records", 0),
                "validation_by_decision_kind": json.dumps(item["validation"].get("by_decision_kind", {}), sort_keys=True),
                "validation_by_phase": json.dumps(item["validation"].get("by_phase", {}), sort_keys=True),
                "validation_by_action_type": json.dumps(item["validation"].get("by_action_type", {}), sort_keys=True),
                "validation_by_matchup": json.dumps(item["validation"].get("by_matchup", {}), sort_keys=True),
            })


if __name__ == "__main__":
    main()
