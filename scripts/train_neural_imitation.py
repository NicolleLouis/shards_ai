#!/usr/bin/env python3
"""Train the first action-conditioned neural imitation model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from shards_ai.ai import (
    NeuralModelConfig, SUPPORTED_ARCHITECTURES, build_neural_scorer, evaluate_epoch, iter_jsonl_records,
    load_training_profile, seed_training, train_epoch,
)
from shards_ai.ai.neural_training import (
    is_targeted_mercenary_record, matches_imitation_slice, split_for_game_id,
)
from shards_ai.ai.neural_reporting import write_training_report


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _fingerprint(value: dict) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load model weights from a parent checkpoint and start with a fresh optimizer.",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming weights, start with a fresh Adam state.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"))
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-validation-records", type=int)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--chart-output", type=Path)
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES)
    parser.add_argument(
        "--strategic-action",
        dest="strategic_actions",
        action="append",
        help="Action type whose training loss receives --strategic-weight; repeatable.",
    )
    parser.add_argument("--strategic-weight", type=float, default=1.0)
    parser.add_argument(
        "--targeted-mercenary-weight",
        type=float,
        default=1.0,
        help="Additional loss weight for decisions offering BuyCard and RecruitMercenary for one mercenary.",
    )
    parser.add_argument("--target-action", dest="target_actions", action="append")
    parser.add_argument("--target-legal-actions-min", type=int)
    parser.add_argument("--target-legal-actions-max", type=int)
    args = parser.parse_args()

    profile = load_training_profile(args.profile) if args.profile else None
    if profile is None:
        if args.dataset is None or args.output is None:
            parser.error("--dataset and --output are required when --profile is absent")
        settings = {
            "dataset": args.dataset, "output": args.output, "epochs": args.epochs or 1,
            "validation_dataset": args.validation_dataset,
            "strategic_actions": args.strategic_actions or (),
            "strategic_weight": args.strategic_weight,
            "targeted_mercenary_weight": args.targeted_mercenary_weight,
            "target_actions": args.target_actions or (),
            "target_legal_actions_min": args.target_legal_actions_min,
            "target_legal_actions_max": args.target_legal_actions_max,
            "learning_rate": args.learning_rate or 1e-3, "split": args.split or "train",
            "split_seed": args.split_seed if args.split_seed is not None else 0,
            "max_records": args.max_records, "max_validation_records": args.max_validation_records or 10000,
            "seed": args.seed if args.seed is not None else 0,
            "torch_threads": args.torch_threads or 1, "model": NeuralModelConfig(),
            "architecture": args.architecture or "independent_action",
        }
    else:
        settings = {
            "dataset": args.dataset or profile.resolve_path(profile.dataset),
            "output": args.output or profile.resolve_path(profile.output),
            "validation_dataset": (
                args.validation_dataset
                or profile.resolve_path(profile.validation_dataset)
                if profile.validation_dataset
                else args.validation_dataset
            ),
            "strategic_actions": args.strategic_actions or (),
            "strategic_weight": args.strategic_weight,
            "targeted_mercenary_weight": args.targeted_mercenary_weight,
            "target_actions": args.target_actions or (),
            "target_legal_actions_min": args.target_legal_actions_min,
            "target_legal_actions_max": args.target_legal_actions_max,
            "epochs": args.epochs if args.epochs is not None else profile.epochs,
            "learning_rate": args.learning_rate if args.learning_rate is not None else profile.learning_rate,
            "split": args.split or "train", "split_seed": args.split_seed if args.split_seed is not None else profile.split_seed,
            "max_records": args.max_records if args.max_records is not None else profile.max_records,
            "max_validation_records": args.max_validation_records if args.max_validation_records is not None else profile.max_validation_records,
            "seed": args.seed if args.seed is not None else profile.seed,
            "torch_threads": args.torch_threads if args.torch_threads is not None else profile.torch_threads,
            "model": profile.resolved_model_config(),
            "architecture": args.architecture or profile.metadata.get("architecture", "independent_action"),
        }
    dataset = Path(settings["dataset"])
    output = Path(settings["output"])
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    validation_dataset = settings["validation_dataset"]
    if validation_dataset is not None:
        validation_dataset = Path(validation_dataset)
        if not validation_dataset.exists():
            raise FileNotFoundError(f"Validation dataset not found: {validation_dataset}")
    if args.strategic_weight <= 0:
        parser.error("--strategic-weight must be positive")
    if args.targeted_mercenary_weight <= 0:
        parser.error("--targeted-mercenary-weight must be positive")
    if args.initialize_from is not None and args.resume_from is not None:
        parser.error("--initialize-from and --resume-from are mutually exclusive")
    if (args.target_legal_actions_min is not None and args.target_legal_actions_min <= 0) or (
        args.target_legal_actions_max is not None and args.target_legal_actions_max <= 0
    ):
        parser.error("target legal-action bounds must be positive")
    if (
        args.target_legal_actions_min is not None
        and args.target_legal_actions_max is not None
        and args.target_legal_actions_min > args.target_legal_actions_max
    ):
        parser.error("target legal-action minimum cannot exceed maximum")
    if settings["epochs"] <= 0 or settings["learning_rate"] <= 0 or settings["torch_threads"] <= 0:
        parser.error("epochs, learning rate and torch threads must be positive")

    seed_training(settings["seed"], torch_threads=settings["torch_threads"])
    initialize_from = args.initialize_from
    if initialize_from is None and profile is not None and profile.initial_checkpoint:
        initialize_from = profile.resolve_path(profile.initial_checkpoint)
    checkpoint = None
    is_initialization = initialize_from is not None
    checkpoint_path = initialize_from or args.resume_from
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = NeuralModelConfig(**checkpoint["model_config"])
        architecture = checkpoint.get("architecture", "independent_action")
    else:
        model_config = settings["model"]
        architecture = settings["architecture"]
    if profile and checkpoint is not None and not is_initialization and checkpoint.get("profile_id") not in (None, profile.profile_id):
        raise ValueError(
            f"Checkpoint belongs to profile {checkpoint['profile_id']!r}, "
            f"cannot resume with {profile.profile_id!r}"
        )
    model = build_neural_scorer(architecture, model_config)
    if checkpoint is not None:
        if tuple(checkpoint.get("card_ids", ())) != model.card_ids:
            raise ValueError("Checkpoint card vocabulary does not match the current card catalog")
        model.load_state_dict(checkpoint["model_state_dict"])
    # The training loop performs one optimizer step per legal-action decision.
    # Adam's CPU foreach implementation reduces Python/dispatch overhead without
    # changing the batch size, record order, or loss definition.
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], foreach=True)
    if checkpoint is not None and not is_initialization and "optimizer_state_dict" in checkpoint and not args.reset_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = settings["learning_rate"]
    metrics = list(checkpoint.get("metrics", [])) if checkpoint is not None and not is_initialization else []
    first_epoch = int(checkpoint.get("epoch", len(metrics))) if checkpoint is not None and not is_initialization else 0
    metrics_output = args.metrics_output or output.with_suffix(".metrics.json")
    chart_output = args.chart_output or output.with_suffix(".svg")

    def records_for(requested_split: str):
        for record in iter_jsonl_records(dataset):
            game_id = str(record.get("game_id", record.get("seed", record["_line_number"])))
            if requested_split == "all" or split_for_game_id(game_id, seed=settings["split_seed"]) == requested_split:
                yield record

    def records():
        return (
            record for record in records_for(settings["split"])
            if matches_imitation_slice(
                record,
                action_types=frozenset(settings["target_actions"]),
                min_legal_actions=settings["target_legal_actions_min"],
                max_legal_actions=settings["target_legal_actions_max"],
            )
        )

    def validation_records():
        if validation_dataset is not None:
            yield from iter_jsonl_records(validation_dataset)
            return
        yield from records_for("validation")

    strategic_actions = frozenset(args.strategic_actions or ())

    def record_weight(record: dict) -> float:
        action_type = record.get("chosen_action", {}).get("action_type")
        weight = float(record.get("sample_weight", 1.0))
        if weight <= 0:
            raise ValueError("sample_weight must be positive")
        weight *= args.strategic_weight if action_type in strategic_actions else 1.0
        if is_targeted_mercenary_record(record):
            weight *= args.targeted_mercenary_weight
        return weight

    for epoch_offset in range(settings["epochs"]):
        train_result = train_epoch(
            model,
            records(),
            optimizer,
            max_records=settings["max_records"],
            record_weight=record_weight,
        )
        validation_result = evaluate_epoch(
            model,
            validation_records(),
            max_records=settings["max_validation_records"],
        )
        metrics.append({
            "epoch": first_epoch + epoch_offset + 1,
            "train": asdict(train_result),
            "validation": asdict(validation_result),
        })
        print(json.dumps(metrics[-1], sort_keys=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "profile_id": profile.profile_id if profile else None,
        "profile_fingerprint": profile.fingerprint if profile else None,
        "dataset": str(dataset), "output": str(output), "architecture": architecture,
        "settings": _jsonable(settings),
    }
    run_metadata["effective_fingerprint"] = _fingerprint(run_metadata["settings"])
    metrics_output.write_text(json.dumps({"metadata": run_metadata, "epochs": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_metrics_csv(metrics, metrics_output.with_suffix(".csv"))
    write_training_report(metrics, metrics_output.with_suffix(".html"))
    if not args.no_chart:
        chart_output.parent.mkdir(parents=True, exist_ok=True)
        _write_metrics_svg(metrics, chart_output)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "card_ids": model.card_ids,
            "metrics": metrics,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": metrics[-1]["epoch"] if metrics else first_epoch,
            "seed": settings["seed"],
            "split": settings["split"],
            "split_seed": settings["split_seed"],
            "profile_id": profile.profile_id if profile else None,
            "architecture": architecture,
            "profile_fingerprint": profile.fingerprint if profile else None,
            "effective_fingerprint": run_metadata["effective_fingerprint"],
            "training_config": run_metadata,
        },
        output,
    )


def _write_metrics_csv(metrics: list[dict], path: Path) -> None:
    fields = (
        "epoch", "train_records", "train_mean_loss", "validation_records", "validation_mean_loss",
        "validation_top1_accuracy", "validation_mean_chosen_rank",
        "validation_mean_normalized_chosen_rank", "validation_pairwise_accuracy",
        "validation_pairwise_pairs",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                "epoch": item["epoch"],
                "train_records": item["train"]["records"],
                "train_mean_loss": item["train"]["mean_loss"],
                **{f"validation_{key}": value for key, value in item["validation"].items()},
            })


def _write_metrics_svg(metrics: list[dict], path: Path) -> None:
    width, height = 900, 680
    panels = [
        ("Loss", ("train_mean_loss", "validation_mean_loss"), ("#2563eb", "#dc2626")),
        ("Action choisie", ("validation_top1_accuracy", "validation_mean_normalized_chosen_rank"), ("#16a34a", "#9333ea")),
        ("Classement paire-à-paire", ("validation_pairwise_accuracy",), ("#ea580c",)),
    ]
    values = {
        key: [
            item["train"]["mean_loss"] if key == "train_mean_loss" else
            item["validation"][key.removeprefix("validation_")]
            for item in metrics
        ]
        for _title, keys, _colors in panels for key in keys
    }
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family: sans-serif;font-size:13px}.title{font-size:16px;font-weight:bold}</style>',
             '<rect width="100%" height="100%" fill="white"/>']
    panel_height = 200
    for panel_index, (title, keys, colors) in enumerate(panels):
        x, y, w, h = 70, 35 + panel_index * panel_height, 780, 130
        series = [value for key in keys for value in values[key]]
        minimum, maximum = min(series or [0]), max(series or [1])
        if maximum == minimum:
            maximum = minimum + 1
        parts.append(f'<text x="{x}" y="{y - 12}" class="title">{title}</text>')
        parts.append(f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="#999"/>')
        for key, color in zip(keys, colors):
            points = []
            for index, value in enumerate(values[key]):
                px = x + (w * index / max(1, len(metrics) - 1))
                py = y + h - (h * (value - minimum) / (maximum - minimum))
                points.append(f"{px:.1f},{py:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{x+w-180}" y="{y+15+keys.index(key)*18}" fill="{color}">{key.replace("validation_", "")}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
