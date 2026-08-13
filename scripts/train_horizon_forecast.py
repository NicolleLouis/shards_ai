#!/usr/bin/env python3
"""Train and compare the turn-number baseline and enriched horizon classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shards_ai.ai.horizon_forecast import (
    HORIZON_FEATURE_SET_BASELINE,
    HORIZON_FEATURE_SET_V1,
    evaluate_model,
    iter_jsonl,
    save_model,
    split_for_game_id,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--baseline-dataset",
        type=Path,
        help="Projected turn-number-only JSONL; defaults to <dataset stem>_baseline.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=51200)
    parser.add_argument("--split-seed", type=int, default=51201)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    args = parser.parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise SystemExit("epochs and learning-rate must be positive")
    torch.set_num_threads(1)
    enriched_records = list(iter_jsonl(args.dataset))
    baseline_path = args.baseline_dataset or args.dataset.with_name(args.dataset.stem + "_baseline.jsonl")
    baseline_records = list(iter_jsonl(baseline_path))
    enriched_ids = {(record["game_id"], record["decision_index"]) for record in enriched_records}
    baseline_ids = {(record["game_id"], record["decision_index"]) for record in baseline_records}
    if enriched_ids != baseline_ids:
        raise SystemExit("Baseline and enriched datasets do not contain the same decision keys")
    split_records = {
        feature_set: {
            name: [record for record in values if split_for_game_id(str(record["game_id"]), args.split_seed) == name]
            for name in ("train", "validation", "test")
        }
        for feature_set, values in ((HORIZON_FEATURE_SET_BASELINE, baseline_records), (HORIZON_FEATURE_SET_V1, enriched_records))
    }
    for feature_set, splits in split_records.items():
        if not splits["train"] or not splits["validation"] or not splits["test"]:
            raise SystemExit(f"Every split needs records for {feature_set}: { {key: len(value) for key, value in splits.items()} }")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(args.dataset),
        "baseline_dataset": str(baseline_path),
        "records": len(enriched_records),
        "splits": {feature_set: {key: len(value) for key, value in splits.items()} for feature_set, splits in split_records.items()},
        "models": {},
    }
    for feature_set, filename in ((HORIZON_FEATURE_SET_BASELINE, "baseline.pt"), (HORIZON_FEATURE_SET_V1, "v1.pt")):
        splits = split_records[feature_set]
        model, history = train_model(splits["train"], feature_set, seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate)
        metrics = {split: evaluate_model(model, values, feature_set) for split, values in splits.items()}
        metadata = {"dataset": str(args.dataset), "split_seed": args.split_seed, "training_seed": args.seed, "history": history, "metrics": metrics}
        save_model(model, args.output_dir / filename, feature_set=feature_set, metadata=metadata)
        report["models"][feature_set] = {"checkpoint": str(args.output_dir / filename), "metrics": metrics, "epochs_ran": len(history)}
    report["comparison"] = {
        "test_accuracy_delta_v1_minus_baseline": report["models"][HORIZON_FEATURE_SET_V1]["metrics"]["test"]["accuracy"] - report["models"][HORIZON_FEATURE_SET_BASELINE]["metrics"]["test"]["accuracy"],
        "test_balanced_accuracy_delta_v1_minus_baseline": report["models"][HORIZON_FEATURE_SET_V1]["metrics"]["test"]["balanced_accuracy"] - report["models"][HORIZON_FEATURE_SET_BASELINE]["metrics"]["test"]["balanced_accuracy"],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
