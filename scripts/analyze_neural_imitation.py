#!/usr/bin/env python3
"""Measure offline agreement and heuristic regret on an imitation JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from shards_ai.ai import NeuralModelConfig, build_neural_scorer
from shards_ai.ai.neural_training import iter_jsonl_records, split_for_game_id
from shards_ai.analysis.neural_imitation import analyze_records, write_html, write_json


DEFAULT_DATASET = Path("artifacts/imitation_dataset/v008_vs_random_v007_1m.jsonl")
DEFAULT_CHECKPOINT = Path("artifacts/neural_training/checkpoint.pt")
DEFAULT_OUTPUT = Path("artifacts/analysis/neural_imitation_v008_1m.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--split", choices=("non_train", "train", "validation", "test", "all"), default="non_train",
        help="Default non_train evaluates validation and test only.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--game-id-prefix", help="Optional prefix filter, useful to isolate DAgGER records.")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if not args.dataset.exists():
        parser.error(f"Dataset not found: {args.dataset}")
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    if args.progress_every <= 0 or args.torch_threads <= 0:
        parser.error("--progress-every and --torch-threads must be positive")

    torch.set_num_threads(args.torch_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config")
    model = build_neural_scorer(
        str(checkpoint.get("architecture", "independent_action")),
        NeuralModelConfig(**config) if config is not None else None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    def selected_records():
        selected = 0
        for record in iter_jsonl_records(args.dataset):
            if args.game_id_prefix is not None:
                game_id = str(record.get("game_id", record.get("seed", record["_line_number"])))
                if not game_id.startswith(args.game_id_prefix):
                    continue
            if args.split != "all":
                game_id = str(record.get("game_id", record.get("seed", record["_line_number"])))
                split = split_for_game_id(game_id, seed=args.split_seed)
                if args.split == "non_train" and split == "train":
                    continue
                if args.split != "non_train" and split != args.split:
                    continue
            yield record
            selected += 1
            if selected % args.progress_every == 0:
                print(f"analysed={selected}", flush=True)
            if args.max_records is not None and selected >= args.max_records:
                return

    result = analyze_records(model, selected_records())
    metadata = {
        "dataset": str(args.dataset), "checkpoint": str(args.checkpoint),
        "split": args.split, "split_seed": args.split_seed, "game_id_prefix": args.game_id_prefix,
        "records": result.overall.records,
        "checkpoint_profile": checkpoint.get("profile_id", "unknown"),
    }
    write_html(result, args.output, metadata=metadata)
    if args.json_output:
        write_json(result, args.json_output, metadata=metadata)
    print(json.dumps({"output": str(args.output), **result.overall.as_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
