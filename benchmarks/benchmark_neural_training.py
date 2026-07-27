"""Reproducible CPU benchmark for neural imitation training and validation."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from shards_ai.ai import NeuralActionScorer, evaluate_epoch, iter_jsonl_records, seed_training, train_epoch
from shards_ai.ai.neural_training import split_for_game_id


def records(path: str, split: str, limit: int):
    count = 0
    for record in iter_jsonl_records(path):
        game_id = str(record.get("game_id", record["_line_number"]))
        if split_for_game_id(game_id) != split:
            continue
        yield record
        count += 1
        if count >= limit:
            return


def run(path: str, train_records: int, validation_records: int, torch_threads: int) -> tuple[float, float, int, int]:
    seed_training(123, torch_threads=torch_threads)
    model = NeuralActionScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.perf_counter()
    train_result = train_epoch(model, records(path, "train", train_records), optimizer)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    validation_result = evaluate_epoch(model, records(path, "validation", validation_records))
    validation_seconds = time.perf_counter() - started
    return train_seconds, validation_seconds, train_result.records, validation_result.records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-records", type=int, default=100)
    parser.add_argument("--validation-records", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    measurements = [run(args.dataset, args.train_records, args.validation_records, args.torch_threads) for _ in range(args.repetitions)]
    for index, measurement in enumerate(measurements, 1):
        print(f"run={index} train_s={measurement[0]:.4f} validation_s={measurement[1]:.4f} "
              f"train_records={measurement[2]} validation_records={measurement[3]}")
    train_median = statistics.median(item[0] for item in measurements)
    validation_median = statistics.median(item[1] for item in measurements)
    print(f"median_train_s={train_median:.4f}")
    print(f"median_validation_s={validation_median:.4f}")
    print(f"median_total_s={train_median + validation_median:.4f}")


if __name__ == "__main__":
    main()
