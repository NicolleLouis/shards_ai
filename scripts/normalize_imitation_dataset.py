#!/usr/bin/env python3
"""Create a deterministic, phase-balanced imitation training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from shards_ai.ai.neural_training import split_for_game_id


DEFAULT_TARGET_RECORDS = 100_000
DEFAULT_SPLIT_SEED = 0
DEFAULT_SELECTION_SEED = 88008
DEFAULT_PHASE_FRACTIONS = {"buy": 0.60, "attack": 0.10, "play": 0.30}
DEFAULT_PLAY_STRATEGIC_FRACTION = 0.05


def _records(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def _action_type(record: dict) -> str:
    return str(record["chosen_action"]["action_type"])


def _phase(record: dict) -> str:
    return str(record["observation"]["phase"])


def _game_id(record: dict) -> str:
    return str(record.get("game_id", record.get("game_seed", "")))


def _allocate(total: int, weights: Counter[str] | dict[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must not be negative")
    positive = {key: float(value) for key, value in weights.items() if float(value) > 0}
    if total and not positive:
        raise ValueError("Cannot allocate a positive total without positive weights")
    weight_total = sum(positive.values())
    raw = {key: total * value / weight_total for key, value in positive.items()}
    allocation = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def _target_quotas(
    reference_counts: dict[str, Counter[str]],
    target_records: int,
    phase_fractions: dict[str, float],
    play_strategic_fraction: float,
) -> dict[tuple[str, str], int]:
    if abs(sum(phase_fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Phase fractions must sum to 1")
    if not 0 <= play_strategic_fraction <= 1:
        raise ValueError("play strategic fraction must be between 0 and 1")

    phase_targets = _allocate(target_records, phase_fractions)
    quotas: dict[tuple[str, str], int] = {}
    buy_targets = _allocate(phase_targets.get("buy", 0), reference_counts.get("buy", Counter()))
    for action_type, count in buy_targets.items():
        quotas[("buy", action_type)] = count
    attack_targets = _allocate(phase_targets.get("attack", 0), reference_counts.get("attack", Counter()))
    for action_type, count in attack_targets.items():
        quotas[("attack", action_type)] = count

    play_total = phase_targets.get("play", 0)
    strategic_total = min(round(target_records * play_strategic_fraction), play_total)
    play_counts = reference_counts.get("play", Counter())
    special_counts = Counter({key: play_counts[key] for key in ("banish_card", "skip_banish")})
    special_targets = _allocate(strategic_total, special_counts)
    for action_type, count in special_targets.items():
        quotas[("play", action_type)] = count
    remaining_counts = Counter(play_counts)
    remaining_counts.subtract(special_counts)
    remaining_targets = _allocate(play_total - strategic_total, remaining_counts)
    for action_type, count in remaining_targets.items():
        quotas[("play", action_type)] = count
    return quotas


def _rank(seed: int, bucket: tuple[str, str], line_number: int, record: dict) -> int:
    payload = f"{seed}:{bucket[0]}:{bucket[1]}:{_game_id(record)}:{record.get('decision_index', '')}:{line_number}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def _select_lines(
    input_path: Path,
    quotas: dict[tuple[str, str], int],
    *,
    selection_seed: int,
    split_seed: int,
) -> tuple[dict[tuple[str, str], set[int]], Counter[tuple[str, str]], Counter[str]]:
    selected: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    available: Counter[tuple[str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    for line_number, record in _records(input_path):
        split = split_for_game_id(_game_id(record), seed=split_seed)
        split_counts[split] += 1
        if split != "train":
            continue
        bucket = (_phase(record), _action_type(record))
        if bucket not in quotas:
            continue
        available[bucket] += 1
        limit = quotas[bucket]
        if limit <= 0:
            continue
        ranked = selected[bucket]
        item = (_rank(selection_seed, bucket, line_number, record), line_number)
        if len(ranked) < limit:
            ranked.append(item)
            ranked.sort(reverse=True)
        elif item < ranked[0]:
            ranked[0] = item
            ranked.sort(reverse=True)
    return (
        {bucket: {line for _rank_value, line in values} for bucket, values in selected.items()},
        available,
        split_counts,
    )


def normalize_dataset(
    input_path: Path,
    output_path: Path,
    *,
    reference_path: Path | None = None,
    validation_output: Path | None = None,
    test_output: Path | None = None,
    target_records: int = DEFAULT_TARGET_RECORDS,
    split_seed: int = DEFAULT_SPLIT_SEED,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    play_strategic_fraction: float = DEFAULT_PLAY_STRATEGIC_FRACTION,
) -> dict:
    if target_records <= 0:
        raise ValueError("target_records must be positive")
    reference_path = reference_path or input_path
    reference_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _line_number, record in _records(reference_path):
        reference_counts[_phase(record)][_action_type(record)] += 1
    quotas = _target_quotas(
        reference_counts,
        target_records,
        DEFAULT_PHASE_FRACTIONS,
        play_strategic_fraction,
    )
    selected, available, split_counts = _select_lines(
        input_path,
        quotas,
        selection_seed=selection_seed,
        split_seed=split_seed,
    )
    selected_lines = set().union(*selected.values()) if selected else set()
    validation_output = validation_output or output_path.with_suffix(".validation.jsonl")
    test_output = test_output or output_path.with_suffix(".test.jsonl")
    for path in (output_path, validation_output, test_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary_paths: list[tuple[Path, Path]] = []
    try:
        streams = {}
        for label, path in (("train", output_path), ("validation", validation_output), ("test", test_output)):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary_paths.append((temporary, path))
            streams[label] = temporary.open("w", encoding="utf-8")
        selected_counts: Counter[tuple[str, str]] = Counter()
        natural_counts: Counter[str] = Counter()
        for line_number, record in _records(input_path):
            split = split_for_game_id(_game_id(record), seed=split_seed)
            if split == "train":
                if line_number not in selected_lines:
                    continue
                selected_counts[(_phase(record), _action_type(record))] += 1
            else:
                natural_counts[split] += 1
            streams[split].write(json.dumps(record, sort_keys=True) + "\n")
        for stream in streams.values():
            stream.close()
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    except Exception:
        for stream in locals().get("streams", {}).values():
            if not stream.closed:
                stream.close()
        for temporary, _destination in temporary_paths:
            if temporary.exists():
                temporary.unlink()
        raise

    manifest = {
        "schema_version": 1,
        "input": str(input_path),
        "reference": str(reference_path),
        "output": str(output_path),
        "validation_output": str(validation_output),
        "test_output": str(test_output),
        "target_records": target_records,
        "split_seed": split_seed,
        "selection_seed": selection_seed,
        "phase_fractions": DEFAULT_PHASE_FRACTIONS,
        "play_strategic_fraction": play_strategic_fraction,
        "source_split_counts": dict(split_counts),
        "quotas": {f"{phase}:{action}": count for (phase, action), count in sorted(quotas.items())},
        "available_train": {f"{phase}:{action}": count for (phase, action), count in sorted(available.items())},
        "selected_train": {f"{phase}:{action}": count for (phase, action), count in sorted(selected_counts.items())},
        "shortfalls": {
            f"{phase}:{action}": quotas[(phase, action)] - selected_counts[(phase, action)]
            for phase, action in quotas
            if selected_counts[(phase, action)] < quotas[(phase, action)]
        },
        "natural_split_counts": dict(natural_counts),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--target-records", type=int, default=DEFAULT_TARGET_RECORDS)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--play-strategic-fraction", type=float, default=DEFAULT_PLAY_STRATEGIC_FRACTION)
    args = parser.parse_args()
    manifest = normalize_dataset(
        args.input,
        args.output,
        reference_path=args.reference_dataset,
        validation_output=args.validation_output,
        test_output=args.test_output,
        target_records=args.target_records,
        split_seed=args.split_seed,
        selection_seed=args.selection_seed,
        play_strategic_fraction=args.play_strategic_fraction,
    )
    print(json.dumps({"output": manifest["output"], "selected_train": manifest["selected_train"], "shortfalls": manifest["shortfalls"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
