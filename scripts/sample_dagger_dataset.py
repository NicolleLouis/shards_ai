#!/usr/bin/env python3
"""Assemble historical and prioritized on-policy DAgGER records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shards_ai.analysis.dagger_dataset import sample_dataset, sample_on_policy_dataset


def parse_action_weights(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"action weight must use action=multiplier: {value}")
        action, multiplier = value.split("=", 1)
        result[action] = float(multiplier)
    if any(value <= 0 for value in result.values()):
        raise ValueError("action weights must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dataset", type=Path)
    parser.add_argument("--dagger-dataset", type=Path, required=True)
    parser.add_argument("--on-policy-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--target-records", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--action-weight", action="append", default=[], metavar="ACTION=MULTIPLIER")
    args = parser.parse_args()
    if not args.on_policy_only and args.old_dataset is None:
        parser.error("--old-dataset is required unless --on-policy-only is used")
    action_weights = parse_action_weights(args.action_weight)
    if args.on_policy_only:
        manifest = sample_on_policy_dataset(args.dagger_dataset, args.output, args.validation_output, target_records=args.target_records, seed=args.seed, action_weights=action_weights)
    else:
        manifest = sample_dataset(args.old_dataset, args.dagger_dataset, args.output, args.validation_output, target_records=args.target_records, seed=args.seed, action_weights=action_weights)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
