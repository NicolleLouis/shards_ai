#!/usr/bin/env python3
"""Generate a standalone HTML report from neural training metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from shards_ai.ai.neural_reporting import load_metrics, write_training_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_training_report(load_metrics(args.metrics), args.output)


if __name__ == "__main__":
    main()
