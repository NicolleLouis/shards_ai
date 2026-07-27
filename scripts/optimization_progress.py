#!/usr/bin/env python3
"""Display persisted progress for a heuristic optimization checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    with args.checkpoint.open(encoding="utf-8") as stream:
        checkpoint = json.load(stream)

    consumed = float(checkpoint.get("compute_seconds_consumed", 0.0))
    target = float(checkpoint.get("compute_seconds_target", 0.0))
    remaining = max(0.0, target - consumed)
    percentage = 100.0 * consumed / target if target else 0.0

    print(f"checkpoint={args.checkpoint}")
    print(f"phase={checkpoint.get('phase', 'unknown')}")
    print(f"next_batch={checkpoint.get('next_batch', 0)}")
    print(
        "consecutive_failed_batches="
        f"{checkpoint.get('consecutive_failed_batches', 0)}"
    )
    print(f"compute_consumed={_format_duration(consumed)} ({consumed:.2f}s)")
    print(f"compute_target={_format_duration(target)} ({target:.2f}s)")
    print(f"compute_remaining={_format_duration(remaining)} ({remaining:.2f}s)")
    print(f"progress_percent={percentage:.2f}")


if __name__ == "__main__":
    main()
