from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


# These prefixes are deliberately repository-relative and checked before any candidate commit.
FORBIDDEN_PATH_PREFIXES = (
    "shards_ai/game/",
    "shards_ai/ai/heuristic_",
    "configs/heuristic_profiles/",
    "configs/neural_profiles/",
)

FORBIDDEN_PATHS = {"configs/neural_training_profiles/active.yaml"}


def validate_changed_paths(paths: list[str] | tuple[str, ...]) -> None:
    """Reject changes outside the neural-experiment surface."""
    forbidden = []
    for raw_path in paths:
        path = PurePosixPath(raw_path).as_posix()
        stable_training_profile = (
            path.startswith("configs/neural_training_profiles/")
            and not path.startswith("configs/neural_training_profiles/candidates/")
        )
        if path in FORBIDDEN_PATHS or path.startswith(FORBIDDEN_PATH_PREFIXES) or stable_training_profile:
            forbidden.append(path)
    if forbidden:
        joined = ", ".join(sorted(forbidden))
        raise ValueError(f"experiment changed protected paths: {joined}")


def validate_campaign_settings(settings: dict[str, object]) -> None:
    """Protect benchmark invariants from an agent-provided configuration."""
    for key in ("seed", "opponents", "baseline_profile", "acceptance_rule"):
        if key not in settings:
            raise ValueError(f"missing protected campaign setting: {key}")
    if settings["baseline_profile"] != "v008":
        raise ValueError("the protected baseline profile must be v008")
    if not settings["opponents"]:
        raise ValueError("the protected opponent panel cannot be empty")


def evaluate_performance_gate(performance: dict[str, Any], max_regression: float = 0.05) -> dict[str, Any]:
    """Evaluate elapsed-time or throughput regression without changing the workload."""
    baseline = performance.get("baseline", {})
    candidate = performance.get("candidate", {})
    regressions = []
    if baseline.get("elapsed_seconds") and candidate.get("elapsed_seconds"):
        regressions.append((candidate["elapsed_seconds"] - baseline["elapsed_seconds"]) / baseline["elapsed_seconds"])
    if baseline.get("throughput") and candidate.get("throughput"):
        regressions.append((baseline["throughput"] - candidate["throughput"]) / baseline["throughput"])
    worst = max(regressions, default=0.0)
    return {
        "available": bool(regressions),
        "max_regression": worst,
        "accepted": bool(regressions) and worst <= max_regression,
        "threshold": max_regression,
    }
