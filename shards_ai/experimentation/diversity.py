from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


EXPERIMENT_FAMILIES = (
    "ppo",
    "imitation",
    "dagger",
    "data",
    "objective",
    "inference",
    "monte_carlo",
    "architecture",
    "representation",
    "search",
    "performance",
    "other",
)


def family_guidance(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize method history without imposing a hard family constraint."""
    families = [
        str(item.get("experiment_family"))
        for item in history
        if item.get("experiment_family") in EXPERIMENT_FAMILIES
    ]
    counts = Counter(families)
    minimum = min(counts.values(), default=0)
    underexplored = [family for family in EXPERIMENT_FAMILIES if counts[family] == minimum]
    last_family = families[-1] if families else None
    consecutive = 0
    if last_family:
        for family in reversed(families):
            if family != last_family:
                break
            consecutive += 1
    recommendation = "choose freely"
    if last_family == "ppo" and consecutive >= 2:
        recommendation = "prefer a non-PPO family unless the PPO change is substantially different and justified"
    return {
        "counts": dict(sorted(counts.items())),
        "last_family": last_family,
        "consecutive_last_family": consecutive,
        "underexplored_families": underexplored,
        "recommendation": recommendation,
    }
