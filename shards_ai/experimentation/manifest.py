from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ExperimentStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    INTERRUPTED = "interrupted"


@dataclass
class ExperimentManifest:
    experiment_id: str
    campaign_id: str
    experiment_kind: str
    parent_commit: str
    parent_profile: str
    hypothesis: str
    experiment_family: str = "other"
    novelty: str | None = None
    status: ExperimentStatus = ExperimentStatus.INTERRUPTED
    allowed_changes: list[str] = field(default_factory=list)
    dataset: str | None = None
    dataset_sha256: str | None = None
    dataset_records: int | None = None
    teacher_profile: str | None = None
    seed: int | None = None
    budget_seconds: int = 3600
    training_budget_seconds: int = 2400
    screening_budget_seconds: int = 750
    overhead_budget_seconds: int = 450
    commands: list[str] = field(default_factory=list)
    training_recipe: dict[str, Any] = field(default_factory=dict)
    architecture_fingerprint: str | None = None
    baseline_checkpoint_sha256: str | None = None
    candidate_checkpoint_sha256: str | None = None
    screening: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    tests: dict[str, Any] = field(default_factory=dict)
    decision_metrics: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    performance_gate: dict[str, Any] = field(default_factory=dict)
    commit: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["status"] = self.status.value
        return values

    def write_json(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
