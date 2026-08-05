"""Primitives for reproducible neural improvement campaigns."""

from .manifest import ExperimentManifest, ExperimentStatus
from .policy import (
    FORBIDDEN_PATH_PREFIXES,
    evaluate_performance_gate,
    validate_campaign_settings,
    validate_changed_paths,
)
from .report import render_experiment_report
from .diversity import EXPERIMENT_FAMILIES, family_guidance

__all__ = [
    "ExperimentManifest",
    "ExperimentStatus",
    "FORBIDDEN_PATH_PREFIXES",
    "evaluate_performance_gate",
    "validate_campaign_settings",
    "render_experiment_report",
    "validate_changed_paths",
    "EXPERIMENT_FAMILIES",
    "family_guidance",
]
