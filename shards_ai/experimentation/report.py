from __future__ import annotations

from typing import Any

from .manifest import ExperimentManifest


def _section(title: str, value: Any) -> str:
    if isinstance(value, (dict, list)):
        import json

        body = "```json\n" + json.dumps(value, indent=2, sort_keys=True) + "\n```"
    else:
        body = str(value or "Non renseigné")
    return f"## {title}\n\n{body}\n"


def render_experiment_report(manifest: ExperimentManifest, result: dict[str, Any] | None = None) -> str:
    """Render a durable human report, including failed and interrupted attempts."""
    result = result or {}
    parts = [
        f"# Expérience {manifest.experiment_id}\n",
        _section("Décision", manifest.status.value),
        _section("Hypothèse", manifest.hypothesis),
        _section("Évolution du catalogue d'idées", result.get("ideas_diff", "Aucune modification détectée.")),
        _section("Provenance", {
            "campaign_id": manifest.campaign_id,
            "experiment_kind": manifest.experiment_kind,
            "parent_commit": manifest.parent_commit,
            "parent_profile": manifest.parent_profile,
            "dataset": manifest.dataset,
            "seed": manifest.seed,
            "budget_seconds": manifest.budget_seconds,
        }),
        _section("Changements autorisés", manifest.allowed_changes),
        _section("Recette d'entraînement", manifest.training_recipe),
        _section("Commandes", manifest.commands),
        _section("Résultats offline", result.get("offline", {})),
        _section("Résultats de screening", manifest.screening),
        _section("Validation", manifest.validation),
        _section("Promotion", result.get("promotion", {})),
        _section("Tests fixes", manifest.tests),
        _section("Décision déterministe", manifest.decision_metrics),
        _section("Performance", manifest.performance),
        _section("Performance gate", manifest.performance_gate),
        _section("Analyse humaine", result.get("analysis", "")),
    ]
    if manifest.error:
        parts.append(_section("Erreur", manifest.error))
    parts.append("## Limites et suite\n\nÀ compléter par l'agent ou l'analyse humaine.\n")
    return "\n".join(parts)
