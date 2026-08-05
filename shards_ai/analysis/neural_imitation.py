"""Offline metrics for an action-conditioned neural imitation checkpoint."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from shards_ai.ai.action_representation import ActionRepresentation
from shards_ai.ai.neural_training import observation_from_dict
from shards_ai.ai.neural_model import NeuralActionScorer


DECISION_TYPES = {
    "Achat": frozenset({"buy_card"}),
    "Attaque": frozenset({"assign_power"}),
    "Recrutement": frozenset({"recruit_free_card", "recruit_mercenary"}),
}


@dataclass
class Metrics:
    records: int = 0
    top1: int = 0
    top3: int = 0
    heuristic_score_total: float = 0.0
    regret_total: float = 0.0

    def add(self, *, top1: bool, top3: bool, heuristic_score: float, regret: float) -> None:
        self.records += 1
        self.top1 += int(top1)
        self.top3 += int(top3)
        self.heuristic_score_total += heuristic_score
        self.regret_total += regret

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.records:
            return {
                "records": 0, "top1_agreement": None, "top3_agreement": None,
                "mean_heuristic_score": None, "mean_heuristic_regret": None,
            }
        return {
            "records": self.records,
            "top1_agreement": self.top1 / self.records,
            "top3_agreement": self.top3 / self.records,
            "mean_heuristic_score": self.heuristic_score_total / self.records,
            "mean_heuristic_regret": self.regret_total / self.records,
        }


@dataclass
class AnalysisResult:
    overall: Metrics = field(default_factory=Metrics)
    by_phase: dict[str, Metrics] = field(default_factory=dict)
    by_decision_type: dict[str, Metrics] = field(default_factory=dict)
    by_action_type: dict[str, Metrics] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall.as_dict(),
            "by_phase": {key: value.as_dict() for key, value in sorted(self.by_phase.items())},
            "by_decision_type": {key: value.as_dict() for key, value in sorted(self.by_decision_type.items())},
            "by_action_type": {key: value.as_dict() for key, value in sorted(self.by_action_type.items())},
        }


def decision_types_for(action: dict) -> tuple[str, ...]:
    """Return requested families; targeting intentionally may overlap another family."""
    action_type = str(action.get("action_type", "unknown"))
    result = [name for name, action_types in DECISION_TYPES.items() if action_type in action_types]
    if action.get("target") is not None:
        result.append("Ciblage")
    return tuple(result)


def analyze_records(
    model: NeuralActionScorer,
    records: Iterable[dict],
    *,
    max_records: int | None = None,
) -> AnalysisResult:
    result = AnalysisResult()
    model.eval()
    with torch.inference_mode():
        for record_number, record in enumerate(records):
            if max_records is not None and record_number >= max_records:
                break
            representations = [ActionRepresentation(**value) for value in record["action_representations"]]
            teacher_index = int(record["chosen_action_index"])
            teacher_scores = [float(value) for value in record["heuristic_scores"]]
            if not representations or len(representations) != len(teacher_scores):
                raise ValueError(f"Invalid action/score lengths at dataset line {record.get('_line_number', '?')}")
            if not 0 <= teacher_index < len(representations):
                raise ValueError(f"Invalid chosen_action_index at dataset line {record.get('_line_number', '?')}")
            scores = model(observation_from_dict(record["observation"]), representations).tolist()
            if len(scores) != len(representations):
                raise ValueError(f"Model returned an invalid score count at dataset line {record.get('_line_number', '?')}")
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            neural_index = order[0]
            values = {
                "top1": neural_index == teacher_index,
                "top3": teacher_index in order[:3],
                "heuristic_score": teacher_scores[teacher_index],
                "regret": teacher_scores[teacher_index] - teacher_scores[neural_index],
            }
            result.overall.add(**values)
            phase = str(record["observation"].get("phase", "unknown"))
            result.by_phase.setdefault(phase, Metrics()).add(**values)
            chosen = record.get("chosen_action", {})
            chosen_representation = representations[teacher_index].to_dict()
            action_type = str(chosen.get("action_type", chosen_representation["action_type"]))
            result.by_action_type.setdefault(action_type, Metrics()).add(**values)
            for decision_type in decision_types_for({**chosen_representation, **chosen}):
                result.by_decision_type.setdefault(decision_type, Metrics()).add(**values)
    return result


def write_json(result: AnalysisResult, output: str | Path, *, metadata: dict | None = None) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"metadata": metadata or {}, "metrics": result.as_dict()}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_html(result: AnalysisResult, *, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    sections = [
        ("Global", {"global": result.overall}),
        ("Par phase", result.by_phase),
        ("Par famille de décision (recouvrement possible)", result.by_decision_type),
        ("Par type d’action", result.by_action_type),
    ]
    tables = "".join(_render_table(title, groups) for title, groups in sections)
    meta = " · ".join(f"{html.escape(str(key))}: {html.escape(str(value))}" for key, value in metadata.items())
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analyse imitation neural</title><style>
:root {{ color-scheme: light; font-family: system-ui,sans-serif; color:#172033; background:#f5f7fb; }}
body {{ max-width:1500px; margin:auto; padding:28px; }} h1 {{ margin-bottom:4px; }}
.meta,.definition {{ color:#596579; margin:8px 0 22px; }} section {{ background:white; padding:18px; margin:16px 0; border-radius:12px; box-shadow:0 2px 12px #17203318; overflow:auto; }}
table {{ border-collapse:collapse; width:100%; white-space:nowrap; }} th,td {{ padding:9px 12px; border-bottom:1px solid #e8ecf2; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:#596579; }}
</style></head><body><h1>Analyse offline de l’imitation neural</h1><div class="meta">{meta}</div>
<div class="definition">Top-1/top-3 = présence du choix heuristique dans le classement neural. Regret = score heuristique du choix teacher − score heuristique du choix neural.</div>{tables}</body></html>"""


def write_html(result: AnalysisResult, output: str | Path, *, metadata: dict | None = None) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result, metadata=metadata), encoding="utf-8")


def _render_table(title: str, groups: dict[str, Metrics]) -> str:
    rows = []
    for name, metrics in sorted(groups.items()):
        values = metrics.as_dict()
        def percent(value):
            return "n/a" if value is None else f"{value:.2%}"
        def number(value):
            return "n/a" if value is None else f"{value:.4f}"
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{values['records']}</td>"
            f"<td>{percent(values['top1_agreement'])}</td><td>{percent(values['top3_agreement'])}</td>"
            f"<td>{number(values['mean_heuristic_score'])}</td><td>{number(values['mean_heuristic_regret'])}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="6">Aucune décision</td></tr>')
    return f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>Groupe</th><th>Décisions</th><th>Top-1</th><th>Top-3</th><th>Score heuristique moyen</th><th>Regret moyen</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
