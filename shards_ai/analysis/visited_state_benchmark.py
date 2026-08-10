"""Aggregates for the counterfactual v008 comparison on NeuralPlayer trajectories."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class VisitedMetrics:
    records: int = 0
    top1: int = 0
    top3: int = 0
    divergences: int = 0
    regret_total: float = 0.0
    heuristic_score_total: float = 0.0
    heuristic_rank_total: float = 0.0

    def add(self, *, top1: bool, rank: int, regret: float, heuristic_score: float) -> None:
        self.records += 1
        self.top1 += int(top1)
        self.top3 += int(rank <= 3)
        self.divergences += int(not top1)
        self.regret_total += regret
        self.heuristic_score_total += heuristic_score
        self.heuristic_rank_total += rank

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.records:
            return {
                "records": 0, "top1_agreement": None, "top3_agreement": None,
                "divergence_rate": None, "mean_heuristic_regret": None,
                "mean_heuristic_score": None, "mean_heuristic_rank": None,
            }
        return {
            "records": self.records,
            "top1_agreement": self.top1 / self.records,
            "top3_agreement": self.top3 / self.records,
            "divergence_rate": self.divergences / self.records,
            "mean_heuristic_regret": self.regret_total / self.records,
            "mean_heuristic_score": self.heuristic_score_total / self.records,
            "mean_heuristic_rank": self.heuristic_rank_total / self.records,
        }


@dataclass
class VisitedStateResult:
    overall: VisitedMetrics = field(default_factory=VisitedMetrics)
    by_phase: dict[str, VisitedMetrics] = field(default_factory=dict)
    by_neural_action_type: dict[str, VisitedMetrics] = field(default_factory=dict)
    by_heuristic_action_type: dict[str, VisitedMetrics] = field(default_factory=dict)
    by_action_cardinality: dict[str, VisitedMetrics] = field(default_factory=dict)
    by_neural_card: dict[str, VisitedMetrics] = field(default_factory=dict)
    by_heuristic_card: dict[str, VisitedMetrics] = field(default_factory=dict)
    first_divergence_by_game: list[dict[str, object]] = field(default_factory=list)
    games: int = 0

    def add_decision(
        self,
        *,
        phase: str,
        neural_action_type: str,
        heuristic_action_type: str,
        top1: bool,
        rank: int,
        regret: float,
        heuristic_score: float,
        legal_action_count: int = 0,
        neural_card_id: str | None = None,
        heuristic_card_id: str | None = None,
    ) -> None:
        values = {
            "top1": top1,
            "rank": rank,
            "regret": regret,
            "heuristic_score": heuristic_score,
        }
        self.overall.add(**values)
        self.by_phase.setdefault(phase, VisitedMetrics()).add(**values)
        self.by_neural_action_type.setdefault(neural_action_type, VisitedMetrics()).add(**values)
        self.by_heuristic_action_type.setdefault(heuristic_action_type, VisitedMetrics()).add(**values)
        action_cardinality = f"{neural_action_type} | {legal_action_count} actions légales"
        self.by_action_cardinality.setdefault(action_cardinality, VisitedMetrics()).add(**values)
        if neural_card_id is not None:
            neural_key = f"{neural_action_type} | {neural_card_id}"
            self.by_neural_card.setdefault(neural_key, VisitedMetrics()).add(**values)
        if heuristic_card_id is not None:
            heuristic_key = f"{heuristic_action_type} | {heuristic_card_id}"
            self.by_heuristic_card.setdefault(heuristic_key, VisitedMetrics()).add(**values)

    def as_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "overall": self.overall.as_dict(),
            "by_phase": {key: value.as_dict() for key, value in sorted(self.by_phase.items())},
            "by_neural_action_type": {
                key: value.as_dict() for key, value in sorted(self.by_neural_action_type.items())
            },
            "by_heuristic_action_type": {
                key: value.as_dict() for key, value in sorted(self.by_heuristic_action_type.items())
            },
            "by_action_cardinality": {
                key: value.as_dict() for key, value in sorted(self.by_action_cardinality.items())
            },
            "by_neural_card": {
                key: value.as_dict() for key, value in sorted(self.by_neural_card.items())
            },
            "by_heuristic_card": {
                key: value.as_dict() for key, value in sorted(self.by_heuristic_card.items())
            },
            "first_divergence_by_game": self.first_divergence_by_game,
        }


def write_json(result: VisitedStateResult, output: str | Path, *, metadata: dict | None = None) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"metadata": metadata or {}, "metrics": result.as_dict()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_html(result: VisitedStateResult, output: str | Path, *, metadata: dict | None = None) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result, metadata=metadata), encoding="utf-8")


def render_html(result: VisitedStateResult, *, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    meta = " · ".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in metadata.items())
    sections = [
        ("Global", {"global": result.overall}),
        ("Par phase", result.by_phase),
        ("Par type d’action Neural", result.by_neural_action_type),
        ("Par action choisie par v008", result.by_heuristic_action_type),
        ("Par action et cardinalité légale", result.by_action_cardinality),
        ("Par carte choisie par Neural", result.by_neural_card),
        ("Par carte choisie par v008", result.by_heuristic_card),
    ]
    tables = "".join(_table(title, groups) for title, groups in sections)
    divergences = "".join(
        f"<tr><td>{html.escape(str(item['seed']))}</td><td>{item['first_divergence_decision'] or 'aucune'}</td>"
        f"<td>{html.escape(str(item.get('neural_action', '')))}</td>"
        f"<td>{html.escape(str(item.get('heuristic_action', '')))}</td></tr>"
        for item in result.first_divergence_by_game
    )
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>États visités Neural vs v008</title>
<style>:root {{ color-scheme:light; font-family:system-ui,sans-serif; color:#172033; background:#f5f7fb; }}
body {{ max-width:1500px; margin:auto; padding:28px; }} section {{ background:white; padding:18px; margin:16px 0; border-radius:12px; overflow:auto; }}
.meta,.definition {{ color:#596579; margin:8px 0 20px; }} table {{ border-collapse:collapse; width:100%; white-space:nowrap; }}
th,td {{ padding:8px 10px; border-bottom:1px solid #e8ecf2; text-align:right; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:#596579; }}</style></head>
<body><h1>États visités : NeuralPlayer contre v008</h1><div class="meta">{meta}</div>
<div class="definition">L’action v008 est calculée contrefactuellement sur l’état réellement visité ; elle n’est jamais appliquée.</div>
{tables}<section><h2>Première divergence par partie</h2><table><thead><tr><th>Seed</th><th>Décision</th><th>Neural</th><th>v008</th></tr></thead><tbody>{divergences}</tbody></table></section></body></html>"""


def _table(title: str, groups: dict[str, VisitedMetrics]) -> str:
    rows = []
    for name, metrics in sorted(groups.items()):
        value = metrics.as_dict()
        def pct(key: str) -> str:
            item = value[key]
            return "n/a" if item is None else f"{item:.2%}"
        def num(key: str) -> str:
            item = value[key]
            return "n/a" if item is None else f"{item:.4f}"
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{value['records']}</td>"
            f"<td>{pct('top1_agreement')}</td><td>{pct('top3_agreement')}</td>"
            f"<td>{num('mean_heuristic_rank')}</td><td>{num('mean_heuristic_regret')}</td>"
            f"<td>{pct('divergence_rate')}</td></tr>"
        )
    rendered_rows = "".join(rows) or '<tr><td colspan="7">Aucune donnée</td></tr>'
    return f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>Groupe</th><th>Décisions</th><th>Top-1</th><th>Top-3</th><th>Rang v008</th><th>Regret</th><th>Divergence</th></tr></thead><tbody>{rendered_rows}</tbody></table></section>"
