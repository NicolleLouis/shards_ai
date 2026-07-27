"""Self-contained HTML reports for neural imitation training runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable


def load_metrics(path: str | Path) -> list[dict]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return document["epochs"] if isinstance(document, dict) else document


def write_training_report(metrics: Iterable[dict], output: str | Path) -> None:
    rows = list(metrics)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render_report(rows), encoding="utf-8")


def _render_report(rows: list[dict]) -> str:
    best = min(rows, key=lambda row: row["validation"]["mean_loss"]) if rows else None
    summary = (
        f"{len(rows)} epochs · meilleure validation à l'epoch {best['epoch']} · "
        f"loss {best['validation']['mean_loss']:.4f} · top-1 {best['validation']['top1_accuracy']:.1%}"
        if best else "Aucune métrique disponible"
    )
    charts = "".join(_chart_card(*chart) for chart in _chart_definitions(rows))
    table = _render_table(rows)
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Progression entraînement neural</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#f5f7fb; color:#172033; }}
body {{ margin:0; }}
main {{ max-width:1400px; margin:0 auto; padding:32px; }}
h1 {{ margin:0 0 8px; font-size:30px; }}
.subtitle {{ color:#596579; margin-bottom:28px; }}
.summary {{ background:#172033; color:white; padding:18px 22px; border-radius:14px; margin-bottom:22px; box-shadow:0 8px 24px #17203322; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:18px; }}
.card {{ background:white; border-radius:14px; padding:18px; box-shadow:0 3px 14px #17203312; overflow:auto; }}
.card h2 {{ margin:0 0 8px; font-size:18px; }}
.legend {{ color:#596579; font-size:12px; margin-bottom:8px; }}
svg {{ width:100%; min-width:390px; height:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; white-space:nowrap; }}
th,td {{ text-align:right; padding:9px 10px; border-bottom:1px solid #e8ecf2; }}
th {{ color:#596579; font-weight:600; }}
th:first-child,td:first-child {{ text-align:left; }}
.muted {{ color:#718096; }}
@media (max-width:600px) {{ main {{ padding:18px; }} .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Progression de l'entraînement neural</h1>
<div class="subtitle">Rapport généré depuis les métriques JSON · les courbes utilisent les valeurs exactes de chaque epoch.</div>
<div class="summary">{html.escape(summary)}</div>
<section class="grid">{charts}</section>
<section class="card" style="margin-top:18px"><h2>Détail par epoch</h2>{table}</section>
</main></body></html>
"""


def _chart_definitions(rows: list[dict]) -> list[tuple[str, list[tuple[str, str, list[float]]]]]:
    epochs = [float(row["epoch"]) for row in rows]
    return [
        ("Loss", [("Train", "#2563eb", [row["train"]["mean_loss"] for row in rows]),
                   ("Validation", "#dc2626", [row["validation"]["mean_loss"] for row in rows])]),
        ("Imitation de l'action choisie", [("Top-1", "#16a34a", [row["validation"]["top1_accuracy"] for row in rows]),
                                             ("Rang normalisé", "#9333ea", [row["validation"]["mean_normalized_chosen_rank"] for row in rows])]),
        ("Qualité du classement", [("Paires correctes", "#ea580c", [row["validation"]["pairwise_accuracy"] for row in rows])]),
        ("Volume évalué", [("Décisions validation", "#0891b2", [row["validation"]["records"] for row in rows]),
                             ("Paires comparées", "#ca8a04", [row["validation"]["pairwise_pairs"] for row in rows])]),
    ]


def _chart_card(title: str, series: list[tuple[str, str, list[float]]]) -> str:
    maximum = max((value for _label, _color, values in series for value in values), default=1)
    minimum = min((value for _label, _color, values in series for value in values), default=0)
    if maximum == minimum:
        maximum += 1
    width, height = 620, 250
    left, top, plot_width, plot_height = 48, 20, 545, 175
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    lines.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#aab4c3"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#aab4c3"/>')
    for index, (label, color, values) in enumerate(series):
        points = []
        for position, value in enumerate(values):
            x = left + plot_width * position / max(1, len(values) - 1)
            y = top + plot_height - plot_height * (value - minimum) / (maximum - minimum)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        lines.append(f'<text x="{left+plot_width-145}" y="{top+18+index*18}" fill="{color}" font-size="12">{html.escape(label)}</text>')
    lines.append(f'<text x="4" y="{top+8}" fill="#718096" font-size="11">{maximum:.3g}</text>')
    lines.append(f'<text x="4" y="{top+plot_height}" fill="#718096" font-size="11">{minimum:.3g}</text>')
    lines.append('</svg>')
    return f'<article class="card"><h2>{html.escape(title)}</h2><div class="legend">Epoch en abscisse</div>{"".join(lines)}</article>'


def _render_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">Aucune donnée.</p>'
    output = ['<table><thead><tr><th>Epoch</th><th>Loss train</th><th>Loss validation</th><th>Top-1</th><th>Rang normalisé</th><th>Paires correctes</th><th>Décisions</th></tr></thead><tbody>']
    for row in rows:
        train, validation = row["train"], row["validation"]
        output.append(
            f'<tr><td>{row["epoch"]}</td><td>{train["mean_loss"]:.5f}</td><td>{validation["mean_loss"]:.5f}</td>'
            f'<td>{validation["top1_accuracy"]:.1%}</td><td>{validation["mean_normalized_chosen_rank"]:.1%}</td>'
            f'<td>{validation["pairwise_accuracy"]:.1%}</td><td>{validation["records"]}</td></tr>'
        )
    output.append('</tbody></table>')
    return "".join(output)
