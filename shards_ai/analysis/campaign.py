from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..game import CARD_CATALOG, GameRunner, GameStatus, Faction
from ..game.cards.definitions import HOMODEUS_CARDS, MAQUIS_CARDS, ORDER_CARDS, SPECTRA_CARDS
from ..game.state import GameState


BASE_CARD_IDS = frozenset({"crystal", "blaster", "shard_reactor", "infinity_shard"})
CENTRAL_DECK_GROUPS = (MAQUIS_CARDS, SPECTRA_CARDS, ORDER_CARDS, HOMODEUS_CARDS)


def central_copy_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for deck_group in CENTRAL_DECK_GROUPS:
        for definition, count in deck_group:
            if definition.card_id in counts and counts[definition.card_id] != count:
                raise ValueError(f"Card has inconsistent central multiplicities: {definition.card_id}")
            counts[definition.card_id] = count
    return counts


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    duration_seconds: float = 60.0
    games: int | None = None
    seed: int | None = None
    max_actions: int = GameRunner.DEFAULT_MAX_ACTIONS
    max_turns: int | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.games is not None and self.games <= 0:
            raise ValueError("games must be positive")
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("max_turns must be positive")


@dataclass(slots=True)
class CampaignResult:
    config: CampaignConfig
    root_seed: int
    elapsed_seconds: float
    attempted: int = 0
    completed: int = 0
    wins: Counter[str] = field(default_factory=Counter)
    draws: int = 0
    errors: list[dict[str, object]] = field(default_factory=list)
    winner_decks: list[dict[str, object]] = field(default_factory=list)
    loser_decks: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        card_rows, faction_rows, grouped_rows = build_statistics(self.winner_decks)
        loser_card_rows, loser_faction_rows, loser_grouped_rows = build_statistics(self.loser_decks)
        card_delta_rows, faction_delta_rows, grouped_delta_rows = build_delta_statistics(
            self.winner_decks, self.loser_decks
        )
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "root_seed": self.root_seed,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "summary": {
                "attempted": self.attempted,
                "completed": self.completed,
                "wins": dict(sorted(self.wins.items())),
                "draws": self.draws,
                "errors": len(self.errors),
                "error_details": self.errors,
            },
            "winner_decks": self.winner_decks,
            "loser_decks": self.loser_decks,
            "cards": card_rows,
            "cards_by_copy_count": grouped_rows,
            "factions": faction_rows,
            "loser_cards": loser_card_rows,
            "loser_cards_by_copy_count": loser_grouped_rows,
            "loser_factions": loser_faction_rows,
            "cards_delta": card_delta_rows,
            "cards_delta_by_copy_count": grouped_delta_rows,
            "factions_delta": faction_delta_rows,
        }


def _part_seed(root_seed: int, game_index: int) -> int:
    payload = f"shards-ai-analysis:{root_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _player_cards(state: GameState, player_id) -> Counter[str]:
    player = state.players[player_id]
    cards = (*player.hand, *player.draw_pile, *player.discard_pile, *player.play_zone)
    return Counter(card.definition.card_id for card in cards)


def run_campaign(
    config: CampaignConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
    runner_factory: Callable[[int, CampaignConfig], GameRunner] | None = None,
) -> CampaignResult:
    root_seed = config.seed if config.seed is not None else random.SystemRandom().randrange(2**63)
    started_at = clock()
    result = CampaignResult(config=config, root_seed=root_seed, elapsed_seconds=0.0)

    while True:
        if config.games is not None and result.attempted >= config.games:
            break
        if result.attempted > 0 and clock() - started_at >= config.duration_seconds:
            break

        game_index = result.attempted
        part_seed = _part_seed(root_seed, game_index)
        result.attempted += 1
        try:
            runner = (
                runner_factory(part_seed, config)
                if runner_factory is not None
                else GameRunner.random_duel(
                    seed=part_seed,
                    max_actions=config.max_actions,
                    max_turns=config.max_turns,
                )
            )
            state = runner.run()
        except Exception as error:  # noqa: BLE001 - tolerant campaign mode is intentional.
            if config.strict:
                raise
            result.errors.append(
                {
                    "game_index": game_index,
                    "seed": part_seed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue

        result.completed += 1
        if state.status is GameStatus.DRAW or state.winner is None:
            result.draws += 1
            continue
        winner = str(int(state.winner))
        result.wins[winner] += 1
        result.winner_decks.append(
            {
                "game_index": game_index,
                "seed": part_seed,
                "winner": int(state.winner),
                "cards": dict(sorted(_player_cards(state, state.winner).items())),
            }
        )
        loser_id = state.winner.opponent
        result.loser_decks.append(
            {
                "game_index": game_index,
                "seed": part_seed,
                "loser": int(loser_id),
                "cards": dict(sorted(_player_cards(state, loser_id).items())),
            }
        )

    result.elapsed_seconds = clock() - started_at
    return result


def _metadata(card_id: str, copy_counts: Mapping[str, int]) -> dict[str, object]:
    definition = CARD_CATALOG[card_id]
    faction = definition.faction.value if definition.faction is not None else Faction.NEUTRAL.value
    return {
        "card_id": card_id,
        "name": definition.name,
        "faction": faction,
        "cost": definition.cost,
        "central_copy_count": copy_counts.get(card_id),
    }


def build_statistics(
    winner_decks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    copy_counts = central_copy_counts()
    metadata_by_card_id = {
        card_id: _metadata(card_id, copy_counts) for card_id in CARD_CATALOG
    }
    victories = len(winner_decks)
    totals: Counter[str] = Counter()
    presences: Counter[str] = Counter()
    faction_totals: Counter[str] = Counter()
    for snapshot in winner_decks:
        cards = snapshot["cards"]
        assert isinstance(cards, dict)
        for card_id, count in cards.items():
            count = int(count)
            totals[card_id] += count
            presences[card_id] += count > 0
            faction = metadata_by_card_id[card_id]["faction"]
            assert isinstance(faction, str)
            faction_totals[faction] += count

    cards: list[dict[str, object]] = []
    for card_id, metadata in metadata_by_card_id.items():
        cards.append(
            {
                **metadata,
                "average_number": round(totals[card_id] / victories, 6) if victories else 0.0,
                "presence_rate": round(presences[card_id] / victories, 6) if victories else 0.0,
            }
        )
    cards.sort(key=lambda row: (-float(row["average_number"]), str(row["card_id"])))
    for rank, row in enumerate(cards, start=1):
        row["rank"] = rank

    factions = [
        {
            "faction": faction,
            "average_number": round(faction_totals[faction] / victories, 6) if victories else 0.0,
        }
        for faction in sorted({row["faction"] for row in cards})
    ]
    total_average = sum(float(row["average_number"]) for row in factions)
    for row in factions:
        row["share"] = round(float(row["average_number"]) / total_average, 6) if total_average else 0.0

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cards:
        count = row["central_copy_count"]
        if row["card_id"] in BASE_CARD_IDS or count is None:
            continue
        grouped[str(count)].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda row: (-float(row["average_number"]), str(row["card_id"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    grouped = dict(sorted(grouped.items(), key=lambda item: int(item[0])))
    return cards, factions, grouped


def build_delta_statistics(
    winner_decks: list[dict[str, object]],
    loser_decks: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    if len(winner_decks) != len(loser_decks):
        raise ValueError("Winner and loser snapshots must have the same length")

    winner_cards, winner_factions, _ = build_statistics(winner_decks)
    loser_cards, loser_factions, _ = build_statistics(loser_decks)
    loser_by_id = {row["card_id"]: row for row in loser_cards}
    cards_delta: list[dict[str, object]] = []
    for winner_row in winner_cards:
        loser_row = loser_by_id[winner_row["card_id"]]
        winner_average = float(winner_row["average_number"])
        loser_average = float(loser_row["average_number"])
        winner_presence = float(winner_row["presence_rate"])
        loser_presence = float(loser_row["presence_rate"])
        cards_delta.append(
            {
                "card_id": winner_row["card_id"],
                "name": winner_row["name"],
                "faction": winner_row["faction"],
                "cost": winner_row["cost"],
                "central_copy_count": winner_row["central_copy_count"],
                "winner_average_number": round(winner_average, 6),
                "loser_average_number": round(loser_average, 6),
                "delta_average_number": round(winner_average - loser_average, 6),
                "winner_presence_rate": round(winner_presence, 6),
                "loser_presence_rate": round(loser_presence, 6),
                "delta_presence_rate": round(winner_presence - loser_presence, 6),
            }
        )
    cards_delta.sort(
        key=lambda row: (-float(row["delta_average_number"]), str(row["card_id"]))
    )
    for rank, row in enumerate(cards_delta, start=1):
        row["rank"] = rank

    loser_factions_by_name = {row["faction"]: row for row in loser_factions}
    factions_delta: list[dict[str, object]] = []
    for winner_row in winner_factions:
        loser_row = loser_factions_by_name[winner_row["faction"]]
        winner_average = float(winner_row["average_number"])
        loser_average = float(loser_row["average_number"])
        winner_share = float(winner_row["share"])
        loser_share = float(loser_row["share"])
        factions_delta.append(
            {
                "faction": winner_row["faction"],
                "winner_average_number": round(winner_average, 6),
                "loser_average_number": round(loser_average, 6),
                "delta_average_number": round(winner_average - loser_average, 6),
                "winner_share": round(winner_share, 6),
                "loser_share": round(loser_share, 6),
                "delta_share": round(winner_share - loser_share, 6),
            }
        )
    factions_delta.sort(
        key=lambda row: (-float(row["delta_average_number"]), str(row["faction"]))
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cards_delta:
        count = row["central_copy_count"]
        if row["card_id"] in BASE_CARD_IDS or count is None:
            continue
        grouped[str(count)].append(dict(row))
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (-float(row["delta_average_number"]), str(row["card_id"]))
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    grouped = dict(sorted(grouped.items(), key=lambda item: int(item[0])))
    return cards_delta, factions_delta, grouped


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pie_svg(rows: list[dict[str, object]], width: int = 480, height: int = 320) -> str:
    colors = {
        "neutral": "#000000",
        "homodeus": "#6b7280",
        "spectra": "#9333ea",
        "maquis": "#16a34a",
        "order": "#2563eb",
    }
    cx, cy, radius = 155, 155, 105
    angle = -math.pi / 2
    paths: list[str] = []
    legend: list[str] = []
    for index, row in enumerate(rows):
        share = float(row["share"])
        if share <= 0:
            continue
        end = angle + share * math.tau
        large = 1 if end - angle > math.pi else 0
        x1, y1 = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        x2, y2 = cx + radius * math.cos(end), cy + radius * math.sin(end)
        faction = str(row["faction"])
        color = colors.get(faction, "#64748b")
        paths.append(f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large} 1 {x2:.2f} {y2:.2f} Z" fill="{color}"/>')
        legend.append(f'<rect x="290" y="{45 + index * 30}" width="14" height="14" fill="{color}"/><text x="312" y="{57 + index * 30}">{html.escape(str(row["faction"]))} ({share:.1%})</text>')
        angle = end
    return f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Répartition par faction">{"".join(paths)}{"".join(legend)}</svg>'


def _line_svg(
    rows: list[dict[str, object]],
    title: str,
    *,
    value_key: str = "average_number",
    value_label: str = "Moyenne",
    width: int = 760,
    height: int = 300,
) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(title)} : aucune donnée.</p>'
    left, top, chart_width, chart_height = 55, 35, width - 80, height - 75
    values = [float(row[value_key]) for row in rows]
    minimum = min(0.0, min(values))
    maximum = max(0.0, max(values))
    value_range = maximum - minimum or 1.0
    tick_step = max(0.01, math.ceil(value_range / 5 * 100) / 100)
    axis_minimum = math.floor(minimum / tick_step) * tick_step
    axis_maximum = math.ceil(maximum / tick_step) * tick_step
    if axis_minimum == axis_maximum:
        axis_minimum -= tick_step
        axis_maximum += tick_step
    axis_range = axis_maximum - axis_minimum
    tick_count = max(1, round((axis_maximum - axis_minimum) / tick_step))
    points = []
    labels = []
    markers = []
    tooltip_label = "carte(s) en moyenne" if value_key == "average_number" else "delta"
    for index, row in enumerate(rows):
        displayed_rank = index + 1
        x = left + chart_width * index / max(len(rows) - 1, 1)
        value = float(row[value_key])
        y = top + chart_height * (1 - (value - axis_minimum) / axis_range)
        points.append(f"{x:.1f},{y:.1f}")
        labels.append(f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle">{displayed_rank}</text>')
        markers.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb">'
            f'<title>Rang {displayed_rank} — {html.escape(str(row["name"]))} : '
            f'{value:.2f} {tooltip_label}</title></circle>'
        )
    y_ticks = []
    for tick_index in range(tick_count + 1):
        value = axis_minimum + tick_step * tick_index
        y = top + chart_height * (1 - (value - axis_minimum) / axis_range)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#cbd5e1"/>'
            f'<text x="{left - 8}" y="{y + 5:.1f}" text-anchor="end">{value:.2f}</text>'
        )
    grid = f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#475569"/><line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#475569"/>{"".join(y_ticks)}'
    return f'<h3>{html.escape(title)}</h3><svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">{grid}<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="3"/>{"".join(markers)}{"".join(labels)}<text x="{width / 2}" y="{height - 4}" text-anchor="middle">Rang</text><text x="12" y="{height / 2}" transform="rotate(-90 12 {height / 2})" text-anchor="middle">{html.escape(value_label)}</text></svg>'


def _table(rows: list[dict[str, object]], fields: list[str]) -> str:
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def write_report(result: CampaignResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    cards = payload["cards"]
    factions = payload["factions"]
    grouped = payload["cards_by_copy_count"]
    loser_cards = payload["loser_cards"]
    loser_factions = payload["loser_factions"]
    loser_grouped = payload["loser_cards_by_copy_count"]
    cards_delta = payload["cards_delta"]
    factions_delta = payload["factions_delta"]
    grouped_delta = payload["cards_delta_by_copy_count"]
    assert all(
        isinstance(value, list)
        for value in (cards, factions, loser_cards, loser_factions, cards_delta, factions_delta)
    )
    assert all(isinstance(value, dict) for value in (grouped, loser_grouped, grouped_delta))
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    card_fields = ["rank", "card_id", "name", "average_number", "presence_rate", "faction", "cost", "central_copy_count"]
    _write_csv(output_dir / "cards.csv", cards, card_fields)
    _write_csv(output_dir / "factions.csv", factions, ["faction", "average_number", "share"])
    grouped_rows = [{"central_copy_count": count, **row} for count, rows in grouped.items() for row in rows]
    _write_csv(output_dir / "cards_by_copy_count.csv", grouped_rows, ["central_copy_count", *card_fields])
    _write_csv(output_dir / "loser_cards.csv", loser_cards, card_fields)
    _write_csv(output_dir / "loser_factions.csv", loser_factions, ["faction", "average_number", "share"])
    loser_grouped_rows = [
        {"central_copy_count": count, **row}
        for count, rows in loser_grouped.items()
        for row in rows
    ]
    _write_csv(output_dir / "loser_cards_by_copy_count.csv", loser_grouped_rows, ["central_copy_count", *card_fields])
    delta_card_fields = [
        "rank", "card_id", "name", "winner_average_number", "loser_average_number",
        "delta_average_number", "winner_presence_rate", "loser_presence_rate",
        "delta_presence_rate", "faction", "cost", "central_copy_count",
    ]
    _write_csv(output_dir / "cards_delta.csv", cards_delta, delta_card_fields)
    _write_csv(
        output_dir / "factions_delta.csv",
        factions_delta,
        [
            "faction", "winner_average_number", "loser_average_number",
            "delta_average_number", "winner_share", "loser_share", "delta_share",
        ],
    )
    grouped_delta_rows = [
        {"central_copy_count": count, **row}
        for count, rows in grouped_delta.items()
        for row in rows
    ]
    _write_csv(
        output_dir / "cards_delta_by_copy_count.csv",
        grouped_delta_rows,
        ["central_copy_count", *delta_card_fields],
    )

    summary = payload["summary"]
    assert isinstance(summary, dict)
    pie_chart = _pie_svg(factions)
    non_base = [row for row in cards if row["card_id"] not in BASE_CARD_IDS]
    delta_non_base = [row for row in cards_delta if row["card_id"] not in BASE_CARD_IDS]
    chart_sections = [
        f'<div class="chart-row"><div class="chart-panel">{_line_svg(non_base, "Cartes hors base")}</div>'
        f'<div class="chart-panel">{_line_svg(delta_non_base, "Delta cartes hors base", value_key="delta_average_number", value_label="Delta")}</div></div>'
    ]
    for count, rows in grouped.items():
        chart_sections.append(
            f'<div class="chart-row"><div class="chart-panel">{_line_svg(rows, f"Cartes en ×{count}")}</div>'
            f'<div class="chart-panel">{_line_svg(grouped_delta[count], f"Delta cartes en ×{count}", value_key="delta_average_number", value_label="Delta")}</div></div>'
        )
    cards_table = _table(cards, card_fields)
    grouped_table = _table(grouped_rows, ["central_copy_count", *card_fields])
    loser_cards_table = _table(loser_cards, card_fields)
    loser_grouped_table = _table(loser_grouped_rows, ["central_copy_count", *card_fields])
    delta_cards_table = _table(cards_delta, delta_card_fields)
    delta_factions_table = _table(
        factions_delta,
        [
            "faction", "winner_average_number", "loser_average_number",
            "delta_average_number", "winner_share", "loser_share", "delta_share",
        ],
    )
    grouped_delta_table = _table(grouped_delta_rows, ["central_copy_count", *delta_card_fields])
    title = "Analyse statistique des parties"
    document = f'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:14px system-ui,sans-serif;color:#172033;margin:2rem}}table{{border-collapse:collapse;margin:1rem 0 2rem;width:100%}}th,td{{border:1px solid #cbd5e1;padding:.35rem;text-align:left}}th{{background:#e2e8f0}}.chart-row{{display:flex;gap:1.5rem;align-items:flex-start;margin-bottom:1.5rem;flex-wrap:wrap}}.chart-panel{{flex:1 1 520px;min-width:0}}.chart{{max-width:760px;width:100%;height:auto;background:#f8fafc;margin:0}}.empty{{color:#64748b}}</style></head>
<body><h1>{title}</h1><p>Seed racine : <code>{result.root_seed}</code> — durée : {result.elapsed_seconds:.3f}s</p>
<p>Parties tentées : {summary["attempted"]} — terminées : {summary["completed"]} — victoires : {sum(summary["wins"].values())} — nuls : {summary["draws"]} — erreurs : {summary["errors"]}</p>
<h2>Répartition moyenne par faction</h2>{pie_chart}
<h2>Cartes des gagnants</h2>{cards_table}<h2>Comparaison gagnants par multiplicité hors cartes de base</h2>{grouped_table}
<h2>Cartes des perdants</h2>{loser_cards_table}<h2>Comparaison perdants par multiplicité hors cartes de base</h2>{loser_grouped_table}
<h2>Delta gagnant − perdant</h2>{delta_cards_table}<h2>Delta par faction</h2>{delta_factions_table}<h2>Delta comparable par multiplicité hors cartes de base</h2>{grouped_delta_table}
<h2>Graphiques</h2>{"".join(chart_sections)}</body></html>'''
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
