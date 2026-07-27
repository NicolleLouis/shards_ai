"""Run NeuralPlayer against a deterministic Random/v007/v008 opponent mix."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile
from shards_ai.analysis.campaign import _line_svg, central_copy_counts
from shards_ai.game import CARD_CATALOG, Game, GameRandom, GameRunner, GameStatus, PlayerId
from shards_ai.game.actions import BuyCard, GainMastery, RecruitMercenary


OPPONENTS = ("random", "v007", "v008")


def opponent_for_game(game_index: int) -> str:
    position = game_index % 10
    return "random" if position < 2 else "v007" if position < 7 else "v008"


def _deck_counts(player) -> dict[str, int]:
    cards = (*player.hand, *player.draw_pile, *player.discard_pile, *player.play_zone, *player.champions)
    return dict(sorted(Counter(card.definition.card_id for card in cards).items()))


def play_game(
    seed: int,
    checkpoint: Path,
    scorer,
    opponent: str,
    profiles: dict[str, HeuristicProfile],
    max_actions: int,
    max_turns: int | None,
    torch_threads: int,
) -> dict[str, object]:
    torch.set_num_threads(torch_threads)
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = neural_id.opponent
    neural = NeuralPlayer(neural_id, None, root_rng.derive("neural"), scorer=scorer)
    if opponent == "random":
        other = RandomPlayer(opponent_id, root_rng.derive("opponent"))
    else:
        profile = profiles[opponent]
        other = HeuristicPlayer(
            opponent_id,
            profile.weights,
            profile.card_acquisition_weights,
            profile.constraint_weights,
        )
    runner = GameRunner(
        game,
        {neural_id: neural, opponent_id: other},
        max_actions=max_actions,
        max_turns=max_turns,
    )
    mercenary_events: list[dict[str, object]] = []
    mastery_events: list[dict[str, object]] = []

    def observe_decision(observation, legal_actions, action, player_id) -> None:
        if player_id is not neural_id:
            return
        mastery_opportunity = any(isinstance(candidate, GainMastery) for candidate in legal_actions)
        if mastery_opportunity:
            mastery = observation.active_player.mastery
            mastery_events.append({
                "turn": observation.turn_number,
                "mastery": mastery,
                "activated": isinstance(action, GainMastery),
            })
        if isinstance(action, (RecruitMercenary, BuyCard)):
            card = game.state.river[action.river_slot]
            if card is not None and card.definition.is_mercenary:
                mercenary_events.append({
                    "card_id": card.definition.card_id,
                    "mode": "immediate" if isinstance(action, RecruitMercenary) else "long_term",
                    "turn": game.state.turn_number,
                })
    started = time.perf_counter()
    state = runner.run(decision_observer=observe_decision)
    return {
        "seed": seed,
        "opponent": opponent,
        "status": state.status.value,
        "neural_won": state.winner is neural_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "turns": state.turn_number,
        "turns_per_player": state.turn_number / len(runner.players),
        "actions": runner.actions_played,
        "elapsed_seconds": time.perf_counter() - started,
        "neural_decisions": neural.decisions,
        "neural_inference_seconds": neural.total_inference_seconds,
        "neural_health": state.players[neural_id].health,
        "opponent_health": state.players[opponent_id].health,
        "neural_mastery": state.players[neural_id].mastery,
        "opponent_mastery": state.players[opponent_id].mastery,
        "neural_deck": _deck_counts(state.players[neural_id]),
        "opponent_deck": _deck_counts(state.players[opponent_id]),
        "mercenary_events": mercenary_events,
        "mastery_events": mastery_events,
    }


def _summary(records: list[dict]) -> dict[str, object]:
    games = len(records)

    def average(key: str) -> float:
        return statistics.mean(record[key] for record in records) if records else 0.0

    def numeric(key: str) -> dict[str, float | int | None]:
        values = [record[key] for record in records]
        return {
            "mean": round(statistics.mean(values), 3) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    def decks(key: str) -> list[dict[str, object]]:
        total = Counter()
        presence = Counter()
        for record in records:
            cards = record[key]
            total.update(cards)
            presence.update(cards.keys())
        return [
            {
                "card_id": card_id,
                "average_copies": round(total[card_id] / games, 3),
                "presence_rate": round(presence[card_id] / games, 3),
            }
            for card_id in sorted(total)
        ]

    mercenaries = Counter()
    for record in records:
        for event in record["mercenary_events"]:
            mercenaries[(event["card_id"], event["mode"])] += 1
    mercenary_stats = []
    for card_id in sorted({card_id for card_id, _mode in mercenaries}):
        immediate = mercenaries[(card_id, "immediate")]
        long_term = mercenaries[(card_id, "long_term")]
        total = immediate + long_term
        mercenary_stats.append({
            "card_id": card_id,
            "immediate": immediate,
            "long_term": long_term,
            "total": total,
            "immediate_rate": round(immediate / total, 4) if total else 0.0,
        })

    mastery_by_turn = Counter()
    mastery_by_level = Counter()
    for record in records:
        for event in record["mastery_events"]:
            mastery_by_turn[(event["turn"], "opportunity")] += 1
            mastery_by_level[(event["mastery"], "opportunity")] += 1
            if event["activated"]:
                mastery_by_turn[(event["turn"], "activated")] += 1
                mastery_by_level[(event["mastery"], "activated")] += 1

    def mastery_rows(counter: Counter) -> list[dict[str, object]]:
        keys = sorted({key for key, _kind in counter})
        return [
            {
                "value": value,
                "opportunities": counter[(value, "opportunity")],
                "activations": counter[(value, "activated")],
                "activation_rate": round(counter[(value, "activated")] / counter[(value, "opportunity")], 4),
            }
            for value in keys
        ]

    return {
        "games": games,
        "neural_wins": sum(record["neural_won"] for record in records),
        "opponent_wins": sum(record["opponent_won"] for record in records),
        "draws": sum(record["draw"] for record in records),
        "neural_win_rate": round(sum(record["neural_won"] for record in records) / games, 4) if games else 0.0,
        "opponent_win_rate": round(sum(record["opponent_won"] for record in records) / games, 4) if games else 0.0,
        "draw_rate": round(sum(record["draw"] for record in records) / games, 4) if games else 0.0,
        "turns": numeric("turns"),
        "turns_per_player": numeric("turns_per_player"),
        "actions": numeric("actions"),
        "elapsed_seconds": numeric("elapsed_seconds"),
        "neural_decisions": numeric("neural_decisions"),
        "neural_inference_seconds": numeric("neural_inference_seconds"),
        "average_neural_inference_ms": round(1000 * average("neural_inference_seconds") / average("neural_decisions"), 3) if average("neural_decisions") else 0.0,
        "neural_health": numeric("neural_health"),
        "opponent_health": numeric("opponent_health"),
        "neural_mastery": numeric("neural_mastery"),
        "opponent_mastery": numeric("opponent_mastery"),
        "neural_deck": decks("neural_deck"),
        "opponent_deck": decks("opponent_deck"),
        "mercenary_stats": mercenary_stats,
        "mastery_by_turn": mastery_rows(mastery_by_turn),
        "mastery_by_level": mastery_rows(mastery_by_level),
    }


def _copy_count_charts(item: dict[str, object], copy_count: int) -> str:
    """Render Neural/opponent/delta charts for one central-deck multiplicity."""
    neural_by_id = {row["card_id"]: row for row in item["neural_deck"]}
    opponent_by_id = {row["card_id"]: row for row in item["opponent_deck"]}
    card_ids = [card_id for card_id, count in central_copy_counts().items() if count == copy_count]

    neural_rows = []
    opponent_rows = []
    delta_rows = []
    for card_id in card_ids:
        neural = neural_by_id.get(card_id, {"average_copies": 0.0, "presence_rate": 0.0})
        opponent = opponent_by_id.get(card_id, {"average_copies": 0.0, "presence_rate": 0.0})
        name = CARD_CATALOG[card_id].name
        neural_average = float(neural["average_copies"])
        opponent_average = float(opponent["average_copies"])
        neural_rows.append({"card_id": card_id, "name": name, "average_number": neural_average})
        opponent_rows.append({"card_id": card_id, "name": name, "average_number": opponent_average})
        delta_rows.append({
            "card_id": card_id,
            "name": name,
            "delta_average_number": neural_average - opponent_average,
        })

    neural_rows.sort(key=lambda row: (-row["average_number"], row["card_id"]))
    opponent_rows.sort(key=lambda row: (-row["average_number"], row["card_id"]))
    delta_rows.sort(key=lambda row: (-abs(row["delta_average_number"]), row["card_id"]))
    return (
        '<div class="chart-row">'
        f'<div class="chart-panel">{_line_svg(neural_rows, f"NeuralPlayer — cartes en ×{copy_count}")}</div>'
        f'<div class="chart-panel">{_line_svg(opponent_rows, f"Adversaire — cartes en ×{copy_count}")}</div>'
        '</div><div class="chart-row">'
        f'<div class="chart-panel">{_line_svg(delta_rows, f"Delta NeuralPlayer − adversaire — cartes en ×{copy_count}", value_key="delta_average_number", value_label="Delta copies moyennes")}</div>'
        '</div>'
    )


def _render_report(payload: dict) -> str:
    summary = payload["summary_by_opponent"]
    rows = []
    for opponent in OPPONENTS:
        item = summary[opponent]
        rows.append(
            f"<tr><td>{opponent}</td><td>{item['games']}</td><td>{item['neural_win_rate']:.1%}</td>"
            f"<td>{item['opponent_win_rate']:.1%}</td><td>{item['draw_rate']:.1%}</td>"
            f"<td>{item['turns']['mean']:.1f}</td><td>{item['turns_per_player']['mean']:.1f}</td>"
            f"<td>{item['neural_health']['mean']:.1f}</td>"
            f"<td>{item['neural_mastery']['mean']:.1f}</td><td>{item['average_neural_inference_ms']:.2f} ms</td></tr>"
        )
    sections = []
    for opponent in OPPONENTS:
        item = summary[opponent]
        sections.append(
            f"<article><h2>Contre {opponent}</h2><p>{item['games']} parties · "
            f"victoire Neural {item['neural_win_rate']:.1%} · durée moyenne {item['elapsed_seconds']['mean']:.2f}s · "
            f"{item['turns']['mean']:.1f} tours · {item['turns_per_player']['mean']:.1f} tours / joueur · "
            f"{item['actions']['mean']:.1f} actions</p>"
            f"<h3>État final moyen</h3><p>Neural : {item['neural_health']['mean']:.1f} PV, "
            f"{item['neural_mastery']['mean']:.1f} maîtrise · Adversaire : {item['opponent_health']['mean']:.1f} PV, "
            f"{item['opponent_mastery']['mean']:.1f} maîtrise</p>"
            f"<h3>Deck final moyen</h3>{_deck_table(item['neural_deck'], 'NeuralPlayer')}"
            f"{_deck_table(item['opponent_deck'], 'Adversaire')}"
            f"<h3>Graphiques par multiplicité du deck central</h3>{''.join(_copy_count_charts(item, count) for count in (1, 2, 3))}"
            f"<h3>Mercenaires : immédiat vs long terme</h3>{_mercenary_table(item['mercenary_stats'])}"
            f"<h3>GainMastery par tour</h3>{_mastery_table(item['mastery_by_turn'], 'Tour')}"
            f"<h3>GainMastery par maîtrise initiale</h3>{_mastery_table(item['mastery_by_level'], 'Maîtrise')}</article>"
        )
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark NeuralPlayer — mix d'adversaires</title><style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:30px}}h1{{margin-bottom:5px}}.muted{{color:#637083}}.card,article{{background:white;border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 3px 14px #17203318}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}.metric{{font-size:28px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e5e9ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.bar{{height:18px;background:#e5e9ef;border-radius:10px;overflow:hidden;margin:8px 0 18px}}.fill{{height:100%;background:#2563eb}}.deck{{display:inline-block;vertical-align:top;width:48%;min-width:300px;margin-right:1%}}@media(max-width:700px){{.deck{{width:100%}}}}
</style></head><body><main><h1>Benchmark NeuralPlayer — mix d'adversaires</h1>
<p class="muted">{payload['config']['games']} parties · répartition Random 20 %, v007 50 %, v008 30 % · seed {payload['config']['seed']}</p>
<section class="card"><h2>Taux de victoire</h2><div class="grid">{''.join(f'<div><strong>{opponent}</strong><div class="metric">{summary[opponent]["neural_win_rate"]:.1%}</div><div class="bar"><div class="fill" style="width:{summary[opponent]["neural_win_rate"]*100:.1f}%"></div></div></div>' for opponent in OPPONENTS)}</div>
<table><thead><tr><th>Adversaire</th><th>Parties</th><th>Neural win</th><th>Adversaire win</th><th>Nuls</th><th>Tours moyens</th><th>Tours / joueur</th><th>PV Neural</th><th>Maîtrise Neural</th><th>Inférence</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(sections)}</main></body></html>"""


def _deck_table(cards: list[dict], title: str) -> str:
    rows = "".join(f"<tr><td>{html.escape(card['card_id'])}</td><td>{card['average_copies']:.2f}</td><td>{card['presence_rate']:.1%}</td></tr>" for card in cards)
    return f"<div class='deck'><h4>{title}</h4><table><thead><tr><th>Carte</th><th>Copies moy.</th><th>Présence</th></tr></thead><tbody>{rows}</tbody></table></div>"


def _mercenary_table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='muted'>Aucun mercenaire recruté ou acheté par le NeuralPlayer.</p>"
    content = "".join(
        f"<tr><td>{html.escape(row['card_id'])}</td><td>{row['immediate']}</td><td>{row['long_term']}</td><td>{row['immediate_rate']:.1%}</td></tr>"
        for row in rows
    )
    return f"<table><thead><tr><th>Mercenaire</th><th>Immédiat</th><th>Long terme</th><th>Taux immédiat</th></tr></thead><tbody>{content}</tbody></table>"


def _mastery_table(rows: list[dict], label: str) -> str:
    if not rows:
        return "<p class='muted'>Aucune opportunité GainMastery observée.</p>"
    content = "".join(
        f"<tr><td>{label} {row['value']}</td><td>{row['opportunities']}</td><td>{row['activations']}</td><td>{row['activation_rate']:.1%}</td></tr>"
        for row in rows
    )
    return f"<table><thead><tr><th>{label}</th><th>Opportunités</th><th>Activations</th><th>Taux</th></tr></thead><tbody>{content}</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/neural_imitation/baseline.pt"))
    parser.add_argument("--profile-v007", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_benchmark/neural_mix.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/neural_benchmark/neural_mix.html"))
    args = parser.parse_args()
    if args.games <= 0 or not args.checkpoint.exists():
        parser.error("games must be positive and checkpoint must exist")
    for path in (args.profile_v007, args.profile_v008):
        if not path.exists():
            parser.error(f"heuristic profile not found: {path}")
    if args.games % 10:
        print("warning: exact 20/50/30 proportions require a multiple of 10 games")
    profiles = {"v007": load_profile(args.profile_v007), "v008": load_profile(args.profile_v008)}
    torch.set_num_threads(args.torch_threads)
    scorer = NeuralPlayer.load_scorer(args.checkpoint)
    results = []
    for index in range(args.games):
        opponent = opponent_for_game(index)
        results.append(play_game(args.seed + index, args.checkpoint, scorer, opponent, profiles, args.max_actions, args.max_turns, args.torch_threads))
        if (index + 1) % 100 == 0:
            print(f"completed={index + 1}/{args.games}")
    payload = {
        "config": {"checkpoint": str(args.checkpoint), "games": args.games, "seed": args.seed, "torch_threads": args.torch_threads, "distribution": {opponent: sum(opponent_for_game(i) == opponent for i in range(args.games)) for opponent in OPPONENTS}},
        "summary_by_opponent": {opponent: _summary([record for record in results if record["opponent"] == opponent]) for opponent in OPPONENTS},
        "games": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps(payload["config"]["distribution"], sort_keys=True))


if __name__ == "__main__":
    main()
