"""Run HybridPlayer against the promotion panel and render final decks."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import torch

from benchmarks.benchmark_neural_mix import _deck_counts, _deck_table
from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer, build_hybrid_player, build_neural_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


OPPONENTS = ("heuristic:v008",)
WEIGHTS = {"heuristic:v008": 1.0}


def _make_opponent(label, player_id, game, rng, heuristic_profiles, neural_scorer):
    if label.startswith("heuristic:"):
        profile = heuristic_profiles[label.removeprefix("heuristic:")]
        return HeuristicPlayer(player_id, profile.weights, profile.card_acquisition_weights, profile.constraint_weights)
    return build_neural_player(player_id, game, rng, scorer=neural_scorer)


def play_game(seed, opponent, hybrid_profile, heuristic_profiles, neural_scorer, max_actions, max_turns):
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    hybrid_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = hybrid_id.opponent
    hybrid = build_hybrid_player(hybrid_id, game, root_rng.derive("hybrid"), profile=hybrid_profile)
    other = _make_opponent(opponent, opponent_id, game, root_rng.derive("opponent"), heuristic_profiles, neural_scorer)
    runner = GameRunner(game, {hybrid_id: hybrid, opponent_id: other}, max_actions=max_actions, max_turns=max_turns)
    started = time.perf_counter()
    state = runner.run()
    hybrid_deck = _deck_counts(state.players[hybrid_id])
    opponent_deck = _deck_counts(state.players[opponent_id])
    return {
        "seed": seed,
        "opponent": opponent,
        "status": state.status.value,
        "hybrid_won": state.winner is hybrid_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "turns": state.turn_number,
        "actions": runner.actions_played,
        "elapsed_seconds": time.perf_counter() - started,
        "hybrid_health": state.players[hybrid_id].health,
        "opponent_health": state.players[opponent_id].health,
        "hybrid_mastery": state.players[hybrid_id].mastery,
        "opponent_mastery": state.players[opponent_id].mastery,
        "hybrid_deck": hybrid_deck,
        "opponent_deck": opponent_deck,
        "hybrid_deck_size": sum(hybrid_deck.values()),
        "opponent_deck_size": sum(opponent_deck.values()),
    }


def _summary(records):
    games = len(records)
    wins = sum(record["hybrid_won"] for record in records)
    losses = sum(record["opponent_won"] for record in records)
    draws = sum(record["draw"] for record in records)
    def mean(key):
        return sum(float(record[key]) for record in records) / games if games else 0.0
    return {
        "games": games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "draw_rate": draws / games if games else 0.0,
        "turns_mean": mean("turns"), "actions_mean": mean("actions"),
        "elapsed_mean": mean("elapsed_seconds"),
        "hybrid_health_mean": mean("hybrid_health"), "opponent_health_mean": mean("opponent_health"),
        "hybrid_mastery_mean": mean("hybrid_mastery"), "opponent_mastery_mean": mean("opponent_mastery"),
        "deck_delta_mean": mean("hybrid_deck_size") - mean("opponent_deck_size"),
        "hybrid_deck": _merge_decks(records, "hybrid_deck"),
        "opponent_deck": _merge_decks(records, "opponent_deck"),
    }


def _merge_decks(records, key):
    counts = {}
    presence = {}
    for record in records:
        for card_id, count in record[key].items():
            counts[card_id] = counts.get(card_id, 0.0) + count / len(records)
        for card_id in record[key]:
            presence[card_id] = presence.get(card_id, 0) + 1
    return [
        {
            "card_id": card_id,
            "average_copies": average,
            "presence_rate": presence.get(card_id, 0) / len(records),
        }
        for card_id, average in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _deck_rows(deck):
    if isinstance(deck, list):
        return deck
    return [
        {"card_id": card_id, "average_copies": float(count), "presence_rate": 1.0}
        for card_id, count in deck.items()
    ]


def _gate(summary):
    weighted = 0.0
    total = 0.0
    rows = {}
    for opponent, item in summary.items():
        delta = item["win_rate"] - 0.5
        weight = WEIGHTS[opponent]
        weighted += weight * delta
        total += weight
        rows[opponent] = {"win_rate": item["win_rate"], "delta_vs_50_percent": delta, "weight": weight}
    mean_delta = weighted / total if total else 0.0
    return {"accepted": mean_delta > 0.0, "weighted_mean_delta": mean_delta, "minimum": 0.0, "by_opponent": rows}


def render_html(payload):
    summary = payload["summary_by_opponent"]
    rows = []
    sections = []
    for opponent in OPPONENTS:
        item = summary[opponent]
        gate = payload["gate"]["by_opponent"][opponent]
        rows.append(
            f"<tr><th>{html.escape(opponent)}</th><td>{item['games']}</td><td>{item['wins']}/{item['games']} ({item['win_rate']:.1%})</td>"
            f"<td>{item['losses']}</td><td>{item['draws']}</td><td>{gate['delta_vs_50_percent']:+.1%}</td><td>{item['deck_delta_mean']:+.1f}</td></tr>"
        )
        sections.append(
            f"<article><h2>Contre {html.escape(opponent)}</h2><p>{item['wins']}/{item['games']} victoires · "
            f"PV moyens {item['hybrid_health_mean']:.1f} contre {item['opponent_health_mean']:.1f} · "
            f"maîtrise moyenne {item['hybrid_mastery_mean']:.1f} contre {item['opponent_mastery_mean']:.1f}</p>"
            f"<h3>Deck final moyen du Hybrid</h3>{_deck_table(_deck_rows(item['hybrid_deck']), 'Hybrid') }"
            f"<h3>Deck final moyen de l'adversaire</h3>{_deck_table(_deck_rows(item['opponent_deck']), 'Adversaire')}</article>"
        )
    gate = payload["gate"]
    status = "ACCEPTÉE" if gate["accepted"] else "REJETÉE"
    return f"""<!doctype html><html lang='fr'><meta charset='utf-8'><title>Hybrid promotion panel</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:2rem auto;color:#172033;background:#f5f7fb}}article,section{{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.55rem;border-bottom:1px solid #dde3eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}.gate{{font-size:1.4rem;font-weight:700}}.ok{{color:#087443}}.ko{{color:#a21d2d}}.deck{{display:inline-block;vertical-align:top;width:48%;min-width:300px;margin-right:1%}}</style>
<h1>Hybrid V2 — panel de promotion</h1><p>Profil {html.escape(payload['hybrid_profile'])} · {payload['games_per_opponent']} parties par adversaire · seed {payload['seed']}</p>
<section><div class='gate {'ok' if gate['accepted'] else 'ko'}'>Gate : {status}</div><p>Moyenne pondérée des deltas par rapport à 50 % : {gate['weighted_mean_delta']:+.2%}</p>
<table><thead><tr><th>Adversaire</th><th>Parties</th><th>Victoires</th><th>Défaites</th><th>Nuls</th><th>Delta vs 50 %</th><th>Delta taille deck</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(sections)}</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-profile", default="hybrid-v002")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hybrid_benchmark/hybrid-v002-panel.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/hybrid_benchmark/hybrid-v002-panel.html"))
    parser.add_argument("--render-from-json", type=Path)
    args = parser.parse_args()
    if args.render_from_json is not None:
        payload = json.loads(args.render_from_json.read_text(encoding="utf-8"))
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_html(payload), encoding="utf-8")
        print(f"rendered={args.html_output}")
        return
    torch.set_num_threads(args.torch_threads)
    profiles = {"v008": load_profile("configs/heuristic_profiles/v008.yaml")}
    neural_scorer = None
    records = []
    for index, opponent in enumerate(OPPONENTS):
        for offset in range(args.games):
            records.append(play_game(args.seed + index * args.games + offset, opponent, args.hybrid_profile, profiles, neural_scorer, args.max_actions, args.max_turns))
        print(f"completed={opponent} games={args.games}", flush=True)
    summary = {opponent: _summary([record for record in records if record["opponent"] == opponent]) for opponent in OPPONENTS}
    payload = {"hybrid_profile": args.hybrid_profile, "games_per_opponent": args.games, "seed": args.seed, "opponents": list(OPPONENTS), "gate": _gate(summary), "summary_by_opponent": summary, "games": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.html_output.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))


if __name__ == "__main__":
    main()
