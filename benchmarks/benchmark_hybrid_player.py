"""Benchmark one versioned HybridPlayer against selected reference players."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer, build_hybrid_player, build_neural_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


def _make_player(name, player_id, game, rng, *, hybrid_profile, heuristic_profiles, neural_scorer):
    if name == "Random":
        return RandomPlayer(player_id, rng)
    if name in heuristic_profiles:
        profile = heuristic_profiles[name]
        return HeuristicPlayer(player_id, profile.weights, profile.card_acquisition_weights, profile.constraint_weights)
    if name == "Neural V6":
        return build_neural_player(player_id, game, rng, scorer=neural_scorer)
    if name == "Hybrid V2":
        return build_hybrid_player(player_id, game, rng, profile=hybrid_profile)
    raise ValueError(f"Unknown player: {name}")


def play_match(name, opponent, games, first_seed, *, hybrid_profile, heuristic_profiles, neural_scorer, max_actions, max_turns):
    wins = losses = draws = 0
    for index in range(games):
        seed = first_seed + index
        root_rng = GameRandom(seed)
        game = Game.new(seed=seed, rng=root_rng.derive("engine"))
        row_id = PlayerId.PLAYER_1 if index % 2 == 0 else PlayerId.PLAYER_2
        column_id = row_id.opponent
        players = {
            row_id: _make_player(name, row_id, game, root_rng.derive("row"), hybrid_profile=hybrid_profile, heuristic_profiles=heuristic_profiles, neural_scorer=neural_scorer),
            column_id: _make_player(opponent, column_id, game, root_rng.derive("column"), hybrid_profile=hybrid_profile, heuristic_profiles=heuristic_profiles, neural_scorer=neural_scorer),
        }
        state = GameRunner(game, players, max_actions=max_actions, max_turns=max_turns).run()
        if state.status is GameStatus.DRAW or state.winner is None:
            draws += 1
        elif state.winner is row_id:
            wins += 1
        else:
            losses += 1
    return {"games": games, "wins": wins, "losses": losses, "draws": draws, "win_rate": wins / games}


def render_html(payload):
    rows = []
    for opponent, result in payload["results"].items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(opponent)}</th><td>{result['games']}</td>"
            f"<td>{result['wins']}</td><td>{result['losses']}</td><td>{result['draws']}</td>"
            f"<td>{result['win_rate']:.2%}</td></tr>"
        )
    return f"""<!doctype html><html lang='fr'><meta charset='utf-8'><title>Benchmark Hybrid V2</title>
<style>body{{font:16px system-ui;max-width:900px;margin:2rem auto;color:#172033}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5e0;padding:.6rem;text-align:center}}th{{text-align:left;background:#eef2f7}}.note{{color:#64748b}}</style>
<h1>Hybrid V2 contre références</h1><p class='note'>Profil {html.escape(payload['hybrid_profile'])} · {payload['games_per_opponent']} parties par adversaire · seed {payload['seed']}</p>
<table><thead><tr><th>Adversaire</th><th>Parties</th><th>Victoires</th><th>Défaites</th><th>Nulles</th><th>Taux de victoire</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-profile", default="hybrid-v002")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hybrid_benchmark/hybrid-v002.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/hybrid_benchmark/hybrid-v002.html"))
    args = parser.parse_args()
    if args.games <= 0 or args.torch_threads <= 0:
        parser.error("--games and --torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    heuristic_profiles = {
        "Heuristic V7": load_profile("configs/heuristic_profiles/v007.yaml"),
        "Heuristic V8": load_profile("configs/heuristic_profiles/v008.yaml"),
    }
    neural_scorer = NeuralPlayer.load_scorer("configs/neural_profiles/v006.pt")
    opponents = ("Random", "Heuristic V8", "Heuristic V7", "Neural V6")
    results = {}
    for index, opponent in enumerate(opponents):
        results[opponent] = play_match(
            "Hybrid V2", opponent, args.games, args.seed + index * args.games,
            hybrid_profile=args.hybrid_profile, heuristic_profiles=heuristic_profiles,
            neural_scorer=neural_scorer, max_actions=args.max_actions, max_turns=args.max_turns,
        )
        print(f"completed=Hybrid V2 vs {opponent} {results[opponent]}", flush=True)
    payload = {"hybrid_profile": args.hybrid_profile, "games_per_opponent": args.games, "seed": args.seed, "opponents": list(opponents), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.html_output.write_text(render_html(payload), encoding="utf-8")
    print(f"wrote={args.output} html={args.html_output}")


if __name__ == "__main__":
    main()
