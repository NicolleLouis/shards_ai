"""Benchmark NeuralPlayer against Neural, Heuristic V8, and three hybrids."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, HybridPlayer, NeuralPlayer
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId

OPPONENTS = (
    "neural",
    "heuristic_v008",
    "hybrid_purchase_recruitment",
    "hybrid_play_phase",
    "hybrid_banish",
)
POLICIES = {
    "hybrid_purchase_recruitment": "purchase_recruitment",
    "hybrid_play_phase": "play_phase",
    "hybrid_banish": "banish",
}


def opponent_for_game(game_index: int) -> str:
    return OPPONENTS[game_index % len(OPPONENTS)]


def _heuristic(profile: HeuristicProfile, player_id: PlayerId) -> HeuristicPlayer:
    return HeuristicPlayer(
        player_id,
        profile.weights,
        profile.card_acquisition_weights,
        profile.constraint_weights,
    )


def play_game(
    seed: int,
    checkpoint: Path,
    scorer,
    profile: HeuristicProfile,
    opponent: str,
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
    if opponent == "neural":
        other = NeuralPlayer(opponent_id, None, root_rng.derive("opponent"), scorer=scorer)
    elif opponent == "heuristic_v008":
        other = _heuristic(profile, opponent_id)
    else:
        other = HybridPlayer(
            opponent_id,
            game,
            root_rng.derive("opponent"),
            scorer=scorer,
            policy=POLICIES[opponent],
            weights=profile.weights,
            acquisition_weights=profile.card_acquisition_weights,
            constraint_weights=profile.constraint_weights,
        )
    runner = GameRunner(
        game,
        {neural_id: neural, opponent_id: other},
        max_actions=max_actions,
        max_turns=max_turns,
    )
    started = time.perf_counter()
    state = runner.run()
    return {
        "seed": seed,
        "opponent": opponent,
        "status": state.status.value,
        "neural_won": state.winner is neural_id,
        "opponent_won": state.winner is opponent_id,
        "draw": state.status is GameStatus.DRAW,
        "turns": state.turn_number,
        "actions": runner.actions_played,
        "elapsed_seconds": time.perf_counter() - started,
        "neural_decisions": neural.decisions,
        "neural_inference_seconds": neural.total_inference_seconds,
        "opponent_decisions": getattr(other, "decisions", None),
        "opponent_inference_seconds": getattr(other, "total_inference_seconds", 0.0),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    games = len(records)
    wins = sum(bool(record["neural_won"]) for record in records)
    losses = sum(bool(record["opponent_won"]) for record in records)
    draws = sum(bool(record["draw"]) for record in records)

    def mean(key: str) -> float:
        values = [float(record[key]) for record in records]
        return statistics.mean(values) if values else 0.0

    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "non_draw_win_rate": wins / (wins + losses) if wins + losses else 0.0,
        "mean_turns": mean("turns"),
        "mean_actions": mean("actions"),
        "mean_neural_decisions": mean("neural_decisions"),
        "mean_neural_inference_seconds": mean("neural_inference_seconds"),
    }


def write_html(result: dict[str, object], path: Path) -> None:
    rows = []
    for opponent, summary in result["summary_by_opponent"].items():
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    opponent,
                    summary["games"],
                    summary["wins"],
                    summary["losses"],
                    summary["draws"],
                    f"{summary['win_rate']:.1%}",
                    f"{summary['mean_turns']:.1f}",
                )
            )
            + "</tr>"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Neural hybrid benchmark</title>"
        "<style>body{font:16px sans-serif;max-width:1000px;margin:2rem auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.45rem}</style>"
        "<h1>Neural hybrid benchmark</h1>"
        "<p>Le NeuralPlayer est évalué contre cinq types d'adversaires."
        " La répartition est déterministe et chaque ligne agrège les parties du même type.</p>"
        "<table><thead><tr><th>Adversaire</th><th>Parties</th><th>Victoires</th>"
        "<th>Défaites</th><th>N nulles</th><th>Win rate</th><th>Tours moyens</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    args = parser.parse_args()
    if args.games <= 0 or args.games % len(OPPONENTS):
        parser.error(f"--games must be a positive multiple of {len(OPPONENTS)}")
    profile = load_profile(args.profile)
    scorer = NeuralPlayer.load_scorer(args.checkpoint)
    records = [
        play_game(
            args.seed + index,
            args.checkpoint,
            scorer,
            profile,
            opponent_for_game(index),
            args.max_actions,
            args.max_turns,
            args.torch_threads,
        )
        for index in range(args.games)
    ]
    grouped = {opponent: [record for record in records if record["opponent"] == opponent] for opponent in OPPONENTS}
    result = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "heuristic_profile": profile.profile_id,
            "games": args.games,
            "seed": args.seed,
            "opponents": list(OPPONENTS),
            "allocation": "game_index modulo 5",
        },
        "summary_by_opponent": {opponent: summarize(rows) for opponent, rows in grouped.items()},
        "games": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_html(result, args.html_output)
    print(json.dumps(result["summary_by_opponent"], indent=2))


if __name__ == "__main__":
    main()
