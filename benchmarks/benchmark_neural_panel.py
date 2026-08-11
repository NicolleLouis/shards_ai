"""Run a neural checkpoint against the complete promoted neural/heuristic panel."""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import torch

from benchmarks.benchmark_neural_mix import (
    _copy_count_charts,
    _deck_counts,
    _deck_table,
    _mastery_table,
    _mercenary_table,
    _summary,
)
from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer, build_neural_player
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId
from shards_ai.game.actions import BuyCard, GainMastery, PassPlayPhase, PlayCard, RecruitMercenary


OPPONENTS = (
    "random",
    "v007",
    "v008",
    "neural:v001",
    "neural:v002",
    "neural:v003",
    "neural:v004",
    "neural:v005",
)
NEURAL_PROFILE_PATHS = {
    "v001": Path("configs/neural_profiles/v001.pt"),
    "v002": Path("configs/neural_profiles/v002.pt"),
    "v003": Path("configs/neural_profiles/v003.pt"),
    "v004": Path("configs/neural_profiles/v004.pt"),
    "v005": Path("configs/neural_profiles/v005.pt"),
}


def play_game(
    seed: int,
    scorer,
    opponent: str,
    heuristic_profiles: dict[str, HeuristicProfile],
    neural_scorers: dict[str, object],
    max_actions: int,
    max_turns: int | None,
) -> dict[str, object]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = neural_id.opponent
    neural = build_neural_player(
        neural_id, game, root_rng.derive("neural"), scorer=scorer,
    )
    if opponent == "random":
        other = RandomPlayer(opponent_id, root_rng.derive("opponent"))
    elif opponent.startswith("neural:"):
        other = build_neural_player(
            opponent_id,
            game,
            root_rng.derive("opponent"),
            scorer=neural_scorers[opponent.removeprefix("neural:")],
        )
    else:
        profile = heuristic_profiles[opponent]
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
    neural_passes_with_playable_cards = 0

    def observe_decision(observation, legal_actions, action, player_id) -> None:
        nonlocal neural_passes_with_playable_cards
        if player_id is not neural_id:
            return
        if isinstance(action, PassPlayPhase) and any(isinstance(candidate, PlayCard) for candidate in legal_actions):
            neural_passes_with_playable_cards += 1
        if any(isinstance(candidate, GainMastery) for candidate in legal_actions):
            mastery_events.append({
                "turn": observation.turn_number,
                "mastery": observation.active_player.mastery,
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
    neural_deck = _deck_counts(state.players[neural_id])
    opponent_deck = _deck_counts(state.players[opponent_id])
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
        "neural_macro_decisions": getattr(neural, "macro_decisions", 0),
        "neural_inference_seconds": neural.total_inference_seconds,
        "neural_health": state.players[neural_id].health,
        "opponent_health": state.players[opponent_id].health,
        "neural_mastery": state.players[neural_id].mastery,
        "opponent_mastery": state.players[opponent_id].mastery,
        "neural_deck": neural_deck,
        "opponent_deck": opponent_deck,
        "neural_deck_size": sum(neural_deck.values()),
        "opponent_deck_size": sum(opponent_deck.values()),
        "deck_size_delta": sum(neural_deck.values()) - sum(opponent_deck.values()),
        "neural_passes_with_playable_cards": neural_passes_with_playable_cards,
        "neural_passed_with_playable_cards": neural_passes_with_playable_cards > 0,
        "mercenary_events": mercenary_events,
        "mastery_events": mastery_events,
    }


def _render_report(payload: dict[str, object]) -> str:
    summary = payload["summary_by_opponent"]
    config = payload["config"]
    rows = []
    for opponent in OPPONENTS:
        item = summary[opponent]
        rows.append(
            f"<tr><td>{html.escape(opponent)}</td><td>{item['games']}</td>"
            f"<td>{item['neural_wins']}/{item['games']} ({item['neural_win_rate']:.1%})</td>"
            f"<td>{item['opponent_wins']}/{item['games']} ({item['opponent_win_rate']:.1%})</td>"
            f"<td>{item['draw_rate']:.1%}</td><td>{item['elapsed_seconds']['mean']:.2f}s</td>"
            f"<td>{item['actions']['mean']:.1f}</td><td>{item['neural_decisions']['mean']:.1f}</td>"
            f"<td>{item['average_neural_inference_ms']:.2f} ms</td>"
            f"<td>{item['deck_size_delta']['mean']:+.1f}</td></tr>"
        )

    sections = []
    for opponent in OPPONENTS:
        item = summary[opponent]
        sections.append(
            f"<article><h2>Contre {html.escape(opponent)}</h2>"
            f"<p>{item['games']} parties · victoire Neural {item['neural_win_rate']:.1%} · "
            f"durée moyenne {item['elapsed_seconds']['mean']:.2f}s · "
            f"{item['actions']['mean']:.1f} actions · {item['neural_decisions']['mean']:.1f} décisions neural</p>"
            f"<h3>État final moyen</h3><p>Neural : {item['neural_health']['mean']:.1f} PV, "
            f"{item['neural_mastery']['mean']:.1f} maîtrise · adversaire : "
            f"{item['opponent_health']['mean']:.1f} PV, {item['opponent_mastery']['mean']:.1f} maîtrise</p>"
            f"<h3>Deck final moyen</h3>{_deck_table(item['neural_deck'], 'Neural testé')}"
            f"{_deck_table(item['opponent_deck'], 'Adversaire')}"
            f"<h3>Graphiques par multiplicité du deck central</h3>"
            f"{''.join(_copy_count_charts(item, count) for count in (1, 2, 3))}"
            f"<h3>Mercenaires</h3>{_mercenary_table(item['mercenary_stats'])}"
            f"<h3>GainMastery par tour</h3>{_mastery_table(item['mastery_by_turn'], 'Tour')}"
            f"<h3>GainMastery par maîtrise initiale</h3>{_mastery_table(item['mastery_by_level'], 'Maîtrise')}</article>"
        )

    cards = "".join(
        f"<div><strong>{html.escape(opponent)}</strong><div class='metric'>{summary[opponent]['neural_win_rate']:.1%}</div>"
        f"<div class='bar'><div class='fill' style='width:{summary[opponent]['neural_win_rate'] * 100:.1f}%'></div></div></div>"
        for opponent in OPPONENTS
    )
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark NeuralPlayer — panel complet</title><style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:30px}}h1{{margin-bottom:5px}}.muted{{color:#637083}}.card,article{{background:white;border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 3px 14px #17203318}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.metric{{font-size:28px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #e5e9ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.bar{{height:18px;background:#e5e9ef;border-radius:10px;overflow:hidden;margin:8px 0 18px}}.fill{{height:100%;background:#2563eb}}.deck{{display:inline-block;vertical-align:top;width:48%;min-width:300px;margin-right:1%}}@media(max-width:700px){{.deck{{width:100%}}}}
</style></head><body><main><h1>Benchmark NeuralPlayer — panel complet</h1>
<p class="muted">{config['games_per_opponent']} parties par adversaire · {config['total_games']} parties · seed {config['seed']} · {config['torch_threads']} thread(s) Torch</p>
<p class="muted">Checkpoint testé : <code>{html.escape(config['checkpoint'])}</code></p>
<section class="card"><h2>Résumé du panel</h2><div class="grid">{cards}</div>
<table><thead><tr><th>Adversaire</th><th>Parties</th><th>Victoires Neural</th><th>Victoires adversaire</th><th>Nuls</th><th>Durée moy.</th><th>Actions moy.</th><th>Décisions moy.</th><th>Inférence moy.</th><th>Delta deck</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(sections)}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("configs/neural_profiles/v005.pt"))
    parser.add_argument("--profile-v007", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--profile-v001-checkpoint", type=Path, default=NEURAL_PROFILE_PATHS["v001"])
    parser.add_argument("--profile-v002-checkpoint", type=Path, default=NEURAL_PROFILE_PATHS["v002"])
    parser.add_argument("--profile-v003-checkpoint", type=Path, default=NEURAL_PROFILE_PATHS["v003"])
    parser.add_argument("--profile-v004-checkpoint", type=Path, default=NEURAL_PROFILE_PATHS["v004"])
    parser.add_argument("--profile-v005-checkpoint", type=Path, default=NEURAL_PROFILE_PATHS["v005"])
    parser.add_argument("--games", type=int, default=200, help="Number of games per opponent.")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_benchmark/neural_panel.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/neural_benchmark/neural_panel.html"))
    args = parser.parse_args()

    paths = {
        "checkpoint": args.checkpoint,
        "v001": args.profile_v001_checkpoint,
        "v002": args.profile_v002_checkpoint,
        "v003": args.profile_v003_checkpoint,
        "v004": args.profile_v004_checkpoint,
        "v005": args.profile_v005_checkpoint,
        "v007": args.profile_v007,
        "v008": args.profile_v008,
    }
    if args.games <= 0:
        parser.error("games must be positive")
    for label, path in paths.items():
        if not path.exists():
            parser.error(f"{label} file not found: {path}")

    profiles = {label: load_profile(paths[label]) for label in ("v007", "v008")}
    torch.set_num_threads(args.torch_threads)
    scorer = NeuralPlayer.load_scorer(args.checkpoint)
    neural_scorers = {
        profile_id: NeuralPlayer.load_scorer(paths[profile_id])
        for profile_id in ("v001", "v002", "v003", "v004", "v005")
    }
    records: list[dict[str, object]] = []
    for opponent in OPPONENTS:
        for index in range(args.games):
            records.append(play_game(
                args.seed + index,
                scorer,
                opponent,
                profiles,
                neural_scorers,
                args.max_actions,
                args.max_turns,
            ))
        print(f"completed={opponent} games={args.games}", flush=True)

    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "games_per_opponent": args.games,
            "total_games": args.games * len(OPPONENTS),
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "opponents": list(OPPONENTS),
            "neural_opponent_checkpoints": {profile_id: str(neural_scorers_path) for profile_id, neural_scorers_path in paths.items() if profile_id in {"v001", "v002", "v003", "v004", "v005"}},
        },
        "summary_by_opponent": {
            opponent: _summary([record for record in records if record["opponent"] == opponent])
            for opponent in OPPONENTS
        },
        "games": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(_render_report(payload), encoding="utf-8")
    print(json.dumps({opponent: payload["summary_by_opponent"][opponent]["neural_win_rate"] for opponent in OPPONENTS}, sort_keys=True))


if __name__ == "__main__":
    main()
