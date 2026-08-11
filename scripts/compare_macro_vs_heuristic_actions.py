#!/usr/bin/env python3
"""Compare exact atomic actions of the macro policy with Heuristic V8."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from shards_ai.ai import (
    HeuristicPlayer,
    MacroNeuralPlayer,
    NeuralModelConfig,
    build_neural_scorer,
)
from shards_ai.ai.action_representation import representation_for_neural_action
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, PlayerId


def _load_scorer(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scorer = build_neural_scorer(
        checkpoint.get("architecture", "independent_action"),
        NeuralModelConfig(**checkpoint["model_config"]),
    )
    scorer.load_state_dict(checkpoint["model_state_dict"])
    scorer.eval()
    return scorer


def _action_record(action, observation):
    representation = representation_for_neural_action(action, observation)
    return {
        "action_type": representation.action_type,
        "card_definition_id": representation.card_definition_id,
        "card_instance_id": representation.card_instance_id,
        "parameters": asdict(action),
    }


def _semantic_key(record):
    parameters = record["parameters"]
    return (
        record["action_type"],
        record["card_definition_id"],
        parameters.get("target"),
        parameters.get("amount"),
        parameters.get("river_slot"),
        parameters.get("choice_id"),
    )


def _play(seed, macro_scorer, profile, max_actions, max_turns):
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    candidate_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = candidate_id.opponent

    def choose_macro(_game, observation, candidates):
        with torch.inference_mode():
            return int(macro_scorer(observation, candidates).argmax().item())

    candidate = MacroNeuralPlayer(
        candidate_id,
        game,
        candidate_scorer=choose_macro,
    )
    opponent = HeuristicPlayer(
        opponent_id,
        profile.weights,
        profile.card_acquisition_weights,
        profile.constraint_weights,
    )
    counterfactual = HeuristicPlayer(
        candidate_id,
        profile.weights,
        profile.card_acquisition_weights,
        profile.constraint_weights,
    )
    runner = GameRunner(
        game,
        {candidate_id: candidate, opponent_id: opponent},
        max_actions=max_actions,
        max_turns=max_turns,
    )
    records = []
    first_divergence = None
    first_exact_divergence = None
    decision_number = 0

    def observe(observation, legal_actions, chosen, player_id):
        nonlocal decision_number, first_divergence, first_exact_divergence
        if player_id is not candidate_id:
            return
        heuristic_action = counterfactual.choose_action(
            game.observation_for(candidate_id), legal_actions,
        )
        candidate_record = _action_record(chosen, observation)
        heuristic_record = _action_record(heuristic_action, observation)
        disagreement = chosen != heuristic_action
        semantic_disagreement = _semantic_key(candidate_record) != _semantic_key(heuristic_record)
        item = {
            "seed": seed,
            "decision": decision_number,
            "turn": observation.turn_number,
            "phase": observation.phase,
            "decision_kind": candidate.last_action_kind,
            "candidate": candidate_record,
            "heuristic_v8": heuristic_record,
            "disagreement": disagreement,
            "semantic_disagreement": semantic_disagreement,
        }
        records.append(item)
        if semantic_disagreement and first_divergence is None:
            first_divergence = item
        if disagreement and first_exact_divergence is None:
            first_exact_divergence = item
        decision_number += 1

    state = runner.run(decision_observer=observe)
    return {
        "seed": seed,
        "status": state.status.value,
        "decisions": records,
        "first_divergence": first_divergence,
        "first_exact_divergence": first_exact_divergence,
        "candidate_won": state.winner is candidate_id,
    }


def _render(report):
    overall = report["summary"]
    pairs = report["disagreement_pairs"]
    pair_rows = "".join(
        f"<tr><td>{html.escape(item['candidate_action'])}</td><td>{html.escape(item['heuristic_action'])}</td>"
        f"<td>{item['count']}</td><td>{item['rate']:.2%}</td></tr>"
        for item in pairs[:50]
    ) or "<tr><td colspan='4'>Aucun désaccord</td></tr>"
    first_rows = "".join(
        f"<tr><td>{item['seed']}</td><td>{item['decision']}</td><td>{html.escape(str(item['phase']))}</td>"
        f"<td>{html.escape(str(item['candidate']['action_type']))}</td>"
        f"<td>{html.escape(str(item['heuristic_v8']['action_type']))}</td></tr>"
        for item in report["first_divergences"]
    ) or "<tr><td colspan='5'>Aucune divergence</td></tr>"
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Comparaison actions macro vs V8</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:28px;background:#f5f7fb;color:#172033}}section{{background:white;border-radius:12px;padding:18px;margin:16px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #e5e9ef;text-align:right}}th:first-child,td:first-child{{text-align:left}}.metric{{font-size:28px;font-weight:700}}</style></head><body>
<h1>Actions précises : politique macro contre Heuristic V8</h1>
<p>Le V8 est calculé contrefactuellement sur chaque état réellement visité par la politique macro ; son action n'est jamais appliquée.</p>
<section><h2>Résumé</h2><p><span class="metric">{overall['decisions']}</span> décisions · <span class="metric">{overall['disagreements']}</span> désaccords sémantiques ({overall['disagreement_rate']:.2%}) · {overall['exact_disagreements']} désaccords exacts ({overall['exact_disagreement_rate']:.2%}) · {report['games']} parties</p>
<p>Macro choices : {overall['macro_choices']} · replay canonique : {overall['macro_replays']}</p></section>
<section><h2>Désaccords par action</h2><table><thead><tr><th>Action candidate</th><th>Action V8</th><th>Nombre</th><th>Part des désaccords</th></tr></thead><tbody>{pair_rows}</tbody></table></section>
<section><h2>Première divergence par partie</h2><table><thead><tr><th>Seed</th><th>Décision</th><th>Phase</th><th>Candidate</th><th>V8</th></tr></thead><tbody>{first_rows}</tbody></table></section>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/neural_training/checkpoint.pt"))
    parser.add_argument("--profile", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/macro_vs_v008_actions.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/analysis/macro_vs_v008_actions.html"))
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("games must be positive")
    torch.set_num_threads(args.torch_threads)
    macro_scorer = _load_scorer(args.checkpoint)
    profile = load_profile(args.profile)
    games = [
        _play(args.seed + index, macro_scorer, profile, args.max_actions, args.max_turns)
        for index in range(args.games)
    ]
    decisions = [item for game in games for item in game["decisions"]]
    exact_disagreements = [item for item in decisions if item["disagreement"]]
    disagreements = [item for item in decisions if item["semantic_disagreement"]]
    pair_counts = Counter(
        (item["candidate"]["action_type"], item["heuristic_v8"]["action_type"])
        for item in disagreements
    )
    report = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "profile": str(args.profile),
            "games": args.games,
            "seed": args.seed,
        },
        "games": games,
        "summary": {
            "decisions": len(decisions),
            "disagreements": len(disagreements),
            "disagreement_rate": len(disagreements) / len(decisions) if decisions else 0.0,
            "exact_disagreements": len(exact_disagreements),
            "exact_disagreement_rate": len(exact_disagreements) / len(decisions) if decisions else 0.0,
            "macro_choices": sum(item["decision_kind"] == "macro_choice" for item in decisions),
            "macro_replays": sum(item["decision_kind"] == "macro_replay" for item in decisions),
        },
        "disagreement_pairs": [
            {
                "candidate_action": candidate,
                "heuristic_action": heuristic,
                "count": count,
                "rate": count / len(disagreements) if disagreements else 0.0,
            }
            for (candidate, heuristic), count in pair_counts.most_common()
        ],
        "first_divergences": [game["first_divergence"] for game in games if game["first_divergence"]],
        "first_exact_divergences": [game["first_exact_divergence"] for game in games if game["first_exact_divergence"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(_render(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
