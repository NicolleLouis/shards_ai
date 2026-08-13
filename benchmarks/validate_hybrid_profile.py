"""Run the usual weighted promotion gate for two HybridPlayer profiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, build_hybrid_player, build_neural_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId
from scripts.validate_neural_profile import QUALITY_OPPONENT_WEIGHTS


OPPONENTS = tuple(QUALITY_OPPONENT_WEIGHTS)


def _opponent(label, player_id, game, rng, heuristics, neural_scorers):
    if label.startswith("neural:"):
        return build_neural_player(
            player_id, game, rng, scorer=neural_scorers[label.removeprefix("neural:")]
        )
    profile = heuristics[label]
    return HeuristicPlayer(
        player_id,
        profile.weights,
        profile.card_acquisition_weights,
        profile.constraint_weights,
    )


def play_game(seed, candidate_profile, reference_profile, candidate_neural, reference_neural, opponent, heuristics, neural_scorers, max_actions, max_turns):
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    candidate_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = candidate_id.opponent
    candidate = (
        build_neural_player(candidate_id, game, root_rng.derive("candidate"), scorer=candidate_neural)
        if candidate_neural is not None
        else build_hybrid_player(candidate_id, game, root_rng.derive("candidate"), profile=candidate_profile)
    )
    reference = (
        build_neural_player(candidate_id, game, root_rng.derive("reference"), scorer=reference_neural)
        if reference_neural is not None
        else build_hybrid_player(candidate_id, game, root_rng.derive("reference"), profile=reference_profile)
    )
    other = _opponent(
        opponent,
        opponent_id,
        game,
        root_rng.derive("opponent"),
        heuristics,
        neural_scorers,
    )
    def run(player):
        state = GameRunner(
            game,
            {candidate_id: player, opponent_id: other},
            max_actions=max_actions,
            max_turns=max_turns,
        ).run()
        return {
            "candidate_won": state.winner is candidate_id,
            "opponent_won": state.winner is opponent_id,
            "draw": state.status is GameStatus.DRAW,
        }

    # The engine mutates state, so each profile must receive a fresh game.
    candidate_result = run(candidate)
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    reference_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    reference_opponent_id = reference_id.opponent
    reference = (
        build_neural_player(reference_id, game, root_rng.derive("reference"), scorer=reference_neural)
        if reference_neural is not None
        else build_hybrid_player(reference_id, game, root_rng.derive("reference"), profile=reference_profile)
    )
    reference_other = _opponent(
        opponent,
        reference_opponent_id,
        game,
        root_rng.derive("opponent"),
        heuristics,
        neural_scorers,
    )
    state = GameRunner(
        game,
        {reference_id: reference, reference_opponent_id: reference_other},
        max_actions=max_actions,
        max_turns=max_turns,
    ).run()
    reference_result = {
        "candidate_won": state.winner is reference_id,
        "opponent_won": state.winner is reference_opponent_id,
        "draw": state.status is GameStatus.DRAW,
    }
    return {"seed": seed, "candidate": candidate_result, "reference": reference_result}


def aggregate(records, side):
    games = len(records)
    wins = sum(record[side]["candidate_won"] for record in records)
    losses = sum(record[side]["opponent_won"] for record in records)
    draws = sum(record[side]["draw"] for record in records)
    return {"games": games, "wins": wins, "losses": losses, "draws": draws, "win_rate": wins / games}


def validate(args):
    heuristics = {version: load_profile(f"configs/heuristic_profiles/{version}.yaml") for version in ("v007", "v008")}
    neural_scorers = {
        version: NeuralPlayer.load_scorer(f"configs/neural_profiles/{version}.pt")
        for version in ("v001", "v002", "v004", "v005", "v006")
    }
    reference_neural = neural_scorers.get(args.reference_neural_profile)
    candidate_neural = neural_scorers.get(args.candidate_neural_profile)
    results = {}
    for index, opponent in enumerate(OPPONENTS):
        records = []
        for offset in range(args.games):
            records.append(play_game(
                args.seed + index * args.games + offset,
                args.candidate_profile,
                args.reference_profile,
                candidate_neural,
                reference_neural,
                opponent,
                heuristics,
                neural_scorers,
                args.max_actions,
                args.max_turns,
            ))
        candidate = aggregate(records, "candidate")
        reference = aggregate(records, "reference")
        results[opponent] = {
            "candidate": candidate,
            "reference": reference,
            "delta_win_rate": candidate["win_rate"] - reference["win_rate"],
            "weight": QUALITY_OPPONENT_WEIGHTS[opponent],
        }
        print(f"completed={opponent} games={args.games}", flush=True)
    weighted = sum(item["weight"] * item["delta_win_rate"] for item in results.values())
    total_weight = sum(item["weight"] for item in results.values())
    mean_delta = weighted / total_weight
    return {
        "candidate_profile": args.candidate_profile,
        "candidate_neural_profile": args.candidate_neural_profile,
        "reference_profile": args.reference_profile,
        "reference_neural_profile": args.reference_neural_profile,
        "games_per_opponent": args.games,
        "seed": args.seed,
        "opponents": list(OPPONENTS),
        "opponent_weights": QUALITY_OPPONENT_WEIGHTS,
        "results": results,
        "decision_metrics": {"accepted": mean_delta > 0.0, "weighted_mean_delta": mean_delta, "minimum": 0.0},
        "decision": "accepted" if mean_delta > 0.0 else "rejected",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-profile", default="hybrid-v002")
    parser.add_argument("--candidate-neural-profile", default=None)
    parser.add_argument("--reference-profile", default="hybrid-v001")
    parser.add_argument("--reference-neural-profile", default="v006")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hybrid_benchmark/hybrid-v002-vs-neural-v006-promotion-gate.json"))
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    torch.set_num_threads(args.torch_threads)
    started = time.perf_counter()
    report = validate(args)
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for opponent, result in report["results"].items():
        print(f"{opponent}: candidat={result['candidate']['win_rate']:.2%} référence={result['reference']['win_rate']:.2%} delta={result['delta_win_rate']:+.2%}")
    print(f"Gate: {report['decision'].upper()} ({report['decision_metrics']['weighted_mean_delta']:+.2%})")
    return 0 if report["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
