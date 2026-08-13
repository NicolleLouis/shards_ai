"""Measure macro and atomic decision cardinality against Heuristic V7/V8."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, build_neural_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


def _play_game(
    seed: int,
    scorer,
    opponent: str,
    profiles: dict[str, object],
    *,
    max_actions: int,
    max_turns: int | None,
) -> dict[str, object]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    neural_id = PlayerId.PLAYER_1 if seed % 2 == 0 else PlayerId.PLAYER_2
    opponent_id = neural_id.opponent
    neural = build_neural_player(
        neural_id,
        game,
        root_rng.derive("neural"),
        scorer=scorer,
    )
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
    macro_candidate_counts: list[int] = []

    def observe_macro(payload, player_id: PlayerId) -> None:
        if player_id is neural_id:
            macro_candidate_counts.append(len(payload.candidate_representations))

    started = time.perf_counter()
    state = runner.run(macro_decision_observer=observe_macro)
    macro_decisions = int(getattr(neural, "macro_decisions", 0))
    micro_decisions = int(getattr(neural, "atomic_decisions", 0))
    atomic_replays = int(getattr(neural, "atomic_replays", 0))
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
        "macro_decisions": macro_decisions,
        "macro_candidates_total": sum(macro_candidate_counts),
        "macro_candidates_per_decision": macro_candidate_counts,
        "micro_decisions": micro_decisions,
        "atomic_replays": atomic_replays,
        "exposed_decisions": macro_decisions + micro_decisions,
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    def values(key: str) -> list[float]:
        return [float(record[key]) for record in records]

    macro_decisions = values("macro_decisions")
    candidates_total = values("macro_candidates_total")
    micro_decisions = values("micro_decisions")
    exposed_decisions = values("exposed_decisions")
    return {
        "games": len(records),
        "macro_decisions": _stats(macro_decisions),
        "macro_candidates_total": _stats(candidates_total),
        "macro_candidates_per_macro_decision": _stats(
            [
                total / macro
                for total, macro in zip(candidates_total, macro_decisions)
                if macro
            ]
        ),
        "micro_decisions": _stats(micro_decisions),
        "atomic_replays": _stats(values("atomic_replays")),
        "exposed_decisions": _stats(exposed_decisions),
        "turns": _stats(values("turns")),
        "actions": _stats(values("actions")),
        "neural_win_rate": sum(bool(record["neural_won"]) for record in records) / len(records),
    }


def _stats(items: list[float]) -> dict[str, float | int]:
    if not items:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "min": min(items),
        "max": max(items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("configs/neural_profiles/v006.pt"))
    parser.add_argument("--profile-v007", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--profile-v008", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--games", type=int, default=50, help="Number of games against each opponent.")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/neural_benchmark/macro_decision_measurement.json"),
    )
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("games must be positive")
    if args.torch_threads <= 0:
        parser.error("torch-threads must be positive")
    for path in (args.checkpoint, args.profile_v007, args.profile_v008):
        if not path.exists():
            parser.error(f"file not found: {path}")

    profiles = {
        "v007": load_profile(args.profile_v007),
        "v008": load_profile(args.profile_v008),
    }
    torch.set_num_threads(args.torch_threads)
    scorer = NeuralPlayer.load_scorer(args.checkpoint)
    records: list[dict[str, object]] = []
    for opponent in ("v007", "v008"):
        for index in range(args.games):
            records.append(
                _play_game(
                    args.seed + index,
                    scorer,
                    opponent,
                    profiles,
                    max_actions=args.max_actions,
                    max_turns=args.max_turns,
                )
            )
        print(f"completed={opponent} games={args.games}", flush=True)

    summary = {opponent: _aggregate([record for record in records if record["opponent"] == opponent]) for opponent in ("v007", "v008")}
    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "games_per_opponent": args.games,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "opponents": ["v007", "v008"],
        },
        "summary_by_opponent": summary,
        "games": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for opponent in ("v007", "v008"):
        item = summary[opponent]
        print(
            f"{opponent}: macro={item['macro_decisions']['mean']:.2f} "
            f"candidates={item['macro_candidates_total']['mean']:.2f} "
            f"micro={item['micro_decisions']['mean']:.2f} "
            f"replays={item['atomic_replays']['mean']:.2f}",
            flush=True,
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
