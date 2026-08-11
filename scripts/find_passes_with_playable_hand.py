"""Find neural passes made while at least one card remains legally playable."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from benchmarks.benchmark_neural_mix import opponent_for_game, play_game
from shards_ai.ai import NeuralModelConfig, NeuralPlayer, build_neural_scorer
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import GameRunner


def _load_scorers(checkpoint: Path):
    document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = document.get("architecture", "independent_action")
    if architecture not in {
        "structured_semantic_v5_macro_deck_state_v1",
        "structured_semantic_v5_macro_root_action_v2",
        "structured_semantic_v5_macro_known_consequence_v1",
        "structured_semantic_v5_macro_tactical_action_v1",
    }:
        return architecture, NeuralPlayer.load_scorer(checkpoint), None

    macro_scorer = build_neural_scorer(
        architecture,
        NeuralModelConfig(**document["model_config"]),
    )
    macro_scorer.load_state_dict(document["model_state_dict"])
    macro_scorer.eval()
    return (
        architecture,
        macro_scorer,
        macro_scorer,
    )


def _card_counts(cards: list[dict[str, str]]) -> str:
    counts = Counter(card["card_id"] for card in cards)
    return ", ".join(
        f"{card_id} x{count}" if count > 1 else card_id
        for card_id, count in sorted(counts.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/neural_training/checkpoint.pt"),
    )
    parser.add_argument(
        "--profile-v007",
        type=Path,
        default=Path("configs/heuristic_profiles/v007.yaml"),
    )
    parser.add_argument(
        "--profile-v008",
        type=Path,
        default=Path("configs/heuristic_profiles/v008.yaml"),
    )
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--opponent",
        choices=("mix", "random", "v007", "v008"),
        default="mix",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    for path in (
        args.checkpoint,
        args.profile_v007,
        args.profile_v008,
    ):
        if not path.exists():
            parser.error(f"file not found: {path}")

    torch.set_num_threads(args.torch_threads)
    architecture, scorer, macro_scorer = _load_scorers(args.checkpoint)
    profiles = {
        "v007": load_profile(args.profile_v007),
        "v008": load_profile(args.profile_v008),
    }

    examples: list[dict[str, object]] = []
    pass_events = 0
    games_scanned = 0
    for index in range(args.games):
        opponent = (
            opponent_for_game(index) if args.opponent == "mix" else args.opponent
        )
        record = play_game(
            args.seed + index,
            args.checkpoint,
            scorer,
            opponent,
            profiles,
            args.max_actions,
            args.max_turns,
            args.torch_threads,
            macro_scorer=macro_scorer,
            pass_example_limit=args.max_examples - len(examples),
        )
        games_scanned += 1
        pass_events += int(record["neural_passes_with_playable_cards"])
        examples.extend(record["pass_with_playable_cards_examples"])
        if len(examples) >= args.max_examples:
            break
        if games_scanned % 100 == 0:
            print(f"scanned={games_scanned}/{args.games} examples={len(examples)}")

    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_architecture": architecture,
            "requested_games": args.games,
            "seed": args.seed,
            "opponent": args.opponent,
            "max_examples": args.max_examples,
        },
        "games_scanned": games_scanned,
        "pass_events": pass_events,
        "examples": examples[: args.max_examples],
    }

    print(
        f"games_scanned={games_scanned} pass_events={pass_events} "
        f"examples={len(payload['examples'])}"
    )
    for index, example in enumerate(payload["examples"], start=1):
        print(
            f"\nExample {index}: seed={example['seed']} "
            f"opponent={example['opponent']} turn={example['turn']} "
            f"source={example['action_source']}"
        )
        print(f"  remaining_hand: {_card_counts(example['remaining_hand'])}")
        print(f"  playable_cards: {_card_counts(example['playable_cards'])}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
