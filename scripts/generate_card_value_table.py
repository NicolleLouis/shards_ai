"""Generate a fixed card-value table from heuristic profile v008."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from shards_ai.ai.heuristic_features import _card_acquisition_value
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import CARD_CATALOG, CardInstance, Game, GameRandom, PlayerId


def generate_values(profile_path: str | Path) -> dict[str, float]:
    profile = load_profile(profile_path)
    game = Game.new(seed=58008, rng=GameRandom(58008))
    state = game.observation_for(PlayerId.PLAYER_1)
    player = replace(
        state.players[PlayerId.PLAYER_1],
        gems=10,
        mastery=15,
        hand=[],
        draw_pile=[],
        discard_pile=[],
        play_zone=[],
        champions=[],
    )
    state = replace(
        state,
        players={**state.players, PlayerId.PLAYER_1: player},
    )
    values = {}
    for card_id, definition in sorted(CARD_CATALOG.items()):
        card = CardInstance(f"value-{card_id}", definition)
        values[card_id] = round(
            _card_acquisition_value(
                state,
                player,
                card,
                profile.card_acquisition_weights,
                profile.constraint_weights,
            ),
            6,
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/neural_training_profiles/card_values_v008.yaml"),
    )
    args = parser.parse_args()
    values = generate_values(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source_profile": str(args.profile),
                "method": "v008_neutral_card_acquisition_value",
                "values": values,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(values)} card values in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
