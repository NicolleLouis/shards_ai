from __future__ import annotations

import json
from pathlib import Path

from scripts.normalize_imitation_dataset import normalize_dataset
from shards_ai.ai.neural_training import split_for_game_id


def _write_dataset(path: Path) -> None:
    records = []
    decision_index = 0
    for game_index in range(30):
        for phase, action_type in (
            ("buy", "buy_card"),
            ("buy", "stop_buying"),
            ("attack", "assign_power"),
            ("play", "play_card"),
            ("play", "banish_card"),
            ("play", "skip_banish"),
        ):
            records.append({
                "game_id": f"game-{game_index}",
                "decision_index": decision_index,
                "observation": {"phase": phase},
                "chosen_action": {"action_type": action_type},
            })
            decision_index += 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_normalization_is_reproducible_and_preserves_natural_holdouts(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_dataset(source)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    normalize_dataset(source, first, target_records=10, selection_seed=44)
    normalize_dataset(source, second, target_records=10, selection_seed=44)

    assert first.read_text() == second.read_text()
    first_records = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(first_records) == 10
    validation_records = [
        json.loads(line)
        for line in (tmp_path / "first.validation.jsonl").read_text().splitlines()
    ]
    test_records = [
        json.loads(line)
        for line in (tmp_path / "first.test.jsonl").read_text().splitlines()
    ]
    train_games = {record["game_id"] for record in first_records}
    assert train_games.isdisjoint({record["game_id"] for record in validation_records})
    assert train_games.isdisjoint({record["game_id"] for record in test_records})
    assert all(split_for_game_id(game_id) == "train" for game_id in train_games)
    manifest = json.loads(first.with_suffix(first.suffix + ".manifest.json").read_text())
    assert sum(manifest["selected_train"].values()) == 10
    assert manifest["shortfalls"] == {}


def test_normalization_reports_shortfalls_without_duplicating_records(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_dataset(source)
    output = tmp_path / "normalized.jsonl"
    normalize_dataset(source, output, target_records=1000, selection_seed=45)

    records = output.read_text().splitlines()
    manifest = json.loads(output.with_suffix(output.suffix + ".manifest.json").read_text())
    assert len(records) <= 30 * 6
    assert manifest["shortfalls"]
