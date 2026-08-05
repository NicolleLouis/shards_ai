import json
import importlib.util
from pathlib import Path

from shards_ai.analysis.dagger_dataset import priority_weight, sample_dataset, sample_on_policy_dataset


def _merge_module():
    path = Path(__file__).parents[2] / "scripts" / "merge_dagger_datasets.py"
    spec = importlib.util.spec_from_file_location("merge_dagger_datasets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_records(path: Path, count: int, *, phase: str, prefix: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(count):
            stream.write(json.dumps({
                "game_id": f"{prefix}-{index}",
                "decision_index": index,
                "observation": {"phase": phase},
                "regret": float(index % 3),
                "teacher_rank": 4 if index % 2 else 1,
                "strategic_divergence": phase == "play" and index % 2 == 0,
            }) + "\n")


def test_sample_dataset_respects_mix_quotas(tmp_path: Path) -> None:
    old = tmp_path / "old.jsonl"
    dagger = tmp_path / "dagger.jsonl"
    output = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_records(old, 100, phase="buy", prefix="old")
    _write_records(dagger, 100, phase="play", prefix="dagger-play")
    with dagger.open("a", encoding="utf-8") as stream:
        for index in range(100):
            stream.write(json.dumps({
                "game_id": f"dagger-other-{index}", "decision_index": index,
                "observation": {"phase": "buy"}, "regret": 0.0,
                "teacher_rank": 1, "strategic_divergence": False,
            }) + "\n")
    manifest = sample_dataset(old, dagger, output, validation, target_records=20, seed=11)

    assert manifest["selected_records"] == 20
    assert manifest["selected_by_bucket"] == {"old": 9, "dagger_play": 7, "dagger_other": 4}
    assert sum(1 for _ in output.open()) == 20


def test_priority_weight_uses_teacher_action_type(tmp_path: Path) -> None:
    record = {
        "observation": {"phase": "buy"},
        "teacher_action_type": "recruit_mercenary",
        "neural_action_type": "play_card",
        "regret": 0.0,
        "teacher_rank": 1,
    }
    assert priority_weight(record, {"recruit_mercenary": 3.0}) == 3.0


def test_sample_on_policy_dataset_applies_action_weights(tmp_path: Path) -> None:
    dagger = tmp_path / "dagger.jsonl"
    output = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    with dagger.open("w", encoding="utf-8") as stream:
        for index in range(30):
            stream.write(json.dumps({
                "game_id": f"dagger_2-{index}",
                "decision_index": index,
                "observation": {"phase": "buy"},
                "teacher_action_type": "assign_power" if index < 10 else "buy_card",
                "teacher_profile_id": "v008",
                "teacher_scores": [1.0],
                "heuristic_scores": [1.0],
                "teacher_action_index": 0,
                "chosen_action_index": 0,
                "regret": 0.0,
                "teacher_rank": 1,
            }) + "\n")
    manifest = sample_on_policy_dataset(
        dagger, output, validation, target_records=10, seed=11,
        action_weights={"assign_power": 3.0},
    )
    assert manifest["selected_records"] == 10
    assert all(json.loads(line)["dagger_sample_bucket"] == "dagger_2_weighted" for line in output.open())


def test_merge_marks_provenance_and_requires_v008_teacher(tmp_path: Path) -> None:
    historical = tmp_path / "historical.jsonl"
    dagger = tmp_path / "dagger.jsonl"
    output = tmp_path / "merged.jsonl"
    historical.write_text('{"game_id":"old-1"}\n', encoding="utf-8")
    dagger.write_text(json.dumps({
        "game_id": "dagger_2-1", "teacher_profile_id": "v008",
        "teacher_scores": [1.0], "heuristic_scores": [1.0],
        "teacher_action_index": 0, "chosen_action_index": 0,
    }) + "\n", encoding="utf-8")
    manifest = _merge_module().merge([("historical", historical), ("dagger_2", dagger)], output)
    records = [json.loads(line) for line in output.open()]
    assert manifest["records_by_source"] == {"historical": 1, "dagger_2": 1}
    assert records[0]["dataset_source"] == "historical"
    assert records[1]["dagger_stage"] == "dagger_2"

    dagger.write_text('{"teacher_profile_id":"v007"}\n', encoding="utf-8")
    try:
        _merge_module().merge([("dagger_2", dagger)], output)
    except ValueError as error:
        assert "v008" in str(error)
    else:
        raise AssertionError("a non-v008 DAgGER record must be rejected")
