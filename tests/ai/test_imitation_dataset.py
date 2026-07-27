from __future__ import annotations

import json
from pathlib import Path

from shards_ai.ai import DatasetCampaignConfig, generate_dataset


PROFILE = Path("configs/heuristic_profiles/v007.yaml")
PROFILE_V008 = Path("configs/heuristic_profiles/v008.yaml")


def test_dataset_contains_masked_decisions_and_final_outcomes(tmp_path: Path) -> None:
    result = generate_dataset(
        DatasetCampaignConfig(
            profile_paths=(PROFILE,),
            output_path=tmp_path / "dataset.jsonl",
            seed=501,
            games=1,
        )
    )

    records = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    manifest = json.loads(result.manifest_path.read_text())

    assert records
    assert result.decision_count == len(records)
    assert manifest["decision_count"] == len(records)
    assert all(record["final_outcome"] in {"win", "loss", "draw"} for record in records)
    assert all("hand" not in record["observation"]["opponent"] for record in records)
    assert all("draw_pile_counts" not in record["observation"]["opponent"] for record in records)
    assert all(
        len(record["legal_actions"])
        == len(record["action_representations"])
        == len(record["heuristic_scores"])
        for record in records
    )
    assert all(
        record["legal_actions"][record["chosen_action_index"]] == record["chosen_action"]
        for record in records
    )


def test_target_decisions_stops_after_a_completed_game(tmp_path: Path) -> None:
    result = generate_dataset(
        DatasetCampaignConfig(
            profile_paths=(PROFILE,),
            output_path=tmp_path / "target.jsonl",
            seed=502,
            target_decisions=20,
            max_games=2,
        )
    )

    assert result.decision_count >= 20
    assert result.completed_games == 1


def test_same_seed_and_configuration_produce_identical_dataset(tmp_path: Path) -> None:
    first = generate_dataset(
        DatasetCampaignConfig(
            profile_paths=(PROFILE,),
            output_path=tmp_path / "first.jsonl",
            seed=503,
            games=1,
        )
    )
    second = generate_dataset(
        DatasetCampaignConfig(
            profile_paths=(PROFILE,),
            output_path=tmp_path / "second.jsonl",
            seed=503,
            games=1,
        )
    )

    assert first.output_path.read_text() == second.output_path.read_text()
    first_manifest = json.loads(first.manifest_path.read_text())
    second_manifest = json.loads(second.manifest_path.read_text())
    assert first_manifest == second_manifest


def test_targeted_teachers_keep_only_teacher_decisions(tmp_path: Path) -> None:
    from shards_ai.ai import DatasetCampaignConfig, MatchupSpec
    from shards_ai.ai.heuristic_profiles import load_profile

    profiles = (PROFILE_V008, PROFILE)
    loaded = {path: load_profile(path) for path in profiles}
    result = generate_dataset(
        DatasetCampaignConfig(
            profile_paths=profiles,
            output_path=tmp_path / "targeted.jsonl",
            seed=504,
            games=2,
            matchups=(
                MatchupSpec(PROFILE_V008),
                MatchupSpec(PROFILE_V008, PROFILE),
            ),
            record_profile_ids=frozenset({loaded[PROFILE_V008].profile_id}),
        )
    )

    records = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    manifest = json.loads(result.manifest_path.read_text())
    assert records
    assert {record["heuristic_profile_id"] for record in records} == {"v008"}
    assert set(manifest["record_profile_ids"]) == {"v008"}
    assert set(manifest["decisions_by_matchup"]) == {"v008_vs_random", "v008_vs_v007"}
