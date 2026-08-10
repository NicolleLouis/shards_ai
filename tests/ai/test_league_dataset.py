from __future__ import annotations

import json
from pathlib import Path

from shards_ai.ai.league_dataset import (
    LeagueDatasetConfig,
    LeaguePlayerSpec,
    collect_league_dataset,
)


PROFILE_V007 = Path("configs/heuristic_profiles/v007.yaml")
PROFILE_V008 = Path("configs/heuristic_profiles/v008.yaml")


def test_league_writes_common_variants_and_masks_opponent(tmp_path: Path) -> None:
    result = collect_league_dataset(
        LeagueDatasetConfig(
            players=(
                LeaguePlayerSpec("random", "random"),
                LeaguePlayerSpec("heuristic", "v007", PROFILE_V007),
            ),
            output_dir=tmp_path,
            seed=801,
            games_per_matchup=1,
        )
    )

    assert result.completed_games == 2
    records = [
        json.loads(line)
        for line in result.variant_paths["control_full_unweighted"].read_text().splitlines()
    ]
    assert {record["teacher_type"] for record in records} == {"random", "heuristic"}
    assert all("hand" not in record["observation"]["opponent"] for record in records)
    assert all(record["sample_weight"] == 1.0 for record in records)
    assert any(record["teacher_scores_available"] for record in records)
    assert any(not record["teacher_scores_available"] for record in records)

    weighted = [
        json.loads(line)
        for line in result.variant_paths["weighted_moderate"].read_text().splitlines()
    ]
    assert all(record["sample_weight"] > 0 for record in weighted)


def test_winner_only_is_subset_of_same_collection(tmp_path: Path) -> None:
    result = collect_league_dataset(
        LeagueDatasetConfig(
            players=(
                LeaguePlayerSpec("heuristic", "v007", PROFILE_V007),
                LeaguePlayerSpec("heuristic", "v008", PROFILE_V008),
            ),
            output_dir=tmp_path,
            seed=802,
            games_per_matchup=1,
        )
    )
    all_records = [
        json.loads(line)
        for line in result.variant_paths["control_full_unweighted"].read_text().splitlines()
    ]
    winner_records = [
        json.loads(line)
        for line in result.variant_paths["winner_only"].read_text().splitlines()
    ]
    assert winner_records
    assert all(record["final_outcome"] == "win" for record in winner_records)
    assert len(winner_records) < len(all_records)
