from pathlib import Path
import json

from shards_ai.ai import (
    MacroDatasetCampaignConfig,
    PlayTurnSolver,
    heuristic_macro_selector,
)
from shards_ai.ai.heuristic_player import HeuristicPlayer
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game import Game


def test_macro_dataset_config_requires_a_collection_limit() -> None:
    profile = Path("configs/heuristic_profiles/v008.yaml")
    try:
        MacroDatasetCampaignConfig(
            teacher_profile_path=profile,
            heuristic_opponent_profile_path=Path("configs/heuristic_profiles/v007.yaml"),
            neural_opponent_checkpoint_path=Path("configs/neural_profiles/v004.pt"),
            output_path=Path("/tmp/macro.jsonl"),
            seed=1,
        )
    except ValueError as error:
        assert "Set games or target_decisions" in str(error)
    else:
        raise AssertionError("missing collection limit should fail")


def test_heuristic_macro_selector_uses_teacher_choice_at_branch_root() -> None:
    game = Game.new(seed=1410)
    resolution = PlayTurnSolver().resolve(game)
    teacher = HeuristicPlayer(game.active_player, **_profile_kwargs("v008"))
    selector = heuristic_macro_selector(teacher)

    index = selector(
        resolution.observation_game,
        resolution.observation_game.neural_observation_for(game.active_player),
        resolution.candidates,
    )

    first_actions = [candidate.atomic_trace[0] for candidate in resolution.candidates]
    expected = teacher.choose_action(
        resolution.observation_game.observation_for(game.active_player),
        first_actions,
    )
    assert first_actions[index] == expected


def test_macro_dataset_generation_writes_unified_v4_candidates(tmp_path: Path) -> None:
    from shards_ai.ai import MacroDatasetCampaignConfig, generate_macro_dataset
    from shards_ai.ai.macro_training import unified_record_diagnostics, unified_records

    result = generate_macro_dataset(
        MacroDatasetCampaignConfig(
            teacher_profile_path=Path("configs/heuristic_profiles/v008.yaml"),
            heuristic_opponent_profile_path=Path("configs/heuristic_profiles/v007.yaml"),
            neural_opponent_checkpoint_path=Path("configs/neural_profiles/v004.pt"),
            output_path=tmp_path / "macro-v2.jsonl",
            seed=1411,
            games=1,
        )
    )
    records = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    macro_records = [record for record in records if record["decision_kind"] == "macro_play"]
    manifest = json.loads(result.manifest_path.read_text())

    assert records
    assert manifest["dataset_schema_version"] == 3
    assert manifest["candidate_schema_version"] == 4
    assert manifest["canonicalization_schema_version"] == 1
    assert manifest["candidate_feature_set"] == "root_action_plus_known_consequence_plus_tactical_v1"
    assert macro_records
    assert all(
        candidate["schema_version"] == 4 and candidate["root_action"] is not None
        for record in macro_records
        for candidate in record["candidates"]
    )
    atomic_records = [record for record in records if record["decision_kind"] == "atomic"]
    assert atomic_records
    assert all(
        candidate["schema_version"] == 4
        and candidate["decision_kind"] == "atomic"
        and candidate["atomic_action_count"] == 1
        for record in atomic_records
        for candidate in record["candidates"]
    )
    assert set(manifest["decision_counts"]["decision_kind"]) == {"atomic", "macro_play"}
    unified = list(unified_records(str(result.output_path)))
    diagnostics = unified_record_diagnostics(unified)
    assert diagnostics["all_records"] == len(unified)
    assert diagnostics["by_decision_kind"] == {"atomic": len(atomic_records), "macro_play": len(macro_records)}
    assert all(
        "delta_gems" in candidate
        and "known_card_definition_ids" in candidate
        and "immediate_victory" in candidate
        for record in macro_records
        for candidate in record["candidates"]
    )


def _profile_kwargs(profile_id):
    profile = load_profile(Path(f"configs/heuristic_profiles/{profile_id}.yaml"))
    return {
        "weights": profile.weights,
        "acquisition_weights": profile.card_acquisition_weights,
        "constraint_weights": profile.constraint_weights,
    }
