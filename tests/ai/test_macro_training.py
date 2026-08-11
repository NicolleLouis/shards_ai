import json
from dataclasses import asdict, replace

import pytest
import torch

from shards_ai.ai import (
    MacroActionScorerV4,
    MacroActionScorerV2,
    NeuralModelConfig,
    PlayTurnSolver,
    evaluate_macro_epoch,
    macro_imitation_loss,
    train_macro_epoch,
    unified_records,
)
from shards_ai.ai.macro_training import macro_record_diagnostics, macro_records
from shards_ai.ai.play_turn_solver import macro_representations_v2
from shards_ai.game import Game, Phase
from shards_ai.ai.macro_player import _atomic_candidate


def test_macro_training_and_evaluation_consume_macro_records_only() -> None:
    game = Game.new(seed=1511)
    resolution = PlayTurnSolver().resolve(game)
    observation = resolution.observation_game.neural_observation_for(game.active_player)
    record = {
        "dataset_schema_version": 2,
        "decision_kind": "macro_play",
        "game_id": "game-1511",
        "observation": asdict(observation),
        "candidates": [asdict(candidate) for candidate in macro_representations_v2(observation, resolution.candidates)],
        "chosen_candidate_index": 0,
    }
    model = MacroActionScorerV2(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train = train_macro_epoch(model, [record], optimizer)
    evaluation = evaluate_macro_epoch(model, [record])

    assert train.records == 1
    assert train.non_trivial_records == 1
    assert train.mean_loss > 0
    assert evaluation.records == 1
    assert 0 <= evaluation.top1_accuracy <= 1
    assert macro_imitation_loss(torch.tensor([1.0, 0.0]), 0).item() > 0


def test_macro_metrics_exclude_singleton_records(tmp_path) -> None:
    singleton = {
        "decision_kind": "macro_play",
        "dataset_schema_version": 2,
        "candidates": [{"schema_version": 1}],
        "chosen_candidate_index": 0,
    }
    diagnostics = macro_record_diagnostics([singleton])

    assert diagnostics["all_records"] == 1
    assert diagnostics["non_trivial_records"] == 0
    path = tmp_path / "macro.jsonl"
    path.write_text(json.dumps(singleton) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate schema V2"):
        list(macro_records(str(path)))


def test_macro_reader_rejects_v1_dataset(tmp_path) -> None:
    path = tmp_path / "macro-v1.jsonl"
    path.write_text(
        json.dumps({
            "dataset_schema_version": 1,
            "decision_kind": "macro_play",
            "candidates": [],
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset_schema_version=2 or 3"):
        list(macro_records(str(path)))


def test_unified_training_consumes_atomic_v4_records(tmp_path) -> None:
    game = Game.new(seed=1518)
    game.state.phase = Phase.BUY
    game.active.gems = 10
    observation = game.neural_observation_for(game.active_player)
    actions = game.legal_actions()
    candidates = [asdict(_atomic_candidate(observation, action, 4).representation) for action in actions]
    record = {
        "dataset_schema_version": 3,
        "decision_kind": "atomic",
        "game_id": "game-1518",
        "phase": observation.phase,
        "opponent_id": "v007",
        "observation": asdict(observation),
        "candidates": candidates,
        "chosen_candidate_index": 0,
    }
    model = MacroActionScorerV4(
        replace(
            NeuralModelConfig(),
            observation_feature_set="deck_state_v1",
            card_embedding_dim=16,
            semantic_hidden_dim=24,
            state_hidden_dim=32,
            action_hidden_dim=16,
            scorer_hidden_dim=24,
            semantic_token_hidden_dim=32,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    path = tmp_path / "unified.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    unified = list(unified_records(str(path)))
    assert len(unified) == 1
    assert unified[0]["decision_kind"] == "atomic"
    train = train_macro_epoch(model, [record], optimizer, record_weight=lambda value: 1.0)
    evaluation = evaluate_macro_epoch(model, [record], record_weight=lambda value: 1.0)

    assert train.records == 1
    assert train.by_decision_kind == {"atomic": 1}
    assert evaluation.records == 1
    assert evaluation.by_phase == {"buy": 1}
