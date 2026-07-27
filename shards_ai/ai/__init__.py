"""Player implementations and decision policies."""

from .heuristic_evaluator import (
    ActionFeatures,
    CardAcquisitionWeights,
    CardConstraintWeights,
    HeuristicWeights,
)
from .heuristic_player import HeuristicPlayer
from .random_player import RandomPlayer
from .state_evaluator import StateRewardWeights
from .action_representation import (
    ACTION_REPRESENTATION_SCHEMA_VERSION,
    ActionRepresentation,
    representation_for_action,
    representation_for_neural_action,
)
from .imitation_dataset import (
    DATASET_SCHEMA_VERSION,
    DatasetCampaignConfig,
    DatasetGenerationResult,
    MatchupSpec,
    default_matchups,
    generate_dataset,
)
from .card_representation import (
    CARD_REPRESENTATION_SCHEMA_VERSION,
    CardSemanticRepresentation,
    ChampionAbilityRepresentation,
    EffectRepresentation,
    EffectStepRepresentation,
    OperationRepresentation,
    clear_representation_cache,
    representation_for_definition,
)
from .neural_model import NeuralActionScorer, NeuralModelConfig
from .neural_player import NeuralPlayer
from .neural_training_profiles import (
    NeuralProfile,
    NeuralTrainingProfile,
    load_active_neural_profile,
    load_active_training_profile,
    load_training_profile,
    next_training_profile_id,
    save_training_profile,
    versioned_training_profiles,
)
from .neural_training import (
    TrainingMetrics,
    chosen_action_loss,
    combined_imitation_loss,
    iter_jsonl_records,
    EvaluationMetrics,
    evaluate_epoch,
    normalized_score_regression_loss,
    observation_from_dict,
    pairwise_ranking_loss,
    seed_training,
    split_for_game_id,
    train_epoch,
    train_jsonl,
)
from .neural_reporting import load_metrics, write_training_report

__all__ = [
    "ActionFeatures",
    "ACTION_REPRESENTATION_SCHEMA_VERSION",
    "ActionRepresentation",
    "DATASET_SCHEMA_VERSION",
    "DatasetCampaignConfig",
    "DatasetGenerationResult",
    "CARD_REPRESENTATION_SCHEMA_VERSION",
    "CardAcquisitionWeights",
    "CardConstraintWeights",
    "CardSemanticRepresentation",
    "ChampionAbilityRepresentation",
    "EffectRepresentation",
    "EffectStepRepresentation",
    "HeuristicPlayer",
    "HeuristicWeights",
    "RandomPlayer",
    "StateRewardWeights",
    "MatchupSpec",
    "default_matchups",
    "generate_dataset",
    "OperationRepresentation",
    "clear_representation_cache",
    "representation_for_definition",
    "representation_for_action",
    "representation_for_neural_action",
    "NeuralActionScorer",
    "NeuralModelConfig",
    "NeuralTrainingProfile",
    "NeuralProfile",
    "NeuralPlayer",
    "load_active_training_profile",
    "load_active_neural_profile",
    "load_training_profile",
    "next_training_profile_id",
    "save_training_profile",
    "versioned_training_profiles",
    "TrainingMetrics",
    "EvaluationMetrics",
    "chosen_action_loss",
    "combined_imitation_loss",
    "iter_jsonl_records",
    "evaluate_epoch",
    "normalized_score_regression_loss",
    "observation_from_dict",
    "pairwise_ranking_loss",
    "seed_training",
    "split_for_game_id",
    "train_epoch",
    "train_jsonl",
    "load_metrics",
    "write_training_report",
]
