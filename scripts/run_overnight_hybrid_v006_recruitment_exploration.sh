#!/usr/bin/env bash
set -euo pipefail

# Long experiment: start from stable Hybrid V6 / neural V9 and increase only
# PPO entropy regularization. The mutable checkpoint remains canonical.
PROFILE="configs/neural_training_profiles/candidates/ppo-deckbuilding-hybrid-v006-recruitment-exploration.yaml"
CHECKPOINT="artifacts/neural_training/checkpoint.pt"
REFERENCE_CHECKPOINT="configs/neural_profiles/v009.pt"
REFERENCE_PROFILE="configs/neural_training_profiles/v009.yaml"
COMPOSITION="configs/hybrid_profiles/hybrid-v006.yaml"
TOTAL_GAMES="${TOTAL_GAMES:-12160}"
SEED="${SEED:-92000}"

mkdir -p artifacts/neural_validation

PYTHONPATH=. poetry run python - <<'PY'
from scripts.validate_neural_profile import QUALITY_OPPONENT_WEIGHTS
from shards_ai.ai.neural_training_profiles import load_training_profile

profile = load_training_profile("configs/neural_training_profiles/candidates/ppo-deckbuilding-hybrid-v006-recruitment-exploration.yaml")
if dict(QUALITY_OPPONENT_WEIGHTS) != dict(profile.opponents):
    raise SystemExit(
        "Training opponent pool/weights differ from QUALITY_OPPONENT_WEIGHTS; "
        "synchronize the candidate profile before starting."
    )
print("gate/training opponent pool synchronized")
PY

PYTHONPATH=. poetry run python scripts/train_neural_rl.py \
  --profile "$PROFILE" \
  --resume-from "$REFERENCE_CHECKPOINT" \
  --output "$CHECKPOINT" \
  --total-games "$TOTAL_GAMES" \
  --games-per-update 128 \
  --optimization-epochs 4 \
  --minibatch-size 2048 \
  --learning-rate 0.001 \
  --entropy-coefficient 0.003 \
  --evaluation-games 20 \
  --evaluation-interval-games 512 \
  --torch-threads 1

PYTHONPATH=. poetry run python scripts/validate_hybrid_deckbuilding_profile.py \
  --candidate-profile "$PROFILE" \
  --candidate-checkpoint "$CHECKPOINT" \
  --candidate-best \
  --reference-profile "$REFERENCE_PROFILE" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --reference-composition-profile "$COMPOSITION" \
  --games 200 \
  --batch-games 20 \
  --seed "$SEED" \
  --torch-threads 1 \
  --output artifacts/neural_validation/hybrid_v006_recruitment_exploration_best.json

PYTHONPATH=. poetry run python scripts/validate_hybrid_deckbuilding_profile.py \
  --candidate-profile "$PROFILE" \
  --candidate-checkpoint "$CHECKPOINT" \
  --reference-profile "$REFERENCE_PROFILE" \
  --reference-checkpoint "$REFERENCE_CHECKPOINT" \
  --reference-composition-profile "$COMPOSITION" \
  --games 200 \
  --batch-games 20 \
  --seed "$SEED" \
  --torch-threads 1 \
  --output artifacts/neural_validation/hybrid_v006_recruitment_exploration_latest.json
