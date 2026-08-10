#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-artifacts/neural_training/league_deck_state_comparison_${RUN_ID}}"
MUTABLE_CHECKPOINT="artifacts/neural_training/checkpoint.pt"
FLAT_PROFILE="configs/neural_training_profiles/candidates/exp00107-league-flat-deck-state.yaml"
WEIGHTED_PROFILE="configs/neural_training_profiles/candidates/exp00108-league-weighted-deck-state.yaml"
GAMES="${GAMES:-50}"
SEED="${SEED:-81000}"

mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/run.log") 2>&1

echo "run_dir=$RUN_DIR"
echo "games_per_benchmark=$GAMES seed=$SEED"

if [[ -f "$RUN_DIR/control-flat.pt" ]]; then
  echo "=== 1/3 train control flat: already completed, reusing checkpoint ==="
else
  echo "=== 1/3 train control flat ==="
  poetry run python scripts/train_neural_imitation.py \
    --profile "$FLAT_PROFILE" \
    --no-chart
  cp "$MUTABLE_CHECKPOINT" "$RUN_DIR/control-flat.pt"
  cp artifacts/neural_training/checkpoint.metrics.json "$RUN_DIR/control-flat.metrics.json"
  cp artifacts/neural_training/checkpoint.metrics.csv "$RUN_DIR/control-flat.csv"
  cp artifacts/neural_training/checkpoint.metrics.html "$RUN_DIR/control-flat.html"
fi

if [[ -f "$RUN_DIR/weighted-moderate.pt" ]]; then
  echo "=== 2/3 train weighted moderate: already completed, reusing checkpoint ==="
else
  echo "=== 2/3 train weighted moderate ==="
  poetry run python scripts/train_neural_imitation.py \
    --profile "$WEIGHTED_PROFILE" \
    --no-chart
  cp "$MUTABLE_CHECKPOINT" "$RUN_DIR/weighted-moderate.pt"
  cp artifacts/neural_training/checkpoint.metrics.json "$RUN_DIR/weighted-moderate.metrics.json"
  cp artifacts/neural_training/checkpoint.metrics.csv "$RUN_DIR/weighted-moderate.csv"
  cp artifacts/neural_training/checkpoint.metrics.html "$RUN_DIR/weighted-moderate.html"
fi

benchmark() {
  local label="$1"
  local checkpoint="$2"
  local opponent="$3"
  local opponent_label="$4"
  local output="$RUN_DIR/${label}-vs-${opponent_label}.json"
  echo "benchmark=$label opponent=$opponent_label"
  if [[ "$opponent" == "heuristic" ]]; then
    poetry run python benchmarks/benchmark_neural_players.py \
      --checkpoint "$checkpoint" \
      --opponent heuristic \
      --profile "configs/heuristic_profiles/${opponent_label}.yaml" \
      --games "$GAMES" \
      --seed "$SEED" \
      --torch-threads 1 \
      --output "$output"
  else
    poetry run python benchmarks/benchmark_neural_players.py \
      --checkpoint "$checkpoint" \
      --opponent "$opponent" \
      --games "$GAMES" \
      --seed "$SEED" \
      --torch-threads 1 \
      --output "$output"
  fi
}

echo "=== 3/3 benchmark real games ==="
for checkpoint_label in control-flat weighted-moderate; do
  checkpoint="$RUN_DIR/${checkpoint_label}.pt"
  benchmark "$checkpoint_label" "$checkpoint" random random
  benchmark "$checkpoint_label" "$checkpoint" heuristic v007
  benchmark "$checkpoint_label" "$checkpoint" heuristic v008
done

echo "completed run_dir=$RUN_DIR"
