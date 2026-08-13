#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="${HORIZON_OUTPUT_DIR:-artifacts/horizon_forecast}"
DATASET="${OUTPUT_DIR}/horizon_v1.jsonl"
BASELINE_DATASET="${OUTPUT_DIR}/horizon_v1_baseline.jsonl"
MODEL_DIR="${OUTPUT_DIR}/models"
GAMES_PER_MATCHUP="${HORIZON_GAMES_PER_MATCHUP:-100}"
SEED="${HORIZON_SEED:-51200}"
MAX_TURNS="${HORIZON_MAX_TURNS:-200}"
EPOCHS="${HORIZON_EPOCHS:-100}"
LEARNING_RATE="${HORIZON_LEARNING_RATE:-1e-2}"

mkdir -p "${OUTPUT_DIR}"

echo "[1/3] Generating horizon datasets"
PYTHONPATH=. poetry run python scripts/generate_horizon_dataset.py \
  --heuristic-profile configs/heuristic_profiles/v007.yaml \
  --heuristic-profile configs/heuristic_profiles/v008.yaml \
  --neural-checkpoint configs/neural_profiles/v006.pt \
  --output "${DATASET}" \
  --games-per-matchup "${GAMES_PER_MATCHUP}" \
  --seed "${SEED}" \
  --max-turns "${MAX_TURNS}"

test -s "${DATASET}"
test -s "${BASELINE_DATASET}"

echo "[2/3] Training baseline and enriched classifiers"
PYTHONPATH=. poetry run python scripts/train_horizon_forecast.py \
  --dataset "${DATASET}" \
  --baseline-dataset "${BASELINE_DATASET}" \
  --output-dir "${MODEL_DIR}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --split-seed "$((SEED + 1))"

test -s "${MODEL_DIR}/baseline.pt"
test -s "${MODEL_DIR}/v1.pt"
test -s "${MODEL_DIR}/report.json"

echo "[3/3] Verification report"
PYTHONPATH=. poetry run python -c '
import json
from pathlib import Path

path = Path("'"${MODEL_DIR}"'/report.json")
report = json.loads(path.read_text(encoding="utf-8"))
baseline = report["models"]["turn_number_v1"]["metrics"]["test"]
enriched = report["models"]["active_state_faction_counts_v1"]["metrics"]["test"]
print(json.dumps({
    "test_records": enriched["records"],
    "baseline_test_accuracy": baseline["accuracy"],
    "enriched_test_accuracy": enriched["accuracy"],
    "accuracy_delta_v1_minus_baseline": report["comparison"]["test_accuracy_delta_v1_minus_baseline"],
    "baseline_short_t0_t2_recall": baseline["short_t0_t2_recall"],
    "enriched_short_t0_t2_recall": enriched["short_t0_t2_recall"],
    "baseline_late_t6_plus_recall": baseline["late_t6_plus_recall"],
    "enriched_late_t6_plus_recall": enriched["late_t6_plus_recall"],
}, indent=2, sort_keys=True))
if enriched["records"] <= 0:
    raise SystemExit("verification failed: empty test split")
'

echo "Completed. Full report: ${MODEL_DIR}/report.json"
