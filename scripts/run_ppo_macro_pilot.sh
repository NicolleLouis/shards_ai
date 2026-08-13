#!/usr/bin/env bash
set -euo pipefail

PROFILE="configs/neural_training_profiles/candidates/ppo-v4-macro-play-turn-solver.yaml"
CHECKPOINT="artifacts/neural_training/checkpoint.pt"
TRAINING_GAMES="${TRAINING_GAMES:-768}"
EVALUATION_GAMES="${EVALUATION_GAMES:-100}"
EVALUATION_INTERVAL_GAMES="${EVALUATION_INTERVAL_GAMES:-700}"
PILOT_DIR="artifacts/neural_training/ppo_macro_pilot"

mkdir -p "$PILOT_DIR"

echo "[1/4] Baseline V006"
PROFILE="$PROFILE" CHECKPOINT="configs/neural_profiles/v006.pt" \
  EVALUATION_GAMES="$EVALUATION_GAMES" OUTPUT="$PILOT_DIR/baseline-v006.json" \
  PYTHONPATH=. poetry run python - <<'PY'
import json
import os
from dataclasses import replace
import torch

from shards_ai.ai import NeuralActorCritic, evaluate_greedy_model, load_training_profile

profile = load_training_profile(os.environ["PROFILE"])
profile = replace(
    profile,
    evaluation_games=int(os.environ["EVALUATION_GAMES"]),
    evaluation_seed=92000,
)
checkpoint = torch.load(os.environ["CHECKPOINT"], map_location="cpu", weights_only=False)
model = NeuralActorCritic.from_checkpoint(checkpoint)
result = evaluate_greedy_model(model, profile)
payload = {
    "checkpoint": os.environ["CHECKPOINT"],
    "architecture": model.architecture,
    "profile_id": checkpoint.get("profile_id"),
    "evaluation_seed": profile.evaluation_seed,
    "evaluation_games": profile.evaluation_games,
    **result,
}
with open(os.environ["OUTPUT"], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "[2/4] PPO training: ${TRAINING_GAMES} games"
PYTHONPATH=. poetry run python scripts/train_neural_rl.py \
  --profile "$PROFILE" \
  --total-games "$TRAINING_GAMES" \
  --games-per-update 128 \
  --evaluation-games "$EVALUATION_GAMES" \
  --evaluation-interval-games "$EVALUATION_INTERVAL_GAMES" \
  --workers 1 \
  --torch-threads 1 \
  --output "$CHECKPOINT" \
  | tee "$PILOT_DIR/training.log"

cp "$CHECKPOINT" "$PILOT_DIR/checkpoint.pt"
cp "${CHECKPOINT%.*}.metrics.json" "$PILOT_DIR/training.metrics.json"

echo "[3/4] Post-training evaluation"
PROFILE="$PROFILE" CHECKPOINT="$CHECKPOINT" \
  EVALUATION_GAMES="$EVALUATION_GAMES" OUTPUT="$PILOT_DIR/post-training.json" \
  PYTHONPATH=. poetry run python - <<'PY'
import json
import os
from dataclasses import replace
import torch

from shards_ai.ai import NeuralActorCritic, evaluate_greedy_model, load_training_profile

profile = load_training_profile(os.environ["PROFILE"])
profile = replace(
    profile,
    evaluation_games=int(os.environ["EVALUATION_GAMES"]),
    evaluation_seed=92000,
)
checkpoint = torch.load(os.environ["CHECKPOINT"], map_location="cpu", weights_only=False)
model = NeuralActorCritic.from_checkpoint(checkpoint)
result = evaluate_greedy_model(model, profile)
payload = {
    "checkpoint": os.environ["CHECKPOINT"],
    "architecture": model.architecture,
    "profile_id": checkpoint.get("profile_id"),
    "games_seen": checkpoint.get("games_seen"),
    "transitions_seen": checkpoint.get("transitions_seen"),
    "evaluation_seed": profile.evaluation_seed,
    "evaluation_games": profile.evaluation_games,
    **result,
}
with open(os.environ["OUTPUT"], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "[4/4] Analysis"
BASELINE="$PILOT_DIR/baseline-v006.json" POST="$PILOT_DIR/post-training.json" \
  TRAINING_METRICS="$PILOT_DIR/training.metrics.json" \
  PYTHONPATH=. poetry run python - <<'PY'
import json
import os

with open(os.environ["BASELINE"], encoding="utf-8") as stream:
    baseline = json.load(stream)
with open(os.environ["POST"], encoding="utf-8") as stream:
    post = json.load(stream)
with open(os.environ["TRAINING_METRICS"], encoding="utf-8") as stream:
    metrics = json.load(stream)

print("\n=== PPO macro pilot summary ===")
print(f"updates={len(metrics)} games_seen={post.get('games_seen')} transitions_seen={post.get('transitions_seen')}")
print("opponent,baseline,post,delta")
for opponent, before in baseline["by_opponent"].items():
    old = float(before["win_rate"])
    new = float(post["by_opponent"][opponent]["win_rate"])
    print(f"{opponent},{old:.3f},{new:.3f},{new-old:+.3f}")
print(f"weighted_baseline={baseline['score']:.4f}")
print(f"weighted_post={post['score']:.4f}")
print(f"weighted_delta={post['score'] - baseline['score']:+.4f}")
for item in metrics:
    ppo = item["ppo"]
    print(
        f"update={item['update']} policy_loss={ppo['policy_loss']:.5f} "
        f"value_loss={ppo['value_loss']:.5f} entropy={ppo['entropy']:.5f} "
        f"win_rate={item['win_rate']:.3f}"
    )
PY

echo "Artifacts: $PILOT_DIR"
