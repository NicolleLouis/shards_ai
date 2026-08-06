.PHONY: heuristic-benchmark-mix neural-rl-train neural-rl-train-resume \
	neural-benchmark-mix neural-hybrid-benchmark neural-dagger-rebuild-baseline neural-dagger-collect neural-dagger-sample neural-dagger-train neural-dagger2-collect neural-dagger2-sample neural-dagger-merge neural-validate neural-validate-batched neural-imitation-analysis neural-visited-state-analysis meta-improve

HEURISTIC_VERSION := v008
HEURISTIC_PUBLISHED_PROFILE := configs/heuristic_profiles/$(HEURISTIC_VERSION).yaml

heuristic-benchmark-mix:
	PYTHONPATH=. poetry run python scripts/benchmark_heuristic_report.py \
		--games $(HEURISTIC_BENCHMARK_GAMES) \
		--seed $(HEURISTIC_BENCHMARK_SEED) \
		--profile $(HEURISTIC_PUBLISHED_PROFILE) \
		--opponent-profile $(HEURISTIC_BENCHMARK_OPPONENT_PROFILE) \
		--output-dir $(HEURISTIC_BENCHMARK_OUTPUT)

# Neural validation and benchmark settings.
# There is exactly one mutable training checkpoint; promoted checkpoints under
# configs/neural_profiles/ are stable and must never be trained in place.
# This is the only mutable checkpoint path. Both train and resume write/read it.
NEURAL_CHECKPOINT ?= artifacts/neural_training/checkpoint.pt
NEURAL_TORCH_THREADS ?= 1
NEURAL_MIX_GAMES ?= 2000
NEURAL_MIX_SEED ?= 104
NEURAL_MIX_OUTPUT ?= artifacts/neural_benchmark/neural_mix.json
NEURAL_MIX_HTML_OUTPUT ?= artifacts/neural_benchmark/neural_mix.html
NEURAL_HYBRID_GAMES ?= 2000
NEURAL_HYBRID_SEED ?= 104
NEURAL_HYBRID_OUTPUT ?= artifacts/neural_benchmark/neural_hybrids.json
NEURAL_HYBRID_HTML_OUTPUT ?= artifacts/neural_benchmark/neural_hybrids.html
NEURAL_IMITATION_DATASET ?= artifacts/imitation_dataset/v008_vs_random_v007_1m.jsonl
NEURAL_IMITATION_ANALYSIS_OUTPUT ?= artifacts/analysis/neural_imitation_v008_1m.html
NEURAL_IMITATION_ANALYSIS_JSON ?= artifacts/analysis/neural_imitation_v008_1m.json
NEURAL_IMITATION_SPLIT ?= non_train
NEURAL_VISITED_GAMES ?= 200
NEURAL_VISITED_SEED ?= 104
NEURAL_VISITED_OUTPUT ?= artifacts/analysis/visited_neural_vs_v008.json
NEURAL_VISITED_HTML_OUTPUT ?= artifacts/analysis/visited_neural_vs_v008.html
NEURAL_DAGGER_GAMES_PER_OPPONENT ?= 1000
NEURAL_DAGGER_SEED ?= 104
NEURAL_DAGGER_RAW_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_1_raw.jsonl
NEURAL_DAGGER_OPPONENTS ?= v008=configs/heuristic_profiles/v008.yaml v007=configs/heuristic_profiles/v007.yaml neural_v001=configs/neural_profiles/v001.pt self=$(NEURAL_CHECKPOINT)
NEURAL_DAGGER_OLD_DATASET ?= artifacts/imitation_dataset/v008_vs_random_v007_1m.jsonl
NEURAL_DAGGER_TRAIN_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_1_train.jsonl
NEURAL_DAGGER_VALIDATION_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_1_validation.jsonl
NEURAL_DAGGER_TARGET_RECORDS ?= 1000000
NEURAL_DAGGER_EPOCHS ?= 1
NEURAL_DAGGER_LEARNING_RATE ?= 0.001
NEURAL_DAGGER_BASELINE_PROFILE ?= configs/neural_training_profiles/candidates/v004.yaml
NEURAL_DAGGER_BASELINE_EPOCHS ?= 3
NEURAL_DAGGER_BASELINE_LEARNING_RATE ?= 0.001
NEURAL_DAGGER2_GAMES_PER_OPPONENT ?= 1000
NEURAL_DAGGER2_SEED ?= 1104
NEURAL_DAGGER2_RAW_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_2_raw.jsonl
NEURAL_DAGGER2_TRAIN_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_2_train.jsonl
NEURAL_DAGGER2_VALIDATION_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_2_validation.jsonl
NEURAL_DAGGER2_TARGET_RECORDS ?= 1000000
NEURAL_DAGGER2_ACTION_WEIGHTS ?= play_card=1.5 recruit_mercenary=3.0 assign_power=3.0 choose_pending_decision=2.0
NEURAL_DAGGER1_RAW_INPUT ?= artifacts/imitation_dataset/dagger_cycle_1_raw.jsonl
NEURAL_DAGGER_HISTORICAL_VALIDATION_INPUT ?= artifacts/imitation_dataset/v008_vs_random_v007_normalized_100k.validation.jsonl
NEURAL_DAGGER_MERGED_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_1_2_merged_train.jsonl
NEURAL_DAGGER_MERGED_VALIDATION_OUTPUT ?= artifacts/imitation_dataset/dagger_cycle_1_2_merged_validation.jsonl

# PPO V002 training uses the single mutable training checkpoint.
NEURAL_RL_PROFILE ?= configs/neural_training_profiles/candidates/v002.yaml
NEURAL_RL_TOTAL_GAMES ?=
NEURAL_RL_GAMES_PER_UPDATE ?=
NEURAL_RL_OPTIMIZATION_EPOCHS ?=
NEURAL_RL_MINIBATCH_SIZE ?=
NEURAL_RL_WORKERS ?= 1

NEURAL_RL_OVERRIDES = \
	$(if $(NEURAL_RL_TOTAL_GAMES),--total-games $(NEURAL_RL_TOTAL_GAMES),) \
	$(if $(NEURAL_RL_GAMES_PER_UPDATE),--games-per-update $(NEURAL_RL_GAMES_PER_UPDATE),) \
	$(if $(NEURAL_RL_OPTIMIZATION_EPOCHS),--optimization-epochs $(NEURAL_RL_OPTIMIZATION_EPOCHS),) \
	$(if $(NEURAL_RL_MINIBATCH_SIZE),--minibatch-size $(NEURAL_RL_MINIBATCH_SIZE),)

neural-rl-train:
	PYTHONPATH=. poetry run python scripts/train_neural_rl.py \
		--profile $(NEURAL_RL_PROFILE) \
		--output $(NEURAL_CHECKPOINT) \
		$(NEURAL_RL_OVERRIDES) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--workers $(NEURAL_RL_WORKERS)

neural-rl-train-resume:
	PYTHONPATH=. poetry run python scripts/train_neural_rl.py \
		--profile $(NEURAL_RL_PROFILE) \
		--output $(NEURAL_CHECKPOINT) \
		--resume-from $(NEURAL_CHECKPOINT) \
		$(NEURAL_RL_OVERRIDES) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--workers $(NEURAL_RL_WORKERS)

neural-benchmark-mix:
	PYTHONPATH=. poetry run python benchmarks/benchmark_neural_mix.py \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--profile-v007 configs/heuristic_profiles/v007.yaml \
		--profile-v008 configs/heuristic_profiles/v008.yaml \
		--games $(NEURAL_MIX_GAMES) \
		--seed $(NEURAL_MIX_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_MIX_OUTPUT) \
		--html-output $(NEURAL_MIX_HTML_OUTPUT)

neural-hybrid-benchmark:
	PYTHONPATH=. poetry run python benchmarks/benchmark_neural_hybrids.py \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--profile configs/heuristic_profiles/v008.yaml \
		--games $(NEURAL_HYBRID_GAMES) \
		--seed $(NEURAL_HYBRID_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_HYBRID_OUTPUT) \
		--html-output $(NEURAL_HYBRID_HTML_OUTPUT)

neural-imitation-analysis:
	PYTHONPATH=. poetry run python scripts/analyze_neural_imitation.py \
		--dataset $(NEURAL_IMITATION_DATASET) \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--split $(NEURAL_IMITATION_SPLIT) \
		--output $(NEURAL_IMITATION_ANALYSIS_OUTPUT) \
		--json-output $(NEURAL_IMITATION_ANALYSIS_JSON) \
		--torch-threads $(NEURAL_TORCH_THREADS)

neural-visited-state-analysis:
	PYTHONPATH=. poetry run python scripts/benchmark_neural_visited_states.py \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--profile configs/heuristic_profiles/v008.yaml \
		--games $(NEURAL_VISITED_GAMES) \
		--seed $(NEURAL_VISITED_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_VISITED_OUTPUT) \
		--html-output $(NEURAL_VISITED_HTML_OUTPUT)

neural-dagger-collect:
	PYTHONPATH=. poetry run python scripts/collect_dagger_dataset.py \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--profile configs/heuristic_profiles/v008.yaml \
		$(foreach opponent,$(NEURAL_DAGGER_OPPONENTS),--opponent $(opponent)) \
		--games-per-opponent $(NEURAL_DAGGER_GAMES_PER_OPPONENT) \
		--seed $(NEURAL_DAGGER_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_DAGGER_RAW_OUTPUT)

neural-dagger2-collect:
	PYTHONPATH=. poetry run python scripts/collect_dagger_dataset.py \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--profile configs/heuristic_profiles/v008.yaml \
		--dagger-stage dagger_2 \
		$(foreach opponent,$(NEURAL_DAGGER_OPPONENTS),--opponent $(opponent)) \
		--games-per-opponent $(NEURAL_DAGGER2_GAMES_PER_OPPONENT) \
		--seed $(NEURAL_DAGGER2_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_DAGGER2_RAW_OUTPUT)

neural-dagger-sample:
	PYTHONPATH=. poetry run python scripts/sample_dagger_dataset.py \
		--old-dataset $(NEURAL_DAGGER_OLD_DATASET) \
		--dagger-dataset $(NEURAL_DAGGER_RAW_OUTPUT) \
		--output $(NEURAL_DAGGER_TRAIN_OUTPUT) \
		--validation-output $(NEURAL_DAGGER_VALIDATION_OUTPUT) \
		--target-records $(NEURAL_DAGGER_TARGET_RECORDS) \
		--seed $(NEURAL_DAGGER_SEED)

neural-dagger2-sample:
	PYTHONPATH=. poetry run python scripts/sample_dagger_dataset.py \
		--dagger-dataset $(NEURAL_DAGGER2_RAW_OUTPUT) \
		--on-policy-only \
		--output $(NEURAL_DAGGER2_TRAIN_OUTPUT) \
		--validation-output $(NEURAL_DAGGER2_VALIDATION_OUTPUT) \
		--target-records $(NEURAL_DAGGER2_TARGET_RECORDS) \
		--seed $(NEURAL_DAGGER2_SEED) \
		$(foreach weight,$(NEURAL_DAGGER2_ACTION_WEIGHTS),--action-weight $(weight))

neural-dagger-merge:
	PYTHONPATH=. poetry run python scripts/merge_dagger_datasets.py \
		--source historical=$(NEURAL_DAGGER_OLD_DATASET) \
		--source dagger_1=$(NEURAL_DAGGER1_RAW_INPUT) \
		--source dagger_2=$(NEURAL_DAGGER2_TRAIN_OUTPUT) \
		--validation-source historical=$(NEURAL_DAGGER_HISTORICAL_VALIDATION_INPUT) \
		--validation-source dagger_2=$(NEURAL_DAGGER2_VALIDATION_OUTPUT) \
		--output $(NEURAL_DAGGER_MERGED_OUTPUT) \
		--validation-output $(NEURAL_DAGGER_MERGED_VALIDATION_OUTPUT)

neural-dagger-rebuild-baseline:
	PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
		--profile $(NEURAL_DAGGER_BASELINE_PROFILE) \
		--dataset $(NEURAL_DAGGER_OLD_DATASET) \
		--output $(NEURAL_CHECKPOINT) \
		--epochs $(NEURAL_DAGGER_BASELINE_EPOCHS) \
		--learning-rate $(NEURAL_DAGGER_BASELINE_LEARNING_RATE) \
		--split train \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--metrics-output artifacts/neural_training/pre_dagger_baseline.metrics.json

neural-dagger-train:
	PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
		--dataset $(NEURAL_DAGGER_TRAIN_OUTPUT) \
		--validation-dataset $(NEURAL_DAGGER_VALIDATION_OUTPUT) \
		--output $(NEURAL_CHECKPOINT) \
		--resume-from $(NEURAL_CHECKPOINT) \
		--epochs $(NEURAL_DAGGER_EPOCHS) \
		--learning-rate $(NEURAL_DAGGER_LEARNING_RATE) \
		--split all \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--reset-optimizer

NEURAL_CANDIDATE_PROFILE ?=
NEURAL_VALIDATION_GAMES ?= 100
NEURAL_VALIDATION_BATCH_GAMES ?= 20
NEURAL_VALIDATION_SEED ?= 90000
NEURAL_VALIDATION_OUTPUT ?= artifacts/neural_validation/latest.json
NEURAL_VALIDATION_PROGRESS_OUTPUT ?= artifacts/neural_validation/latest.progress.json

META_EXPERIMENTS ?= 1
META_BUDGET_SECONDS ?= 3600
META_TRAINING_BUDGET_SECONDS ?= 2400
META_SCREENING_BUDGET_SECONDS ?= 750
META_OVERHEAD_BUDGET_SECONDS ?= 450
META_EXPERIMENT_KIND ?= quality
META_TARGET_ARCHITECTURE ?=
META_AGENT_COMMAND ?= codex exec --sandbox workspace-write --ephemeral -

neural-validate:
	@test -n "$(NEURAL_CANDIDATE_PROFILE)" || (echo "Set NEURAL_CANDIDATE_PROFILE=..." && exit 1)
	PYTHONPATH=. poetry run python scripts/validate_neural_profile.py \
		--candidate-profile $(NEURAL_CANDIDATE_PROFILE) \
		--candidate-checkpoint $(NEURAL_CHECKPOINT) \
		--games $(NEURAL_VALIDATION_GAMES) \
		--seed $(NEURAL_VALIDATION_SEED) \
		--output $(NEURAL_VALIDATION_OUTPUT)

neural-validate-batched:
	@test -n "$(NEURAL_CANDIDATE_PROFILE)" || (echo "Set NEURAL_CANDIDATE_PROFILE=..." && exit 1)
	PYTHONPATH=. poetry run python scripts/validate_neural_profile_batched.py \
		--candidate-profile $(NEURAL_CANDIDATE_PROFILE) \
		--candidate-checkpoint $(NEURAL_CHECKPOINT) \
		--games $(NEURAL_VALIDATION_GAMES) \
		--batch-games $(NEURAL_VALIDATION_BATCH_GAMES) \
		--seed $(NEURAL_VALIDATION_SEED) \
		--output $(NEURAL_VALIDATION_OUTPUT) \
		--progress-output $(NEURAL_VALIDATION_PROGRESS_OUTPUT)

meta-improve:
	PYTHONPATH=. poetry run python scripts/meta_improve.py \
		--experiments $(META_EXPERIMENTS) \
		--budget-seconds $(META_BUDGET_SECONDS) \
		--training-budget-seconds $(META_TRAINING_BUDGET_SECONDS) \
		--screening-budget-seconds $(META_SCREENING_BUDGET_SECONDS) \
		--overhead-budget-seconds $(META_OVERHEAD_BUDGET_SECONDS) \
		--experiment-kind $(META_EXPERIMENT_KIND) \
		$(if $(META_TARGET_ARCHITECTURE),--target-architecture $(META_TARGET_ARCHITECTURE),) \
		--agent-command "$(META_AGENT_COMMAND)"
