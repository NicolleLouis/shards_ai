.PHONY: heuristic-benchmark-mix neural-macro-dataset neural-macro-train \
	neural-benchmark-mix neural-benchmark-panel neural-validate neural-validate-batched \
	neural-imitation-analysis neural-visited-state-analysis neural-rl-train \
	neural-validate-hybrid-deckbuilding meta-improve

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
NEURAL_PANEL_CHECKPOINT ?= configs/neural_profiles/v006.pt
NEURAL_PANEL_GAMES ?= 200
NEURAL_PANEL_SEED ?= 104
NEURAL_PANEL_OUTPUT ?= artifacts/neural_benchmark/neural_panel.json
NEURAL_PANEL_HTML_OUTPUT ?= artifacts/neural_benchmark/neural_panel.html
NEURAL_IMITATION_DATASET ?= artifacts/imitation_dataset/v008_vs_random_v007_1m.jsonl
NEURAL_MACRO_DATASET ?= artifacts/imitation_dataset/unified_v4_v8_vs_v7_v4.jsonl
NEURAL_MACRO_DATASET_GAMES ?= 100
NEURAL_MACRO_DATASET_SEED ?= 1411
NEURAL_MACRO_TRAINING_PROFILE ?= configs/neural_training_profiles/candidates/exp00112-macro-v4-tactical-action.yaml
NEURAL_IMITATION_ANALYSIS_OUTPUT ?= artifacts/analysis/neural_imitation_v008_1m.html
NEURAL_IMITATION_ANALYSIS_JSON ?= artifacts/analysis/neural_imitation_v008_1m.json
NEURAL_IMITATION_SPLIT ?= non_train
NEURAL_VISITED_GAMES ?= 200
NEURAL_VISITED_SEED ?= 104
NEURAL_VISITED_OUTPUT ?= artifacts/analysis/visited_neural_vs_v008.json
NEURAL_VISITED_HTML_OUTPUT ?= artifacts/analysis/visited_neural_vs_v008.html
NEURAL_RL_PROFILE ?= configs/neural_training_profiles/candidates/ppo-deckbuilding-hybrid-v003.yaml
NEURAL_RL_TOTAL_GAMES ?= 100000
NEURAL_RL_GAMES_PER_UPDATE ?= 128
NEURAL_RL_OPTIMIZATION_EPOCHS ?= 4
NEURAL_RL_MINIBATCH_SIZE ?= 2048
NEURAL_HYBRID_VALIDATION_GAMES ?= 200
NEURAL_HYBRID_VALIDATION_BATCH_GAMES ?= 20
NEURAL_HYBRID_VALIDATION_SEED ?= 90000
NEURAL_HYBRID_VALIDATION_OUTPUT ?= artifacts/neural_validation/hybrid_deckbuilding.json
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

neural-benchmark-panel:
	PYTHONPATH=. poetry run python benchmarks/benchmark_neural_panel.py \
		--checkpoint $(NEURAL_PANEL_CHECKPOINT) \
		--games $(NEURAL_PANEL_GAMES) \
		--seed $(NEURAL_PANEL_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_PANEL_OUTPUT) \
		--html-output $(NEURAL_PANEL_HTML_OUTPUT)

neural-imitation-analysis:
	PYTHONPATH=. poetry run python scripts/analyze_neural_imitation.py \
		--dataset $(NEURAL_IMITATION_DATASET) \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--split $(NEURAL_IMITATION_SPLIT) \
		--output $(NEURAL_IMITATION_ANALYSIS_OUTPUT) \
		--json-output $(NEURAL_IMITATION_ANALYSIS_JSON) \
		--torch-threads $(NEURAL_TORCH_THREADS)

neural-macro-dataset:
	PYTHONPATH=. poetry run python scripts/generate_macro_imitation_dataset.py \
		--output $(NEURAL_MACRO_DATASET) \
		--games $(NEURAL_MACRO_DATASET_GAMES) \
		--seed $(NEURAL_MACRO_DATASET_SEED)

neural-macro-train:
	PYTHONPATH=. poetry run python scripts/train_macro_imitation.py \
		--profile $(NEURAL_MACRO_TRAINING_PROFILE) \
		--output $(NEURAL_CHECKPOINT)

neural-visited-state-analysis:
	PYTHONPATH=. poetry run python scripts/benchmark_neural_visited_states.py \
		--checkpoint $(NEURAL_CHECKPOINT) \
		--profile configs/heuristic_profiles/v008.yaml \
		--games $(NEURAL_VISITED_GAMES) \
		--seed $(NEURAL_VISITED_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_VISITED_OUTPUT) \
		--html-output $(NEURAL_VISITED_HTML_OUTPUT)

neural-rl-train:
	PYTHONPATH=. poetry run python scripts/train_neural_rl.py \
		--profile $(NEURAL_RL_PROFILE) \
		--output $(NEURAL_CHECKPOINT) \
		--total-games $(NEURAL_RL_TOTAL_GAMES) \
		--games-per-update $(NEURAL_RL_GAMES_PER_UPDATE) \
		--optimization-epochs $(NEURAL_RL_OPTIMIZATION_EPOCHS) \
		--minibatch-size $(NEURAL_RL_MINIBATCH_SIZE) \
		--torch-threads $(NEURAL_TORCH_THREADS)

neural-validate-hybrid-deckbuilding:
	PYTHONPATH=. poetry run python scripts/validate_hybrid_deckbuilding_profile.py \
		--candidate-profile $(NEURAL_RL_PROFILE) \
		--candidate-checkpoint $(NEURAL_CHECKPOINT) \
		--games $(NEURAL_HYBRID_VALIDATION_GAMES) \
		--batch-games $(NEURAL_HYBRID_VALIDATION_BATCH_GAMES) \
		--seed $(NEURAL_HYBRID_VALIDATION_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_HYBRID_VALIDATION_OUTPUT)

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
