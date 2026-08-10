.PHONY: heuristic-benchmark-mix \
	neural-benchmark-mix neural-benchmark-panel neural-hybrid-benchmark neural-validate neural-validate-batched \
	neural-imitation-analysis neural-visited-state-analysis meta-improve

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
NEURAL_PANEL_CHECKPOINT ?= configs/neural_profiles/v004.pt
NEURAL_PANEL_GAMES ?= 200
NEURAL_PANEL_SEED ?= 104
NEURAL_PANEL_OUTPUT ?= artifacts/neural_benchmark/neural_panel.json
NEURAL_PANEL_HTML_OUTPUT ?= artifacts/neural_benchmark/neural_panel.html
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
