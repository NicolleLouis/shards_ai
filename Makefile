.PHONY: train-resume train-remaining train-v008-resume train-v008-remaining \
	heuristic-benchmark-mix \
	neural-train neural-train-resume neural-train-report neural-benchmark-random \
	neural-benchmark-mix neural-validate

HEURISTIC_VERSION := v008
HEURISTIC_CHECKPOINT := artifacts/heuristic_optimization/$(HEURISTIC_VERSION)/checkpoint.json
HEURISTIC_PUBLISHED_PROFILE := configs/heuristic_profiles/$(HEURISTIC_VERSION).yaml

train-resume:
	systemd-inhibit --what=idle:sleep:handle-lid-switch \
		--why="Shards AI heuristic training" --mode=block \
		 env PYTHONPATH=. nice -n 10 poetry run python scripts/optimize_heuristic.py \
		--resume $(HEURISTIC_CHECKPOINT) \
		--publish-profile $(HEURISTIC_PUBLISHED_PROFILE)

train-remaining:
	PYTHONPATH=. poetry run python scripts/optimization_progress.py \
		--checkpoint $(HEURISTIC_CHECKPOINT)

HEURISTIC_BENCHMARK_GAMES ?= 1000
HEURISTIC_BENCHMARK_SEED ?= 87000
HEURISTIC_BENCHMARK_OPPONENT_PROFILE ?= configs/heuristic_profiles/v007.yaml
HEURISTIC_BENCHMARK_OUTPUT ?= analysis_output/heuristic_$(HEURISTIC_VERSION)_mix_1000

heuristic-benchmark-mix:
	PYTHONPATH=. poetry run python scripts/benchmark_heuristic_report.py \
		--games $(HEURISTIC_BENCHMARK_GAMES) \
		--seed $(HEURISTIC_BENCHMARK_SEED) \
		--profile $(HEURISTIC_PUBLISHED_PROFILE) \
		--opponent-profile $(HEURISTIC_BENCHMARK_OPPONENT_PROFILE) \
		--output-dir $(HEURISTIC_BENCHMARK_OUTPUT)

# Neural training profiles. The YAML profile is the versioned source of truth;
# checkpoints and metrics remain run artifacts.
NEURAL_VERSION := v001
NEURAL_PROFILE := configs/neural_training_profiles/$(NEURAL_VERSION).yaml
NEURAL_DATASET ?= artifacts/imitation_dataset/v008_vs_random_v007_1m.jsonl
# Training writes a candidate artifact by default. Promoted checkpoints under
# configs/neural_profiles/ are written only by validate_neural_profile.py.
NEURAL_MODEL ?= artifacts/neural_imitation/$(NEURAL_VERSION)-candidate.pt
# Source checkpoint for neural-train-resume. Keep it independent from the
# candidate output so an existing baseline is never overwritten accidentally.
NEURAL_RESUME_FROM ?= $(NEURAL_MODEL)
NEURAL_EPOCHS ?= 5
NEURAL_SEED ?= 51x²
NEURAL_TORCH_THREADS ?= 1
# Above the current dataset size: evaluates the complete validation split,
# while split_for_game_id still excludes train and test decisions.
NEURAL_MAX_VALIDATION_RECORDS ?= 2000000
NEURAL_METRICS := $(NEURAL_MODEL:.pt=.metrics.json)
NEURAL_REPORT := $(NEURAL_MODEL:.pt=.html)
NEURAL_BENCHMARK_GAMES ?= 1000
NEURAL_BENCHMARK_SEED ?= 100
NEURAL_BENCHMARK_OUTPUT ?= artifacts/neural_benchmark/neural_vs_random.json
NEURAL_MIX_GAMES ?= 1000
NEURAL_MIX_SEED ?= 101
NEURAL_MIX_OUTPUT ?= artifacts/neural_benchmark/neural_mix.json
NEURAL_MIX_HTML_OUTPUT ?= artifacts/neural_benchmark/neural_mix.html

neural-train:
	PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
		--profile $(NEURAL_PROFILE) \
		--dataset $(NEURAL_DATASET) \
		--output $(NEURAL_MODEL) \
		--epochs $(NEURAL_EPOCHS) \
		--seed $(NEURAL_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--max-validation-records $(NEURAL_MAX_VALIDATION_RECORDS)

neural-train-resume:
	PYTHONPATH=. poetry run python scripts/train_neural_imitation.py \
		--profile $(NEURAL_PROFILE) \
		--dataset $(NEURAL_DATASET) \
		--output $(NEURAL_MODEL) \
		--resume-from $(NEURAL_RESUME_FROM) \
		--epochs $(NEURAL_EPOCHS) \
		--seed $(NEURAL_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--max-validation-records $(NEURAL_MAX_VALIDATION_RECORDS)

neural-train-report:
	PYTHONPATH=. poetry run python scripts/generate_neural_training_report.py \
		--metrics $(NEURAL_METRICS) \
		--output $(NEURAL_REPORT)

neural-benchmark-mix:
	PYTHONPATH=. poetry run python benchmarks/benchmark_neural_mix.py \
		--checkpoint $(NEURAL_MODEL) \
		--profile-v007 configs/heuristic_profiles/v007.yaml \
		--profile-v008 configs/heuristic_profiles/v008.yaml \
		--games $(NEURAL_MIX_GAMES) \
		--seed $(NEURAL_MIX_SEED) \
		--torch-threads $(NEURAL_TORCH_THREADS) \
		--output $(NEURAL_MIX_OUTPUT) \
		--html-output $(NEURAL_MIX_HTML_OUTPUT)

NEURAL_CANDIDATE_PROFILE ?=
NEURAL_CANDIDATE_CHECKPOINT ?=
NEURAL_VALIDATION_GAMES ?= 1000
NEURAL_VALIDATION_SEED ?= 90000
NEURAL_VALIDATION_OUTPUT ?= artifacts/neural_validation/latest.json

neural-validate:
	@test -n "$(NEURAL_CANDIDATE_PROFILE)" || (echo "Set NEURAL_CANDIDATE_PROFILE=..." && exit 1)
	PYTHONPATH=. poetry run python scripts/validate_neural_profile.py \
		--candidate-profile $(NEURAL_CANDIDATE_PROFILE) \
		$(if $(NEURAL_CANDIDATE_CHECKPOINT),--candidate-checkpoint $(NEURAL_CANDIDATE_CHECKPOINT),) \
		--games $(NEURAL_VALIDATION_GAMES) \
		--seed $(NEURAL_VALIDATION_SEED) \
		--output $(NEURAL_VALIDATION_OUTPUT)
