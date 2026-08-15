#!/usr/bin/env python3
"""Run isolated, commit-per-experiment neural improvement campaigns.

The agent command receives META_EXPERIMENT_WORKTREE and META_RESULT_PATH in its environment. It may
edit the temporary worktree and must write its result to META_RESULT_PATH with at least a
``hypothesis`` and, when it finishes, a ``status`` value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
import re
from pathlib import Path
from typing import Any

import yaml
import torch

from shards_ai.ai import load_active_training_profile, load_training_profile
from shards_ai.experimentation import (
    ExperimentManifest,
    ExperimentStatus,
    EXPERIMENT_FAMILIES,
    render_experiment_report,
    evaluate_performance_gate,
    family_guidance,
    validate_campaign_settings,
    validate_changed_paths,
)
from scripts.validate_neural_profile import acceptance_metrics


TERMINAL_STATUSES = {status.value for status in ExperimentStatus}
IGNORED_COMMIT_PREFIXES = ("artifacts/",)


class CampaignError(RuntimeError):
    pass


def _run(repo: Path, *args: str, check: bool = True, timeout: int | None = None) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode:
        raise CampaignError(completed.stderr.strip() or "git command failed")
    return completed.stdout.rstrip("\n")


def _current_branch(repo: Path) -> str:
    branch = _run(repo, "branch", "--show-current")
    if not branch:
        raise CampaignError("campaigns cannot run from a detached HEAD")
    return branch


def _changed_paths(worktree: Path) -> list[str]:
    output = _run(worktree, "status", "--porcelain", "--untracked-files=all")
    paths = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        paths.append(path)
    return paths


def _next_id(repo: Path) -> str:
    experiment_dir = repo / "doc" / "Experiments"
    numbers = []
    if experiment_dir.exists():
        for path in experiment_dir.glob("exp-*.md"):
            try:
                numbers.append(int(path.stem.removeprefix("exp-")))
            except ValueError:
                continue
    for subject in _run(repo, "log", "--all", "--format=%s").splitlines():
        match = re.search(r"exp-(\d+)", subject)
        if match:
            numbers.append(int(match.group(1)))
    return f"exp-{max(numbers, default=0) + 1:05d}"


def _commit_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not path.startswith(IGNORED_COMMIT_PREFIXES)]


def _experiment_history(repo: Path) -> list[dict[str, Any]]:
    history = []
    manifest_experiments = set()
    for path in sorted((repo / "artifacts" / "experiments").glob("*/exp-*/manifest.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            history.append(item)
            manifest_experiments.add(item.get("experiment_id"))
        except (OSError, json.JSONDecodeError):
            continue
    # Reports remain the durable history even when ignored artifacts are absent.
    for path in sorted((repo / "doc" / "Experiments").glob("exp-*.md")):
        experiment_id = path.stem
        if experiment_id in manifest_experiments:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r'"experiment_family":\s*"([^"]+)"', text)
        family = match.group(1) if match else ("ppo" if re.search(r"\bPPO\b", text, re.IGNORECASE) else "other")
        history.append({"experiment_id": experiment_id, "experiment_family": family})
    return history


def _performance_followup_from_history(repo: Path, threshold: float) -> bool:
    """Rebuild the performance-debt flag when a campaign is resumed later."""
    manifests = []
    for path in (repo / "artifacts" / "experiments").glob("*/exp-*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("experiment_id"):
            manifests.append(manifest)

    pending = False
    for manifest in sorted(manifests, key=lambda item: item["experiment_id"]):
        if manifest.get("status") != ExperimentStatus.ACCEPTED.value:
            continue
        if manifest.get("experiment_kind") == "performance":
            pending = False
            continue
        if manifest.get("experiment_kind") != "quality":
            continue
        gate = manifest.get("performance_gate") or {}
        pending = float(gate.get("max_regression", 0.0)) > threshold
    return pending


def _analysis_schedule_from_history(repo: Path, failure_threshold: int) -> tuple[bool, int]:
    """Rebuild the diagnostic-analysis schedule when a campaign is resumed."""
    manifests = []
    for path in (repo / "artifacts" / "experiments").glob("*/exp-*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("experiment_id"):
            manifests.append(manifest)

    failures = 0
    pending = False
    for manifest in sorted(manifests, key=lambda item: item["experiment_id"]):
        kind = manifest.get("experiment_kind")
        status = manifest.get("status")
        if kind == "analysis":
            if status == ExperimentStatus.ACCEPTED.value:
                failures = 0
                pending = False
            continue
        if kind != "quality":
            continue
        if status == ExperimentStatus.ACCEPTED.value:
            failures = 0
            pending = False
        else:
            failures += 1
            pending = pending or failures >= failure_threshold
    return pending, failures


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_dataset(worktree: Path, experiment_dir: Path, result: dict[str, Any]) -> None:
    dataset = result.get("dataset") or result.get("metrics", {}).get("dataset")
    if not dataset:
        return
    source = Path(str(dataset))
    if not source.is_absolute():
        source = worktree / source
    if not source.exists():
        return
    archive_root = experiment_dir / "datasets"
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / source.name
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        result["dataset_archive"] = str(destination.relative_to(experiment_dir))
        return
    shutil.copy2(source, destination)
    result["dataset_archive"] = str(destination.relative_to(experiment_dir))
    result.setdefault("dataset_sha256", _file_digest(source))


def _write_report(worktree: Path, manifest: ExperimentManifest, result: dict[str, Any]) -> Path:
    report_path = worktree / "doc" / "Experiments" / f"{manifest.experiment_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_experiment_report(manifest, result), encoding="utf-8")
    return report_path


class Campaign:
    def __init__(
        self,
        repo: Path,
        agent_command: str,
        budget_seconds: int,
        training_budget_seconds: int,
        screening_budget_seconds: int,
        overhead_budget_seconds: int,
        parent_profile: str | None,
        experiment_kind: str = "quality",
        max_performance_regression: float = 0.05,
        test_command: str | None = "poetry run pytest -q",
        target_architecture: str | None = None,
    ):
        self.repo = repo.resolve()
        self.agent_command = agent_command
        self.budget_seconds = budget_seconds
        self.training_budget_seconds = training_budget_seconds
        self.screening_budget_seconds = screening_budget_seconds
        self.overhead_budget_seconds = overhead_budget_seconds
        if sum((training_budget_seconds, screening_budget_seconds, overhead_budget_seconds)) > budget_seconds:
            raise CampaignError("phase budgets cannot exceed the experiment budget")
        self.experiment_kind = experiment_kind
        self.max_performance_regression = max_performance_regression
        self.test_command = test_command
        self.target_architecture = target_architecture
        settings_path = self.repo / "configs" / "meta_improvement.yaml"
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {
            "seed": 104,
            "opponents": ["hybrid:v006", "hybrid:v004", "hybrid:v005", "v008", "hybrid:v001", "hybrid:v003"],
            "baseline_profile": "v008",
            "acceptance_rule": "weighted_mean_opponent_gain",
        }
        validate_campaign_settings(settings)
        active_path = self.repo / "configs" / "neural_training_profiles" / "active.yaml"
        if parent_profile is None:
            parent_profile = load_active_training_profile(active_path).profile_id
        elif active_path.exists():
            active_profile = load_active_training_profile(
                active_path
            )
            if parent_profile != active_profile.profile_id:
                raise CampaignError(
                    f"parent profile {parent_profile} is not the active neural profile "
                    f"{active_profile.profile_id}"
                )
        self.parent_profile = parent_profile
        self.seed = int(settings["seed"])
        self.performance_maintenance_every = int(settings.get("performance_maintenance_every", 0))
        self.performance_followup_required = _performance_followup_from_history(
            self.repo, self.max_performance_regression
        )
        self.analysis_after_quality_failures = int(settings.get("analysis_after_quality_failures", 4))
        self.analysis_followup_required, self.quality_failure_streak = _analysis_schedule_from_history(
            self.repo, self.analysis_after_quality_failures
        )
        self.branch = _current_branch(self.repo)
        self.parent_commit = _run(self.repo, "rev-parse", "HEAD")
        self.campaign_id = f"campaign-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def preflight(self) -> None:
        if _run(self.repo, "status", "--porcelain", "--untracked-files=all"):
            raise CampaignError("checkout is not clean; commit or stash existing work first")
        if _current_branch(self.repo) != self.branch or _run(self.repo, "rev-parse", "HEAD") != self.parent_commit:
            raise CampaignError("campaign branch changed during preflight")

    def _assert_branch_unchanged(self) -> None:
        if _current_branch(self.repo) != self.branch:
            raise CampaignError(f"branch changed from {self.branch}")
        if _run(self.repo, "rev-parse", "HEAD") != self.parent_commit:
            raise CampaignError("branch advanced outside this campaign")

    def _agent(self, worktree: Path, experiment_dir: Path, prompt: Path) -> dict[str, Any]:
        env = os.environ.copy()
        env.update({
            "META_EXPERIMENT_DIR": str(experiment_dir),
            "META_PROMPT": str(prompt),
            "META_EXPERIMENT_WORKTREE": str(worktree),
            "META_RESULT_PATH": str(worktree / "result.json"),
            "META_PARENT_COMMIT": self.parent_commit,
            "META_EXPERIMENT_BUDGET_SECONDS": str(self.budget_seconds),
            "META_TRAINING_BUDGET_SECONDS": str(self.training_budget_seconds),
            "META_SCREENING_BUDGET_SECONDS": str(self.screening_budget_seconds),
            "META_OVERHEAD_BUDGET_SECONDS": str(self.overhead_budget_seconds),
            "META_EXPERIMENT_KIND": self.experiment_kind,
        })
        started = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            shlex.split(self.agent_command),
            cwd=worktree,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        prompt_text = prompt.read_text(encoding="utf-8")
        try:
            stdout, stderr = process.communicate(input=prompt_text, timeout=self.budget_seconds)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            # Give training scripts a chance to save at an update/epoch boundary.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=min(self.overhead_budget_seconds, 30))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return_code = process.returncode
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            raise
        (experiment_dir / "agent.stdout.log").write_text(stdout, encoding="utf-8")
        (experiment_dir / "agent.stderr.log").write_text(stderr, encoding="utf-8")
        # The agent works in the worktree; keep a copy in the artifact directory
        # after reading it.  The fallback preserves compatibility with agents
        # using the original (ambiguous) contract.
        result_path = worktree / "result.json"
        if not result_path.exists():
            result_path = experiment_dir / "result.json"
        result: dict[str, Any] = {}
        if result_path.exists():
            result_text = result_path.read_text(encoding="utf-8")
            result = json.loads(result_text)
            (experiment_dir / "result.json").write_text(result_text, encoding="utf-8")
        result.setdefault("agent_exit_code", return_code)
        result.setdefault("agent_duration_seconds", round(time.monotonic() - started, 3))
        result["timed_out"] = timed_out or bool(result.get("timed_out"))
        return result

    def _run_fixed_tests(self, worktree: Path, experiment_dir: Path) -> dict[str, Any]:
        if not self.test_command:
            return {"skipped": True}
        started = time.monotonic()
        process = subprocess.Popen(
            shlex.split(self.test_command),
            cwd=worktree,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=self.overhead_budget_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
        (experiment_dir / "fixed-tests.log").write_text(output, encoding="utf-8")
        return {
            "command": self.test_command,
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "duration_seconds": round(time.monotonic() - started, 3),
        }

    def _decide_result(self, result: dict[str, Any], tests: dict[str, Any]) -> tuple[ExperimentStatus, dict[str, Any], str | None]:
        if result.get("timed_out"):
            return ExperimentStatus.INCONCLUSIVE, {}, "agent budget expired before a complete result"
        if tests.get("timed_out"):
            return ExperimentStatus.INCONCLUSIVE, {}, "fixed test budget expired"
        if tests.get("exit_code", 0) != 0:
            return ExperimentStatus.FAILED, {}, "fixed tests failed"
        if self.experiment_kind == "analysis":
            analysis = result.get("analysis") or result.get("observations")
            if not analysis:
                return ExperimentStatus.FAILED, {}, "analysis result has no observations"
            return ExperimentStatus.ACCEPTED, {
                "analysis_complete": True,
                "subject_profile": result.get("analysis_subject_profile", self.parent_profile),
            }, None
        if self.experiment_kind == "performance":
            gate = evaluate_performance_gate(result.get("performance", {}), max_regression=0.02)
            improvement = -float(gate.get("max_regression", 0.0))
            if not gate.get("available") or improvement < 0.02:
                return ExperimentStatus.REJECTED, gate, "performance gain below the robust 2% threshold"
            return ExperimentStatus.ACCEPTED, gate, None
        validation = result.get("validation", {})
        opponent_results = validation.get("results", {})
        if not opponent_results:
            raw_status = str(result.get("status", ExperimentStatus.FAILED.value))
            if raw_status in TERMINAL_STATUSES and raw_status != ExperimentStatus.ACCEPTED.value:
                return ExperimentStatus(raw_status), {}, result.get("error")
            return ExperimentStatus.FAILED, {}, "result has no validation results"
        decision = acceptance_metrics(opponent_results, validation.get("categories"))
        performance = result.get("performance", {})
        performance_gate = evaluate_performance_gate(
            performance,
            max_regression=self.max_performance_regression,
        )
        if not performance_gate["available"]:
            return ExperimentStatus.FAILED, decision, "accepted result has no comparable performance metrics"
        performance_followup_required = (
            float(performance_gate.get("max_regression", 0.0)) > self.max_performance_regression
        )
        decision["performance_gate"] = performance_gate
        decision["performance_followup_required"] = performance_followup_required
        return (
            ExperimentStatus.ACCEPTED if decision["accepted"] else ExperimentStatus.REJECTED,
            decision,
            None if decision["accepted"] else str(decision["reason"]),
        )

    def _promote_quality_candidate(self, worktree: Path, experiment_dir: Path, result: dict[str, Any]) -> None:
        profile = result.get("candidate_profile")
        checkpoint = result.get("candidate_checkpoint")
        if not profile or not checkpoint:
            raise CampaignError("accepted quality experiment must provide candidate_profile and candidate_checkpoint")
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = worktree / profile_path
        candidate_profile = load_training_profile(profile_path)
        candidate_checkpoint_path = Path(checkpoint)
        if not candidate_checkpoint_path.is_absolute():
            candidate_checkpoint_path = worktree / candidate_checkpoint_path
        candidate_checkpoint_document = torch.load(
            candidate_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_architecture = str(
            candidate_checkpoint_document.get("architecture", "independent_action")
        )
        profile_architecture = str(candidate_profile.metadata.get("architecture", "independent_action"))
        if checkpoint_architecture != profile_architecture:
            raise CampaignError(
                "candidate profile architecture does not match checkpoint architecture: "
                f"profile={profile_architecture!r}, checkpoint={checkpoint_architecture!r}"
            )
        active_profile = load_active_training_profile(
            worktree / "configs" / "neural_training_profiles" / "active.yaml"
        )
        active_architecture = str(active_profile.metadata.get("architecture", "independent_action"))
        if candidate_profile.method == "imitation" and profile_architecture != active_architecture:
            declared_initial = result.get("metrics", {}).get("training_initial_checkpoint")
            if declared_initial:
                raise CampaignError(
                    "architecture-transition imitation must initialize from scratch; "
                    "training_initial_checkpoint must be null"
                )
        if candidate_profile.method == "ppo":
            expected_checkpoint = (worktree / "configs" / "neural_profiles" / f"{active_profile.profile_id}.pt").resolve()
            declared_initial = candidate_profile.initial_checkpoint
            if not declared_initial:
                raise CampaignError("PPO candidate must declare the active neural initial checkpoint")
            initial_path = Path(declared_initial)
            if not initial_path.is_absolute():
                initial_path = worktree / initial_path
            if initial_path.resolve() != expected_checkpoint:
                raise CampaignError(
                    "PPO candidate must start from the active neural checkpoint "
                    f"{active_profile.profile_id}.pt"
                )
            declared_by_agent = result.get("metrics", {}).get("training_initial_checkpoint")
            if declared_by_agent:
                agent_initial = Path(str(declared_by_agent))
                if not agent_initial.is_absolute():
                    agent_initial = worktree / agent_initial
                if agent_initial.resolve() != expected_checkpoint:
                    raise CampaignError(
                        "agent training metrics report an initial checkpoint different from the active neural checkpoint"
                    )
        output = experiment_dir / "promotion.json"
        command = [
            "poetry", "run", "python", "scripts/validate_neural_profile_batched.py",
            "--candidate-profile", str(profile),
            "--candidate-checkpoint", str(checkpoint),
            "--games", "200",
            "--batch-games", "20",
            "--seed", "90000",
            "--output", str(output),
            "--progress-output", str(experiment_dir / "promotion.progress.json"),
            "--promote",
        ]
        completed = subprocess.run(
            command,
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=self.screening_budget_seconds,
            check=False,
        )
        (experiment_dir / "promotion.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (experiment_dir / "promotion.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise CampaignError("independent neural promotion validation failed")
        if not output.exists():
            raise CampaignError("promotion validator did not write its report")
        promotion = json.loads(output.read_text(encoding="utf-8"))
        result["promotion"] = promotion
        if promotion.get("decision") != "accepted" or not promotion.get("promotion"):
            raise CampaignError("independent neural promotion gate rejected the candidate")

    def _validate_target_architecture(self, worktree: Path, result: dict[str, Any]) -> None:
        if not self.target_architecture or self.experiment_kind != "quality":
            return
        profile = result.get("candidate_profile")
        if not profile:
            return
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = worktree / profile_path
        candidate_profile = load_training_profile(profile_path)
        architecture = str(candidate_profile.metadata.get("architecture", "independent_action"))
        if architecture != self.target_architecture:
            raise CampaignError(
                "candidate profile architecture does not match campaign target: "
                f"expected={self.target_architecture!r}, got={architecture!r}"
            )

    def run_one(self) -> str:
        self._assert_branch_unchanged()
        experiment_id = _next_id(self.repo)
        temp_root = Path(tempfile.mkdtemp(prefix=f"shards-ai-{experiment_id}-", dir=self.repo.parent))
        worktree = temp_root / "worktree"
        experiment_dir = temp_root / "experiment"
        experiment_dir.mkdir()
        temp_branch = f"meta/{self.campaign_id}/{experiment_id}"
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            campaign_id=self.campaign_id,
            experiment_kind=self.experiment_kind,
            parent_commit=self.parent_commit,
            parent_profile=self.parent_profile,
            hypothesis="Agent did not provide a hypothesis.",
            seed=self.seed,
            budget_seconds=self.budget_seconds,
            training_budget_seconds=self.training_budget_seconds,
            screening_budget_seconds=self.screening_budget_seconds,
            overhead_budget_seconds=self.overhead_budget_seconds,
        )
        try:
            _run(self.repo, "worktree", "add", "-b", temp_branch, str(worktree), self.parent_commit)
            prompt = experiment_dir / "prompt.md"
            diversity = family_guidance(_experiment_history(self.repo))
            if self.experiment_kind == "performance":
                kind_instructions = (
                    "This is a PURE PERFORMANCE experiment. Do not try to improve the neural "
                    "player's playing strength, training objective, dataset or architecture "
                    "unless the change is strictly required to measure or improve execution "
                    "speed. Focus on game simulation, inference, data loading, batching, "
                    "allocation, serialization, multiprocessing or other runtime bottlenecks. "
                    "Run a reproducible before/after benchmark on the same workload, seed, "
                    "checkpoint and number of games. Report elapsed_seconds or throughput for "
                    "both sides and explain why the change is attributable to the code. The "
                    "result must use experiment_family=performance. Do not provide quality "
                    "validation results as evidence of success; the performance gate alone "
                    "decides this experiment."
                )
            elif self.experiment_kind == "analysis":
                kind_instructions = (
                    "This is a PURE DIAGNOSTIC ANALYSIS of the latest active neural player. Do not "
                    "train, modify or promote a checkpoint. Measure where the current neural player "
                    "fails or disagrees with its parent and visible opponents: loss and accuracy by "
                    "phase/action/card, dataset coverage and imbalance, logits or confidence, and "
                    "representative game states. Use only information visible to the neural player. "
                    "You may write temporary analysis code and generate local artifacts, but keep "
                    "the durable result in the Markdown report and Ideas.md. The result must use "
                    "experiment_family=analysis and include analysis_subject_profile, analysis, "
                    "observations, limitations and recommendations. No candidate profile, checkpoint "
                    "or quality validation is required."
                )
            else:
                kind_instructions = "This is a QUALITY experiment. "
                if self.target_architecture:
                    kind_instructions += (
                        f"Target architecture is {self.target_architecture!r}; the candidate profile "
                        "metadata and checkpoint must use it. "
                    )
                kind_instructions += "Optimize playing strength and preserve the required comparable runtime measurements."
            prompt.write_text(
                f"Run one {self.experiment_kind} experiment. {kind_instructions} First read doc/Ideas.md and all "
                "relevant reports under doc/Experiments. Use that history to understand the "
                "latest attempts and avoid repeating an experiment that already failed without "
                "a concrete correction. Then choose freely among resuming a promising incomplete "
                "experiment, correcting an earlier failure, or inventing a new experiment that "
                "could improve the neural player. The catalogue and past reports inform the choice "
                "but do not constrain innovation. If the ideas catalogue is empty, formulate a "
                "new falsifiable hypothesis or resume the strongest promising lead from the reports. "
                "Record the selected idea plus any new future ideas. Classify the experiment with "
                f"one family from {', '.join(EXPERIMENT_FAMILIES)} and explain its novelty. "
                f"The current family history is {json.dumps(diversity, sort_keys=True)}. "
                f"This is guidance only: {diversity['recommendation']}. "
                "If a dataset is created, include dataset, dataset_sha256, dataset_records and "
                "teacher_profile in the result; include the training_recipe for reproducibility. "
                "Do not modify the game engine, "
                "heuristic players, or the information mask. Write the final JSON result to "
                "$META_RESULT_PATH (not to META_EXPERIMENT_DIR). It must contain hypothesis, "
                "status, metrics and analysis. For quality experiments, include performance.baseline, "
                "performance.candidate and validation.results. For performance experiments, include "
                "performance.baseline and performance.candidate. "
                "For a quality experiment, validation.results must contain delta_win_rate for "
                "hybrid v006/v004/v005, v008, and hybrid v001/v003, measured against the active latest neural reference. "
                "The v008 entry is a weighted heuristic signal, not a hard guard or the neural reference. "
                "The final gate excludes Random, retired neural v001-v009, heuristic v007 and hybrid v002, and weights hybrid v006/v004/v005/v008 at 1 and hybrid v001/v003 at 0.75; "
                "the active parent is loaded from active.yaml "
                f"and is currently {self.parent_profile}. "
                f"For a candidate that keeps the active architecture, training must start from the latest active "
                f"neural checkpoint, currently {self.parent_profile}; resetting Adam optimizer state is allowed. "
                "For an architecture-transition candidate such as structured_semantic_v4, do not load the V2 "
                "state dict: initialize the new model from scratch and train it by imitation. Keep V2 as the "
                "validation reference, report the architecture transition and a null training_initial_checkpoint, "
                "and use a candidate profile whose metadata architecture matches the checkpoint. Initializing "
                "from v001 or another older neural version is not allowed. "
                "For quality and performance experiments, performance.baseline and performance.candidate "
                "must contain comparable elapsed_seconds or throughput values. A quality result with "
                "status accepted must also provide candidate_profile and candidate_checkpoint. "
                "For validation beyond 20 games per opponent, use "
                "scripts/validate_neural_profile_batched.py with --batch-games 20 and a "
                "--progress-output file so the run can resume after interruption. "
                "The orchestrator performs the final deterministic gate, so do not report accepted "
                "from an alternate short protocol. After evaluation, update doc/Ideas.md with done "
                "statuses, removals and next steps, then write the final JSON result.\n",
                encoding="utf-8",
            )
            try:
                result = self._agent(worktree, experiment_dir, prompt)
                self._validate_target_architecture(worktree, result)
                manifest.hypothesis = str(result.get("hypothesis", manifest.hypothesis))
                family = str(result.get("experiment_family", "other"))
                manifest.experiment_family = family if family in EXPERIMENT_FAMILIES else "other"
                manifest.novelty = result.get("novelty")
                metrics = result.get("metrics", {})
                manifest.dataset = result.get("dataset") or metrics.get("dataset")
                manifest.dataset_sha256 = result.get("dataset_sha256") or metrics.get("dataset_sha256")
                manifest.dataset_records = result.get("dataset_records") or metrics.get("dataset_records")
                manifest.teacher_profile = result.get("teacher_profile") or metrics.get("teacher_profile")
                manifest.training_recipe = result.get("training_recipe") or metrics.get("training_recipe", {})
                manifest.screening = result.get("screening", {})
                manifest.validation = result.get("validation", {})
                manifest.performance = result.get("performance", {})
                _archive_dataset(worktree, experiment_dir, result)
                manifest.dataset_sha256 = result.get("dataset_sha256") or manifest.dataset_sha256
                tests = self._run_fixed_tests(worktree, experiment_dir)
                manifest.commands = [self.agent_command, tests.get("command", "")]
                result["fixed_tests"] = tests
                manifest.status, decision, manifest.error = self._decide_result(result, tests)
                manifest.tests = tests
                manifest.decision_metrics = decision
                agent_changed = _changed_paths(worktree)
                manifest.allowed_changes = agent_changed
                try:
                    validate_changed_paths(agent_changed)
                except ValueError as exc:
                    manifest.status = ExperimentStatus.FAILED
                    manifest.error = str(exc)
                if manifest.status is ExperimentStatus.ACCEPTED and self.experiment_kind == "quality":
                    self._promote_quality_candidate(worktree, experiment_dir, result)
                if self.experiment_kind == "analysis":
                    manifest.performance_gate = {}
                elif self.experiment_kind == "performance":
                    manifest.performance_gate = decision
                else:
                    manifest.performance_gate = evaluate_performance_gate(
                        manifest.performance,
                        max_regression=self.max_performance_regression,
                    )
                if manifest.status is ExperimentStatus.ACCEPTED:
                    if self.experiment_kind == "analysis":
                        self.analysis_followup_required = False
                        self.quality_failure_streak = 0
                    elif self.experiment_kind == "performance":
                        self.performance_followup_required = False
                    elif self.experiment_kind == "quality":
                        self.quality_failure_streak = 0
                        self.analysis_followup_required = False
                        self.performance_followup_required = (
                            self.performance_followup_required
                            or float(manifest.performance_gate.get("max_regression", 0.0))
                            > self.max_performance_regression
                        )
                elif self.experiment_kind == "quality":
                    self.quality_failure_streak += 1
                    self.analysis_followup_required = (
                        self.analysis_followup_required
                        or self.quality_failure_streak >= self.analysis_after_quality_failures
                    )
                manifest.performance_followup_required = self.performance_followup_required
                result["decision_metrics"] = decision
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                manifest.status = ExperimentStatus.INTERRUPTED
                manifest.error = f"agent interrupted: {type(exc).__name__}"
                result = {"analysis": "L'expérience n'a pas atteint son résultat final."}
            except Exception as exc:  # Report the failure before moving to the next experiment.
                manifest.status = ExperimentStatus.FAILED
                manifest.error = f"{type(exc).__name__}: {exc}"
                result = {"analysis": "Échec technique avant validation."}

            changed = _changed_paths(worktree)
            internal_promotion_paths = {
                path for path in changed
                if path.startswith("configs/neural_profiles/")
                or (
                    path.startswith("configs/neural_training_profiles/")
                    and not path.startswith("configs/neural_training_profiles/candidates/")
                )
            }
            try:
                validate_changed_paths([path for path in changed if path not in internal_promotion_paths])
            except ValueError as exc:
                manifest.status = ExperimentStatus.FAILED
                manifest.error = str(exc)
            manifest.allowed_changes = changed
            if "doc/Ideas.md" in changed:
                result["ideas_diff"] = _run(worktree, "diff", "--", "doc/Ideas.md")
            ideas_path = worktree / "doc" / "Ideas.md"
            ideas_content = ideas_path.read_text(encoding="utf-8") if "doc/Ideas.md" in changed and ideas_path.exists() else None

            if manifest.status is not ExperimentStatus.ACCEPTED or self.experiment_kind == "analysis":
                # Disposable candidate code is removed; only its durable report is committed.
                _run(worktree, "restore", "--worktree", "--staged", ".")
                for path in _changed_paths(worktree):
                    target = worktree / path
                    if target.is_dir():
                        shutil.rmtree(target)
                    elif target.exists():
                        target.unlink()
                if ideas_content is not None:
                    ideas_path.parent.mkdir(parents=True, exist_ok=True)
                    ideas_path.write_text(ideas_content, encoding="utf-8")
                report_path = _write_report(worktree, manifest, result)
                paths_to_add = [str(report_path.relative_to(worktree))]
                if ideas_content is not None:
                    paths_to_add.append("doc/Ideas.md")
                _run(worktree, "add", "--", *paths_to_add)
            else:
                report_path = _write_report(worktree, manifest, result)
                paths = _commit_paths(_changed_paths(worktree))
                if str(report_path.relative_to(worktree)) not in paths:
                    paths.append(str(report_path.relative_to(worktree)))
                if not paths:
                    raise CampaignError("accepted experiment produced no commitable changes")
                _run(worktree, "add", "--", *paths)

            commit_message = f"experiment: {manifest.status.value} {experiment_id}"
            _run(worktree, "commit", "-m", commit_message)
            manifest.commit = _run(worktree, "rev-parse", "HEAD")
            commit_sha = manifest.commit
            _run(self.repo, "cherry-pick", commit_sha)
            self.parent_commit = _run(self.repo, "rev-parse", "HEAD")
            active_path = self.repo / "configs" / "neural_training_profiles" / "active.yaml"
            if active_path.exists():
                self.parent_profile = load_active_training_profile(active_path).profile_id
            manifest.commit = self.parent_commit
            artifact_dir = self.repo / "artifacts" / "experiments" / self.campaign_id / experiment_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            manifest.write_json(artifact_dir / "manifest.json")
            for artifact in experiment_dir.iterdir():
                destination = artifact_dir / artifact.name
                if artifact.is_dir():
                    shutil.copytree(artifact, destination, dirs_exist_ok=True)
                elif artifact.is_file():
                    shutil.copy2(artifact, destination)
            print(f"{experiment_id}: {manifest.status.value} ({self.parent_commit[:12]})")
            return commit_sha
        finally:
            _run(self.repo, "worktree", "remove", "--force", str(worktree), check=False)
            _run(self.repo, "branch", "-D", temp_branch, check=False)
            shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--experiments", type=int, default=1)
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--training-budget-seconds", type=int, default=2400)
    parser.add_argument("--screening-budget-seconds", type=int, default=750)
    parser.add_argument("--overhead-budget-seconds", type=int, default=450)
    parser.add_argument(
        "--parent-profile",
        default=None,
        help="Active neural parent profile override; defaults to configs/neural_training_profiles/active.yaml.",
    )
    parser.add_argument("--experiment-kind", choices=("quality", "performance", "analysis"), default="quality")
    parser.add_argument(
        "--target-architecture",
        default=None,
        help="Require quality candidates to use this neural architecture, e.g. structured_semantic_v4.",
    )
    parser.add_argument("--max-performance-regression", type=float, default=0.05)
    parser.add_argument("--test-command", default="poetry run pytest -q")
    parser.add_argument("--performance-maintenance-every", type=int, default=None)
    parser.add_argument("--performance-agent-command", default=None)
    parser.add_argument("--analysis-agent-command", default=None)
    parser.add_argument(
        "--agent-command",
        default="codex exec --sandbox workspace-write --ephemeral -",
        help="Agent command; its stdin receives the experiment prompt.",
    )
    args = parser.parse_args()
    if args.experiments <= 0 or min(
        args.budget_seconds,
        args.training_budget_seconds,
        args.screening_budget_seconds,
        args.overhead_budget_seconds,
    ) <= 0:
        parser.error("experiment and phase budgets must be positive")

    try:
        campaign = Campaign(
            args.repo,
            args.agent_command,
            args.budget_seconds,
            args.training_budget_seconds,
            args.screening_budget_seconds,
            args.overhead_budget_seconds,
            args.parent_profile,
            args.experiment_kind,
            args.max_performance_regression,
            args.test_command,
            args.target_architecture,
        )
        campaign.preflight()
        maintenance_every = (
            args.performance_maintenance_every
            if args.performance_maintenance_every is not None
            else campaign.performance_maintenance_every
        )
        if maintenance_every < 0:
            parser.error("--performance-maintenance-every cannot be negative")
        performance_command = args.performance_agent_command or args.agent_command
        analysis_command = args.analysis_agent_command or args.agent_command

        def run_performance_experiment() -> None:
            quality_command = campaign.agent_command
            quality_kind = campaign.experiment_kind
            campaign.agent_command = performance_command
            campaign.experiment_kind = "performance"
            try:
                campaign.run_one()
            finally:
                campaign.agent_command = quality_command
                campaign.experiment_kind = quality_kind

        def run_analysis_experiment() -> None:
            quality_command = campaign.agent_command
            quality_kind = campaign.experiment_kind
            campaign.agent_command = analysis_command
            campaign.experiment_kind = "analysis"
            try:
                campaign.run_one()
            finally:
                campaign.agent_command = quality_command
                campaign.experiment_kind = quality_kind

        for index in range(args.experiments):
            if campaign.analysis_followup_required:
                run_analysis_experiment()
                continue
            if campaign.performance_followup_required:
                run_performance_experiment()
                continue
            campaign.run_one()
            if campaign.analysis_followup_required:
                run_analysis_experiment()
                continue
            if campaign.performance_followup_required:
                continue
            if (
                campaign.experiment_kind == "quality"
                and maintenance_every
                and (index + 1) % maintenance_every == 0
            ):
                run_performance_experiment()
    except CampaignError as exc:
        parser.exit(2, f"meta-improve: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
