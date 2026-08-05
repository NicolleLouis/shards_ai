#!/usr/bin/env python3
"""Run isolated, commit-per-experiment neural improvement campaigns.

The agent command receives META_EXPERIMENT_WORKTREE and META_RESULT_PATH in its environment. It may
edit the temporary worktree and must write its result to META_RESULT_PATH with at least a
``hypothesis`` and, when it finishes, a ``status`` value.
"""

from __future__ import annotations

import argparse
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

from shards_ai.ai import load_active_training_profile, load_training_profile
from shards_ai.experimentation import (
    ExperimentManifest,
    ExperimentStatus,
    render_experiment_report,
    evaluate_performance_gate,
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
        settings_path = self.repo / "configs" / "meta_improvement.yaml"
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {
            "seed": 104,
            "opponents": ["random", "v007", "v008"],
            "baseline_profile": "v008",
            "acceptance_rule": "weighted_v008_guard",
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
        if not performance_gate["accepted"]:
            return ExperimentStatus.REJECTED, decision, "performance regression exceeds the protected threshold"
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
        if candidate_profile.method == "ppo":
            active_profile = load_active_training_profile(
                worktree / "configs" / "neural_training_profiles" / "active.yaml"
            )
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
            "poetry", "run", "python", "scripts/validate_neural_profile.py",
            "--candidate-profile", str(profile),
            "--candidate-checkpoint", str(checkpoint),
            "--games", "200",
            "--seed", "90000",
            "--output", str(output),
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
            prompt.write_text(
                f"Run one {self.experiment_kind} experiment. First read doc/Ideas.md and all "
                "relevant reports under doc/Experiments. Use that history to understand the "
                "latest attempts and avoid repeating an experiment that already failed without "
                "a concrete correction. Then choose freely among resuming a promising incomplete "
                "experiment, correcting an earlier failure, or inventing a new experiment that "
                "could improve the neural player. The catalogue and past reports inform the choice "
                "but do not constrain innovation. If the ideas catalogue is empty, formulate a "
                "new falsifiable hypothesis or resume the strongest promising lead from the reports. "
                "Record the selected idea plus any new future ideas. "
                "Do not modify the game engine, "
                "heuristic players, or the information mask. Write the final JSON result to "
                "$META_RESULT_PATH (not to META_EXPERIMENT_DIR). It must contain hypothesis, "
                "status, performance.baseline, performance.candidate, metrics and analysis. "
                "For a quality experiment, validation.results must contain delta_win_rate for "
                "random, v007 and v008, measured against the active latest neural reference. "
                "The v008 entry is the protected heuristic guard, not the neural reference. "
                "Training must start from the latest active neural checkpoint, currently v002; "
                "resetting Adam optimizer state is allowed, but initializing weights from v001 "
                "or an older neural version is not allowed. "
                "performance.baseline and performance.candidate must "
                "contain comparable elapsed_seconds or throughput values. A quality result with "
                "status accepted must also provide candidate_profile and candidate_checkpoint. "
                "The orchestrator performs the final deterministic gate, so do not report accepted "
                "from an alternate short protocol. After evaluation, update doc/Ideas.md with done "
                "statuses, removals and next steps, then write the final JSON result.\n",
                encoding="utf-8",
            )
            try:
                result = self._agent(worktree, experiment_dir, prompt)
                manifest.hypothesis = str(result.get("hypothesis", manifest.hypothesis))
                manifest.screening = result.get("screening", {})
                manifest.validation = result.get("validation", {})
                manifest.performance = result.get("performance", {})
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
                manifest.performance_gate = decision if self.experiment_kind == "performance" else evaluate_performance_gate(
                    manifest.performance,
                    max_regression=self.max_performance_regression,
                )
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

            if manifest.status is not ExperimentStatus.ACCEPTED:
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
            manifest.commit = self.parent_commit
            artifact_dir = self.repo / "artifacts" / "experiments" / self.campaign_id / experiment_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            manifest.write_json(artifact_dir / "manifest.json")
            for artifact in experiment_dir.iterdir():
                if artifact.is_file():
                    shutil.copy2(artifact, artifact_dir / artifact.name)
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
    parser.add_argument("--experiment-kind", choices=("quality", "performance"), default="quality")
    parser.add_argument("--max-performance-regression", type=float, default=0.05)
    parser.add_argument("--test-command", default="poetry run pytest -q")
    parser.add_argument("--performance-maintenance-every", type=int, default=None)
    parser.add_argument("--performance-agent-command", default=None)
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
        for index in range(args.experiments):
            campaign.run_one()
            if (
                campaign.experiment_kind == "quality"
                and maintenance_every
                and (index + 1) % maintenance_every == 0
            ):
                quality_command = campaign.agent_command
                quality_kind = campaign.experiment_kind
                campaign.agent_command = performance_command
                campaign.experiment_kind = "performance"
                try:
                    campaign.run_one()
                finally:
                    campaign.agent_command = quality_command
                    campaign.experiment_kind = quality_kind
    except CampaignError as exc:
        parser.exit(2, f"meta-improve: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
