from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.meta_improve import Campaign


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Campaign Test")
    (repo / "doc").mkdir()
    (repo / "doc" / "Ideas.md").write_text("# Ideas\n\n- [ ] test idea\n", encoding="utf-8")
    (repo / "README.md").write_text("test\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def _agent(tmp_path: Path, result: dict, *, sleep_seconds: int = 0, result_in_worktree: bool = False) -> str:
    script = tmp_path / "agent.py"
    result_line = (
        "result_path = os.environ['META_RESULT_PATH']\n"
        if result_in_worktree
        else "result_path = os.path.join(os.environ['META_EXPERIMENT_DIR'], 'result.json')\n"
    )
    script.write_text(
        "import json, os, sys, time\n"
        f"time.sleep({sleep_seconds})\n"
        "worktree = os.environ['META_EXPERIMENT_WORKTREE']\n"
        "with open(os.path.join(worktree, 'candidate.txt'), 'w') as handle: handle.write('candidate\\n')\n"
        "with open(os.path.join(worktree, 'doc', 'Ideas.md'), 'a') as handle: handle.write('- [ ] generated idea\\n')\n"
        + result_line
        + f"json.dump({result!r}, open(result_path, 'w'))\n",
        encoding="utf-8",
    )
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def _campaign(repo: Path, command: str, budget: int = 30, kind: str = "quality") -> Campaign:
    return Campaign(
        repo,
        command,
        budget_seconds=budget,
        training_budget_seconds=max(1, budget // 3),
        screening_budget_seconds=max(1, budget // 3),
        overhead_budget_seconds=max(1, budget // 3),
        parent_profile="v008",
        experiment_kind=kind,
        test_command="true",
    )


def test_rejected_experiment_commits_report_and_keeps_idea_but_not_code(tmp_path):
    repo = _repo(tmp_path)
    command = _agent(tmp_path, {"status": "rejected", "hypothesis": "bad idea"})
    campaign = _campaign(repo, command)
    campaign.preflight()
    campaign.run_one()

    assert not (repo / "candidate.txt").exists()
    assert "generated idea" in (repo / "doc" / "Ideas.md").read_text(encoding="utf-8")
    assert (repo / "doc" / "Experiments" / "exp-00001.md").exists()
    assert "experiment: rejected exp-00001" in _git(repo, "log", "-1", "--format=%s")


def test_accepted_experiment_uses_independent_validation_and_commits_code(tmp_path):
    repo = _repo(tmp_path)
    result = {
        "status": "accepted",
        "hypothesis": "useful idea",
        "validation": {"results": {
            "random": {"delta_win_rate": -0.01},
            "v007": {"delta_win_rate": -0.01},
            "v008": {"delta_win_rate": 0.03},
        }},
        "performance": {
            "baseline": {"elapsed_seconds": 100.0},
            "candidate": {"elapsed_seconds": 97.0},
        },
    }
    command = _agent(tmp_path, result)
    campaign = _campaign(repo, command, kind="performance")
    campaign.preflight()
    campaign.run_one()

    assert (repo / "candidate.txt").exists()
    assert "experiment: accepted exp-00001" in _git(repo, "log", "-1", "--format=%s")
    manifest = next((repo / "artifacts" / "experiments").glob("*/exp-00001/manifest.json"))
    assert json.loads(manifest.read_text(encoding="utf-8"))["commit"] == _git(repo, "rev-parse", "HEAD")


def test_agent_result_is_read_from_worktree_and_archived(tmp_path):
    repo = _repo(tmp_path)
    command = _agent(tmp_path, {"status": "rejected", "hypothesis": "worktree result"}, result_in_worktree=True)
    campaign = _campaign(repo, command)
    campaign.preflight()
    campaign.run_one()

    report = (repo / "doc" / "Experiments" / "exp-00001.md").read_text(encoding="utf-8")
    assert "worktree result" in report
    artifact = next((repo / "artifacts" / "experiments").glob("*/exp-00001/result.json"))
    assert json.loads(artifact.read_text(encoding="utf-8"))["hypothesis"] == "worktree result"


def test_campaign_uses_active_neural_profile_as_parent(tmp_path):
    repo = _repo(tmp_path)
    profiles = repo / "configs" / "neural_training_profiles"
    profiles.mkdir(parents=True)
    (profiles / "active.yaml").write_text(
        "schema_version: 1\nactive_profile_id: v002\n", encoding="utf-8",
    )
    (profiles / "v002.yaml").write_text(
        "profile_id: v002\nmethod: ppo\ndataset: null\noutput: checkpoint.pt\n",
        encoding="utf-8",
    )

    campaign = Campaign(
        repo,
        _agent(tmp_path, {"status": "rejected"}),
        budget_seconds=30,
        training_budget_seconds=10,
        screening_budget_seconds=10,
        overhead_budget_seconds=10,
        parent_profile=None,
        test_command="true",
    )

    assert campaign.parent_profile == "v002"


def test_interrupted_agent_is_committed_as_inconclusive(tmp_path):
    repo = _repo(tmp_path)
    command = _agent(tmp_path, {"status": "accepted"}, sleep_seconds=5)
    campaign = _campaign(repo, command, budget=3)
    campaign.preflight()
    campaign.run_one()

    report = repo / "doc" / "Experiments" / "exp-00001.md"
    assert report.exists()
    assert "inconclusive" in report.read_text(encoding="utf-8")
    assert "experiment: inconclusive exp-00001" in _git(repo, "log", "-1", "--format=%s")
