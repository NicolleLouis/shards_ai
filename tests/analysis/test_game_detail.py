import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_game_detail_cli_writes_replayable_json_and_html(tmp_path: Path) -> None:
    output_dir = tmp_path / "detail"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/analyze_game_detail.py"),
            "--seed",
            "123",
            "--player1",
            "heuristic",
            "--player2",
            "random",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )

    trace = json.loads((output_dir / "game.json").read_text(encoding="utf-8"))
    report = (output_dir / "report.html").read_text(encoding="utf-8")

    assert "status=finished" in result.stdout
    assert trace["seed"] == 123
    assert trace["events"]
    assert "ranked_alternatives" in trace["events"][0]["explanation"]
    purchase_event = next(
        event for event in trace["events"] if "purchase_analysis" in event
    )
    assert len(purchase_event["purchase_analysis"]["river"]) == 6
    assert all(
        option["score"] is not None
        for entry in purchase_event["purchase_analysis"]["river"]
        for option in entry["options"]
    )
    assert "conclusion" in purchase_event["purchase_analysis"]
    assert "Analyse détaillée d’une partie" in report
    assert "Tour 1" in report
    assert "Analyse de la phase achat" in report
    assert "legend-1" in report
