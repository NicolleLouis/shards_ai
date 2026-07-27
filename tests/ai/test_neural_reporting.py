from __future__ import annotations

from pathlib import Path

from shards_ai.ai.neural_reporting import write_training_report


def _metrics() -> list[dict]:
    return [{
        "epoch": 1,
        "train": {"records": 10, "mean_loss": 1.2},
        "validation": {
            "records": 4, "mean_loss": 1.1, "top1_accuracy": 0.5,
            "mean_chosen_rank": 2.0, "mean_normalized_chosen_rank": 0.7,
            "pairwise_accuracy": 0.8, "pairwise_pairs": 20,
        },
    }]


def test_training_report_is_standalone_html(tmp_path: Path) -> None:
    output = tmp_path / "report.html"

    write_training_report(_metrics(), output)

    content = output.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "Loss" in content
    assert "Top-1" in content
    assert "Paires correctes" in content
    assert "https://" not in content


def test_empty_training_report_is_still_valid(tmp_path: Path) -> None:
    output = tmp_path / "empty.html"

    write_training_report([], output)

    assert "Aucune métrique disponible" in output.read_text(encoding="utf-8")
