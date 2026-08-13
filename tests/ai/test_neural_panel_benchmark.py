from __future__ import annotations

from benchmarks.benchmark_neural_mix import _summary
from benchmarks.benchmark_neural_panel import OPPONENTS, _render_report


def _record(opponent: str) -> dict[str, object]:
    return {
        "seed": 104,
        "opponent": opponent,
        "status": "finished",
        "neural_won": True,
        "opponent_won": False,
        "draw": False,
        "turns": 10,
        "turns_per_player": 5.0,
        "actions": 20,
        "elapsed_seconds": 0.5,
        "neural_decisions": 8,
        "neural_inference_seconds": 0.08,
        "neural_health": 10,
        "opponent_health": 0,
        "neural_mastery": 3,
        "opponent_mastery": 1,
        "neural_deck": {},
        "opponent_deck": {},
        "neural_deck_size": 5,
        "opponent_deck_size": 4,
        "deck_size_delta": 1,
        "neural_passes_with_playable_cards": 0,
        "neural_passed_with_playable_cards": False,
        "mercenary_events": [],
        "mastery_events": [],
    }


def test_panel_contains_heuristic_and_eight_configured_opponents() -> None:
    assert OPPONENTS == (
        "random",
        "v007",
        "v008",
        "neural:v001",
        "neural:v002",
        "neural:v003",
        "neural:v004",
        "neural:v005",
        "neural:v006",
    )


def test_panel_html_renders_each_opponent() -> None:
    payload = {
        "config": {
            "checkpoint": "configs/neural_profiles/v006.pt",
            "games_per_opponent": 1,
            "total_games": 8,
            "seed": 104,
            "torch_threads": 1,
        },
        "summary_by_opponent": {
            opponent: _summary([_record(opponent)]) for opponent in OPPONENTS
        },
    }

    report = _render_report(payload)

    assert "Benchmark NeuralPlayer — panel complet" in report
    assert "configs/neural_profiles/v006.pt" in report
    for opponent in OPPONENTS:
        assert f"Contre {opponent}" in report
