import json
from pathlib import Path

from scripts.benchmark_heuristic_report import run_benchmark, write_reports


def test_heuristic_report_contains_final_deck_statistics_and_choice_deltas(tmp_path: Path) -> None:
    result = run_benchmark(
        duration_seconds=10.0,
        games=4,
        seed=1046,
        profile_path="configs/heuristic_profiles/v008.yaml",
        opponent_profile_path="configs/heuristic_profiles/v007.yaml",
        max_actions=10_000,
        max_turns=None,
        strict=True,
        top_cards=10,
    )

    assert result["overall"]["games"] == 4
    assert set(result["opponents"]) == {"random", "v007"}
    assert [result["opponents"][name]["overall"]["games"] for name in ("random", "v007")] == [2, 2]
    for matchup in result["opponents"].values():
        assert len(matchup["final_decks_by_role"]["heuristic"]) == 2
        assert len(matchup["final_decks_by_role"]["random"]) == 2
        assert "cards" in matchup["final_deck_statistics_by_role"]["heuristic"]
        assert "delta_per_game" in next(
            iter(next(iter(matchup["choice_deltas_heuristic_minus_random"].values())))
        )
        assert "pass_play" in matchup["heuristic_behavior"]
        assert "gain_mastery" in matchup["heuristic_behavior"]

    json_path, html_path = write_reports(result, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload["opponents"]) == {"random", "v007"}
    html = html_path.read_text(encoding="utf-8")
    assert "Graphiques des decks finaux" in html
    assert "Delta deck v008 − Random" in html
    assert "Delta deck v008 − Heuristic v007" in html
    assert "Choix par rôle" not in html
    assert (tmp_path / "random_final_deck_delta_cards.csv").exists()
    assert (tmp_path / "v007_final_deck_delta_cards.csv").exists()
    assert (tmp_path / "random_choice_deltas.csv").exists()
    assert (tmp_path / "v007_choice_deltas.csv").exists()
