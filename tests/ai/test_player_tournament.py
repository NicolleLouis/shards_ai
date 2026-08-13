from benchmarks.benchmark_player_tournament import (
    MatchResult,
    build_matrix,
    order_players_by_global_win_rate,
    render_html,
)


def test_tournament_matrix_has_reciprocal_matchups_and_all_totals() -> None:
    results = {
        ("Random", "Heuristic 7"): MatchResult("Random", "Heuristic 7", 4, 1, 3, 0),
        ("Heuristic 7", "Random"): MatchResult("Heuristic 7", "Random", 4, 3, 1, 0),
        ("Random", "Heuristic 8"): MatchResult("Random", "Heuristic 8", 4, 2, 1, 1),
        ("Heuristic 8", "Random"): MatchResult("Heuristic 8", "Random", 4, 1, 2, 1),
    }

    matrix = build_matrix(results, "signed")
    rows = {row[0]: row for row in matrix}

    assert rows["Random"][3] == "-50.00%"
    assert rows["Heuristic 7"][1] == "+50.00%"
    assert rows["Random"][1] == "-12.50%"
    assert rows["Random"][4] == "+25.00%"
    assert rows["Random"][0] == "Random"
    assert rows["All"][1] == "—"
    assert rows["Heuristic 7"][2] == "+50.00%"
    assert rows["Heuristic 7"][3] == "—"


def test_tournament_players_are_ordered_by_global_win_rate() -> None:
    results = {
        ("Random", "Heuristic 7"): MatchResult("Random", "Heuristic 7", 10, 2, 8, 0),
        ("Heuristic 7", "Random"): MatchResult("Heuristic 7", "Random", 10, 8, 2, 0),
        ("Random", "Heuristic 8"): MatchResult("Random", "Heuristic 8", 10, 6, 4, 0),
        ("Heuristic 8", "Random"): MatchResult("Heuristic 8", "Random", 10, 4, 6, 0),
    }

    assert order_players_by_global_win_rate(results)[:3] == (
        "Heuristic 7",
        "Random",
        "Heuristic 8",
    )

    matrix = build_matrix(results, "win-rate", order_players_by_global_win_rate(results))
    assert [row[0] for row in matrix[:4]] == ["All", "Heuristic 7", "Random", "Heuristic 8"]

    report = render_html(matrix, games=10, seed=1, metric="win-rate", players=order_players_by_global_win_rate(results))
    body = report.split("<tbody>", 1)[1]
    assert body.index("<th>Heuristic 7</th>") < body.index("<th>Random</th>") < body.index("<th>Heuristic 8</th>")
    assert "class='negative' style='background-color:" in report and ">40.00%</td>" in report
    assert "class='positive' style='background-color:" in report and ">60.00%</td>" in report
    assert "supérieure à 50%" in report
