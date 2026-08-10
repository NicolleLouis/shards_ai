from benchmarks.benchmark_player_tournament import MatchResult, build_matrix


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
    assert rows["Heuristic 7"][2] == "+50.00%"
    assert rows["Heuristic 7"][3] == "—"
