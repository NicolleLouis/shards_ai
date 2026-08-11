from __future__ import annotations

from benchmarks.benchmark_neural_mix import (
    _render_report,
    _summary,
    opponent_for_game,
    pass_with_playable_cards_example,
)
from shards_ai.game import Game
from shards_ai.game.actions import PassPlayPhase, PlayCard


def test_mix_schedule_is_exact_for_one_thousand_games() -> None:
    counts = {opponent: sum(opponent_for_game(index) == opponent for index in range(1000)) for opponent in ("random", "v007", "v008")}

    assert counts == {"random": 200, "v007": 300, "v008": 500}


def test_campaign_summary_and_html_contain_final_game_insights() -> None:
    record = {
        "neural_won": True, "opponent_won": False, "draw": False,
        "turns": 7, "turns_per_player": 3.5, "actions": 20, "elapsed_seconds": 1.2,
        "neural_decisions": 10, "neural_inference_seconds": 0.02,
        "neural_health": 35, "opponent_health": 0,
        "neural_mastery": 4, "opponent_mastery": 2,
        "neural_deck": {"crystal": 3}, "opponent_deck": {"crystal": 2},
        "neural_deck_size": 4, "opponent_deck_size": 3, "deck_size_delta": 1,
        "neural_passes_with_playable_cards": 2, "neural_passed_with_playable_cards": True,
        "mercenary_events": [], "mastery_events": [],
    }
    payload = {
        "config": {"games": 1, "seed": 1},
        "summary_by_opponent": {opponent: _summary([record]) for opponent in ("random", "v007", "v008")},
    }

    report = _render_report(payload)

    assert payload["summary_by_opponent"]["random"]["neural_win_rate"] == 1.0
    assert payload["summary_by_opponent"]["random"]["turns_per_player"]["mean"] == 3.5
    assert payload["summary_by_opponent"]["random"]["neural_deck_size"]["mean"] == 4
    assert payload["summary_by_opponent"]["random"]["deck_size_delta"]["mean"] == 1
    assert payload["summary_by_opponent"]["random"]["neural_passed_with_playable_cards_games"] == 1
    assert "PV Neural" in report
    assert "7.0 tours · 3.5 tours / joueur" in report
    assert "Tours / joueur" in report
    assert "Maîtrise Neural" in report
    assert "Deck Neural" in report
    assert "Pass avec carte jouable" in report
    assert "Développement du deck" in report
    assert "Deck final moyen" in report
    assert "crystal" in report


def test_report_contains_sorted_copy_count_charts_and_zeroes_missing_cards() -> None:
    record = {
        "neural_won": True, "opponent_won": False, "draw": False,
        "turns": 7, "turns_per_player": 3.5, "actions": 20, "elapsed_seconds": 1.2,
        "neural_decisions": 10, "neural_inference_seconds": 0.02,
        "neural_health": 35, "opponent_health": 0,
        "neural_mastery": 4, "opponent_mastery": 2,
        "neural_deck": {"aspirant_maquis": 2, "saule_vengeur": 1},
        "opponent_deck": {"aspirant_maquis": 1},
        "mercenary_events": [], "mastery_events": [],
    }
    payload = {
        "config": {"games": 1, "seed": 1},
        "summary_by_opponent": {opponent: _summary([record]) for opponent in ("random", "v007", "v008")},
    }

    report = _render_report(payload)

    assert report.count("<h3>NeuralPlayer — cartes en ×1</h3>") == 3
    assert report.count("<h3>Adversaire — cartes en ×2</h3>") == 3
    assert report.count("<h3>Delta NeuralPlayer − adversaire — cartes en ×3</h3>") == 3
    assert "aspirant_maquis" in report
    assert "saule_vengeur" in report


def test_report_summary_headers_match_their_values() -> None:
    record = {
        "neural_won": True, "opponent_won": False, "draw": False,
        "turns": 7, "turns_per_player": 3.5, "actions": 20, "elapsed_seconds": 1.2,
        "neural_decisions": 10, "neural_macro_decisions": 6,
        "neural_inference_seconds": 0.02,
        "neural_health": 35, "opponent_health": 0,
        "neural_mastery": 4, "opponent_mastery": 2,
        "neural_deck": {"crystal": 3}, "opponent_deck": {"crystal": 2},
        "neural_deck_size": 13, "opponent_deck_size": 12, "deck_size_delta": 1,
        "neural_passes_with_playable_cards": 0,
        "neural_passed_with_playable_cards": False,
        "mercenary_events": [], "mastery_events": [],
    }
    payload = {
        "config": {"games": 1, "seed": 1},
        "summary_by_opponent": {
            opponent: _summary([record]) for opponent in ("random", "v007", "v008")
        },
    }

    report = _render_report(payload)

    assert "<td>3.5</td><td>10.0</td><td>6.0</td><td>13.0</td>" in report
    assert "<td>12.0</td><td>+1.0</td><td>0.0%</td><td>35.0</td><td>4.0</td>" in report


def test_pass_example_lists_remaining_and_playable_cards() -> None:
    game = Game.new(seed=1)
    observation = game.neural_observation_for(game.active_player)
    first, second = observation.active_player.hand[:2]

    example = pass_with_playable_cards_example(
        observation,
        [PlayCard(first.instance_id), PlayCard(second.instance_id), PassPlayPhase()],
        PassPlayPhase(),
        seed=42,
        opponent="v008",
        action_source="macro_choice",
    )

    assert example is not None
    assert example["seed"] == 42
    assert example["opponent"] == "v008"
    assert {card["instance_id"] for card in example["playable_cards"]} == {
        first.instance_id,
        second.instance_id,
    }
    assert len(example["remaining_hand"]) == len(observation.active_player.hand)
