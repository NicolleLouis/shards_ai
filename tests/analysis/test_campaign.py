from pathlib import Path

import pytest

from shards_ai.analysis import (
    CampaignConfig,
    build_delta_statistics,
    build_statistics,
    central_copy_counts,
    run_campaign,
    write_report,
)
from shards_ai.game import CardInstance, Game, GameStatus, PlayerId, card_definition


class StubRunner:
    def __init__(self, state=None, error=None) -> None:
        self.state = state
        self.error = error

    def run(self):
        if self.error is not None:
            raise self.error
        return self.state


def finished_state_with_cards(*card_ids: str):
    state = Game.new(seed=101).state
    state.status = GameStatus.FINISHED
    state.winner = PlayerId.PLAYER_1
    player = state.players[PlayerId.PLAYER_1]
    player.hand.clear()
    player.draw_pile.clear()
    player.discard_pile.clear()
    player.play_zone.clear()
    for index, card_id in enumerate(card_ids):
        player.play_zone.append(CardInstance(f"test-{index}", card_definition(card_id)))
    return state


def test_campaign_config_defaults_to_sixty_seconds() -> None:
    assert CampaignConfig().duration_seconds == 60


def test_central_copy_counts_are_read_from_deck_definitions() -> None:
    counts = central_copy_counts()
    assert counts["aspirant_maquis"] == 3
    assert counts["saule_vengeur"] == 1


def test_statistics_include_zeroes_and_stratify_non_base_cards() -> None:
    cards, factions, grouped = build_statistics(
        [{"cards": {"crystal": 7, "aspirant_maquis": 2, "saule_vengeur": 1}}]
    )

    crystal = next(row for row in cards if row["card_id"] == "crystal")
    assert crystal["average_number"] == 7
    assert crystal["central_copy_count"] is None
    assert all(row["card_id"] not in {"crystal", "blaster", "shard_reactor", "infinity_shard"} for rows in grouped.values() for row in rows)
    assert grouped["3"][0]["card_id"] == "aspirant_maquis"
    assert grouped["1"][0]["card_id"] == "saule_vengeur"
    assert {row["faction"] for row in factions}


def test_campaign_is_reproducible_with_a_fixed_seed() -> None:
    first = run_campaign(CampaignConfig(seed=22, games=2, duration_seconds=10))
    second = run_campaign(CampaignConfig(seed=22, games=2, duration_seconds=10))
    assert first.to_dict()["winner_decks"] == second.to_dict()["winner_decks"]
    assert first.to_dict()["cards"] == second.to_dict()["cards"]


def test_campaign_generates_and_reports_a_seed_when_omitted() -> None:
    result = run_campaign(CampaignConfig(games=1), runner_factory=lambda seed, config: StubRunner(finished_state_with_cards("blaster")))
    assert result.root_seed >= 0
    assert result.to_dict()["root_seed"] == result.root_seed


def test_winner_deck_collects_cards_from_all_owned_zones() -> None:
    state = Game.new(seed=102).state
    state.status = GameStatus.FINISHED
    state.winner = PlayerId.PLAYER_1
    player = state.players[PlayerId.PLAYER_1]
    player.hand = [CardInstance("hand", card_definition("blaster"))]
    player.draw_pile = [CardInstance("draw", card_definition("crystal"))]
    player.discard_pile = [CardInstance("discard", card_definition("crystal"))]
    player.play_zone = [CardInstance("play", card_definition("infinity_shard"))]

    result = run_campaign(CampaignConfig(games=1), runner_factory=lambda seed, config: StubRunner(state))

    assert result.winner_decks[0]["cards"] == {
        "blaster": 1,
        "crystal": 2,
        "infinity_shard": 1,
    }


def test_campaign_stops_between_games_when_duration_is_reached() -> None:
    clock_values = iter([0.0, 2.0, 2.0])
    result = run_campaign(
        CampaignConfig(games=5, duration_seconds=1),
        clock=lambda: next(clock_values),
        runner_factory=lambda seed, config: StubRunner(finished_state_with_cards("crystal")),
    )

    assert result.attempted == 1
    assert result.completed == 1


def test_campaign_continues_after_an_error_by_default_and_records_seed() -> None:
    calls = iter([StubRunner(error=RuntimeError("broken")), StubRunner(finished_state_with_cards("crystal"))])
    result = run_campaign(
        CampaignConfig(games=2),
        runner_factory=lambda seed, config: next(calls),
    )

    assert result.completed == 1
    assert len(result.errors) == 1
    assert result.errors[0]["game_index"] == 0
    assert isinstance(result.errors[0]["seed"], int)


def test_campaign_stores_loser_deck_and_delta_compares_both_sides() -> None:
    state = finished_state_with_cards("aspirant_maquis", "aspirant_maquis", "crystal")
    loser = state.players[PlayerId.PLAYER_2]
    loser.hand.clear()
    loser.draw_pile.clear()
    loser.discard_pile.clear()
    loser.play_zone = [CardInstance("loser-card", card_definition("aspirant_maquis"))]

    result = run_campaign(
        CampaignConfig(games=1),
        runner_factory=lambda seed, config: StubRunner(state),
    )

    assert result.loser_decks[0]["cards"] == {"aspirant_maquis": 1}
    delta_cards, delta_factions, _ = build_delta_statistics(
        result.winner_decks, result.loser_decks
    )
    aspirant = next(row for row in delta_cards if row["card_id"] == "aspirant_maquis")
    assert aspirant["winner_average_number"] == 2
    assert aspirant["loser_average_number"] == 1
    assert aspirant["delta_average_number"] == 1
    maquis = next(row for row in delta_factions if row["faction"] == "maquis")
    assert maquis["delta_average_number"] == 1


def test_strict_campaign_raises_on_the_first_error() -> None:
    with pytest.raises(RuntimeError, match="broken"):
        run_campaign(
            CampaignConfig(games=1, strict=True),
            runner_factory=lambda seed, config: StubRunner(error=RuntimeError("broken")),
        )


def test_report_writes_json_csv_and_html(tmp_path: Path) -> None:
    result = run_campaign(CampaignConfig(seed=23, games=1, duration_seconds=10))
    report = write_report(result, tmp_path)
    assert report.name == "report.html"
    assert {path.name for path in tmp_path.iterdir()} == {
        "result.json",
        "cards.csv",
        "factions.csv",
        "cards_by_copy_count.csv",
        "loser_cards.csv",
        "loser_factions.csv",
        "loser_cards_by_copy_count.csv",
        "cards_delta.csv",
        "factions_delta.csv",
        "cards_delta_by_copy_count.csv",
        "report.html",
    }
    report_html = report.read_text(encoding="utf-8")
    assert "<svg" in report_html
    assert "chart-row" in report_html
    assert "Delta cartes hors base" in report_html
    assert "<circle" in report_html
    assert "carte(s) en moyenne" in report_html
    assert "0.00" in report_html
    assert ".00</text>" in report_html


@pytest.mark.parametrize("kwargs", [{"duration_seconds": 0}, {"games": 0}])
def test_invalid_campaign_limits_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        CampaignConfig(**kwargs)
