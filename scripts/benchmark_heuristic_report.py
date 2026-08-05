#!/usr/bin/env python3
"""Benchmark HeuristicPlayer vs RandomPlayer with card and end-state reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shards_ai.ai import HeuristicPlayer, RandomPlayer
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.analysis.campaign import (
    BASE_CARD_IDS,
    _line_svg,
    _pie_svg,
    build_delta_statistics,
    build_statistics,
)
from shards_ai.game import (
    BanishCard,
    BuyCard,
    CardInstance,
    ChoosePendingDecision,
    Game,
    GameRandom,
    GameRunner,
    GameStatus,
    GainMastery,
    PlayerId,
    PassPlayPhase,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
)
from shards_ai.game.actions import Action


class _RecordingPlayer:
    """Record the chosen action before the runner applies it, without state snapshots."""

    observation_is_read_only = True

    def __init__(self, player, on_action) -> None:
        self._player = player
        self._on_action = on_action

    def choose_action(self, observation, legal_actions):
        action = self._player.choose_action(observation, legal_actions)
        self._on_action(observation, action, self._player.player_id)
        return action


@dataclass
class OutcomeStats:
    games: int = 0
    heuristic_wins: int = 0
    random_wins: int = 0
    draws: int = 0
    errors: int = 0
    winner_health: list[int] = field(default_factory=list)
    winner_mastery: list[int] = field(default_factory=list)
    loser_mastery: list[int] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "games": self.games,
            "heuristic_wins": self.heuristic_wins,
            "random_wins": self.random_wins,
            "draws": self.draws,
            "errors": self.errors,
            "heuristic_win_rate": _percentage(self.heuristic_wins, self.games),
            "winner_health": _numeric_summary(self.winner_health),
            "winner_mastery": _numeric_summary(self.winner_mastery),
            "loser_mastery": _numeric_summary(self.loser_mastery),
            "turns": _numeric_summary(self.turns),
        }


def _percentage(value: int, total: int) -> float:
    return round(100.0 * value / total, 3) if total else 0.0


def _numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "min": min(values),
        "max": max(values),
    }


def _part_seed(root_seed: int, game_index: int) -> int:
    payload = f"shards-ai-heuristic-report:{root_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _player_cards(state, player_id: PlayerId) -> Counter[str]:
    player = state.players[player_id]
    cards = (*player.hand, *player.draw_pile, *player.discard_pile, *player.play_zone)
    return Counter(card.definition.card_id for card in cards)


def _deck_size(state, player_id: PlayerId) -> int:
    """Count cards owned by a player in all persistent/player-owned zones."""

    player = state.players[player_id]
    return sum(
        len(zone)
        for zone in (
            player.hand,
            player.draw_pile,
            player.discard_pile,
            player.play_zone,
            player.champions,
        )
    )


def _deck_snapshot(state, player_id: PlayerId, role: str, result_group: str, game_index: int) -> dict[str, object]:
    return {
        "game_index": game_index,
        "seed": state.seed,
        "role": role,
        "result_group": result_group,
        "player_id": int(player_id),
        "deck_size": _deck_size(state, player_id),
        "cards": dict(sorted(_player_cards(state, player_id).items())),
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _card_from_action(state, player_id: PlayerId, action: Action) -> CardInstance | None:
    player = state.players[player_id]
    if isinstance(action, (PlayCard, BanishCard)):
        zones = (player.hand, player.discard_pile) if isinstance(action, BanishCard) else (player.hand,)
        for zone in zones:
            for card in zone:
                if card.instance_id == action.card_id:
                    return card
    if isinstance(action, (BuyCard, RecruitMercenary, RecruitFreeCard)):
        for card in state.river:
            if card is not None and card.instance_id == action.card_instance_id:
                return card
    if isinstance(action, ChoosePendingDecision):
        for zone in (player.hand, player.discard_pile, player.play_zone, player.champions):
            for card in zone:
                if card.instance_id == action.choice_id:
                    return card
    return None


def _record_action(card_counts: dict[str, Counter[str]], state, player_id: PlayerId, action: Action) -> CardInstance | None:
    card = _card_from_action(state, player_id, action)
    if card is None:
        return None
    if isinstance(action, PlayCard):
        category = "play"
    elif isinstance(action, BanishCard):
        category = "banish"
    elif isinstance(action, BuyCard):
        category = "buy"
    elif isinstance(action, RecruitMercenary):
        category = "recruit_mercenary"
    elif isinstance(action, RecruitFreeCard):
        category = "recruit_free"
    else:
        category = "decision"
    card_counts[category][card.definition.name] += 1
    return card


def _top_cards(counter: Counter[str], limit: int) -> list[dict[str, object]]:
    return [{"card": name, "count": count} for name, count in counter.most_common(limit)]


def _new_heuristic_behavior_stats() -> dict[str, object]:
    return {
        "pass_play_events": [],
        "gain_mastery_actions": 0,
        "gain_mastery_games": 0,
        "gain_mastery_used": False,
        "mercenary_recruitments": Counter(),
        "mercenary_long_term_buys": Counter(),
    }


def _record_heuristic_behavior(
    state,
    action: Action,
    player_id: PlayerId,
    behavior: dict[str, object],
    *,
    card: CardInstance | None = None,
) -> None:
    if isinstance(action, PassPlayPhase):
        hand = [card.definition.name for card in state.players[player_id].hand]
        if hand:
            behavior["pass_play_events"].append({"cards": hand})
    elif isinstance(action, GainMastery):
        behavior["gain_mastery_actions"] += 1
        behavior["gain_mastery_used"] = True
    elif isinstance(action, (BuyCard, RecruitMercenary)):
        if card is None or not card.definition.is_mercenary:
            return
        category = (
            "mercenary_recruitments"
            if isinstance(action, RecruitMercenary)
            else "mercenary_long_term_buys"
        )
        behavior[category][card.definition.name] += 1


def _merge_heuristic_behavior(
    target: dict[str, object],
    source: dict[str, object],
    *,
    game_index: int,
    seed: int,
    result_group: str,
) -> None:
    for event in source["pass_play_events"]:
        target["pass_play_events"].append({
            "game_index": game_index,
            "seed": seed,
            "result_group": result_group,
            "cards": event["cards"],
        })
    target["gain_mastery_actions"] += source["gain_mastery_actions"]
    if source["gain_mastery_used"]:
        target["gain_mastery_games"] += 1
    target["mercenary_recruitments"].update(source["mercenary_recruitments"])
    target["mercenary_long_term_buys"].update(source["mercenary_long_term_buys"])


def _behavior_summary(behavior: dict[str, object], games: int) -> dict[str, object]:
    pass_events = behavior["pass_play_events"]
    remaining_cards = Counter(
        card for event in pass_events for card in event["cards"]
    )
    games_with_remaining = len({event["game_index"] for event in pass_events})
    return {
        "games": games,
        "pass_play": {
            "events": len(pass_events),
            "games_with_remaining_hand": games_with_remaining,
            "game_rate_percent": _percentage(games_with_remaining, games),
            "cards_remaining_total": sum(len(event["cards"]) for event in pass_events),
            "cards": _top_cards(remaining_cards, 100),
            "events_detail": pass_events,
        },
        "gain_mastery": {
            "actions": behavior["gain_mastery_actions"],
            "games_with_action": behavior["gain_mastery_games"],
            "game_rate_percent": _percentage(behavior["gain_mastery_games"], games),
            "actions_per_game": round(behavior["gain_mastery_actions"] / games, 3) if games else 0.0,
        },
        "mercenary_purchases": {
            "recruit_immediate": _top_cards(behavior["mercenary_recruitments"], 100),
            "buy_long_term": _top_cards(behavior["mercenary_long_term_buys"], 100),
        },
    }


def _run_single_benchmark(
    *,
    duration_seconds: float,
    games: int | None,
    seed: int | None,
    profile_path: str | None,
    max_actions: int,
    max_turns: int | None,
    strict: bool,
    top_cards: int,
    opponent_name: str = "random",
    opponent_profile_path: str | None = None,
    game_index_offset: int = 0,
    game_index_step: int = 1,
) -> dict[str, object]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if games is not None and games <= 0:
        raise ValueError("games must be positive")
    root_seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
    profile = load_profile(profile_path) if profile_path else None
    opponent_profile = load_profile(opponent_profile_path) if opponent_profile_path else None
    started = time.monotonic()
    overall = OutcomeStats()
    by_heuristic_result = {"heuristic_win": OutcomeStats(), "heuristic_loss": OutcomeStats(), "draw": OutcomeStats()}
    role_card_counts: dict[str, dict[str, Counter[str]]] = {
        "heuristic": defaultdict(Counter),
        "random": defaultdict(Counter),
    }
    result_card_counts: dict[str, dict[str, Counter[str]]] = {
        group: defaultdict(Counter) for group in by_heuristic_result
    }
    final_decks_by_role: dict[str, list[dict[str, object]]] = {
        "heuristic": [],
        "random": [],
    }
    final_decks_by_result: dict[str, list[dict[str, object]]] = {
        group: [] for group in by_heuristic_result
    }
    heuristic_behavior = _new_heuristic_behavior_stats()
    heuristic_behavior_by_result = {
        group: _new_heuristic_behavior_stats() for group in by_heuristic_result
    }

    attempted = 0
    while games is None or attempted < games:
        if attempted and games is None and time.monotonic() - started >= duration_seconds:
            break
        global_game_index = game_index_offset + attempted * game_index_step
        game_seed = _part_seed(root_seed, global_game_index)
        heuristic_id = PlayerId.PLAYER_1 if global_game_index % 2 == 0 else PlayerId.PLAYER_2
        root_rng = GameRandom(game_seed)
        game = Game.new(seed=game_seed, rng=root_rng.derive("engine"))
        raw_players = {
            player_id: (
                HeuristicPlayer(
                    player_id,
                    profile.weights if profile else None,
                    profile.card_acquisition_weights if profile else None,
                    profile.constraint_weights if profile else None,
                )
                if player_id is heuristic_id
                else (
                    RandomPlayer(player_id, root_rng.derive(f"player-{player_id.value}"))
                    if opponent_name == "random"
                    else HeuristicPlayer(
                        player_id,
                        opponent_profile.weights if opponent_profile else None,
                        opponent_profile.card_acquisition_weights if opponent_profile else None,
                        opponent_profile.constraint_weights if opponent_profile else None,
                    )
                )
            )
            for player_id in PlayerId
        }
        action_counts: dict[str, dict[str, Counter[str]]] = {
            "heuristic": defaultdict(Counter),
            "random": defaultdict(Counter),
        }
        behavior_for_game = _new_heuristic_behavior_stats()

        def record_action(before, action, player_id) -> None:
            role = "heuristic" if player_id is heuristic_id else "random"
            card = _record_action(action_counts[role], before, player_id, action)
            if role == "heuristic":
                _record_heuristic_behavior(
                    before, action, player_id, behavior_for_game, card=card
                )

        players = {
            player_id: _RecordingPlayer(player, record_action)
            for player_id, player in raw_players.items()
        }

        attempted += 1
        try:
            runner = GameRunner(game, players, max_actions=max_actions, max_turns=max_turns)
            state = runner.run()
        except Exception:
            if strict:
                raise
            overall.errors += 1
            continue

        overall.games += 1
        if state.status is GameStatus.DRAW or state.winner is None:
            result_group = "draw"
            overall.draws += 1
        elif state.winner is heuristic_id:
            result_group = "heuristic_win"
            overall.heuristic_wins += 1
        else:
            result_group = "heuristic_loss"
            overall.random_wins += 1
        group = by_heuristic_result[result_group]
        group.games += 1
        if result_group == "heuristic_win":
            group.heuristic_wins += 1
        elif result_group == "heuristic_loss":
            group.random_wins += 1
        else:
            group.draws += 1

        _merge_heuristic_behavior(
            heuristic_behavior,
            behavior_for_game,
                    game_index=global_game_index,
            seed=game_seed,
            result_group=result_group,
        )
        _merge_heuristic_behavior(
            heuristic_behavior_by_result[result_group],
            behavior_for_game,
            game_index=attempted - 1,
            seed=game_seed,
            result_group=result_group,
        )

        if state.winner is not None:
            winner = state.players[state.winner]
            loser = state.players[state.winner.opponent]
            for target, value in ((overall.winner_health, winner.health), (overall.winner_mastery, winner.mastery), (overall.loser_mastery, loser.mastery), (overall.turns, state.turn_number)):
                target.append(value)

        role_by_player = {
            heuristic_id: "heuristic",
            heuristic_id.opponent: "random",
        }
        for player_id, role in role_by_player.items():
            snapshot = _deck_snapshot(state, player_id, role, result_group, global_game_index)
            final_decks_by_role[role].append(snapshot)
            final_decks_by_result[result_group].append(snapshot)
            for target, value in ((group.winner_health, winner.health), (group.winner_mastery, winner.mastery), (group.loser_mastery, loser.mastery), (group.turns, state.turn_number)):
                target.append(value)

        for role, role_counts in action_counts.items():
            for category, counts in role_counts.items():
                role_card_counts[role][category].update(counts)
                result_card_counts[result_group][category].update(counts)

    def cards_section(source: dict[str, Counter[str]]) -> dict[str, list[dict[str, object]]]:
        return {category: _top_cards(counter, top_cards) for category, counter in sorted(source.items())}

    role_deck_statistics: dict[str, dict[str, object]] = {}
    for role, snapshots in final_decks_by_role.items():
        cards, factions, grouped = build_statistics(snapshots)
        role_deck_statistics[role] = {
            "cards": cards,
            "factions": factions,
            "cards_by_copy_count": grouped,
        }
    role_delta_cards, role_delta_factions, role_delta_grouped = build_delta_statistics(
        final_decks_by_role["heuristic"], final_decks_by_role["random"]
    )
    role_delta_cards.sort(
        key=lambda row: (-abs(float(row["delta_average_number"])), str(row["card_id"]))
    )
    for rank, row in enumerate(role_delta_cards, start=1):
        row["rank"] = rank
    role_deck_delta = {
        "cards": role_delta_cards,
        "factions": role_delta_factions,
        "cards_by_copy_count": role_delta_grouped,
    }
    result_deck_statistics: dict[str, dict[str, object]] = {}
    for group, snapshots in final_decks_by_result.items():
        cards, factions, grouped = build_statistics(snapshots)
        result_deck_statistics[group] = {
            "cards": cards,
            "factions": factions,
            "cards_by_copy_count": grouped,
        }
    deck_size_by_role = {
        role: _numeric_summary([int(snapshot["deck_size"]) for snapshot in snapshots])
        for role, snapshots in final_decks_by_role.items()
    }
    heuristic_deck_size_by_result = {
        group: _numeric_summary([
            int(snapshot["deck_size"])
            for snapshot in snapshots
            if snapshot["role"] == "heuristic"
        ])
        for group, snapshots in final_decks_by_result.items()
    }
    choice_deltas: dict[str, list[dict[str, object]]] = {}
    completed_games = overall.games
    categories = set(role_card_counts["heuristic"]) | set(role_card_counts["random"])
    for category in sorted(categories):
        names = set(role_card_counts["heuristic"][category]) | set(role_card_counts["random"][category])
        rows = []
        for name in names:
            heuristic_count = role_card_counts["heuristic"][category][name]
            random_count = role_card_counts["random"][category][name]
            rows.append({
                "card": name,
                "heuristic_count": heuristic_count,
                "random_count": random_count,
                "delta_per_game": round(
                    (heuristic_count - random_count) / completed_games, 6
                ) if completed_games else 0.0,
            })
        rows.sort(key=lambda row: (-abs(float(row["delta_per_game"])), str(row["card"])))
        choice_deltas[category] = rows

    return {
        "schema_version": 1,
        "root_seed": root_seed,
        "profile": profile.profile_id if profile else "default",
        "opponent": opponent_name,
        "opponent_profile": opponent_profile.profile_id if opponent_profile else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "attempted": attempted,
        "duration_seconds": duration_seconds,
        "overall": overall.summary(),
        "heuristic_result_groups": {name: stats.summary() for name, stats in by_heuristic_result.items()},
        "cards_by_role": {
            role: cards_section(counts) for role, counts in role_card_counts.items()
        },
        "cards_by_heuristic_result": {
            name: cards_section(counts) for name, counts in result_card_counts.items()
        },
        "final_decks_by_role": final_decks_by_role,
        "final_decks_by_result": final_decks_by_result,
        "final_deck_statistics_by_role": role_deck_statistics,
        "deck_size_by_role": deck_size_by_role,
        "heuristic_deck_size_by_result": heuristic_deck_size_by_result,
        "final_deck_delta_heuristic_minus_random": role_deck_delta,
        "final_deck_delta_heuristic_minus_opponent": role_deck_delta,
        "final_deck_statistics_by_result": result_deck_statistics,
        "choice_deltas_heuristic_minus_random": choice_deltas,
        "choice_deltas_heuristic_minus_opponent": choice_deltas,
        "heuristic_behavior": _behavior_summary(heuristic_behavior, overall.games),
        "heuristic_behavior_by_result": {
            name: _behavior_summary(stats, by_heuristic_result[name].games)
            for name, stats in heuristic_behavior_by_result.items()
        },
    }


def run_benchmark(
    *,
    duration_seconds: float,
    games: int | None,
    seed: int | None,
    profile_path: str | None,
    opponent_profile_path: str,
    max_actions: int,
    max_turns: int | None,
    strict: bool,
    top_cards: int,
) -> dict[str, object]:
    """Run an exact 50/50 campaign and keep each opponent's analysis separate."""

    root_seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
    if games is None:
        random_games = None
        v007_games = None
        sub_duration = duration_seconds / 2
    else:
        random_games = (games + 1) // 2
        v007_games = games // 2
        sub_duration = duration_seconds

    opponents: dict[str, dict[str, object]] = {}
    if random_games != 0:
        opponents["random"] = _run_single_benchmark(
            duration_seconds=sub_duration,
            games=random_games,
            seed=root_seed,
            profile_path=profile_path,
            max_actions=max_actions,
            max_turns=max_turns,
            strict=strict,
            top_cards=top_cards,
            opponent_name="random",
            game_index_offset=0,
            game_index_step=2,
        )
    if v007_games != 0:
        opponents["v007"] = _run_single_benchmark(
            duration_seconds=sub_duration,
            games=v007_games,
            seed=root_seed,
            profile_path=profile_path,
            opponent_profile_path=opponent_profile_path,
            max_actions=max_actions,
            max_turns=max_turns,
            strict=strict,
            top_cards=top_cards,
            opponent_name="v007",
            game_index_offset=1,
            game_index_step=2,
        )

    overall = {
        "games": sum(int(item["overall"]["games"]) for item in opponents.values()),
        "heuristic_wins": sum(int(item["overall"]["heuristic_wins"]) for item in opponents.values()),
        "random_wins": sum(int(item["overall"]["random_wins"]) for item in opponents.values()),
        "draws": sum(int(item["overall"]["draws"]) for item in opponents.values()),
        "errors": sum(int(item["overall"]["errors"]) for item in opponents.values()),
    }
    overall["heuristic_win_rate"] = _percentage(overall["heuristic_wins"], overall["games"])
    return {
        "schema_version": 2,
        "root_seed": root_seed,
        "profile": next(iter(opponents.values()))["profile"] if opponents else "default",
        "opponent_profiles": {name: item["opponent_profile"] for name, item in opponents.items()},
        "elapsed_seconds": round(sum(float(item["elapsed_seconds"]) for item in opponents.values()), 3),
        "attempted": sum(int(item["attempted"]) for item in opponents.values()),
        "duration_seconds": duration_seconds,
        "overall": overall,
        "opponents": opponents,
    }


def _table(title: str, sections: dict[str, list[dict[str, object]]]) -> str:
    rows = [f"<h3>{html.escape(title)}</h3>"]
    for category, cards in sections.items():
        rows.append(f"<h4>{html.escape(category)}</h4><table><tr><th>Carte</th><th>Choix</th></tr>")
        rows.extend(f"<tr><td>{html.escape(str(item['card']))}</td><td>{item['count']}</td></tr>" for item in cards)
        rows.append("</table>")
    return "".join(rows) if sections else f"<h3>{html.escape(title)}</h3><p>Aucun choix enregistré.</p>"


def _grouped_tables(groups: dict[str, dict[str, list[dict[str, object]]]]) -> str:
    return "".join(_table(group, sections) for group, sections in groups.items())


def write_reports(result: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    """Write the multi-opponent JSON/CSV/HTML report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    html_path = output_dir / "report.html"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def data_table(rows: list[dict[str, object]], fields: list[str]) -> str:
        if not rows:
            return '<p class="empty">Aucune donnée.</p>'
        header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    def top(rows: list[dict[str, object]], key: str = "average_number") -> list[dict[str, object]]:
        return sorted(rows, key=lambda row: (-float(row.get(key, 0.0)), str(row.get("card_id", ""))))[:20]

    html_sections: list[str] = []
    for opponent_name, matchup in result["opponents"].items():
        overall = matchup["overall"]
        opponent_label = "Random" if opponent_name == "random" else "Heuristic v007"
        role_statistics = matchup["final_deck_statistics_by_role"]
        role_delta = matchup["final_deck_delta_heuristic_minus_opponent"]
        heuristic_deck = role_statistics["heuristic"]
        opponent_deck = role_statistics["random"]
        delta_rows = [row for row in role_delta["cards"] if row["card_id"] not in BASE_CARD_IDS]
        heuristic_cards = [row for row in heuristic_deck["cards"] if row["card_id"] not in BASE_CARD_IDS]
        opponent_cards = [row for row in opponent_deck["cards"] if row["card_id"] not in BASE_CARD_IDS]
        chart_html = (
            '<div class="chart-row">'
            f'<div class="chart-panel">{_line_svg(heuristic_cards, "Deck final moyen — v008")}</div>'
            f'<div class="chart-panel">{_line_svg(opponent_cards, f"Deck final moyen — {opponent_label}")}</div>'
            '</div><div class="chart-row">'
            f'<div class="chart-panel">{_line_svg(delta_rows, f"Delta deck v008 − {opponent_label}", value_key="delta_average_number", value_label="Delta")}</div>'
            '</div>'
        )
        choice_rows = []
        for category, rows in matchup["choice_deltas_heuristic_minus_opponent"].items():
            choice_rows.extend({
                "category": category,
                "card": row["card"],
                "heuristic_count": row["heuristic_count"],
                "opponent_count": row["random_count"],
                "delta_per_game": row["delta_per_game"],
            } for row in rows[:10])
        behavior = matchup["heuristic_behavior"]
        recruit = {row["card"]: row["count"] for row in behavior["mercenary_purchases"]["recruit_immediate"]}
        long_term = {row["card"]: row["count"] for row in behavior["mercenary_purchases"]["buy_long_term"]}
        mercenary_rows = [
            {
                "mercenaire": card,
                "recrutement_immediat": recruit.get(card, 0),
                "achat_long_terme": long_term.get(card, 0),
                "total": recruit.get(card, 0) + long_term.get(card, 0),
            }
            for card in sorted(set(recruit) | set(long_term))
        ]
        html_sections.append(
            f'<section><h2>v008 contre {html.escape(opponent_label)}</h2>'
            f'<p>{overall["games"]} parties — profil adversaire <code>{html.escape(str(matchup["opponent_profile"] or "Random"))}</code></p>'
            '<div class="metrics">'
            f'<div class="metric"><strong>{overall["heuristic_win_rate"]}%</strong><span>victoires v008</span></div>'
            f'<div class="metric"><strong>{overall["heuristic_wins"]}</strong><span>victoires v008</span></div>'
            f'<div class="metric"><strong>{overall["random_wins"]}</strong><span>victoires adversaire</span></div>'
            f'<div class="metric"><strong>{overall["draws"]}</strong><span>matchs nuls</span></div>'
            f'<div class="metric"><strong>{overall["turns"]["mean"]}</strong><span>tours moyens</span></div>'
            '</div>'
            '<h3>État final des parties</h3>'
            + data_table([{
                "v008_vie_gagnant": overall["winner_health"]["mean"],
                "maitrise_gagnant": overall["winner_mastery"]["mean"],
                "maitrise_perdant": overall["loser_mastery"]["mean"],
                "tours": overall["turns"]["mean"],
            }], ["v008_vie_gagnant", "maitrise_gagnant", "maitrise_perdant", "tours"])
            + '<h3>Decks finaux</h3>'
            + data_table(top(heuristic_deck["cards"]), ["name", "average_number", "presence_rate", "faction"])
            + f'<h3>Delta deck v008 − {html.escape(opponent_label)}</h3>'
            + data_table(sorted(delta_rows, key=lambda row: -abs(float(row["delta_average_number"])))[:20], ["name", "delta_average_number", "delta_presence_rate", "faction"])
            + '<h3>Comportements de v008</h3>'
            + data_table([{
                "conversion_actions": behavior["gain_mastery"]["actions"],
                "parties_avec_conversion": behavior["gain_mastery"]["games_with_action"],
                "actions_par_partie": behavior["gain_mastery"]["actions_per_game"],
                "pass_events": behavior["pass_play"]["events"],
            }], ["conversion_actions", "parties_avec_conversion", "actions_par_partie", "pass_events"])
            + '<h4>Mercenaires utilisés par v008</h4>'
            + data_table(mercenary_rows, ["mercenaire", "recrutement_immediat", "achat_long_terme", "total"])
            + '<h3>Écarts de choix les plus importants</h3>'
            + data_table(sorted(choice_rows, key=lambda row: -abs(float(row["delta_per_game"])))[:30], ["category", "card", "heuristic_count", "opponent_count", "delta_per_game"])
            + '<h3>Graphiques des decks finaux</h3>' + chart_html
            + '</section>'
        )

        card_fields = ["rank", "card_id", "name", "average_number", "presence_rate", "faction", "cost", "central_copy_count"]
        delta_fields = ["rank", "card_id", "name", "winner_average_number", "loser_average_number", "delta_average_number", "winner_presence_rate", "loser_presence_rate", "delta_presence_rate", "faction", "cost", "central_copy_count"]
        _write_csv(output_dir / f"{opponent_name}_final_deck_v008_cards.csv", heuristic_deck["cards"], card_fields)
        _write_csv(output_dir / f"{opponent_name}_final_deck_delta_cards.csv", role_delta["cards"], delta_fields)
        _write_csv(output_dir / f"{opponent_name}_choice_deltas.csv", choice_rows, ["category", "card", "heuristic_count", "opponent_count", "delta_per_game"])

    overall = result["overall"]
    summary_rows = [
        {"adversaire": name, "parties": item["overall"]["games"], "victoires_v008": item["overall"]["heuristic_wins"], "victoires_adversaire": item["overall"]["random_wins"], "nuls": item["overall"]["draws"], "taux_victoire_v008": item["overall"]["heuristic_win_rate"]}
        for name, item in result["opponents"].items()
    ]
    header = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Benchmark Heuristic multi-adversaires</title>
<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;color:#172033}section{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0}table{border-collapse:collapse;margin-bottom:1rem;width:100%}td,th{border-bottom:1px solid #ddd;padding:.3rem .8rem;text-align:left}th{background:#e2e8f0}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem}.metric{background:#f1f5f9;padding:.7rem;border-radius:8px}.metric strong{display:block;font-size:1.4rem}.chart-row{display:flex;gap:1.5rem;align-items:flex-start;margin-bottom:1.5rem;flex-wrap:wrap}.chart-panel{flex:1 1 520px;min-width:0}.chart{max-width:760px;width:100%;height:auto;background:#f8fafc}.empty{color:#64748b}</style></head><body>
<h1>Benchmark HeuristicPlayer multi-adversaires</h1>"""
    header += f"<p>v008 — seed <code>{result['root_seed']}</code> — durée <code>{result['elapsed_seconds']}s</code></p>"
    body = '<section><h2>Résumé global</h2>'
    body += data_table([{
        "parties": overall["games"], "victoires_v008": overall["heuristic_wins"],
        "victoires_adversaires": overall["random_wins"], "nuls": overall["draws"],
        "taux_victoire_v008": overall["heuristic_win_rate"],
    }], ["parties", "victoires_v008", "victoires_adversaires", "nuls", "taux_victoire_v008"])
    body += '<h3>Répartition et résultats par adversaire</h3>' + data_table(summary_rows, ["adversaire", "parties", "victoires_v008", "victoires_adversaire", "nuls", "taux_victoire_v008"])
    body += '</section>' + ''.join(html_sections) + '</body></html>'
    html_path.write_text(header + body, encoding="utf-8")
    return json_path, html_path


def _write_legacy_reports(result: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    html_path = output_dir / "report.html"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    role_statistics = result["final_deck_statistics_by_role"]
    role_delta = result["final_deck_delta_heuristic_minus_random"]
    choice_deltas = result["choice_deltas_heuristic_minus_random"]
    assert isinstance(role_statistics, dict)
    assert isinstance(role_delta, dict)
    card_fields = [
        "rank", "card_id", "name", "average_number", "presence_rate",
        "faction", "cost", "central_copy_count",
    ]
    delta_fields = [
        "rank", "card_id", "name", "winner_average_number", "loser_average_number",
        "delta_average_number", "winner_presence_rate", "loser_presence_rate",
        "delta_presence_rate", "faction", "cost", "central_copy_count",
    ]
    for role in ("heuristic", "random"):
        statistics = role_statistics[role]
        _write_csv(output_dir / f"final_deck_{role}_cards.csv", statistics["cards"], card_fields)
        _write_csv(output_dir / f"final_deck_{role}_factions.csv", statistics["factions"], ["faction", "average_number", "share"])
    _write_csv(output_dir / "final_deck_delta_cards.csv", role_delta["cards"], delta_fields)
    _write_csv(output_dir / "final_deck_delta_factions.csv", role_delta["factions"], [
        "faction", "winner_average_number", "loser_average_number", "delta_average_number",
        "winner_share", "loser_share", "delta_share",
    ])
    deck_size_rows = [
        {"group": role, **summary}
        for role, summary in result["deck_size_by_role"].items()
    ] + [
        {"group": f"heuristic_{group}", **summary}
        for group, summary in result["heuristic_deck_size_by_result"].items()
    ]
    _write_csv(
        output_dir / "deck_size_summary.csv",
        deck_size_rows,
        ["group", "count", "mean", "min", "max"],
    )
    behavior_by_result = result["heuristic_behavior_by_result"]
    behavior_rows = []
    mastery_rows = []
    mercenary_rows = []
    for result_group, stats in behavior_by_result.items():
        mastery = stats["gain_mastery"]
        mastery_rows.append({
            "result_group": result_group,
            "games": stats["games"],
            "actions": mastery["actions"],
            "games_with_action": mastery["games_with_action"],
            "game_rate_percent": mastery["game_rate_percent"],
            "actions_per_game": mastery["actions_per_game"],
        })
        for event in stats["pass_play"]["events_detail"]:
            behavior_rows.append({
                "result_group": result_group,
                "game_index": event["game_index"],
                "seed": event["seed"],
                "cards": ", ".join(event["cards"]),
            })
        recruit = {row["card"]: row["count"] for row in stats["mercenary_purchases"]["recruit_immediate"]}
        long_term = {row["card"]: row["count"] for row in stats["mercenary_purchases"]["buy_long_term"]}
        for card in sorted(set(recruit) | set(long_term)):
            immediate = recruit.get(card, 0)
            deferred = long_term.get(card, 0)
            total = immediate + deferred
            mercenary_rows.append({
                "result_group": result_group,
                "card": card,
                "recruit_immediate": immediate,
                "buy_long_term": deferred,
                "total": total,
                "recruit_share_percent": _percentage(immediate, total),
            })
    _write_csv(output_dir / "heuristic_pass_play_hand_cards.csv", behavior_rows,
               ["result_group", "game_index", "seed", "cards"])
    _write_csv(output_dir / "heuristic_gain_mastery_summary.csv", mastery_rows,
               ["result_group", "games", "actions", "games_with_action", "game_rate_percent", "actions_per_game"])
    _write_csv(output_dir / "heuristic_mercenary_choices.csv", mercenary_rows,
               ["result_group", "card", "recruit_immediate", "buy_long_term", "total", "recruit_share_percent"])
    overall = result["overall"]
    groups = result["heuristic_result_groups"]
    heuristic_deck = role_statistics["heuristic"]
    random_deck = role_statistics["random"]
    heuristic_cards = [row for row in heuristic_deck["cards"] if row["card_id"] not in BASE_CARD_IDS]
    random_cards = [row for row in random_deck["cards"] if row["card_id"] not in BASE_CARD_IDS]
    delta_cards = [row for row in role_delta["cards"] if row["card_id"] not in BASE_CARD_IDS]
    role_chart_sections = [
        '<div class="chart-row">'
        f'<div class="chart-panel">{_line_svg(heuristic_cards, "Deck final moyen — Heuristic")}</div>'
        f'<div class="chart-panel">{_line_svg(random_cards, "Deck final moyen — Random")}</div>'
        '</div>',
        f'<div class="chart-row"><div class="chart-panel">{_line_svg(delta_cards, "Delta final Heuristic − Random", value_key="delta_average_number", value_label="Delta")}</div></div>',
        '<div class="chart-row">'
        f'<div class="chart-panel"><h3>Deck final moyen — Heuristic</h3>{_pie_svg(heuristic_deck["factions"])}</div>'
        f'<div class="chart-panel"><h3>Deck final moyen — Random</h3>{_pie_svg(random_deck["factions"])}</div>'
        '</div>',
    ]
    for count, rows in heuristic_deck["cards_by_copy_count"].items():
        random_rows = random_deck["cards_by_copy_count"].get(count, [])
        grouped_delta = [
            row for row in role_delta["cards_by_copy_count"].get(count, [])
            if row["card_id"] not in BASE_CARD_IDS
        ]
        role_chart_sections.append(
            f'<div class="chart-row"><div class="chart-panel">{_line_svg(rows, f"Heuristic — cartes en ×{count}")}</div>'
            f'<div class="chart-panel">{_line_svg(random_rows, f"Random — cartes en ×{count}")}</div></div>'
        )
        role_chart_sections.append(
            f'<div class="chart-row"><div class="chart-panel">{_line_svg(grouped_delta, f"Delta — cartes en ×{count}", value_key="delta_average_number", value_label="Delta")}</div></div>'
        )
    def _top_deck_rows(rows: list[dict[str, object]], limit: int = 20) -> list[dict[str, object]]:
        return sorted(rows, key=lambda row: (-float(row.get("average_number", 0.0)), str(row["card_id"])))[:limit]

    def _top_delta_rows(rows: list[dict[str, object]], limit: int = 20) -> list[dict[str, object]]:
        return sorted(rows, key=lambda row: (-abs(float(row.get("delta_average_number", 0.0))), str(row["card_id"])))[:limit]

    def _data_table(rows: list[dict[str, object]], fields: list[str]) -> str:
        header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    def _behavior_tables(label: str, stats: dict[str, object]) -> str:
        pass_play = stats["pass_play"]
        mastery = stats["gain_mastery"]
        recruit = {row["card"]: row["count"] for row in stats["mercenary_purchases"]["recruit_immediate"]}
        long_term = {row["card"]: row["count"] for row in stats["mercenary_purchases"]["buy_long_term"]}
        mercenary = []
        for card in sorted(set(recruit) | set(long_term)):
            immediate = recruit.get(card, 0)
            deferred = long_term.get(card, 0)
            total = immediate + deferred
            mercenary.append({
                "card": card,
                "recruit_immediate": immediate,
                "buy_long_term": deferred,
                "total": total,
                "recruit_share_percent": _percentage(immediate, total),
            })
        return (
            f"<h3>{html.escape(label)}</h3>"
            + _data_table([{
                "pass_events": pass_play["events"],
                "games_with_remaining_hand": pass_play["games_with_remaining_hand"],
                "game_rate_percent": pass_play["game_rate_percent"],
                "cards_remaining_total": pass_play["cards_remaining_total"],
            }], ["pass_events", "games_with_remaining_hand", "game_rate_percent", "cards_remaining_total"])
            + _data_table(pass_play["cards"], ["card", "count"])
            + _data_table([mastery], ["actions", "games_with_action", "game_rate_percent", "actions_per_game"])
            + _data_table(mercenary, ["card", "recruit_immediate", "buy_long_term", "total", "recruit_share_percent"])
        )

    deck_size_rows_html = [
        {"groupe": role, **summary}
        for role, summary in result["deck_size_by_role"].items()
    ] + [
        {"groupe": f"Heuristic — {group}", **summary}
        for group, summary in result["heuristic_deck_size_by_result"].items()
    ]
    deck_size_tables = (
        "<h3>Taille des decks finaux</h3>"
        + _data_table(deck_size_rows_html, ["groupe", "count", "mean", "min", "max"])
    )
    deck_tables = (
        deck_size_tables
        +
        '<h3>Top cartes du deck final moyen — Heuristic</h3>'
        + _data_table(_top_deck_rows(heuristic_deck["cards"]), ["name", "average_number", "presence_rate", "faction"])
        + '<h3>Top cartes du deck final moyen — Random</h3>'
        + _data_table(_top_deck_rows(random_deck["cards"]), ["name", "average_number", "presence_rate", "faction"])
        + '<h3>Plus gros écarts de deck final — Heuristic − Random</h3>'
        + _data_table(_top_delta_rows(role_delta["cards"]), ["name", "delta_average_number", "delta_presence_rate", "faction"])
    )
    choice_tables = "".join(
        f'<h3>Écarts de choix — {html.escape(category)}</h3>'
        + _data_table(rows[:20], ["card", "heuristic_count", "random_count", "delta_per_game"])
        for category, rows in choice_deltas.items()
    )
    header = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Benchmark Heuristic</title>
<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;color:#172033}section{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:1rem 0}table{border-collapse:collapse;margin-bottom:1rem;width:100%}td,th{border-bottom:1px solid #ddd;padding:.3rem .8rem;text-align:left}th{background:#e2e8f0}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem}.metric{background:#f1f5f9;padding:.7rem;border-radius:8px}.metric strong{display:block;font-size:1.4rem}.chart-row{display:flex;gap:1.5rem;align-items:flex-start;margin-bottom:1.5rem;flex-wrap:wrap}.chart-panel{flex:1 1 520px;min-width:0}.chart{max-width:760px;width:100%;height:auto;background:#f8fafc}.empty{color:#64748b}</style></head><body>
<h1>Benchmark HeuristicPlayer vs RandomPlayer</h1>
"""
    header += f"<p>Seed <code>{result['root_seed']}</code> — profil <code>{html.escape(str(result['profile']))}</code> — durée <code>{result['elapsed_seconds']}s</code></p>\n"
    header += """<section><h2>Résultat global</h2><div class="metrics">"""
    metrics = "".join(
            f'<div class="metric"><strong>{html.escape(str(value))}</strong><span>{html.escape(key)}</span></div>'
            for key, value in overall.items()
        )
    body = """</div></section>
<section><h2>État final</h2><pre>"""
    body += html.escape(json.dumps(groups, indent=2, ensure_ascii=False)) + """</pre></section>
<section><h2>Choix par rôle</h2>"""
    body += _grouped_tables(result["cards_by_role"])
    body += """</section><section><h2>Choix selon le résultat de l’Heuristic</h2>"""
    body += "".join(
        _table(name, sections) for name, sections in result["cards_by_heuristic_result"].items()
    )
    body += """</section><section><h2>Comportements spécifiques à l’Heuristic</h2>"""
    body += _behavior_tables("Tous les résultats", result["heuristic_behavior"])
    body += "".join(
        _behavior_tables(name, stats)
        for name, stats in result["heuristic_behavior_by_result"].items()
    )
    body += """</section><section><h2>Decks finaux</h2>""" + deck_tables
    body += "<h2>Écarts de choix Heuristic − Random</h2>" + choice_tables
    body += "<h2>Graphiques des decks finaux</h2>" + "".join(role_chart_sections)
    body += """</section></body></html>"""
    html_path.write_text(header + metrics + body, encoding="utf-8")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--profile",
        type=str,
        default="configs/heuristic_profiles/v008.yaml",
        help="YAML profile for HeuristicPlayer (default: v008)",
    )
    parser.add_argument(
        "--opponent-profile",
        type=str,
        default="configs/heuristic_profiles/v007.yaml",
        help="YAML profile used for the 50%% Heuristic v007 matchups",
    )
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--top-cards", type=int, default=15)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "analysis" / "heuristic_vs_random",
    )
    args = parser.parse_args()
    result = run_benchmark(
        duration_seconds=args.duration_seconds,
        games=args.games,
        seed=args.seed,
        profile_path=args.profile,
        opponent_profile_path=args.opponent_profile,
        max_actions=args.max_actions,
        max_turns=args.max_turns,
        strict=args.strict,
        top_cards=args.top_cards,
    )
    json_path, html_path = write_reports(result, args.output_dir)
    overall = result["overall"]
    print(f"seed={result['root_seed']}")
    print(
        f"attempted={result['attempted']} completed={overall['games']} "
        f"heuristic_wins={overall['heuristic_wins']} ({overall['heuristic_win_rate']:.2f}%) "
        f"random_wins={overall['random_wins']} draws={overall['draws']} errors={overall['errors']}"
    )
    for opponent_name, matchup in result["opponents"].items():
        stats = matchup["overall"]
        print(
            f"opponent={opponent_name} games={stats['games']} "
            f"v008_wins={stats['heuristic_wins']} ({stats['heuristic_win_rate']:.2f}%) "
            f"opponent_wins={stats['random_wins']} draws={stats['draws']}"
        )
    print(f"results={json_path}")
    print(f"report={html_path}")


if __name__ == "__main__":
    main()
