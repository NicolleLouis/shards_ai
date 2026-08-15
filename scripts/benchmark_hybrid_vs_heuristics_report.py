#!/usr/bin/env python3
"""Detailed Hybrid deckbuilding analysis against Heuristic V7 and V8."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from shards_ai.ai import HeuristicPlayer, build_hybrid_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.analysis.campaign import _line_svg, build_delta_statistics
from shards_ai.game import (
    BuyCard,
    Game,
    GameRandom,
    GameRunner,
    GameStatus,
    GainMastery,
    PlayerId,
    RecruitFreeCard,
    RecruitMercenary,
)


def _summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "min": min(values),
        "max": max(values),
    }


def _player_cards(state, player_id: PlayerId) -> Counter[str]:
    player = state.players[player_id]
    zones = (*player.hand, *player.draw_pile, *player.discard_pile, *player.play_zone)
    return Counter(card.definition.card_id for card in zones)


def _deck_size(state, player_id: PlayerId) -> int:
    player = state.players[player_id]
    return sum(len(zone) for zone in (
        player.hand, player.draw_pile, player.discard_pile,
        player.play_zone, player.champions,
    ))


def _role_stats() -> dict[str, object]:
    return {
        "games": 0, "wins": 0, "losses": 0, "draws": 0,
        "mastery": [], "deck_size": [], "turns": [],
        "actions": Counter(), "cards": Counter(), "mercenary_actions": Counter(),
    }


def _play_match(hybrid_profile: str, heuristic_profile: str, games: int, seed: int) -> dict[str, object]:
    hybrid = _role_stats()
    heuristic = _role_stats()
    final_games: list[dict[str, object]] = []
    hybrid_decks: list[dict[str, object]] = []
    heuristic_decks: list[dict[str, object]] = []
    heuristic_weights = load_profile(heuristic_profile)

    for index in range(games):
        root_rng = GameRandom(seed + index)
        game = Game.new(seed=seed + index, rng=root_rng.derive("engine"))
        hybrid_id = PlayerId.PLAYER_1 if index % 2 == 0 else PlayerId.PLAYER_2
        heuristic_id = hybrid_id.opponent
        players = {
            hybrid_id: build_hybrid_player(
                hybrid_id, game, root_rng.derive("hybrid"), profile=hybrid_profile,
            ),
            heuristic_id: HeuristicPlayer(
                heuristic_id,
                heuristic_weights.weights,
                heuristic_weights.card_acquisition_weights,
                heuristic_weights.constraint_weights,
            ),
        }
        action_counts = {hybrid_id: Counter(), heuristic_id: Counter()}
        mercenary_counts = {hybrid_id: Counter(), heuristic_id: Counter()}

        def observe(_before, action, _after, player_id):
            action_counts[player_id][type(action).__name__] += 1
            if isinstance(action, (BuyCard, RecruitMercenary, RecruitFreeCard)):
                card = _before.river[action.river_slot]
                if card is not None and card.definition.is_mercenary:
                    if isinstance(action, BuyCard):
                        mode = "buy"
                    elif isinstance(action, RecruitMercenary):
                        mode = "recruit"
                    else:
                        mode = "free_recruit"
                    mercenary_counts[player_id][(card.definition.card_id, mode)] += 1

        runner = GameRunner(game, players, max_actions=10000, max_turns=200)
        state = runner.run(
            transition_observer=observe,
            observer_receives_detached_state=False,
            players_receive_detached_observation=False,
        )
        result = "draw" if state.status is GameStatus.DRAW or state.winner is None else (
            "hybrid_win" if state.winner is hybrid_id else "heuristic_win"
        )
        hybrid["games"] += 1
        heuristic["games"] += 1
        if result == "hybrid_win":
            hybrid["wins"] += 1
            heuristic["losses"] += 1
        elif result == "heuristic_win":
            hybrid["losses"] += 1
            heuristic["wins"] += 1
        else:
            hybrid["draws"] += 1
            heuristic["draws"] += 1
        for role, player_id, stats in (
            ("hybrid", hybrid_id, hybrid), ("heuristic", heuristic_id, heuristic),
        ):
            player = state.players[player_id]
            stats["mastery"].append(player.mastery)
            stats["deck_size"].append(_deck_size(state, player_id))
            stats["turns"].append(state.turn_number)
            stats["actions"].update(action_counts[player_id])
            stats["cards"].update(_player_cards(state, player_id))
            stats["mercenary_actions"].update(mercenary_counts[player_id])
        hybrid_decks.append({"cards": dict(_player_cards(state, hybrid_id))})
        heuristic_decks.append({"cards": dict(_player_cards(state, heuristic_id))})
        final_games.append({
            "game": index,
            "seed": seed + index,
            "result": result,
            "turns": state.turn_number,
            "hybrid": {"mastery": state.players[hybrid_id].mastery, "deck_size": _deck_size(state, hybrid_id)},
            "heuristic": {"mastery": state.players[heuristic_id].mastery, "deck_size": _deck_size(state, heuristic_id)},
            "cards": {
                "hybrid": dict(_player_cards(state, hybrid_id)),
                "heuristic": dict(_player_cards(state, heuristic_id)),
            },
        })

    def finish(stats: dict[str, object]) -> dict[str, object]:
        games_count = int(stats["games"])
        mercenary_rows = {}
        card_ids = sorted(card_id for card_id, _mode in stats["mercenary_actions"])
        for card_id in card_ids:
            buy = stats["mercenary_actions"][(card_id, "buy")]
            recruit = stats["mercenary_actions"][(card_id, "recruit")]
            free_recruit = stats["mercenary_actions"][(card_id, "free_recruit")]
            recruited = recruit + free_recruit
            total = buy + recruited
            mercenary_rows[card_id] = {
                "buy": buy,
                "recruit": recruit,
                "free_recruit": free_recruit,
                "total": total,
                "buy_frequency": round(buy / total, 4) if total else 0.0,
                "recruit_frequency": round(recruited / total, 4) if total else 0.0,
                "acquisitions_per_game": round(total / games_count, 3) if games_count else 0.0,
            }
        return {
            "games": games_count,
            "wins": stats["wins"], "losses": stats["losses"], "draws": stats["draws"],
            "win_rate": round(stats["wins"] / games_count, 4) if games_count else 0.0,
            "mastery": _summary(stats["mastery"]),
            "deck_size": _summary(stats["deck_size"]),
            "turns": _summary(stats["turns"]),
            "actions": dict(sorted(stats["actions"].items())),
            "actions_per_game": {k: round(v / games_count, 3) for k, v in sorted(stats["actions"].items())},
            "cards": dict(stats["cards"].most_common()),
            "mercenary_acquisition": mercenary_rows,
        }

    delta_cards, _delta_factions, delta_by_copy_count = build_delta_statistics(
        hybrid_decks, heuristic_decks,
    )
    return {
        "hybrid_profile": hybrid_profile,
        "heuristic_profile": heuristic_profile,
        "games": games,
        "hybrid": finish(hybrid),
        "heuristic": finish(heuristic),
        "deck_delta": delta_cards,
        "deck_delta_by_copy_count": delta_by_copy_count,
        "games_detail": final_games,
    }


def _table(title: str, values: dict[str, object]) -> str:
    rows = [f"<h3>{html.escape(title)}</h3><table><tr><th>Indicateur</th><th>Valeur</th></tr>"]
    for key, value in values.items():
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>")
    return "".join(rows) + "</table>"


def _cards(title: str, cards: dict[str, int]) -> str:
    rows = [f"<h3>{html.escape(title)}</h3><table><tr><th>Carte</th><th>Occurrences</th></tr>"]
    rows.extend(f"<tr><td>{html.escape(card)}</td><td>{count}</td></tr>" for card, count in cards.items())
    return "".join(rows) + "</table>"


def _mercenary_table(title: str, rows: dict[str, dict[str, object]]) -> str:
    fields = (
        "Carte", "Achats", "Recrutements", "dont gratuits", "Total",
        "Fréq. achat", "Fréq. recrutement", "Acquisitions/partie",
    )
    body = [f"<h3>{html.escape(title)}</h3><table><tr>{''.join(f'<th>{field}</th>' for field in fields)}</tr>"]
    for card_id, row in rows.items():
        body.append(
            "<tr>"
            f"<td>{html.escape(card_id)}</td><td>{row['buy']}</td>"
            f"<td>{row['recruit']}</td><td>{row['free_recruit']}</td>"
            f"<td>{row['total']}</td><td>{row['buy_frequency']:.1%}</td>"
            f"<td>{row['recruit_frequency']:.1%}</td>"
            f"<td>{row['acquisitions_per_game']:.3f}</td>"
            "</tr>"
        )
    return "".join(body) + "</table>"


def write_report(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    sections = ["<style>body{font-family:sans-serif;max-width:1200px;margin:2em auto}table{border-collapse:collapse;margin-bottom:2em}td,th{border:1px solid #ccc;padding:.4em 1em;text-align:left}h2{border-bottom:2px solid #555;padding-top:1em}</style>"]
    sections.append(
        f"<h1>Analyse {html.escape(payload['hybrid_profile'])}</h1>"
        f"<p>{payload['games_per_opponent']} parties par adversaire</p>"
    )
    for opponent, result in payload["matchups"].items():
        sections.append(f"<h2>Contre {html.escape(opponent)}</h2>")
        sections.append(_table("Résultat et fin de partie — Hybrid", {k: result["hybrid"][k] for k in ("games", "wins", "losses", "draws", "win_rate", "mastery", "deck_size", "turns")}))
        sections.append(_table("Résultat et fin de partie — Heuristic", {k: result["heuristic"][k] for k in ("games", "wins", "losses", "draws", "win_rate", "mastery", "deck_size", "turns")}))
        sections.append(_table("Actions Hybrid", {"totals": result["hybrid"]["actions"], "par partie": result["hybrid"]["actions_per_game"]}))
        sections.append(_table("Actions Heuristic", {"totals": result["heuristic"]["actions"], "par partie": result["heuristic"]["actions_per_game"]}))
        sections.append(_cards("Cartes finales Hybrid", result["hybrid"]["cards"]))
        sections.append(_cards("Cartes finales Heuristic", result["heuristic"]["cards"]))
        sections.append(_mercenary_table(
            "Achats vs recrutements de mercenaires — Hybrid", result["hybrid"]["mercenary_acquisition"],
        ))
        sections.append(_mercenary_table(
            "Achats vs recrutements de mercenaires — Heuristic", result["heuristic"]["mercenary_acquisition"],
        ))
        sections.append(f"<h2>Différences de composition — Hybrid − {html.escape(opponent)}</h2>")
        sections.append(
            "<p>Delta de quantité moyenne par partie. Une valeur positive signifie que Hybrid "
            "joue davantage cette carte.</p>"
        )
        for copy_count in ("1", "2", "3"):
            rows = result["deck_delta_by_copy_count"].get(copy_count, [])
            sections.append(_line_svg(
                rows,
                f"Delta de quantité moyenne — cartes centrales en ×{copy_count}",
                value_key="delta_average_number",
                value_label="Hybrid − adversaire",
            ))
    output.write_text("<!doctype html><html lang='fr'><meta charset='utf-8'>" + "".join(sections) + "</html>", encoding="utf-8")
    print(f"json={json_path}\nhtml={output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-profile", default="configs/hybrid_profiles/hybrid-v006.yaml")
    parser.add_argument("--games", type=int, default=400, help="Parties par adversaire")
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis/hybrid_v006_vs_heuristics.html"))
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    payload = {
        "schema_version": 1,
        "hybrid_profile": args.hybrid_profile,
        "games_per_opponent": args.games,
        "seed": args.seed,
        "matchups": {
            "heuristic:v007": _play_match(args.hybrid_profile, "configs/heuristic_profiles/v007.yaml", args.games, args.seed),
            "heuristic:v008": _play_match(args.hybrid_profile, "configs/heuristic_profiles/v008.yaml", args.games, args.seed + args.games),
        },
    }
    write_report(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
