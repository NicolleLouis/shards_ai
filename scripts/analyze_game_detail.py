#!/usr/bin/env python3
"""Generate a readable, step-by-step trace for one deterministic game."""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shards_ai.ai import HeuristicPlayer, HybridPlayer, RandomPlayer, build_hybrid_player
from shards_ai.ai.heuristic_profiles import load_profile
from shards_ai.game.actions import Action
from shards_ai.game import (
    BanishCard,
    BuyCard,
    CardInstance,
    Game,
    GameRandom,
    GameRunner,
    GameStatus,
    PlayerId,
    PlayCard,
    RecruitFreeCard,
    RecruitMercenary,
)


def _card_payload(card: CardInstance) -> dict[str, Any]:
    return {
        "instance_id": card.instance_id,
        "card_id": card.definition.card_id,
        "name": card.definition.name,
    }


def _state_summary(state, player_id: PlayerId) -> dict[str, Any]:
    player = state.players[player_id]
    opponent = state.players[player_id.opponent]
    return {
        "turn": state.turn_number,
        "phase": state.phase.value,
        "active_player": state.active_player.name,
        "player": {
            "health": player.health,
            "mastery": player.mastery,
            "gems": player.gems,
            "power": player.power,
            "hand": [_card_payload(card) for card in player.hand],
            "discard": [_card_payload(card) for card in player.discard_pile],
            "play_zone": [_card_payload(card) for card in player.play_zone],
            "champions": [_card_payload(card) for card in player.champions],
            "draw_pile_size": len(player.draw_pile),
            "pending_banishes": player.pending_banishes,
        },
        "opponent": {
            "health": opponent.health,
            "mastery": opponent.mastery,
            "gems": opponent.gems,
            "power": opponent.power,
            "champions": [_card_payload(card) for card in opponent.champions],
            "hand_size": len(opponent.hand),
            "discard_size": len(opponent.discard_pile),
        },
    }


def _card_name(state, player_id: PlayerId, instance_id: str) -> str:
    player = state.players[player_id]
    cards = [*player.hand, *player.discard_pile, *player.play_zone, *player.champions]
    cards.extend(card for card in state.river if card is not None)
    for card in cards:
        if card.instance_id == instance_id:
            return card.definition.name
    return instance_id


def _action_data(state, player_id: PlayerId, action: Action) -> dict[str, Any]:
    data = {"type": type(action).__name__}
    if is_dataclass(action):
        data.update(asdict(action))
    if isinstance(action, (PlayCard, BanishCard)):
        data["card_name"] = _card_name(state, player_id, action.card_id)
    elif isinstance(action, (BuyCard, RecruitMercenary, RecruitFreeCard)):
        data["card_name"] = next(
            (
                card.definition.name
                for card in state.river
                if card is not None and card.instance_id == action.card_instance_id
            ),
            action.card_instance_id,
        )
    return data


def _feature_data(player: HeuristicPlayer, observation, action: Action) -> dict[str, Any]:
    features = player.features_for_action(observation, action)
    values = asdict(features)
    weights = asdict(player.weights)
    contributions = {
        name: round(float(values[name]) * float(weights.get(name, 0.0)), 6)
        for name in values
        if name != "projection_supported"
        and values[name]
        and weights.get(name, 0.0)
    }
    return {
        "score": round(player.weights.score(features), 6),
        "features": values,
        "contributions": contributions,
    }


def _decision_explanation(
    player: object,
    observation,
    legal_actions: list[Action],
    chosen: Action,
    player_id: PlayerId,
) -> dict[str, Any]:
    if isinstance(player, HybridPlayer):
        diagnostic = player.last_decision
        return {
            "kind": "hybrid",
            "policy_id": diagnostic.policy_id if diagnostic else None,
            "decision_family": diagnostic.decision_family if diagnostic else None,
            "action_type": diagnostic.action_type if diagnostic else type(chosen).__name__,
            "reason": diagnostic.reason if diagnostic else "diagnostic indisponible",
            "chosen_score": diagnostic.chosen_score if diagnostic else None,
            "ranked_alternatives": list(diagnostic.ranked_alternatives) if diagnostic else [],
        }
    if not isinstance(player, HeuristicPlayer):
        return {"kind": "random", "reason": "choix aléatoire pondéré"}

    ranked = []
    for action in legal_actions:
        try:
            details = _feature_data(player, observation, action)
            features = details["features"]
            ranked.append(
                {
                    "action": _action_data(observation, player_id, action),
                    **details,
                    "terminal_win": features["terminal_win"],
                    "lethal": features["lethal"],
                    "selected": action == chosen,
                }
            )
        except Exception as error:  # pragma: no cover - diagnostic fallback
            ranked.append(
                {
                    "action": _action_data(observation, player_id, action),
                    "score": None,
                    "selected": action == chosen,
                    "error": str(error),
                }
            )
    ranked.sort(
        key=lambda item: (
            item.get("terminal_win", 0),
            item.get("lethal", 0),
            item.get("score") if item.get("score") is not None else float("-inf"),
        ),
        reverse=True,
    )
    chosen_entry = next((item for item in ranked if item["selected"]), None)
    if chosen_entry is None:
        reason = "action choisie par la politique heuristique"
    else:
        better = [item for item in ranked if not item["selected"] and item.get("score") is not None]
        reason = "meilleur classement heuristique"
        if better and chosen_entry["score"] == better[0]["score"]:
            reason = "égalité de score : départage déterministe"
    return {
        "kind": "heuristic",
        "reason": reason,
        "chosen_score": chosen_entry["score"] if chosen_entry else None,
        "ranked_alternatives": ranked,
    }


def _purchase_analysis(
    observation,
    player: object,
    legal_actions: list[Action],
    chosen: Action,
    player_id: PlayerId,
) -> dict[str, Any]:
    if isinstance(player, HybridPlayer):
        diagnostic = player.last_decision
        if diagnostic is not None and diagnostic.decision_family == "acquisition":
            return {
                "kind": "neural",
                "alternatives": list(diagnostic.ranked_alternatives),
                "conclusion": {
                    "action": _action_data(observation, player_id, chosen),
                    "score": diagnostic.chosen_score,
                    "reason": diagnostic.reason,
                },
            }
    legal_actions_set = set(legal_actions)

    river = []
    for slot, card in enumerate(observation.river):
        entry: dict[str, Any] = {
            "river_slot": slot,
            "available": card is not None,
            "card": _card_payload(card) if card is not None else None,
            "options": [],
        }
        theoretical_actions: list[Action] = []
        if card is not None:
            theoretical_actions.append(BuyCard(slot, card.instance_id))
            if card.definition.is_mercenary:
                theoretical_actions.append(RecruitMercenary(slot, card.instance_id))
            theoretical_actions.extend(
                action
                for action in legal_actions
                if isinstance(action, RecruitFreeCard) and action.river_slot == slot
            )
        for action in theoretical_actions:
            option: dict[str, Any] = {"action": _action_data(observation, player_id, action)}
            option["selected"] = action == chosen
            option["legal"] = action in legal_actions_set
            if isinstance(player, HeuristicPlayer):
                try:
                    option.update(_feature_data(player, observation, action))
                    option["threshold"] = player.weights.buy_threshold
                    option["threshold_status"] = (
                        "au-dessus"
                        if not isinstance(action, BuyCard)
                        or option["score"] > player.weights.buy_threshold
                        else "sous le seuil"
                    )
                except Exception as error:  # pragma: no cover - diagnostic fallback
                    option["score"] = None
                    option["error"] = str(error)
            else:
                option["score"] = None
                option["reason"] = "score indisponible pour RandomPlayer"
            entry["options"].append(option)
        river.append(entry)

    chosen_option = next(
        (
            option
            for entry in river
            for option in entry["options"]
            if option["selected"]
        ),
        None,
    )
    return {
        "river": river,
        "conclusion": {
            "action": _action_data(observation, player_id, chosen),
            "score": chosen_option.get("score") if chosen_option else None,
            "reason": (
                "option d'achat retenue"
                if chosen_option
                else "aucune carte achetée — décision de fin ou autre action"
            ),
        },
    }


def _render_purchase_analysis(analysis: dict[str, Any]) -> str:
    if analysis.get("kind") == "neural":
        rows = []
        for alternative in analysis.get("alternatives", []):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(alternative.get('action_type', 'Action')))}</td>"
                f"<td>{html.escape(str(alternative.get('action', '')))}</td>"
                f"<td>{html.escape(str(alternative.get('score')))}</td>"
                f"<td>{'oui' if alternative.get('selected') else 'non'}</td></tr>"
            )
        conclusion = analysis.get("conclusion", {})
        return (
            "<div class='purchase-analysis'><h4>Scores neural de l'acquisition</h4>"
            "<table><thead><tr><th>Type</th><th>Action</th><th>Score</th><th>Retenue</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"<p class='conclusion'><strong>Action retenue :</strong> "
            f"{html.escape(str(conclusion.get('action', {}).get('type', 'Action')))}"
            f" — score {html.escape(str(conclusion.get('score')))}</p></div>"
        )
    rows = []
    for entry in analysis.get("river", []):
        card = entry.get("card")
        card_label = card.get("name", "") if card else "Slot vide"
        options = entry.get("options", [])
        if not options:
            rows.append(
                f"<tr><td>{entry['river_slot'] + 1}</td><td>{html.escape(card_label)}</td>"
                "<td colspan='5' class='muted'>aucune option d'achat théorique</td></tr>"
            )
            continue
        for option in options:
            action = option["action"]
            option_label = action.get("type", "Action")
            legal = "Oui" if option.get("legal") else "Non"
            details = json.dumps(
                {"features": option.get("features"), "contributions": option.get("contributions")},
                ensure_ascii=False,
                indent=2,
            )
            rows.append(
                f"<tr><td>{entry['river_slot'] + 1}</td><td>{html.escape(card_label)}</td>"
                f"<td>{html.escape(option_label)}</td><td>{html.escape(str(option.get('score')))}</td>"
                f"<td>{legal}</td><td>{html.escape(option.get('threshold_status', '—'))}</td>"
                f"<td>{'Oui' if option.get('selected') else 'Non'}"
                f"<details><summary>features</summary><pre>{html.escape(details)}</pre></details></td></tr>"
            )
    conclusion = analysis.get("conclusion", {})
    conclusion_action = conclusion.get("action", {})
    conclusion_label = conclusion_action.get("type", "Action")
    if conclusion_action.get("card_name"):
        conclusion_label += f" — {conclusion_action['card_name']}"
    return (
        "<div class='purchase-analysis'><h4>Analyse de la phase achat</h4>"
        "<table><thead><tr><th>Slot</th><th>Carte</th><th>Option</th><th>Score</th><th>Légale</th><th>Seuil</th><th>Retenue</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<p class='conclusion'><strong>Conclusion du moteur :</strong> {html.escape(conclusion_label)}"
        f" — score {html.escape(str(conclusion.get('score')))} — {html.escape(conclusion.get('reason', ''))}</p></div>"
    )


def _render_action(item: dict[str, Any]) -> str:
    action = item["action"]
    label = action.get("type", "Action")
    if action.get("card_name"):
        label += f" — {action['card_name']}"
    explanation = item.get("explanation", {})
    score = explanation.get("chosen_score")
    reason = item.get("explanation", {}).get("reason", "")
    purchase = (
        _render_purchase_analysis(item["purchase_analysis"])
        if item.get("purchase_analysis")
        else ""
    )
    return (
        f"<div class='action player-{item['player_id'].lower()[-1]}'><div class='action-title'><strong>{html.escape(label)}</strong> "
        + (f"<span>score choisi : {html.escape(str(score))}</span>" if score is not None else "")
        + "</div>"
        f"<div class='reason'>{html.escape(item['player_id'])} — {html.escape(reason)}</div>"
        f"{purchase}"
        f"<details><summary>État avant / après</summary><div class='state-grid'><div><b>Avant</b><pre>{html.escape(json.dumps(item['state_before'], ensure_ascii=False, indent=2))}</pre></div>"
        f"<div><b>Après</b><pre>{html.escape(json.dumps(item.get('state_after', {}), ensure_ascii=False, indent=2))}</pre></div></div></details>"
        f"<details><summary>Actions légales et diagnostic</summary><pre>{html.escape(json.dumps(item.get('explanation', {}).get('ranked_alternatives', item['legal_actions']), ensure_ascii=False, indent=2))}</pre></details></div>"
    )


def _write_html(trace: dict[str, Any], path: Path) -> None:
    turns: dict[int, list[dict[str, Any]]] = {}
    for item in trace["events"]:
        turns.setdefault(item["turn"], []).append(item)
    actions = "".join(
        f"<details class='turn turn-player-{items[0]['player_id'].lower()[-1]}' open><summary><strong>Tour {turn}</strong> — {html.escape(items[0]['player_id'])} — {len(items)} action(s)</summary>"
        + "".join(_render_action(item) for item in items)
        + "</details>"
        for turn, items in turns.items()
    )
    document = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analyse détaillée — partie {trace['seed']}</title>
<style>
body {{ margin:0; padding:2rem; background:#f8fafc; color:#172033; font:14px system-ui,sans-serif; }}
main {{ max-width:1200px; margin:auto; }} .card,.action {{ background:white; border:1px solid #dbe3ef; border-radius:10px; padding:1rem; margin:.8rem 0; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.7rem; }}
.metric strong {{ display:block; font-size:1.35rem; }} .muted,.reason {{ color:#64748b; }}
 .action-title {{ display:flex; justify-content:space-between; gap:1rem; }} .state-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }} .turn {{ margin:1rem 0; border-left:6px solid #94a3b8; }} .turn > summary {{ cursor:pointer; padding:.7rem; background:#e8eef7; border-radius:8px; }} .turn-player-1 {{ border-color:#2563eb; }} .turn-player-2 {{ border-color:#9333ea; }} .player-1 {{ border-left:4px solid #60a5fa; }} .player-2 {{ border-left:4px solid #c084fc; }} .legend {{ display:flex; gap:1rem; margin-top:.7rem; }} .legend span {{ padding:.3rem .6rem; border-radius:5px; }} .legend-1 {{ background:#dbeafe; color:#1d4ed8; }} .legend-2 {{ background:#f3e8ff; color:#7e22ce; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:.45rem; border-bottom:1px solid #e2e8f0; vertical-align:top; }} .purchase-analysis {{ margin:.8rem 0; padding:.7rem; background:#f8fafc; border:1px solid #dbe3ef; border-radius:7px; }} .purchase-analysis h4 {{ margin:.2rem 0 .6rem; }} .purchase-analysis pre {{ margin:.4rem 0 0; }} .conclusion {{ margin:.7rem 0 .1rem; }}
pre {{ white-space:pre-wrap; overflow:auto; background:#f1f5f9; padding:.7rem; border-radius:6px; font-size:12px; }}
@media(max-width:800px) {{ .state-grid {{ grid-template-columns:1fr; }} .action-title {{ display:block; }} }}
</style></head><body><main>
<h1>Analyse détaillée d’une partie</h1><p class="muted">Seed {trace['seed']} — les actions et scores sont capturés avant application.</p>
<div class="legend"><span class="legend-1">Joueur 1</span><span class="legend-2">Joueur 2</span></div>
<section class="card summary"><div class="metric"><strong>{html.escape(str(trace['result']['status']))}</strong><span>résultat</span></div>
<div class="metric"><strong>{html.escape(str(trace['result'].get('winner')))}</strong><span>vainqueur</span></div>
<div class="metric"><strong>{len(trace['events'])}</strong><span>actions</span></div></section>
{actions}</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--player1", choices=("heuristic", "hybrid", "random"), default="heuristic")
    parser.add_argument("--player2", choices=("heuristic", "hybrid", "random"), default="random")
    parser.add_argument("--profile", type=Path, default=PROJECT_ROOT / "configs/heuristic_profiles/v008.yaml")
    parser.add_argument("--hybrid-profile", type=str, default="hybrid-v002")
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: artifacts/analysis/game_detail/<seed>",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "artifacts" / "analysis" / "game_detail" / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.profile)
    root_rng = GameRandom(args.seed)
    game = Game.new(seed=args.seed, rng=root_rng.derive("engine"))
    players = {}
    for player_id, kind in ((PlayerId.PLAYER_1, args.player1), (PlayerId.PLAYER_2, args.player2)):
        if kind == "heuristic":
            players[player_id] = HeuristicPlayer(
                player_id,
                profile.weights,
                profile.card_acquisition_weights,
                profile.constraint_weights,
            )
        elif kind == "hybrid":
            players[player_id] = build_hybrid_player(
                player_id,
                game,
                root_rng.derive(f"player-{player_id.value}"),
                profile=args.hybrid_profile,
            )
        else:
            players[player_id] = RandomPlayer(
                player_id, root_rng.derive(f"player-{player_id.value}")
            )

    events: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    def on_decision(observation, legal_actions, chosen, player_id) -> None:
        nonlocal pending
        player = players[player_id]
        pending = {
            "turn": observation.turn_number,
            "phase": observation.phase.value,
            "player_id": player_id.name,
            "state_before": _state_summary(observation, player_id),
            "action": _action_data(observation, player_id, chosen),
            "legal_actions": [_action_data(observation, player_id, action) for action in legal_actions],
            "explanation": _decision_explanation(
                player, observation, list(legal_actions), chosen, player_id
            ),
        }
        if observation.phase.value == "buy":
            pending["purchase_analysis"] = _purchase_analysis(
                observation, player, list(legal_actions), chosen, player_id
            )

    def on_transition(before, action, after, player_id) -> None:
        nonlocal pending
        if pending is None:
            return
        pending["state_after"] = _state_summary(after, player_id)
        events.append(pending)
        pending = None

    runner = GameRunner(game, players, max_actions=args.max_actions, max_turns=args.max_turns)
    state = runner.run(decision_observer=on_decision, transition_observer=on_transition)
    trace = {
        "seed": args.seed,
        "players": {player_id.name: args.player1 if player_id is PlayerId.PLAYER_1 else args.player2 for player_id in PlayerId},
        "profile": str(args.profile),
        "hybrid_profile": args.hybrid_profile if "hybrid" in (args.player1, args.player2) else None,
        "result": {"status": state.status.value, "winner": state.winner.name if state.winner else None},
        "events": events,
    }
    json_path = output_dir / "game.json"
    html_path = output_dir / "report.html"
    json_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(trace, html_path)
    print(f"seed={args.seed}")
    print(f"status={state.status.value} winner={state.winner.name if state.winner else 'none'} actions={len(events)}")
    print(f"json={json_path}")
    print(f"report={html_path}")


if __name__ == "__main__":
    main()
