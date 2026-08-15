"""Run a round-robin tournament and render a matchup win-rate matrix.

By default, the matrix contains the signed advantage of the row player:
``(wins - losses) / games``.  Thus 0% is an even matchup, positive values
favour the row and negative values favour the column.  Use ``--metric
win-rate`` when the usual 0..100% win rate is preferred.

Example::

    PYTHONPATH=. poetry run python benchmarks/benchmark_player_tournament.py \
        --games 100 \
        --output artifacts/player_tournament/tournament.json \
        --html-output artifacts/player_tournament/tournament.html
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer, build_hybrid_player, build_neural_player
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile
from shards_ai.game import GainMastery, Game, GameRandom, GameRunner, GameStatus, PlayerId


PLAYERS = (
    "Random", "Heuristic 8",
    "Hybrid 1", "Hybrid 3", "Hybrid 4", "Hybrid 5", "Hybrid 6",
)
DEFAULT_HYBRID_PROFILES = {
    "Hybrid 1": Path("configs/hybrid_profiles/hybrid-v001.yaml"),
    "Hybrid 3": Path("configs/hybrid_profiles/hybrid-v003.yaml"),
    "Hybrid 4": Path("configs/hybrid_profiles/hybrid-v004.yaml"),
    "Hybrid 5": Path("configs/hybrid_profiles/hybrid-v005.yaml"),
    "Hybrid 6": Path("configs/hybrid_profiles/hybrid-v006.yaml"),
}


@dataclass(frozen=True)
class MatchResult:
    row: str
    column: str
    games: int
    wins: int
    losses: int
    draws: int
    row_gain_mastery_actions: int = 0
    column_gain_mastery_actions: int = 0
    row_boundary_gain_mastery_conversions: int = 0
    column_boundary_gain_mastery_conversions: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def signed_advantage(self) -> float:
        return (self.wins - self.losses) / self.games if self.games else 0.0


def play_match(
    row: str,
    column: str,
    games: int,
    first_seed: int,
    make_player: Callable[[str, PlayerId, GameRandom], object],
    max_actions: int,
    max_turns: int | None,
) -> MatchResult:
    wins = losses = draws = 0
    row_gain_mastery_actions = column_gain_mastery_actions = 0
    row_boundary_conversions = column_boundary_conversions = 0
    for index in range(games):
        seed = first_seed + index
        root_rng = GameRandom(seed)
        game = Game.new(seed=seed, rng=root_rng.derive("engine"))
        row_id = PlayerId.PLAYER_1 if index % 2 == 0 else PlayerId.PLAYER_2
        column_id = row_id.opponent
        players = {
            row_id: make_player(row, row_id, root_rng.derive("row"), game),
            column_id: make_player(column, column_id, root_rng.derive("column"), game),
        }
        def observe_transition(_before, action, _after, player_id):
            nonlocal row_gain_mastery_actions, column_gain_mastery_actions
            if isinstance(action, GainMastery):
                if player_id is row_id:
                    row_gain_mastery_actions += 1
                else:
                    column_gain_mastery_actions += 1

        runner = GameRunner(game, players, max_actions=max_actions, max_turns=max_turns)
        state = runner.run(
            transition_observer=observe_transition,
            observer_receives_detached_state=False,
            players_receive_detached_observation=False,
        )
        row_middleware = runner.players[row_id]
        column_middleware = runner.players[column_id]
        row_boundary_conversions += getattr(row_middleware, "boundary_gain_mastery_conversions", 0)
        column_boundary_conversions += getattr(column_middleware, "boundary_gain_mastery_conversions", 0)
        if state.status is GameStatus.DRAW or state.winner is None:
            draws += 1
        elif state.winner is row_id:
            wins += 1
        else:
            losses += 1
    return MatchResult(
        row,
        column,
        games,
        wins,
        losses,
        draws,
        row_gain_mastery_actions,
        column_gain_mastery_actions,
        row_boundary_conversions,
        column_boundary_conversions,
    )


def _build_factory(
    heuristic_profiles: dict[str, HeuristicProfile],
    neural_scorers: dict[str, object],
) -> Callable[[str, PlayerId, GameRandom], object]:
    def make_player(name: str, player_id: PlayerId, rng: GameRandom, current_game: Game) -> object:
        if name == "Random":
            return RandomPlayer(player_id, rng)
        if name == "Heuristic 8":
            profile = heuristic_profiles[name]
            return HeuristicPlayer(
                player_id,
                profile.weights,
                profile.card_acquisition_weights,
                profile.constraint_weights,
            )
        if name in DEFAULT_HYBRID_PROFILES:
            return build_hybrid_player(
                player_id,
                current_game,
                rng,
                profile=DEFAULT_HYBRID_PROFILES[name],
            )
        if name.startswith("Neuronal "):
            return build_neural_player(
                player_id,
                current_game,
                rng,
                scorer=neural_scorers[name],
            )
        raise ValueError(f"Unknown tournament player: {name}")

    return make_player


def _aggregate(results: dict[tuple[str, str], MatchResult], row: str, column: str) -> MatchResult:
    selected = [result for (result_row, result_column), result in results.items() if result_row == row and result_column == column]
    if not selected:
        raise KeyError((row, column))
    return selected[0]


def _behavior_summary(results: dict[tuple[str, str], MatchResult]) -> dict[str, dict[str, float | int]]:
    """Aggregate actual GainMastery actions and V3 boundary conversions by player."""
    summary = {
        player: {
            "games": 0,
            "gain_mastery_actions": 0,
            "gain_mastery_actions_per_game": 0.0,
            "boundary_gain_mastery_conversions": 0,
            "boundary_conversion_rate": 0.0,
        }
        for player in PLAYERS
    }
    for (row, column), result in results.items():
        if PLAYERS.index(row) >= PLAYERS.index(column):
            continue
        row_summary = summary[row]
        column_summary = summary[column]
        row_summary["games"] += result.games
        column_summary["games"] += result.games
        row_summary["gain_mastery_actions"] += result.row_gain_mastery_actions
        column_summary["gain_mastery_actions"] += result.column_gain_mastery_actions
        row_summary["boundary_gain_mastery_conversions"] += result.row_boundary_gain_mastery_conversions
        column_summary["boundary_gain_mastery_conversions"] += result.column_boundary_gain_mastery_conversions
    for item in summary.values():
        games = int(item["games"])
        if games:
            item["gain_mastery_actions_per_game"] = item["gain_mastery_actions"] / games
            item["boundary_conversion_rate"] = item["boundary_gain_mastery_conversions"] / games
    return summary


def _format_cell(value: float, metric: str) -> str:
    return f"{value:+.2%}" if metric == "signed" else f"{value:.2%}"


def _cell_background(number: float, metric: str) -> str:
    """Return a red-neutral-green background for a rendered percentage."""
    if metric == "signed":
        position = (number + 100.0) / 200.0
    else:
        position = number / 100.0
    position = max(0.0, min(1.0, position))

    red = (239, 111, 121)
    neutral = (243, 245, 247)
    green = (69, 201, 139)
    if position <= 0.5:
        start, end, ratio = red, neutral, position * 2
    else:
        start, end, ratio = neutral, green, (position - 0.5) * 2
    channels = tuple(round(start[index] + (end[index] - start[index]) * ratio) for index in range(3))
    return "rgb(" + ", ".join(str(channel) for channel in channels) + ")"


def order_players_by_global_win_rate(results: dict[tuple[str, str], MatchResult]) -> tuple[str, ...]:
    """Order players from highest to lowest win rate across all matchups."""
    totals = {
        player: (
            sum(result.wins for result in results.values() if result.row == player),
            sum(result.games for result in results.values() if result.row == player),
        )
        for player in PLAYERS
    }
    return tuple(
        sorted(
            PLAYERS,
            key=lambda player: (
                totals[player][0] / totals[player][1] if totals[player][1] else 0.0,
                -PLAYERS.index(player),
            ),
            reverse=True,
        )
    )


def build_matrix(
    results: dict[tuple[str, str], MatchResult],
    metric: str,
    players: tuple[str, ...] = PLAYERS,
) -> list[list[str]]:
    """Return rows suitable for the HTML/JSON table, including All totals."""
    matrix: list[list[str]] = []
    for row in ("All", *players):
        values = []
        for column in ("All", *players):
            if row == column == "All":
                values.append("—")
                continue
            if row == column:
                values.append("—")
                continue
            matches = [
                result for (result_row, result_column), result in results.items()
                if (row == "All" or result_row == row)
                and (column == "All" or result_column == column)
                and result_row != result_column
            ]
            games = sum(result.games for result in matches)
            numerator = sum(
                result.wins - result.losses if metric == "signed" else result.wins
                for result in matches
            )
            values.append(_format_cell(numerator / games if games else 0.0, metric))
        matrix.append([row, *values])
    return matrix


def render_html(
    matrix: list[list[str]],
    games: int,
    seed: int,
    metric: str,
    players: tuple[str, ...] = PLAYERS,
) -> str:
    headers = ["Joueur", "All", *players]
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for row in matrix:
        cells = [f"<th>{html.escape(row[0])}</th>"]
        for column_index, value in enumerate(row[1:], start=1):
            classes = ""
            if value != "—":
                number = float(value.removesuffix("%").replace("+", ""))
                if row[0] == "All" and column_index == 1:
                    classes = " neutral"
                else:
                    comparison = number if metric == "signed" else number - 50.0
                    classes = " positive" if comparison > 0 else " negative" if comparison < 0 else " neutral"
            style = ""
            if value != "—" and not (row[0] == "All" and column_index == 1):
                style = f" style='background-color:{_cell_background(number, metric)}'"
            cells.append(f"<td class='{classes.strip()}'{style}>{html.escape(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    metric_label = "avantage signé (victoires − défaites)" if metric == "signed" else "taux de victoire"
    note = (
        "Une valeur positive signifie que le joueur en ligne gagne davantage que le joueur en colonne."
        if metric == "signed"
        else "Une valeur supérieure à 50% signifie que le joueur en ligne gagne davantage que le joueur en colonne."
    )
    return f"""<!doctype html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Tournoi Shards AI</title><style>
body{{font-family:system-ui,sans-serif;margin:0;padding:28px;background:#f5f7fb;color:#172033}}main{{max-width:1500px;margin:auto;background:white;padding:24px;border-radius:14px;box-shadow:0 3px 14px #17203318}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d5dbe3;padding:9px;text-align:center;white-space:nowrap}}thead th{{background:#eef1f5}}tbody th{{background:#eef1f5;text-align:left}}td{{background:#fff}}td.positive{{background:#b9e8d2}}td.negative{{background:#f6c0c4}}td.neutral{{background:#edf5f3}}.note{{color:#637083}}
</style></head><body><main><h1>Tournoi Shards AI</h1><p class='note'>{games} parties par match-up · seed {seed} · métrique : {html.escape(metric_label)}</p>
<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class='note'>{note} Les diagonales sont volontairement vides.</p></main></body></html>"""


def render_html_from_json(input_path: Path) -> str:
    """Render a previously persisted tournament payload without rerunning games."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    players = tuple(payload["players"])
    return render_html(
        payload["matrix"],
        int(payload["games_per_matchup"]),
        int(payload["seed"]),
        payload["metric"],
        players,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="Number of games per unordered matchup.")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--metric", choices=("signed", "win-rate"), default="signed")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--heuristic-8", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/player_tournament/tournament.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/player_tournament/tournament.html"))
    parser.add_argument(
        "--render-from-json",
        type=Path,
        help="Render an existing tournament JSON without running any games.",
    )
    args = parser.parse_args()
    if args.games <= 0 or args.torch_threads <= 0:
        parser.error("--games and --torch-threads must be positive")

    if args.render_from_json is not None:
        if not args.render_from_json.exists():
            parser.error(f"file not found: {args.render_from_json}")
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_html_from_json(args.render_from_json), encoding="utf-8")
        print(f"rendered={args.html_output} from={args.render_from_json}")
        return

    heuristic_paths = {"Heuristic 8": args.heuristic_8}
    neural_paths = {}
    for path in (*heuristic_paths.values(), *DEFAULT_HYBRID_PROFILES.values()):
        if not path.exists():
            parser.error(f"file not found: {path}")
    torch.set_num_threads(args.torch_threads)
    heuristic_profiles = {name: load_profile(path) for name, path in heuristic_paths.items()}
    neural_scorers = {name: NeuralPlayer.load_scorer(path) for name, path in neural_paths.items()}
    make_player = _build_factory(heuristic_profiles, neural_scorers)

    results: dict[tuple[str, str], MatchResult] = {}
    for row_index, row in enumerate(PLAYERS):
        for column in PLAYERS[row_index + 1:]:
            result = play_match(row, column, args.games, args.seed + len(results) * args.games, make_player, args.max_actions, args.max_turns)
            results[(row, column)] = result
            results[(column, row)] = MatchResult(
                column,
                row,
                result.games,
                result.losses,
                result.wins,
                result.draws,
                result.column_gain_mastery_actions,
                result.row_gain_mastery_actions,
                result.column_boundary_gain_mastery_conversions,
                result.row_boundary_gain_mastery_conversions,
            )
            print(f"completed={row} vs {column} games={args.games} result={result.wins}-{result.losses}-{result.draws}", flush=True)

    player_order = order_players_by_global_win_rate(results)
    matrix = build_matrix(results, args.metric, player_order)
    payload = {
        "players": list(player_order),
        "games_per_matchup": args.games,
        "seed": args.seed,
        "metric": args.metric,
        "results": {f"{row} vs {column}": asdict(result) for (row, column), result in results.items() if PLAYERS.index(row) < PLAYERS.index(column)},
        "matrix": matrix,
        "behavior_by_player": _behavior_summary(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(
        render_html(matrix, args.games, args.seed, args.metric, player_order),
        encoding="utf-8",
    )
    print(f"wrote={args.output} html={args.html_output}")


if __name__ == "__main__":
    main()
