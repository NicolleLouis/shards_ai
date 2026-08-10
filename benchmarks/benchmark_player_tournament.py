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

from shards_ai.ai import HeuristicPlayer, NeuralPlayer, RandomPlayer
from shards_ai.ai.heuristic_profiles import HeuristicProfile, load_profile
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


PLAYERS = ("Random", "Heuristic 7", "Heuristic 8", "Neuronal 1", "Neuronal 2", "Neuronal 3", "Neuronal 4")
DEFAULT_NEURAL_CHECKPOINTS = {
    "Neuronal 1": Path("configs/neural_profiles/v001.pt"),
    "Neuronal 2": Path("configs/neural_profiles/v002.pt"),
    "Neuronal 3": Path("configs/neural_profiles/v003.pt"),
    "Neuronal 4": Path("configs/neural_profiles/v004.pt"),
}


@dataclass(frozen=True)
class MatchResult:
    row: str
    column: str
    games: int
    wins: int
    losses: int
    draws: int

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
    for index in range(games):
        seed = first_seed + index
        root_rng = GameRandom(seed)
        game = Game.new(seed=seed, rng=root_rng.derive("engine"))
        row_id = PlayerId.PLAYER_1 if index % 2 == 0 else PlayerId.PLAYER_2
        column_id = row_id.opponent
        players = {
            row_id: make_player(row, row_id, root_rng.derive("row")),
            column_id: make_player(column, column_id, root_rng.derive("column")),
        }
        state = GameRunner(game, players, max_actions=max_actions, max_turns=max_turns).run()
        if state.status is GameStatus.DRAW or state.winner is None:
            draws += 1
        elif state.winner is row_id:
            wins += 1
        else:
            losses += 1
    return MatchResult(row, column, games, wins, losses, draws)


def _build_factory(
    heuristic_profiles: dict[str, HeuristicProfile],
    neural_scorers: dict[str, object],
) -> Callable[[str, PlayerId, GameRandom], object]:
    def make_player(name: str, player_id: PlayerId, rng: GameRandom) -> object:
        if name == "Random":
            return RandomPlayer(player_id, rng)
        if name == "Heuristic 7" or name == "Heuristic 8":
            profile = heuristic_profiles[name]
            return HeuristicPlayer(
                player_id,
                profile.weights,
                profile.card_acquisition_weights,
                profile.constraint_weights,
            )
        if name.startswith("Neuronal "):
            return NeuralPlayer(player_id, None, rng, scorer=neural_scorers[name])
        raise ValueError(f"Unknown tournament player: {name}")

    return make_player


def _aggregate(results: dict[tuple[str, str], MatchResult], row: str, column: str) -> MatchResult:
    selected = [result for (result_row, result_column), result in results.items() if result_row == row and result_column == column]
    if not selected:
        raise KeyError((row, column))
    return selected[0]


def _format_cell(value: float, metric: str) -> str:
    return f"{value:+.2%}" if metric == "signed" else f"{value:.2%}"


def build_matrix(results: dict[tuple[str, str], MatchResult], metric: str) -> list[list[str]]:
    """Return rows suitable for the HTML/JSON table, including All totals."""
    matrix: list[list[str]] = []
    for row in ("All", *PLAYERS):
        values = []
        for column in ("All", *PLAYERS):
            if row == column == "All":
                values.append("0.00%")
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


def render_html(matrix: list[list[str]], games: int, seed: int, metric: str) -> str:
    headers = ["Joueur", "All", *PLAYERS]
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for row in matrix:
        cells = [f"<th>{html.escape(row[0])}</th>"]
        for value in row[1:]:
            classes = ""
            if value != "—":
                number = float(value.removesuffix("%").replace("+", ""))
                classes = " positive" if number > 0 else " negative" if number < 0 else " neutral"
            cells.append(f"<td class='{classes.strip()}'>{html.escape(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    metric_label = "avantage signé (victoires − défaites)" if metric == "signed" else "taux de victoire"
    return f"""<!doctype html><html lang='fr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Tournoi Shards AI</title><style>
body{{font-family:system-ui,sans-serif;margin:0;padding:28px;background:#f5f7fb;color:#172033}}main{{max-width:1500px;margin:auto;background:white;padding:24px;border-radius:14px;box-shadow:0 3px 14px #17203318}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d5dbe3;padding:9px;text-align:center;white-space:nowrap}}thead th{{background:#eef1f5}}tbody th{{background:#eef1f5;text-align:left}}td{{background:#fff}}td.positive{{background:#b9e8d2}}td.negative{{background:#f6c0c4}}td.neutral{{background:#edf5f3}}.note{{color:#637083}}
</style></head><body><main><h1>Tournoi Shards AI</h1><p class='note'>{games} parties par match-up · seed {seed} · métrique : {html.escape(metric_label)}</p>
<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class='note'>Une valeur positive signifie que le joueur en ligne gagne davantage que le joueur en colonne. Les diagonales sont volontairement vides.</p></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="Number of games per unordered matchup.")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--metric", choices=("signed", "win-rate"), default="signed")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=GameRunner.DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--heuristic-7", type=Path, default=Path("configs/heuristic_profiles/v007.yaml"))
    parser.add_argument("--heuristic-8", type=Path, default=Path("configs/heuristic_profiles/v008.yaml"))
    neural_argument_names = {}
    for index, (name, path) in enumerate(DEFAULT_NEURAL_CHECKPOINTS.items(), start=1):
        argument_name = f"neural_{index}"
        neural_argument_names[name] = argument_name
        parser.add_argument(f"--{name.lower().replace(' ', '-')}", dest=argument_name, type=Path, default=path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/player_tournament/tournament.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/player_tournament/tournament.html"))
    args = parser.parse_args()
    if args.games <= 0 or args.torch_threads <= 0:
        parser.error("--games and --torch-threads must be positive")

    heuristic_paths = {"Heuristic 7": args.heuristic_7, "Heuristic 8": args.heuristic_8}
    neural_paths = {name: getattr(args, argument_name) for name, argument_name in neural_argument_names.items()}
    for path in (*heuristic_paths.values(), *neural_paths.values()):
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
            results[(column, row)] = MatchResult(column, row, result.games, result.losses, result.wins, result.draws)
            print(f"completed={row} vs {column} games={args.games} result={result.wins}-{result.losses}-{result.draws}", flush=True)

    matrix = build_matrix(results, args.metric)
    payload = {
        "players": list(PLAYERS),
        "games_per_matchup": args.games,
        "seed": args.seed,
        "metric": args.metric,
        "results": {f"{row} vs {column}": asdict(result) for (row, column), result in results.items() if PLAYERS.index(row) < PLAYERS.index(column)},
        "matrix": matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.write_text(render_html(matrix, args.games, args.seed, args.metric), encoding="utf-8")
    print(f"wrote={args.output} html={args.html_output}")


if __name__ == "__main__":
    main()
