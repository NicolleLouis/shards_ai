#!/usr/bin/env python3
"""Run HeuristicPlayer vs RandomPlayer games and write a win-rate HTML report."""

from __future__ import annotations

import argparse
import hashlib
import html
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shards_ai.ai import HeuristicPlayer, RandomPlayer
from shards_ai.game import Game, GameRandom, GameRunner, GameStatus, PlayerId


@dataclass(slots=True)
class Result:
    root_seed: int
    elapsed_seconds: float
    attempted: int
    completed: int
    heuristic_wins: int
    random_wins: int
    draws: int
    errors: int

    @property
    def heuristic_rate(self) -> float:
        return _rate(self.heuristic_wins, self.completed)

    @property
    def random_rate(self) -> float:
        return _rate(self.random_wins, self.completed)

    @property
    def draw_rate(self) -> float:
        return _rate(self.draws, self.completed)


def _rate(value: int, total: int) -> float:
    return (100.0 * value / total) if total else 0.0


def _part_seed(root_seed: int, game_index: int) -> int:
    payload = f"shards-ai-heuristic-vs-random:{root_seed}:{game_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _runner_for(seed: int, game_index: int, max_actions: int, max_turns: int | None) -> tuple[GameRunner, PlayerId]:
    root_rng = GameRandom(seed)
    game = Game.new(seed=seed, rng=root_rng.derive("engine"))
    heuristic_id = PlayerId.PLAYER_1 if game_index % 2 == 0 else PlayerId.PLAYER_2
    players = {
        player_id: (
            HeuristicPlayer(player_id)
            if player_id is heuristic_id
            else RandomPlayer(player_id, root_rng.derive(f"player-{player_id.value}"))
        )
        for player_id in PlayerId
    }
    return GameRunner(game, players, max_actions=max_actions, max_turns=max_turns), heuristic_id


def run_campaign(
    *,
    duration_seconds: float,
    games: int | None,
    seed: int | None,
    max_actions: int,
    max_turns: int | None,
    strict: bool,
) -> Result:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if games is not None and games <= 0:
        raise ValueError("games must be positive")

    root_seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
    started_at = time.monotonic()
    attempted = completed = heuristic_wins = random_wins = draws = errors = 0

    while games is None or attempted < games:
        if attempted and games is None and time.monotonic() - started_at >= duration_seconds:
            break
        game_seed = _part_seed(root_seed, attempted)
        game_index = attempted
        attempted += 1
        try:
            runner, heuristic_id = _runner_for(game_seed, game_index, max_actions, max_turns)
            state = runner.run()
        except Exception:
            if strict:
                raise
            errors += 1
            continue

        completed += 1
        if state.status is GameStatus.DRAW or state.winner is None:
            draws += 1
        elif state.winner is heuristic_id:
            heuristic_wins += 1
        else:
            random_wins += 1

    return Result(
        root_seed=root_seed,
        elapsed_seconds=time.monotonic() - started_at,
        attempted=attempted,
        completed=completed,
        heuristic_wins=heuristic_wins,
        random_wins=random_wins,
        draws=draws,
        errors=errors,
    )


def _bar(label: str, percentage: float, color: str) -> str:
    width = max(0.0, min(100.0, percentage)) * 5.8
    return (
        f'<div class="bar-row"><div class="bar-label">{html.escape(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}px;'
        f'background:{color}">{percentage:.2f}%</div></div></div>'
    )


def write_report(result: Result, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_report = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HeuristicPlayer vs RandomPlayer</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; padding: 2rem; color: #172033; background: #f8fafc; }}
    main {{ max-width: 900px; margin: auto; }}
    h1 {{ margin-bottom: .35rem; }}
    .subtitle {{ color: #64748b; margin-top: 0; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 14px;
      padding: 1.25rem; margin-top: 1rem; box-shadow: 0 5px 18px #0f172a0d; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: .75rem; }}
    .metric strong {{ display: block; font-size: 1.65rem; }}
    .metric span {{ color: #64748b; font-size: .9rem; }}
    .bar-row {{ display: flex; align-items: center; gap: .75rem; margin: 1rem 0; }}
    .bar-label {{ width: 180px; font-weight: 600; }}
    .bar-track {{ width: 580px; max-width: calc(100vw - 280px); height: 34px;
      background: #e2e8f0; border-radius: 8px; overflow: hidden; }}
    .bar-fill {{ height: 100%; color: white; display: flex; align-items: center;
      padding-left: .65rem; box-sizing: border-box; font-weight: 700; min-width: 0; }}
    .note {{ color: #64748b; font-size: .9rem; line-height: 1.5; }}
    code {{ background: #f1f5f9; padding: .15rem .3rem; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>HeuristicPlayer vs RandomPlayer</h1>
  <p class="subtitle">Taux de victoire sur les parties terminées</p>
  <section class="card metrics">
    <div class="metric"><strong>{result.completed}</strong><span>parties terminées</span></div>
    <div class="metric"><strong>{result.attempted}</strong><span>parties tentées</span></div>
    <div class="metric"><strong>{result.elapsed_seconds:.2f}s</strong><span>durée</span></div>
    <div class="metric"><strong>{result.root_seed}</strong><span>seed racine</span></div>
  </section>
  <section class="card">
    {_bar("HeuristicPlayer", result.heuristic_rate, "#2563eb")}
    {_bar("RandomPlayer", result.random_rate, "#9333ea")}
    {_bar("Parties nulles", result.draw_rate, "#64748b")}
  </section>
  <section class="card note">
    Les rôles sont alternés entre Player 1 et Player 2 à chaque partie. Les pourcentages sont
    calculés sur les parties terminées ; les erreurs ({result.errors}) sont exclues du dénominateur.
    Commande : <code>poetry run python scripts/analyze_heuristic_vs_random.py</code>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_report, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--games", type=int, default=None, help="Override the duration with a fixed number of games.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "analysis"
            / "heuristic_vs_random"
            / "report.html"
        ),
    )
    args = parser.parse_args()
    result = run_campaign(
        duration_seconds=args.duration_seconds,
        games=args.games,
        seed=args.seed,
        max_actions=args.max_actions,
        max_turns=args.max_turns,
        strict=args.strict,
    )
    report_path = write_report(result, args.output)
    print(f"seed={result.root_seed}")
    print(
        f"attempted={result.attempted} completed={result.completed} "
        f"heuristic_wins={result.heuristic_wins} ({result.heuristic_rate:.2f}%) "
        f"random_wins={result.random_wins} ({result.random_rate:.2f}%) "
        f"draws={result.draws} ({result.draw_rate:.2f}%) errors={result.errors}"
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
