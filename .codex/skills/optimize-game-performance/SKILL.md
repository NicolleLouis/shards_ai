---
name: optimize-game-performance
description: Optimize the Shards AI engine through reproducible benchmarks, profiling, targeted code changes, and before/after validation. Use when the user asks to speed up simulations, benchmarks, the engine, a RandomPlayer, a heuristic player, or an orchestrator, optionally with a workload constraint such as a player type, number of parties, or duration.
---

# Optimize Game Performance

Improve runtime performance while preserving the documented Shards rules and current feature
scope. Every accepted optimization should have a comparable baseline, profiler evidence or a clear
hot-path rationale, passing functional tests, and a faster final benchmark under the same workload.
The validation may be run by the user outside Codex. When the user reports that it was run, accept
that external validation as the current evidence and avoid relaunching the same expensive campaign
unless its result is missing or contradictory.
Run optimization in iterative passes: after an accepted optimization, use the resulting code as
the baseline for another pass. Stop at the first pass whose robust improvement is below 2% or whose
benchmark is flat/slower; report the cumulative gain and the stopping pass.

## Repository guardrails

- Read `doc/Current state/` and the applicable rules documentation before changing behavior.
- Do not implement rules or features that are not in the current state.
- Do not add caches, multiprocessing, or multithreading without explicit user approval.
- Do not accept a patch that makes the controlled benchmark slower.
- Keep rules, players, and orchestration separated.

## Workflow

### 1. Establish the workload

Read the user's extra constraint before acting. It may specify only V0 games, only `RandomPlayer`,
only a heuristic player, a fixed number of parties, or a fixed runtime duration. If no workload is
specified, use the repository's existing benchmark and default workload.

Choose one controlled mode:

- **N parties:** preferred for deterministic comparisons; use the same count, seeds, players and
  action policy before and after the change.
- **N seconds:** use the same duration, warm-up policy and measurement loop before and after; report
  throughput and note that this mode is more sensitive to run-to-run noise.

If no suitable benchmark exists, generate or adapt one that represents the requested workload. Keep
benchmark code outside the engine's rule logic and do not change the workload after measuring the
baseline.

### 2. Capture the baseline

Run the benchmark before editing production code. Record the exact command, workload, seed policy,
player types, runtime when relevant, elapsed time or throughput, and correctness indicators such as
completed games and winners.

For noisy measurements, run at least three comparable repetitions and use the median. The benchmark
command and measurement method must remain identical after the change. Never optimize from intuition
without a measured baseline.

For an iterative campaign, the baseline for pass `n+1` is the accepted result of pass `n`. Keep the
workload, seeds, correctness checks and measurement method identical across all passes so cumulative
improvement remains attributable.

### 3. Profile representative parties

Profile one or more representative parties, or the smallest workload that exercises the requested
constraint. Prefer the least invasive available profiler, usually:

```bash
poetry run python -m cProfile -s cumulative path/to/profile_or_benchmark.py
```

Use a temporary output outside tracked documentation when a profiler produces a file. Inspect
cumulative time, call counts, and hot paths. Distinguish engine time from player/orchestrator time
when the workload includes players.

Connect profiler findings to the benchmark score; do not optimize an isolated profiler artifact.

### 4. Inspect rules and code boundaries

Before changing a hot path, inspect `doc/Current state/`, the relevant rules document in `doc/`,
the engine/player/runner/benchmark code, and tests covering affected transitions.

Performance changes may alter data structures, allocations, loops, imports, or call boundaries, but
must preserve legal actions, card movement, phases, random reproducibility, and all documented
behavior.

### 5. Implement a targeted optimization

Use `apply_patch` for code changes. Keep each patch narrow enough to attribute its impact. Do not
introduce a cache, multiprocessing, multithreading, JIT, or similar execution model without the
user's explicit approval.

Run focused tests after each meaningful change. If the change modifies observable rules or APIs,
stop and request direction rather than presenting it as a performance optimization.

### 6. Re-run validation and benchmark

Run the full relevant test suite, then repeat the exact baseline benchmark with the same workload.
For a throughput score, higher is better; for elapsed time, lower is better. Report both absolute
values and the percentage change:

```text
absolute delta = final score - baseline score
improvement %  = (final score - baseline score) / baseline score * 100
```

For elapsed time, invert the comparison so a reduction is an improvement.

Accept the patch only when functional tests pass, correctness remains equivalent, and final
performance is better than the baseline. If the result is flat or slower, remove or revert only the
changes made during this attempt, without destructive repository-wide commands, and report the
rejected result.

### 6a. Decide whether to continue iterating

After each accepted pass, immediately start another profiling pass using the new implementation as
the baseline. A pass is significant only when its robust benchmark improvement is at least 2%
relative to its immediate baseline. Use the median of at least three comparable runs, or a fixed-party
benchmark with an equivalent repeatability check when runtime noise is low.

- **Gain >= 2%:** accept the pass and continue with another iteration.
- **Positive gain < 2%:** the change may remain accepted if correctness is preserved, but stop the
  iterative optimization campaign because the marginal gain is not significant.
- **Zero or negative gain:** reject/revert only the current pass and stop the campaign.

Do not change the benchmark, seeds, player policies, or termination rules to manufacture a 2% gain.

### 7. Report the result

Summarize:

- initial score and exact workload;
- profiler finding or hot path addressed;
- action taken and files changed;
- final score;
- absolute delta and percentage improvement or regression;
- tests and validation commands;
- whether the patch was accepted or rejected.

If an accepted optimization changes a component's documented behavior or performance
characteristics, update `doc/Current state/`. Do not rewrite completed architecture history merely
to reflect the optimization.

## Additional rules

- Never trade rule correctness for speed.
- Never compare different seeds, player policies, party counts, termination conditions, or output
  modes and call the result an optimization.
- Never include profiler output, logs, or generated data in `doc/`.
- Never hide a slower result by changing the benchmark after the baseline.
- If the benchmark is too noisy for an honest conclusion, repeat with a more stable workload and
  say so explicitly.
- When several passes are performed, report a pass table containing each immediate baseline,
  resulting score, percentage gain, cumulative gain, and the reason the campaign stopped.
