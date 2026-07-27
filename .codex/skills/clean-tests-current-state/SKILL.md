---
name: clean-tests-current-state
description: Audit and clean unit tests and `doc/Current state/` documentation so they match the implemented behavior. Use after a feature, refactor, architecture change, or whenever tests or current-state docs may be stale, incomplete, misleading, redundant, or inconsistent.
---

# Clean Tests and Current State

Keep the repository's executable tests and current-state documentation synchronized with the code.
Treat the implemented behavior as the source of truth, while using architecture and rules documents
to identify intended scope and avoid silently dropping required behavior.

## Workflow

### 1. Establish scope

Read the relevant `doc/Current state/` pages, the applicable architecture or rules documents, the
implementation, and the test suite before editing. Identify the feature or behavior under review
and list its public flows, state transitions, error cases, edge cases, and player policies.

For this repository, inspect at minimum:

- `shards_ai/` implementation files affected by the feature;
- `tests/` unit tests and fixtures;
- `doc/Current state/` pages;
- the relevant `doc/Architecture/` and rules reference when behavior is rule-sensitive.

### 2. Build a behavior matrix

Map each implemented behavior to one or more tests. Include both happy paths and rejection paths:

| Behavior | Test(s) | Status | Action |
|---|---|---|---|
| valid transition | test name | covered/missing/stale | keep/add/update/remove |
| invalid input | test name | covered/missing/stale | keep/add/update/remove |

Check setup, state ownership, public actions, legal-action generation, mutations,
randomness/reproducibility, errors, cleanup, termination, and integration/orchestration behavior.

### 3. Clean tests

Keep tests that still protect current behavior. Update tests when the behavior remains valid but its
API, naming, fixtures, or data model changed. Remove a test only when it asserts behavior that is
explicitly retired, duplicated without additional protection, or tied to an obsolete API that has no
compatibility requirement.

Do not delete a failing test merely to make the suite green. First determine whether the failure is
an implementation regression, a stale expectation, or a fixture defect. Preserve deterministic seeds
and test the smallest observable contract rather than incidental internal ordering.

Add focused tests when the matrix shows an important missing behavior. Prefer a small test for one
invariant over broad scenario tests. Cover atomicity for operations that can partially mutate state:
invalid input must leave all relevant state unchanged.

Use `apply_patch` for edits. Do not modify production behavior as part of documentation/test cleanup
unless the audit proves that the implementation violates the documented current scope; if that
happens, report the behavior change explicitly.

### 4. Decide whether coverage is useful

Do not add a coverage dependency or percentage gate by default. A behavior matrix is the primary
coverage method because line coverage cannot prove that rules, state transitions, or rejection paths
are correct.

Use an existing coverage tool only when one of these applies:

- the behavior matrix has ambiguous or unreachable branches;
- a recent refactor may have left dead code;
- a critical module has many untested branches and targeted tests need guidance;
- the user explicitly requests coverage metrics.

If coverage is used, run it with the same test suite, report the command and the actionable missing
branches, and add tests for meaningful behavior rather than chasing a percentage. If no coverage
tool is installed, do not install one solely for this audit; report that the matrix and tests were
used instead.

### 5. Synchronize `Current state`

Update only documentation that describes behavior actually available in the code. Keep it concise
and implementation-specific:

- components and file paths;
- current state model and public actions;
- phase/flow behavior and edge cases;
- player/strategy policies;
- execution limits and reproducibility;
- tests and benchmark facts when measured.

Remove stale V0 claims, obsolete file paths, old test counts, and promises about unimplemented
features. Link related `Current state` pages when useful. Do not rewrite completed architecture
history merely to make it look current; architecture documents record decisions, while
`Current state` records delivered behavior.

### 6. Validate and report

Run the narrowest relevant tests while iterating, then the full suite. Also run type/compile checks or
the repository's standard validation command when available. Inspect the final diff for accidental
deletions, stale test names, contradictory documentation, and generated artifacts.

Report:

- tests kept, updated, added, or removed and why;
- current-state pages changed;
- validation commands and results;
- whether coverage was used and why;
- unresolved gaps or assumptions.

## Quality bar

- Every documented implemented behavior has a corresponding test or an explicit reason it is not
  unit-testable.
- Obsolete behavior is removed from tests and `Current state`, not silently preserved as misleading
  documentation.
- Tests assert observable rules and invariants, not fragile implementation details.
- Reproducibility, invalid actions, state atomicity, and termination limits are covered when present.
- No coverage percentage is treated as a substitute for behavioral review.
