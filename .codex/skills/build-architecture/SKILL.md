---
name: build-architecture
description: Generate a challenged target architecture plan as Markdown and save it under `doc/Architecture/{my_file}.md` in this repository. Use when the user asks to create, draft, build, or update an architecture document, architecture.md, target architecture, technical architecture, or architectural plan before implementation, especially when Codex must first clarify the need, inspect existing code impact, challenge scalability/performance, and record durable decisions.
---

# Build Architecture

## Core Rule

Create a durable architecture document before implementation work starts.

Do not rush to the document. First understand the user need, ask the necessary questions, inspect the existing system, ask integration questions, then challenge scalability and performance. The goal is the best architecture possible, not the fastest plausible plan.

For this repository, always write the result to:

```text
doc/Architecture/{my_file}.md
```

If the user provides `{my_file}`, normalize only unsafe path characters and prefix it with the next
unused three-digit architecture number, unless it already has the correct prefix. If the user does
not provide a filename, derive a short kebab-case filename from the feature or task name and prefix
it with the next unused number. Do not write outside `doc/Architecture/` unless the user explicitly
requests another location. Never reuse an architecture number.

## Workflow

1. Understand the user need.
2. Ask all necessary product and behavior questions until the requested change is clear enough to evaluate.
3. Inspect the relevant existing code, docs, routes, models, tests, migrations, frontend files, permissions, and data flows.
4. Explain the main impact on the existing system and ask follow-up questions about the integration points, compatibility, ownership, and migration path.
5. Challenge the proposal from a scalability and performance perspective. Ask questions that expose expected volume, latency, concurrency, storage growth, query patterns, caching, background processing, backfills, and operational failure modes.
6. Be critical of weak ideas, hidden coupling, avoidable complexity, risky data models, unclear ownership, and premature abstractions. Propose stronger alternatives when needed and explain the tradeoffs.
7. Create `doc/Architecture/` if it does not already exist.
8. Draft or update the target architecture file only after the key decisions are understood or explicitly recorded as assumptions.
9. Keep unresolved questions in `Open Questions`; move answered decisions into `Key Decisions` or the relevant section.
10. Stop after the architecture document unless the user explicitly asks for implementation or a separate plan.

## Clarification Passes

Use staged questioning. Do not collapse all concerns into one generic question dump.

### 1. Need And Behavior

Before reading code, clarify:

- product goal and expected outcome
- users or systems affected
- current pain or limitation
- target behavior and happy path
- variants, permissions, and failure states
- explicit non-goals
- deadline or rollout constraints
- intended architecture filename, when not obvious

### 2. Existing-System Impact

After the initial answers, inspect the codebase. Prefer `rg`, routes, model associations, service/actor boundaries, serializers, OpenAPI specs, React pages/hooks, migrations, and existing tests.

Then summarize what the new system or change touches:

- existing tables, associations, scopes, validations, and lifecycle callbacks
- controllers, policies, API contracts, serializers, and OpenAPI generation
- actors, services, jobs, queues, idempotency, and retries
- frontend routes, components, hooks, generated API client usage, translations, and state flow
- tests and fixtures that will need to change
- compatibility concerns with existing data and workflows

Ask integration questions after this summary. Focus on the "jointure" between old and new behavior: ownership boundaries, source of truth, migration strategy, backward compatibility, data consistency, rollout order, and how existing users should experience the transition.

### 3. Scalability And Performance Challenge

Before writing the final architecture, challenge the design:

- expected row counts, request volume, batch sizes, and growth rate
- hot paths, query shapes, N+1 risks, indexes, locks, and transactions
- synchronous versus asynchronous work
- cacheability, invalidation, stale data tolerance, and precomputation
- concurrency, idempotency, retries, duplicate events, and race conditions
- external service limits, timeouts, and degraded modes
- memory use, payload size, pagination, streaming, and export limits
- observability needed to detect slow paths or operational failures

If a proposed approach is fragile or unlikely to scale, say so directly and offer a better option with tradeoffs.

## Document Format

Use this structure unless the repository or user gives a stronger convention:

```markdown
# <Feature> Architecture

## Objective
Describe the problem, user or system outcome, and success criteria.

## Current State
Summarize the existing behavior, files, data flow, dependencies, and constraints.

## Target Behavior
Describe the intended behavior, main flows, variants, and compatibility requirements.

## Non-Goals
List what must stay out of scope.

## Key Decisions
Record explicit decisions that future implementation agents must follow.

## Open Questions
Track unresolved decisions. Mark whether each question is blocking or non-blocking.

## Proposed Architecture
Explain components, responsibilities, boundaries, data ownership, and important interactions.

## Data Model
Describe new or changed tables, columns, indexes, API contracts, types, state, retention, and migration needs.

## Backend Flow
Cover controllers, services, actors, jobs, policies, validations, idempotency, retries, concurrency, and error handling.

## Frontend Flow
Cover pages, components, hooks, generated API client usage, loading states, empty states, errors, and navigation.

## Authorization And Feature Gates
Describe controller policies, non-controller permission checks, role assignment needs, and rollout gates.

## Observability And Operations
Describe logs, metrics, audit trail, debugging paths, alerts, backfills, and recovery behavior.

## Edge Cases
List empty inputs, invalid inputs, partial failures, stale data, duplicate events, race conditions, and missing configuration.

## Testing Strategy
List the RSpec, frontend, Cypress, fixtures, mocks, and migration-dependent validation expected for implementation.

## Rollout And Migration
Describe deployment order, compatibility shims, feature flags, data migrations, backfills, and rollback considerations.

## Files Expected To Change
List likely file paths or areas. Mark uncertain paths as tentative.
```

## Questioning Standard

Ask questions before writing when critical decisions are missing. Keep questions concrete and tied to implementation risk:

```markdown
I need decisions on these points before writing the architecture:

1. <question> Why it matters: <impact>
2. <question> Why it matters: <impact>
3. <question> Why it matters: <impact>
```

If a reasonable assumption is low risk, write the assumption in `Key Decisions` or `Open Questions` instead of blocking progress.

Prefer several focused rounds of questions over one oversized list. After each answer, update the mental model, inspect or re-inspect code if needed, and ask the next set of questions that materially improves the architecture.

## Critical Review Standard

Challenge ideas when they create hidden complexity, unclear ownership, data duplication, permission bypasses, migration risk, poor performance, or operational ambiguity. Do not present criticism as taste; tie it to concrete failure modes and implementation consequences.

When challenging, use this pattern:

```markdown
I would challenge <idea> because <concrete risk>.
The stronger option is <alternative>.
Tradeoff: <cost or limitation>.
Decision needed: <question>.
```

## Quality Bar

- Make the document usable by a future implementation agent without relying on chat history.
- Prefer repository-specific file paths, classes, routes, and commands over generic architecture prose.
- Call out interactions with migrations, permissions, SPA routing, OpenAPI generation, translations, and tests when relevant.
- Keep prose concise but decision-dense.
- Do not start coding from the architecture document unless the user explicitly asks.
