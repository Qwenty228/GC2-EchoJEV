# Design — Audio deepfake detection workshop (GC2)

Date: 2026-09-20 · Status: accepted · Scope: scaffold + Stages 0–4

## Problem

Turn the handoff brief (`docs/handoff.md`) into a working repository that a
team of agents can build out without drifting from its scientific discipline. The brief is
detailed and opinionated; the risk is not "what to build" but **unverified implementation** —
a wrong EER convention, a leaked split, or a literature number quoted as if it were ours.

## Decisions

**1. Repo root is the existing GC2 folder**, with the source material filed under `docs/`: the brief becomes `docs/handoff.md` and the architecture render moves to `docs/assets/`.
`docs/assets/architecture-full.png` (1.8 MB) is gitignored; `docs/assets/architecture.jpg` (272 KB,
sips-compressed) is the committed render, embedded in the README.

**2. Structure follows the handoff layout**, with three additions:
- `research/` — the researcher agent's knowledge base (below).
- `tasks/` — orchestration: `TASKS.md`, `CONVENTIONS.md`, `BACKLOG.md`.
- `.claude/agents/` — `implementer` and `researcher` subagent definitions, both Sonnet.

**3. `research/` is the verification layer, not documentation.** Three sub-parts with
different jobs:
- `knowledge/` — notes from primary sources only. Cite or delete. `Findings` (what the
  source says) is structurally separate from `Implications for us` (our reading).
- `fact-checks/` — every implementer ends with a *claim block*: falsifiable statements,
  `file.py:L##` evidence, and what it relied on. The researcher returns
  `confirmed` / `contradicted` / `unsupported` / `out-of-scope`. A task is done only on
  `confirmed`.
- `decisions/` — ADRs with run IDs as evidence and an explicit reversal condition.

The researcher also owns **performance explanation**: for each benchmarked model, the number,
its protocol, its efficiency cost, and a mechanistic expectation written before seeing our
result where possible — so a surprise registers as a surprise rather than being rationalised.

**4. Orchestration is wave-based with disjoint file ownership.** 14 tasks in 5 waves. Within
a wave no two tasks touch the same file, so the wave dispatches in parallel; between waves the
researcher fact-checks. This trades some throughput for the property that a parallel wave can
never produce a merge conflict or a silent interface drift.

**5. Stages 5–10 are backlog, gated on evidence.** The handoff is explicit that the final
architecture must not be optimised before the diagnostic stages. Each backlog item names the
Stage 0–4 result that unlocks it.

**6. Everything runs through `uv`**, Python 3.11, device `auto` resolving cuda → mps → cpu.
Tests run with **no dataset present** — synthetic audio fixtures — so the whole scaffold is
reviewable today, with real-data tests marked `needs_data`.

## Alternatives rejected

- *Stub every module with `NotImplementedError` up front.* Rejected: stubs get shipped. File
  creation belongs to the owning task, with the interface pinned in `TASKS.md` instead.
- *One agent builds everything sequentially.* Rejected: the task graph is genuinely wide at
  waves 1–2, and sequential work makes the fact-check loop arrive too late to be corrective.
- *Fact-check at the end.* Rejected: a wrong EER convention in Wave 1 invalidates every number
  in Waves 3–5. Checking is per-wave.

## Risks

| Risk | Mitigation |
|---|---|
| EER convention wrong → every number wrong | T04 tested against analytic cases; R01 must pin the official ASVspoof definition before T04 is confirmed |
| Dataset shortcut learned instead of synthesis evidence | Stage 0 leakage audit (T09) runs before any modelling and fails loudly |
| Representation comparison confounded by backend | T12/T13 run all six variants through an identical backend, split and seed set |
| Parallel agents colliding | Disjoint file ownership per task; touching an unowned file means stopping and reporting |
| Large files pushed | `docs/assets/architecture-full.png` and all audio/checkpoints/caches gitignored; `git status` check in the per-wave checklist |

## Definition of done (this scope)

Locked uv env; manifest + leakage audit; EDA notebooks running on synthetic data; the A–F
feature ablation grid; linear/MLP baselines and CNN/ResNet baselines with 3 seeds each, all
in the experiment registry; every claim carrying a `confirmed` fact-check; `final_report.md`
sections 1–6 with every number traceable to a `run_id`.
