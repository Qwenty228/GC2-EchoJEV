# `Miscellaneous/research/` — the researcher agent's knowledge base

This folder exists so that **no implementation claim in this repo is unsourced and no
number is unexplained**. It is written and maintained by the *researcher agent* and read
by every implementation agent before they write code.

It has three jobs:

1. **Knowledge** (`knowledge/`) — what the literature and the dataset protocols actually say.
2. **Fact-checks** (`fact-checks/`) — verdicts on specific claims made by implementation agents.
3. **Decisions** (`decisions/`) — which way we went, on what evidence, and what would change our mind.

## Hard rules

- **Cite or delete.** Every factual statement in `knowledge/` carries a URL or a repo path +
  commit. No claim from memory.
- **Literature numbers are context, never comparison.** Any EER copied from a paper is tagged
  `source: literature` and may never appear in the same column as our own runs without that tag.
  Datasets, protocols and training data differ; see the original project brief, §"Reporting and experiment discipline".
- **Separate observation from interpretation.** Notes have a `Findings` section (what the
  source says) and an `Implications for us` section (our reading). Never blend them.
- **Uncertainty is recorded, not resolved by confidence.** Use `confidence: high|medium|low`
  and say what evidence would raise it.
- **Reasoning artifacts belong here.** When a result is surprising, the researcher writes down
  *why we expected otherwise* — that trail is what makes the final report defensible.

## Layout

```text
research/
├── knowledge/
│   ├── index.md            # catalogue: every note, its status and confidence
│   ├── papers/             # one note per paper (see _TEMPLATE.md)
│   ├── datasets/           # protocol facts: splits, metadata fields, official metrics
│   ├── methods/            # LFCC/DCT, EER, layer probing, pooling, MTL — the mechanics
│   └── benchmarks/         # published numbers, each tagged with dataset+protocol+source
├── fact-checks/            # FC-###-<topic>.md — verdict on one implementation claim
└── decisions/              # ADR-###-<topic>.md — a choice, its evidence, its reversal condition
```

## The fact-check loop

Implementation agents do not self-certify. The loop is:

1. An implementation agent finishes a task and writes a **claim block** into its task report:
   what it implemented, what it asserts is correct, and which sources it relied on.
2. The researcher agent opens `fact-checks/FC-###-<topic>.md`, checks the claim against
   `knowledge/` (fetching sources if the note does not exist yet), and reads the code.
3. Verdict is one of `confirmed` / `contradicted` / `unsupported` / `out-of-scope`, with the
   specific file:line and the specific source that settles it.
4. `contradicted` or `unsupported` sends the task back with the verdict attached. It is not a
   suggestion.

### What must be fact-checked

- Any formula implemented in code (LFCC/DCT pipeline, EER computation, attention, loss weighting).
- Any claim about a dataset protocol (which split, which metadata field, which official metric).
- Any claim of the form "X outperforms Y" or "this matches the paper".
- Any reproduced baseline (RawNet2, AASIST, SSL front-ends): checkpoint, commit, config.
- Any performance number that will appear in `reports/final_report.md`.

### What must be *performance-explained*, not just checked

For every benchmarked model the researcher records, in `knowledge/benchmarks/`:
- the number **and** the protocol it was produced under,
- the efficiency cost (params, wall time, RTF) that bought it,
- a plain-language account of *why* this model would be expected to behave this way given
  its inductive bias — written **before** looking at our result where possible, so that a
  surprise is visible as a surprise.

## Templates

Copy `knowledge/_TEMPLATE.md`, `fact-checks/_TEMPLATE.md` or `decisions/_TEMPLATE.md`.
Register every new note in `knowledge/index.md` in the same commit.
