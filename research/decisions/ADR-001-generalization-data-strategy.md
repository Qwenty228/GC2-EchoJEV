---
id: ADR-001-generalization-data-strategy
status: proposed
date: 2026-09-21
supersedes: none
---

# ADR-001 — How we get unseen-generator and cross-domain evidence

## Context

`always-ignore/revisedata.md` proposes a three-level evaluation (known-attack,
unseen-generator, cross-dataset) and adds a workstream that does not exist anywhere in
`tasks/`: **generating our own test set with several contemporary TTS / voice-cloning
systems**, holding at least one out of training. It also introduces **In-the-Wild** as an
external set and a **controlled MLAAD subset** in place of full MLAAD.

`always-ignore/concern.md` otherwise ratifies the built plan. Two of its warnings had already
been independently arrived at: §2's pre-modelling leakage audit is T09, and §7's "do not
create a cache larger than the original dataset without noticing" is the measured 1.24 TB vs
6.22 GB constraint now pinned in `configs/ssl_probe.yaml`.

The orchestrator and the researcher agent formed views independently and compared.

## The problem with self-generated fakes

If we synthesise fakes and pair them against bonafide from a different corpus, recording
chain or processing pipeline, a detector reaches near-perfect EER by learning **corpus**, not
**synthesis**. The number looks like a triumph and means nothing. This is `concern.md` §2's
shortcut in its most seductive form, because we would have built the shortcut ourselves.

Evidence this is the default outcome rather than a tail risk, all from our own knowledge base:
- `methods/leakage-audit.md` — a single silence-duration scalar yields 15.1% EER on
  ASVspoof 2019 LA (arXiv:2106.12914). TTS output is cleanly trimmed; natural recordings
  retain leading/trailing silence. Pair them naively and that is the whole classifier.
- `datasets/mlaad.md` — MLAAD's authors **fix** the bonafide source to M-AILABS, drive
  synthesis from the *same transcripts*, and deliberately do not redistribute the originals.
  That is a designed control, not an oversight.
- `datasets/in-the-wild.md` — its authors pair scraped deepfakes with manually sourced
  genuine clips of the same speaker on a **best-effort, unaudited** basis, and say so.

Beyond corpus and silence, **duration distribution** must match: TTS duration is a near
deterministic function of text length and speaking rate, so it is a usable scalar shortcut on
its own.

## Options considered

| Option | Evidence for | Evidence against | Cost |
|---|---|---|---|
| A. Generate our own multi-system TTS set, as written | Fully controlled; certainly unseen | We would re-solve, with less validation, a pairing problem MLAAD already solved; XTTS-v2 is non-commercial-only with no licence path since Coqui shut down | High: generation, licences, pairing design, audit |
| B. Hold out architectures from MLAAD's own catalogue | Bonafide pairing already validated by its authors; `meta.csv` carries model, architecture, language, duration; supports whole-family holdout | Non-commercial licence (CC-BY-NC-4.0); per-architecture duration wildly unbalanced, so naive sampling hides family imbalance | Low: download a pinned subset |
| C. ASVspoof 2019 LA official eval | **Free — already parsed.** Train/dev use A01–A06, eval uses A07–A19: thirteen generators never seen in training | Single corpus, single recording chain (VCTK) | Zero |
| D. In-the-Wild as frozen external eval | 8.16 GB, CC-BY-SA-4.0, no training needed; the expected EER collapse is itself a citable result | **No generator labels and no official split** | Low: download + audit |

## Decision

**Get unseen-generator evidence from C and B before considering A.**

1. **Level 2 is already free.** ASVspoof 2019 LA's official protocol is itself an
   unseen-generator test — A07–A19 at eval, A01–A06 at train/dev. `revisedata.md` frames
   level 2 as requiring new audio; it does not.
2. **Prefer holding out MLAAD architectures over generating our own.** This keeps
   `revisedata.md`'s scientific goal — at least one generator excluded from training —
   while inheriting a bonafide pairing that MLAAD's authors already validated. Pin a named
   subset of 8–12 architectures with ≤2 held out, recorded in a reproducible manifest, and
   pin the MLAAD **version**: it is a moving target, v1 was 63.0 h / 21 models and v11 is
   1153.6 h / 205 models, so "MLAAD" without a version is not a reproducible statement.
3. **In-the-Wild is a frozen, zero-training external set.** Because it has no generator
   labels, per-generator EER on it is impossible — do not promise it. With only 58 speakers
   and no official split, any internal split we construct must be speaker-disjoint and
   justified in writing.
4. **Self-generated TTS is optional and scoped to one system.** If kept: OpenVoice
   (MIT, verified free for commercial and research use), driven from the transcripts and
   speakers of a bonafide corpus we already manifest, so speaker identity, recording chain
   and transcript distribution are inherited rather than invented. Treat it as one more
   MLAAD-style architecture, not a parallel dataset. **Avoid XTTS-v2**: verified
   non-commercial-only under CPML, and Coqui shut down in January 2024, so no licence path
   exists.
5. **Any constructed or subsetted set must pass `scripts/audit_leakage.py` before it is
   permitted to produce a reported number.** Duration, RMS and silence are already audited
   there. Silence and duration *distribution* must be actively equalised across classes, not
   assumed harmless.

## Not scheduled

**SLS** (ACM MM 2024, 10.1145/3664647.3681345) and **MSLA-XLS-R** (Computer Speech & Language,
10.1016/j.csl.2026.102044) are real and citable, but both are paywalled and we could not read
their mechanisms from primary sources. Writing a spec from secondary descriptions would break
this project's cite-or-delete rule in the same way `concern.md` §8 forbids inventing an
auxiliary task because the name sounds right. They stay gated behind T22. If NUS library
access resolves the paywalls, revisit — that is the single action that would change this.

## Reversal condition

If the ASVspoof A07–A19 unseen-attack result and the MLAAD held-out-architecture result
**agree closely**, the marginal value of a self-generated set is near zero and option A should
be dropped entirely. If they **disagree sharply**, that disagreement is itself the interesting
finding and is worth a controlled, single-system generated set to arbitrate — built to the
constraints in §4 above.
