---
id: musan
type: dataset
sources:
  - https://www.openslr.org/17/ (OpenSLR SLR17 resource page — fetched directly: download size, top-level license statement)
  - https://ar5iv.labs.arxiv.org/html/1510.08484 (Snyder, Chen, Povey, "MUSAN: A Music, Speech, and Noise Corpus," arXiv:1510.08484 — fetched directly, quoted below)
confidence: high
last_verified: 2026-09-30
related: []
---

# MUSAN (OpenSLR SLR17) — download size, noise subset, licensing

## Claim in one sentence
MUSAN downloads as a single 11 GB `musan.tar.gz` from OpenSLR under a top-level **CC BY 4.0** attribution, but the underlying technical report states that **per-source licensing actually varies** (Free Sound content is public domain, Sound Bible content is various CC variants, documented per-directory) — the `noise/` subset specifically contains **929 files, ~6 hours total**, split between technical noises (DTMF tones, dialtones, fax noises) and ambient sounds (car idling, thunder, wind, footsteps, paper rustling, rain, animal noises, and crowd noise with indistinct — not intelligible — voices).

## Findings

### Download (OpenSLR SLR17 page, fetched directly)
- Identifier: **SLR17**. Summary: "A corpus of music, speech, and noise."
- Single archive: **`musan.tar.gz`, 11 GB**, with EU/EU/CN mirrors listed.
- **License field on the OpenSLR page itself: "Attribution 4.0 International (CC BY 4.0)."**

### The `noise/` subset (arXiv:1510.08484, fetched directly, quoted)
- **"This portion of the corpus contains 929 files of assorted noises, with a total duration of about 6 hours."**
- Two informal subcategories per the paper's own text:
  - Technical/synthetic noises: DTMF tones, dialtones, fax machine noises.
  - Ambient/environmental sounds: car idling, thunder, wind, footsteps, paper rustling, rain, animal noises.
- The corpus explicitly **excludes recordings with intelligible speech**, but does include crowd noise with indistinct (unintelligible) voices — relevant if this project ever worries about the noise-augmentation set itself leaking speech content into a "noise-only" augmentation.
- No per-file size-on-disk figure for `noise/` alone was found (only file count and duration); the 11 GB figure is for the whole corpus (speech + music + noise combined), not `noise/` in isolation.

### Licensing — a real discrepancy between the distribution page and the technical report
- OpenSLR's page-level license tag is **CC BY 4.0** for the packaged download as a whole.
- The paper itself states the underlying per-file licensing is **not uniform**: "The Free Sound part of the corpus is Public Domain material and the Sound Bible part is CC licensed" (paraphrased from the fetched text; exact CC sub-variant per file was not itemized in the fetched excerpt) — attribution requirements are documented in **per-subdirectory LICENSE files** inside the archive, not summarized as a single blanket license in the paper.
- **For this project's purposes:** treat the OpenSLR-page CC BY 4.0 tag as the operative license for using the packaged corpus as redistributed (that is what OpenSLR itself asserts control over), but do not assume every individual noise clip is CC BY 4.0 specifically if per-file attribution is ever required (e.g. in a published report's data-availability section) — the underlying sources are mixed (public domain + various CC).

## Mechanism / derivation
N/A — corpus-description note, not a formula-derivation note.

## Implications for us
- If `noise/` is used as an augmentation source (e.g. MUSAN-noise mixing for robustness testing), 929 files / ~6 h is the effective size of that augmentation pool — small enough that repeated reuse across many training epochs is expected and should be accounted for if diversity of augmentation is a concern.
- The "no intelligible speech" guarantee (but not "no voices at all," since crowd noise with indistinct voices is included) is worth flagging if this project ever computes a leakage-style audit (`leakage-audit.md`) on any pipeline that uses MUSAN noise for augmentation — the crowd-noise subset is not silence, and is not free of all vocal content.
- Licensing citation for a data-availability section should point to the OpenSLR SLR17 page's CC BY 4.0 statement as the primary claim, with a footnote that the underlying paper documents mixed-source per-file licensing.

## Open questions / what would change this
- The exact CC sub-variant(s) used for the Sound Bible-sourced files were not itemized from the fetched paper excerpt (the tool returned a paraphrase, not the itemized per-source license table if the paper has one) — if a strict per-file license audit is ever needed, the archive's own per-directory LICENSE files (inside `musan.tar.gz`) are the actual primary source, not the paper's prose summary.
- No independent figure for `noise/`'s size-on-disk (as opposed to duration/file-count) was found; if disk-space planning needs an exact number, this would require downloading and measuring the subfolder directly.
