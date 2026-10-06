"""The EDA figure vocabulary for Stage 1 (waveform/spectral) and Stage 2 (feature-space) EDA.

Every function takes an `ax=None` and returns the `Axes` it drew on, so a notebook can
compose several of these into one figure instead of getting one figure per call.

Scientific discipline this module encodes (from the original project brief, Stage 1 / "EDA in
feature space"):

- **Aggregate views by default.** The handoff is explicit that the analysis must not
  cherry-pick one impressive real/fake pair; distributions must be aggregated. Nothing here
  takes a single waveform-pair-comparison shortcut: `plot_mean_spectrum_with_ci` plots a mean
  and a confidence band over many examples, `plot_class_distributions` plots a full
  distribution per class, never a single scalar per class.
- **Generator-conditioned overlays matter.** A statistical cue that separates real speech from
  one TTS/VC generator but reverses sign for another generator is most plausibly a property of
  that generator's synthesis artifacts, not evidence of "fake speech" in general. Averaging
  across generators can hide exactly this reversal. The `*_by_attack` functions exist so a
  reversal is *visible* as crossing/diverging lines rather than cancelling out in a pooled
  mean — call them whenever more than one attack/generator label is available, not only the
  pooled `*_by_class` version.
- **Colour is never the only channel.** Every function below draws from one fixed
  class -> (colour, linestyle, marker) mapping (`CLASS_STYLE`, keyed by class name) so the same
  class always looks the same across every figure in the module, and so classes/attacks remain
  distinguishable in greyscale or for colour-blind readers.

This module computes nothing statistical itself (no effect sizes, no CIs, no AUCs) — it only
plots arrays the caller already computed (in `stats.py` / `features.py`), consistent with the
module boundary in `CONVENTIONS.md` ("visualization.py: figures, must not: compute
statistics").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "CLASS_COLORS",
    "CLASS_STYLE",
    "plot_waveform",
    "plot_spectrogram",
    "plot_mean_spectrum_with_ci",
    "plot_mean_spectrum_by_attack",
    "plot_class_distributions",
    "plot_class_distributions_by_attack",
    "plot_effect_size_per_bin",
    "plot_feature_trajectory",
    "plot_eer_vs_layer",
    "save_figure",
]

# A single class/attack -> (colour, linestyle, marker) mapping, reused by every function in
# this module so the same label always looks the same across every figure. Colour alone never
# carries the distinction: linestyle and marker are cycled together with colour.
CLASS_COLORS: dict[str, str] = {
    "real": "#1b9e77",
    "bona-fide": "#1b9e77",
    "bonafide": "#1b9e77",
    "fake": "#d95f02",
    "spoof": "#d95f02",
}

_STYLE_CYCLE: list[tuple[str, str, str]] = [
    ("#1b9e77", "-", "o"),
    ("#d95f02", "--", "s"),
    ("#7570b3", "-.", "^"),
    ("#e7298a", ":", "D"),
    ("#66a61e", "-", "v"),
    ("#e6ab02", "--", "P"),
    ("#a6761d", "-.", "X"),
    ("#666666", ":", "*"),
]

# label -> (colour, linestyle, marker), populated lazily so new labels get a stable slot the
# first time they are seen, deterministically (insertion order), and reused afterwards.
CLASS_STYLE: dict[str, tuple[str, str, str]] = {}


def _style_for(label: str) -> tuple[str, str, str]:
    """Return a stable (colour, linestyle, marker) triple for `label`.

    Known class names use `CLASS_COLORS` (with a linestyle/marker assigned on first sight);
    any other label (e.g. an attack/generator id) is assigned the next slot in `_STYLE_CYCLE`
    and remembered in `CLASS_STYLE` for the rest of the process.
    """
    if label in CLASS_STYLE:
        return CLASS_STYLE[label]
    if label in CLASS_COLORS:
        idx = len(CLASS_STYLE) % len(_STYLE_CYCLE)
        _, linestyle, marker = _STYLE_CYCLE[idx]
        style = (CLASS_COLORS[label], linestyle, marker)
    else:
        idx = len(CLASS_STYLE) % len(_STYLE_CYCLE)
        style = _STYLE_CYCLE[idx]
    CLASS_STYLE[label] = style
    return style


def _ensure_ax(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_waveform(x: np.ndarray, sr: int, ax: Axes | None = None, **kw: Any) -> Axes:
    """Plot a single waveform's amplitude against time in seconds.

    A single waveform is inherently a "one example" view (there is no aggregate waveform);
    use it for a spot-check, not as the evidence figure — the evidence figures are the
    distribution/spectrum functions below.
    """
    ax = _ensure_ax(ax)
    x = np.asarray(x)
    t = np.arange(x.shape[-1]) / float(sr)
    color = kw.pop("color", "#1b9e77")
    ax.plot(t, x, color=color, linewidth=kw.pop("linewidth", 0.8), **kw)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_xlim(t[0], t[-1] if t.size else 1.0)
    return ax


def plot_spectrogram(
    S: np.ndarray,
    sr: int,
    hop_length: int,
    ax: Axes | None = None,
    **kw: Any,
) -> Axes:
    """Plot a log-power spectrogram `S` (freq_bins x time_frames) with a time/Hz axis.

    `S` is expected to already be a power (or magnitude) spectrogram; this function applies
    the log compression for display (`10 * log10`), it does not compute the STFT.
    """
    ax = _ensure_ax(ax)
    S = np.asarray(S, dtype=np.float64)
    S_db = 10.0 * np.log10(np.maximum(S, 1e-12))
    n_freq, n_frames = S_db.shape
    duration = n_frames * hop_length / float(sr)
    nyquist = sr / 2.0
    cmap = kw.pop("cmap", "magma")
    im = ax.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=(0, duration, 0, nyquist),
        cmap=cmap,
        **kw,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig = ax.get_figure()
    if fig is not None:
        fig.colorbar(im, ax=ax, label="power (dB)")
    return ax


def _mean_and_ci(arr: np.ndarray, ci: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and a normal-approximation CI of the mean across axis 0 (examples).

    This is a display-only convenience (not a statistical claim of significance); `stats.py`
    owns bootstrap CIs and effect sizes for anything reported as a result.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(max(n, 1)) if n > 1 else np.zeros_like(mean)
    z = 1.96 if ci == 0.95 else 1.0
    return mean, mean - z * sem, mean + z * sem


def plot_mean_spectrum_with_ci(
    spectra_by_class: dict[str, np.ndarray],
    freqs: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Mean spectrum per class with a filled confidence band, aggregated over examples.

    `spectra_by_class[label]` is (n_examples, n_freq_bins). This is the aggregate view the
    handoff asks for in place of a single impressive real/fake pair: every class curve is a
    mean over its examples, with the band showing the mean's CI, not per-example spread.
    """
    ax = _ensure_ax(ax)
    freqs = np.asarray(freqs)
    for label, spectra in spectra_by_class.items():
        color, linestyle, marker = _style_for(label)
        mean, lo, hi = _mean_and_ci(spectra)
        ax.plot(
            freqs,
            mean,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=max(len(freqs) // 12, 1),
            label=label,
        )
        # Pylance's bundled matplotlib stub types y1/y2 as scalars; arrays are the documented use.
        ax.fill_between(freqs, lo, hi, color=color, alpha=0.25, linewidth=0)  # pyright: ignore[reportArgumentType]
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude")
    ax.legend()
    return ax


def plot_mean_spectrum_by_attack(
    spectra_by_attack: dict[str, np.ndarray],
    freqs: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Generator-conditioned mean spectra, one curve+band per attack/generator label.

    Same computation as `plot_mean_spectrum_with_ci`; kept as a separate name (per the task
    interface) so a call site's intent is explicit: this is the plot that can reveal a cue
    that separates one generator but reverses for another, which a pooled `real` vs `fake`
    plot would average away.
    """
    return plot_mean_spectrum_with_ci(spectra_by_attack, freqs, ax=ax)


def plot_class_distributions(
    values_by_class: dict[str, np.ndarray],
    ax: Axes | None = None,
) -> Axes:
    """Overlaid histograms (as step outlines) of a scalar feature, one per class.

    Uses unfilled step histograms rather than solid bars so overlapping classes both stay
    visible, and each class keeps its fixed colour/linestyle so no two classes are
    distinguished by colour alone.
    """
    ax = _ensure_ax(ax)
    all_vals = np.concatenate([np.asarray(v).ravel() for v in values_by_class.values()])
    finite = all_vals[np.isfinite(all_vals)]
    bins = np.histogram_bin_edges(finite, bins="auto").tolist() if finite.size else 10
    for label, values in values_by_class.items():
        color, linestyle, _marker = _style_for(label)
        ax.hist(
            np.asarray(values).ravel(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend()
    return ax


def plot_class_distributions_by_attack(
    values_by_attack: dict[str, np.ndarray],
    ax: Axes | None = None,
) -> Axes:
    """Overlaid distributions conditioned on attack/generator rather than pooled class.

    Identical drawing logic to `plot_class_distributions`; a distinct name keeps call sites
    explicit about aggregating within attack rather than within class, so a
    generator-specific shift is visible instead of averaged into "fake."
    """
    return plot_class_distributions(values_by_attack, ax=ax)


def plot_effect_size_per_bin(
    effect_sizes: np.ndarray,
    freqs: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Standardized effect size (e.g. Cohen's d) per frequency bin.

    Plotted as a simple line against frequency, with a zero-reference line, so the reader
    sees where and how strongly classes diverge without the plot implying uncorrected
    pointwise significance (per the handoff's warning against treating thousands of
    pointwise tests as evidence — that judgement belongs in `stats.py`/prose, not this plot).
    """
    ax = _ensure_ax(ax)
    freqs = np.asarray(freqs)
    effect_sizes = np.asarray(effect_sizes)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.plot(freqs, effect_sizes, color="#7570b3", linestyle="-", marker="o", markersize=3)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("effect size")
    return ax


def plot_feature_trajectory(
    feature_over_time: np.ndarray | dict[str, np.ndarray],
    ax: Axes | None = None,
) -> Axes:
    """Plot a scalar feature's evolution over frames/time.

    Accepts either a single 1D array, or a `{label: array}` dict (e.g. per class) so a
    caller can overlay several trajectories on shared axes with the shared style mapping.
    """
    ax = _ensure_ax(ax)
    if isinstance(feature_over_time, dict):
        for label, values in feature_over_time.items():
            color, linestyle, marker = _style_for(label)
            values = np.asarray(values)
            ax.plot(
                np.arange(values.shape[-1]),
                values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=max(values.shape[-1] // 12, 1),
                label=label,
            )
        ax.legend()
    else:
        values = np.asarray(feature_over_time)
        ax.plot(np.arange(values.shape[-1]), values, color="#1b9e77", linestyle="-")
    ax.set_xlabel("frame")
    ax.set_ylabel("feature value")
    return ax


def plot_eer_vs_layer(eer_by_layer: dict[int, float], ax: Axes | None = None) -> Axes:
    """EER against SSL transformer layer index (Stage 6 layer probe).

    Lower EER is better, so the y-axis is inverted: the top of the plot reads as "better",
    matching how the reader's eye scans up for improvement. The axis is labelled explicitly
    ("EER (lower is better)") so the inversion is never ambiguous on its own.
    """
    ax = _ensure_ax(ax)
    layers = sorted(eer_by_layer)
    eers = [eer_by_layer[layer] for layer in layers]
    ax.plot(layers, eers, color="#d95f02", linestyle="-", marker="o")
    ax.set_xlabel("transformer layer index")
    ax.set_ylabel("EER (lower is better)")
    ax.set_xticks(layers)
    lo, hi = min(eers), max(eers)
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    ax.set_ylim(hi + pad, lo - pad)  # inverted: better (lower EER) reads at the top
    return ax


def save_figure(fig: Figure, name: str, out_dir: str = "reports/figures") -> Path:
    """Save `fig` as `<out_dir>/<name>.png`, creating `out_dir` if needed. Returns the path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = name if name.endswith(".png") else f"{name}.png"
    path = out_path / filename
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    return path
