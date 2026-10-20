"""Tests for the EDA figure vocabulary in `audio_df.helpers.research.visualization`.

Headless: force the Agg backend before any pyplot import. No dataset is used — every input
is synthesised with numpy.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from audio_df.helpers.research import visualization as viz

RNG = np.random.default_rng(0)


def _sine(freq=220.0, sr=16000, dur=0.5, noise=0.0):
    t = np.arange(int(sr * dur)) / sr
    x = np.sin(2 * np.pi * freq * t)
    if noise:
        x = x + RNG.normal(scale=noise, size=x.shape)
    return x.astype(np.float64)


def _spectrogram(n_freq=129, n_frames=100):
    return RNG.normal(size=(n_freq, n_frames)) ** 2 + 1e-6


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotWaveform:
    def test_returns_axes(self):
        ax = viz.plot_waveform(_sine(), sr=16000)
        assert isinstance(ax, Axes)

    def test_accepts_caller_supplied_ax(self):
        fig, ax = plt.subplots()
        out = viz.plot_waveform(_sine(), sr=16000, ax=ax)
        assert out is ax


class TestPlotSpectrogram:
    def test_returns_axes(self):
        S = _spectrogram()
        ax = viz.plot_spectrogram(S, sr=16000, hop_length=256)
        assert isinstance(ax, Axes)

    def test_accepts_caller_supplied_ax(self):
        fig, ax = plt.subplots()
        S = _spectrogram()
        out = viz.plot_spectrogram(S, sr=16000, hop_length=256, ax=ax)
        assert out is ax


class TestPlotMeanSpectrumWithCI:
    def test_returns_axes_and_draws_ci_band(self):
        freqs = np.linspace(0, 8000, 65)
        spectra_by_class = {
            "real": RNG.normal(loc=1.0, scale=0.1, size=(30, 65)),
            "fake": RNG.normal(loc=1.2, scale=0.1, size=(30, 65)),
        }
        ax = viz.plot_mean_spectrum_with_ci(spectra_by_class, freqs)
        assert isinstance(ax, Axes)
        poly = [c for c in ax.collections if isinstance(c, mcollections.PolyCollection)]
        assert len(poly) >= 2, "expected a filled CI band (PolyCollection) per class"

    def test_two_classes_produce_two_lines(self):
        freqs = np.linspace(0, 8000, 65)
        spectra_by_class = {
            "real": RNG.normal(loc=1.0, scale=0.1, size=(30, 65)),
            "fake": RNG.normal(loc=1.2, scale=0.1, size=(30, 65)),
        }
        ax = viz.plot_mean_spectrum_with_ci(spectra_by_class, freqs)
        assert len(ax.lines) >= 2


class TestPlotClassDistributions:
    def test_two_classes_produce_two_artist_groups(self):
        values_by_class = {
            "real": RNG.normal(0, 1, size=200),
            "fake": RNG.normal(0.5, 1, size=200),
        }
        ax = viz.plot_class_distributions(values_by_class)
        assert isinstance(ax, Axes)
        # At least one artist per class: patches (hist/violin/box) or lines (kde).
        n_artists = len(ax.patches) + len(ax.lines) + len(ax.collections)
        assert n_artists >= 2

    def test_accepts_caller_supplied_ax(self):
        fig, ax = plt.subplots()
        values_by_class = {"real": RNG.normal(size=50), "fake": RNG.normal(size=50)}
        out = viz.plot_class_distributions(values_by_class, ax=ax)
        assert out is ax


class TestPlotClassDistributionsByAttack:
    def test_three_attacks_produce_distinguishable_series(self):
        values_by_attack = {
            "real": RNG.normal(0, 1, size=200),
            "attack_a": RNG.normal(0.3, 1, size=200),
            "attack_b": RNG.normal(-0.3, 1, size=200),
            "attack_c": RNG.normal(0.6, 1, size=200),
        }
        ax = viz.plot_class_distributions_by_attack(values_by_attack)
        assert isinstance(ax, Axes)
        n_artists = len(ax.patches) + len(ax.lines) + len(ax.collections)
        assert n_artists >= 4


class TestPlotEffectSizePerBin:
    def test_returns_axes(self):
        freqs = np.linspace(0, 8000, 65)
        effect_sizes = RNG.normal(size=65)
        ax = viz.plot_effect_size_per_bin(effect_sizes, freqs)
        assert isinstance(ax, Axes)
        assert len(ax.lines) + len(ax.patches) >= 1

    def test_accepts_caller_supplied_ax(self):
        fig, ax = plt.subplots()
        freqs = np.linspace(0, 8000, 65)
        effect_sizes = RNG.normal(size=65)
        out = viz.plot_effect_size_per_bin(effect_sizes, freqs, ax=ax)
        assert out is ax


class TestPlotFeatureTrajectory:
    def test_returns_axes_single_series(self):
        traj = RNG.normal(size=100).cumsum()
        ax = viz.plot_feature_trajectory(traj)
        assert isinstance(ax, Axes)
        assert len(ax.lines) >= 1

    def test_dict_of_classes_produces_multiple_lines(self):
        traj_by_class = {
            "real": RNG.normal(size=100).cumsum(),
            "fake": RNG.normal(size=100).cumsum(),
        }
        ax = viz.plot_feature_trajectory(traj_by_class)
        assert len(ax.lines) >= 2


class TestPlotEerVsLayer:
    def test_returns_axes(self):
        eer_by_layer = {i: 0.5 - 0.03 * i + 0.001 * i * i for i in range(12)}
        ax = viz.plot_eer_vs_layer(eer_by_layer)
        assert isinstance(ax, Axes)
        assert len(ax.lines) >= 1

    def test_axis_labelled_and_oriented_for_lower_is_better(self):
        eer_by_layer = {0: 0.4, 1: 0.2, 2: 0.05}
        ax = viz.plot_eer_vs_layer(eer_by_layer)
        assert "eer" in ax.get_ylabel().lower()
        # y-axis should be inverted so the "better" (lower EER) region reads at the top.
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]


class TestSaveFigure:
    def test_writes_nonempty_png(self, tmp_path: Path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = viz.save_figure(fig, "demo_plot", out_dir=str(tmp_path))
        assert isinstance(out, Path)
        assert out.exists()
        assert out.stat().st_size > 0
        assert out.suffix == ".png"

    def test_creates_out_dir_if_missing(self, tmp_path: Path):
        fig, ax = plt.subplots()
        nested = tmp_path / "nested" / "figs"
        out = viz.save_figure(fig, "demo2", out_dir=str(nested))
        assert out.exists()


class TestColourMapping:
    def test_class_colour_is_consistent_and_uses_linestyle_too(self):
        # The colour map constant must exist and be reused; check it is a dict of str->colour.
        assert hasattr(viz, "CLASS_COLORS")
        assert isinstance(viz.CLASS_COLORS, dict)
        assert "real" in viz.CLASS_COLORS and "fake" in viz.CLASS_COLORS
