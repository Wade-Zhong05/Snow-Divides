"""Exploratory plots; save the plotted counts and quantify all clipped tails."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from gradients import Features, Gradients


class DistributionSummary(TypedDict):
    valid_count: int
    active_count: int
    all_zero_fraction: float
    active_zero_fraction: float
    active_negative_fraction: float
    active_positive_fraction: float
    active_quantiles: list[float]
    display_limit: float
    active_clipped_fraction: float


def distributions(
    g: Gradients, f: Features, out: Path, period: str
) -> dict[str, DistributionSummary]:
    """Compare conditional probabilities, retaining zero changes in count histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    summary: dict[str, DistributionSummary] = {}
    for axis, values, active, name, unit, signed in [
        (axes[0, 0], g.temporal, g.time_active, "Temporal change", "cm/day", True),
        (axes[0, 1], g.spatial, g.space_active, "Spatial magnitude", "cm/km", False),
    ]:
        all_values = values[np.isfinite(values)]
        selected = values[active]
        limit = max(float(np.quantile(np.abs(selected), 0.995)), 1e-6)
        edges = np.linspace(-limit if signed else 0, limit, 101)
        all_hist, _ = np.histogram(all_values, edges)
        active_hist, _ = np.histogram(selected, edges)
        axis.stairs(all_hist / all_values.size, edges, label="All valid", color="#52788E")
        axis.stairs(active_hist / selected.size, edges, label="Snow-active", color="#CF733B")
        axis.set(
            yscale="log",
            xlabel=f"{name} ({unit})",
            ylabel="Fraction per bin",
            title=f"{name} | exactly zero: {np.mean(all_values == 0):.1%} of valid samples",
        )
        axis.legend()
        summary[name] = {
            "valid_count": int(all_values.size),
            "active_count": int(selected.size),
            "all_zero_fraction": float(np.mean(all_values == 0)),
            "active_zero_fraction": float(np.mean(selected == 0)),
            "active_negative_fraction": float(np.mean(selected < 0)),
            "active_positive_fraction": float(np.mean(selected > 0)),
            "active_quantiles": np.quantile(
                selected, [0, 0.005, 0.25, 0.5, 0.75, 0.995, 1]
            ).tolist(),
            "display_limit": limit,
            "active_clipped_fraction": float(np.mean(np.abs(selected) > limit)),
        }
        np.savez_compressed(
            out / f"{name.split()[0].lower()}_histogram.npz",
            edges=edges,
            all_counts=all_hist,
            active_counts=active_hist,
        )
        tail = summary[name]["active_clipped_fraction"]
        axis.text(
            0.02,
            0.97,
            f"Active tail outside view: {tail:.2%}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    for axis, raw, extent, title, unit in [
        (axes[1, 0], f.temporal_raw, (-90, 90), "Temporal oriented histogram", "|change|, cm/day"),
        (axes[1, 1], f.spatial_raw, (-180, 180), "Spatial oriented histogram", "magnitude, cm/km"),
    ]:
        weights = raw.sum(axis=(0, 1, 2))
        edges = np.linspace(*extent, weights.size + 1)
        proportions = weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)
        axis.bar(
            (edges[:-1] + edges[1:]) / 2, proportions, width=np.diff(edges) * 0.88, color="#52788E"
        )
        axis.set(
            xlabel="Signed orientation (degrees)",
            ylabel="Fraction of total gradient weight",
            title=title,
        )
        axis.set_ylim(0, max(float(proportions.max()) * 1.25, 0.01))
        axis.text(
            0.02,
            0.96,
            f"Weight: {unit}; flat samples have zero weight",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
    fig.suptitle(f"Snow-depth gradient distributions | {period}", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "Region: 73–105°E, 26–40°N rectangle; no plateau mask or 30-snow-day filter.\n"
        "Time: D(t)-D(t-1), cm/day. Space: centered E/N differences, cm/km; "
        "magnitude = sqrt(E²+N²). Missing stencils excluded.\n"
        "Snow-active: at least one stencil depth >0. Top: 100 bins, log y, "
        "active |gradient| 99.5th-percentile limits; probabilities retain full denominators.\n"
        "Bottom: 18 hard signed-angle bins, magnitude-weighted, global L1. "
        "Temporal angle = atan(change / 1 cm/day); spatial angle = atan2(N,E).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(out / "01_distributions.png", dpi=160)
    plt.close(fig)
    return summary


def joint_distributions(g: Gradients, out: Path, period: str) -> dict[str, float]:
    """Left: east × north (cm/km). Right: time (cm/day) × spatial norm (cm/km).

    These are joint count distributions, not angle-weighted HOG features. The
    mixed-unit right panel does not define a spatiotemporal orientation angle.
    """
    paired = np.isfinite(g.temporal) & np.isfinite(g.spatial) & (g.time_active | g.space_active)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    clipped: dict[str, float] = {}
    for axis, x, y, name, xlabel, ylabel, y_signed in [
        (
            axes[0],
            g.east[g.space_active],
            g.north[g.space_active],
            "east_north",
            "Eastward gradient (cm/km)",
            "Northward gradient (cm/km)",
            True,
        ),
        (
            axes[1],
            g.temporal[paired],
            g.spatial[paired],
            "time_space",
            "Temporal change (cm/day)",
            "Spatial magnitude (cm/km)",
            False,
        ),
    ]:
        xlim = max(float(np.quantile(np.abs(x), 0.995)), 1e-6)
        ylim = max(float(np.quantile(np.abs(y), 0.995)), 1e-6)
        xe = np.linspace(-xlim, xlim, 101)
        ye = np.linspace(-ylim if y_signed else 0, ylim, 101)
        counts, _, _ = np.histogram2d(x, y, bins=(xe, ye))
        clipped[name] = 1 - float(counts.sum() / x.size)
        mesh = axis.pcolormesh(
            xe,
            ye,
            np.ma.masked_equal(counts.T, 0),
            norm=LogNorm(vmin=1, vmax=max(2, counts.max())),
            cmap="magma",
        )
        fig.colorbar(mesh, ax=axis, label="Sample count (log color)")
        axis.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{name.replace('_', ' × ')} | outside view: {clipped[name]:.2%}",
        )
        np.savez_compressed(
            out / f"{name}_histogram2d.npz",
            counts=counts,
            x_edges=xe,
            y_edges=ye,
            total_count=x.size,
        )
    fig.suptitle(f"Two distinct 2D histograms | {period}", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "Region: 73–105°E, 26–40°N rectangle. Each panel: 100 × 100 bins; "
        "log color = raw sample count; white = empty bin.\n"
        "Left: centered east/north derivatives (cm/km), north positive; "
        "valid five-point stencil with any depth >0.\n"
        "Right: D(t)-D(t-1) (cm/day) versus same-day sqrt(E²+N²) (cm/km); "
        "both gradients valid, either stencil snow-active.\n"
        "Independent 99.5th-percentile |gradient| limits; outside-view percentage "
        "shown above. Zero changes retained; missing dates never bridged.\n"
        "Joint count histograms are distinct from magnitude-weighted HOG. "
        "No combined time-space angle is defined because axis units differ.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.19, 1, 0.95))
    fig.savefig(out / "02_joint_distributions.png", dpi=160)
    plt.close(fig)
    return clipped


def seasonal_features(f: Features, out: Path, period: str) -> None:
    """Pool gradient weights by calendar month, then normalize each month's angles."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for axis, raw, extent, title in [
        (axes[0], f.temporal_raw, (-90, 90), "Temporal: negative = decrease, positive = increase"),
        (axes[1], f.spatial_raw, (-180, 180), "Spatial: 0° east, 90° north, ±180° west"),
    ]:
        pooled = raw.sum(axis=(1, 2))
        totals = pooled.sum(axis=1, keepdims=True)
        weights = np.divide(pooled, totals, out=np.zeros_like(pooled), where=totals > 0)
        mesh = axis.pcolormesh(
            np.linspace(*extent, raw.shape[-1] + 1),
            np.arange(0.5, 13.5),
            weights,
            cmap="viridis",
            vmin=0,
        )
        fig.colorbar(mesh, ax=axis, label="Fraction of monthly gradient weight")
        axis.set(
            xlabel="Signed orientation (degrees)",
            ylabel="Calendar month",
            yticks=np.arange(1, 13),
        )
        axis.set_title(title, fontsize=11, pad=12)
    fig.suptitle(f"Monthly oriented histograms | {period}", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "Region: 73–105°E, 26–40°N rectangle; valid snow-active stencils only. "
        "All available years pooled by calendar month.\n"
        "Left: angle = atan(change / 1 cm/day), 18 bins over [-90°,90°]; "
        "weight = |change|. Decrease alone does not prove physical snowmelt.\n"
        "Right: angle = atan2(north,east), 18 signed bins over [-180°,180°); "
        "weight = spatial gradient magnitude (cm/km).\n"
        "Color: each month's angle weights sum to one; color scales are separate. "
        "Flat samples contribute zero weight. No cell/block L2-Hys.\n"
        "Saved features normalize per pixel/month; this display instead pools "
        "raw weights across pixels before normalizing each month.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.19, 1, 0.95))
    fig.savefig(out / "03_monthly_hog.png", dpi=160)
    plt.close(fig)
