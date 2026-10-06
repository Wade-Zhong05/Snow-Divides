"""Figures for temporal histograms, snow graph partitions, and geographic checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm
from numpy.typing import NDArray

MONTHS = ("Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul")


def _map_axes(axis: Axes, latitude: NDArray[np.float64], longitude: NDArray[np.float64]) -> None:
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_xlim(longitude.min() - 0.125, longitude.max() + 0.125)
    axis.set_ylim(latitude.min() - 0.125, latitude.max() + 0.125)
    axis.set_aspect("equal")


def feature_and_region(
    output: Path,
    labels: NDArray[np.int32],
    selected: NDArray[np.int64],
    counts: NDArray[np.int32],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
    angle_edges: NDArray[np.float64],
) -> None:
    """Show the snow-only partition beside each regime's average 2D feature distribution."""
    windows = counts.shape[0]
    grid = np.full(latitude.size * longitude.size, -1, dtype=np.int32)
    grid[selected] = labels
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    image = axes[0, 0].imshow(
        np.ma.masked_less(grid.reshape(latitude.size, longitude.size), 0),
        cmap=ListedColormap(("#31688e", "#35b779", "#fde725")),
        vmin=-0.5,
        vmax=2.5,
        extent=(
            longitude.min() - 0.125,
            longitude.max() + 0.125,
            latitude.min() - 0.125,
            latitude.max() + 0.125,
        ),
        origin="upper",
        interpolation="nearest",
    )
    image.cmap.set_bad("#dddddd")
    axes[0, 0].set_title("Snow-feature graph partition (gray: excluded)")
    _map_axes(axes[0, 0], latitude, longitude)
    fig.colorbar(image, ax=axes[0, 0], ticks=(0, 1, 2), label="Cluster (arbitrary ID)")
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        counts, totals, out=np.zeros_like(counts, dtype=np.float32), where=totals > 0
    )
    cluster_images = []
    for cluster, axis in enumerate((axes[0, 1], axes[1, 0], axes[1, 1])):
        mean_hist = probabilities[:, selected[labels == cluster]].mean(axis=1)
        cluster_images.append(
            axis.pcolormesh(
                np.arange(windows + 1),
                angle_edges,
                np.maximum(mean_hist.T, 0.001),
                cmap="magma",
                norm=LogNorm(vmin=0.001, vmax=1),
            )
        )
        axis.set_title(
            f"Cluster {cluster}: mean gradient histogram, n={(labels == cluster).sum():,}"
        )
        axis.set_ylabel("atan(gradient / scale) (degrees)")
        axis.set_xlabel("Snow-year time window (Aug to Jul)")
        axis.set_xticks(np.arange(0, windows, windows // 12) + windows / 24, MONTHS, rotation=45)
    fig.colorbar(
        cluster_images[-1],
        ax=(axes[0, 1], axes[1, 0], axes[1, 1]),
        label="Mean conditional fraction per gradient bin (log scale)",
    )
    fig.suptitle(
        "Snow-only regions and signed daily-gradient distributions; zero changes remain visible",
        fontsize=14,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def occurrence_profiles(
    output: Path,
    labels: NDArray[np.int32],
    selected: NDArray[np.int64],
    counts: NDArray[np.int32],
    snow_days: NDArray[np.int32],
    valid_days: NDArray[np.int32],
) -> None:
    """Separate snow absence from zero daily gradient in each snow-derived cluster."""
    windows = counts.shape[0]
    occurrence = np.divide(
        snow_days,
        valid_days,
        out=np.zeros(snow_days.shape, dtype=np.float32),
        where=valid_days > 0,
    )
    active_pair_days = counts.sum(axis=-1)
    activity = np.divide(
        active_pair_days,
        valid_days,
        out=np.zeros(active_pair_days.shape, dtype=np.float32),
        where=valid_days > 0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = ("#31688e", "#35b779", "#d4ad00")
    for cluster, color in enumerate(colors):
        members = selected[labels == cluster]
        x = np.arange(windows) + 0.5
        axes[0].plot(
            x, occurrence[:, members].mean(axis=1), color=color, label=f"Cluster {cluster}"
        )
        axes[1].plot(x, activity[:, members].mean(axis=1), color=color, label=f"Cluster {cluster}")
    for axis in axes:
        axis.set_xticks(np.arange(0, windows, windows // 12) + windows / 24, MONTHS, rotation=45)
        axis.set_xlabel("Snow-year time window (Aug to Jul)")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set(ylabel="Fraction of valid days with depth > 0", title="Snow occurrence by cluster")
    axes[1].set(
        ylabel="Snow-active gradient pairs / valid days",
        title="Gradient sample availability by cluster",
    )
    fig.suptitle("Snow occurrence distinguishes snow-free and stable-snow zero gradients")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def diagnostics(
    output: Path,
    k_values: NDArray[np.int32],
    component_counts: NDArray[np.int32],
    largest_fractions: NDArray[np.float64],
    edge_lengths: NDArray[np.float64],
) -> None:
    """Expose graph connectivity and distances before interpreting regions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    axes[0].plot(k_values, component_counts, "o-", color="#31688e")
    axes[0].set(
        xlabel="Snow-feature neighbors k",
        ylabel="Connected components",
        title="Graph connectivity across k",
    )
    axes[0].set_yticks(np.unique(component_counts))
    axes[0].grid(alpha=0.25)
    axes[1].plot(k_values, largest_fractions, "o-", color="#35b779")
    axes[1].set(
        xlabel="Snow-feature neighbors k",
        ylabel="Largest component fraction",
        ylim=(0, 1.02),
        title="Fraction in largest component",
    )
    axes[1].grid(alpha=0.25)
    axes[2].hist(edge_lengths, bins=50, color="#31688e", alpha=0.85)
    axes[2].set(
        xlabel="Geographic edge length (km)",
        ylabel="Snow-graph edges",
        title="Geography measured after construction",
    )
    fig.suptitle(
        "Snow-only graph diagnostics; edges are feature neighbors, with no geographic cutoff",
        fontsize=13,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def boundary_map(
    output: Path,
    disagreement: NDArray[np.int32],
    support: NDArray[np.int32],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
    year_count: int,
) -> None:
    """Map the share of eligible neighboring cell-year pairs assigned different labels."""
    probability = np.divide(
        disagreement, support, out=np.full(disagreement.shape, np.nan), where=support > 0
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    extent = (
        longitude.min() - 0.125,
        longitude.max() + 0.125,
        latitude.min() - 0.125,
        latitude.max() + 0.125,
    )
    image = axes[0].imshow(
        probability,
        origin="upper",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="magma",
        interpolation="nearest",
    )
    axes[0].set_title("Neighbor disagreement fraction across snow years")
    _map_axes(axes[0], latitude, longitude)
    fig.colorbar(image, ax=axes[0], label="Different-cluster adjacent pairs / eligible pairs")
    image_support = axes[1].imshow(
        support, origin="upper", extent=extent, cmap="viridis", interpolation="nearest"
    )
    axes[1].set_title("Eligible adjacent cell-year pairs")
    _map_axes(axes[1], latitude, longitude)
    fig.colorbar(image_support, ax=axes[1], label="Number of comparisons")
    fig.suptitle(
        f"Yearly partitions: geographic boundary evidence from {year_count} snow years", fontsize=13
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
