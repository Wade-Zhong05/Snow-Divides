from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray

from snow_divides.e5_laplacian.spectral import fiedler_partition


def _grid(
    labels: NDArray[np.int32], selected: NDArray[np.int64], shape: tuple[int, int]
) -> NDArray[np.int32]:
    grid = np.full(shape[0] * shape[1], -1, dtype=np.int32)
    grid[selected] = labels
    return grid.reshape(shape)


def _extent(
    latitude: NDArray[np.float64], longitude: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    return (
        float(longitude.min() - 0.125),
        float(longitude.max() + 0.125),
        float(latitude.min() - 0.125),
        float(latitude.max() + 0.125),
    )


def cluster_maps(
    output: Path,
    partitions: dict[int, NDArray[np.int32]],
    selected: NDArray[np.int64],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
) -> None:
    columns = min(3, len(partitions))
    rows = int(np.ceil(len(partitions) / columns))
    figure_size = (10, 5) if len(partitions) == 1 else (5 * columns, 4.5 * rows)
    fig, axes = plt.subplots(
        rows, columns, figsize=figure_size, constrained_layout=True, squeeze=False
    )
    extent = _extent(latitude, longitude)
    for axis, (count, labels) in zip(axes.flat, sorted(partitions.items()), strict=False):
        grid = np.ma.masked_less(_grid(labels, selected, (latitude.size, longitude.size)), 0)
        colors = [plt.get_cmap("tab10")(index) for index in range(count)]
        cmap = ListedColormap(colors)
        cmap.set_bad("#dddddd")
        image = axis.imshow(
            grid,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=count - 0.5,
        )
        axis.set(
            title=f"{count} clusters (n={labels.size:,})",
            xlabel="Longitude (°E)",
            ylabel="Latitude (°N)",
        )
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, ticks=np.arange(count), label="Arbitrary cluster ID")
    for axis in list(axes.flat)[len(partitions) :]:
        axis.set_visible(False)
    fig.suptitle("Normalized-Laplacian spectral clusters on original 0.25° geographic grid")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fiedler_maps(
    output: Path,
    vector: NDArray[np.float64],
    bands: tuple[float, ...],
    selected: NDArray[np.int64],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
) -> None:
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4.8), constrained_layout=True)
    extent = _extent(latitude, longitude)
    for axis, band in zip(np.atleast_1d(axes), bands, strict=True):
        labels = fiedler_partition(vector, band)
        grid = np.ma.masked_less(_grid(labels, selected, (latitude.size, longitude.size)), 0)
        cmap = ListedColormap(("#31688e", "#dddddd", "#d95f02"))
        cmap.set_bad("white")
        image = axis.imshow(
            grid,
            origin="upper",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=2.5,
        )
        axis.set(
            title=f"Near-zero band: {band:.0%} of |q₂|",
            xlabel="Longitude (°E)",
            ylabel="Latitude (°N)",
        )
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, ticks=(0, 1, 2), label="negative / uncertain / positive")
    fig.suptitle("Fiedler sign cut; gray is near-zero uncertainty, white is excluded")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def spectrum_plot(output: Path, values: NDArray[np.float64], candidates: tuple[int, ...]) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    positions = np.arange(1, values.size + 1)
    axes[0].plot(positions, values, "o-", ms=4, color="#31688e")
    axes[0].set(
        xlabel="Eigenvalue index (1-based)",
        ylabel="Normalized Laplacian eigenvalue",
        title=f"First {values.size} normalized-Laplacian eigenvalues",
    )
    axes[0].grid(alpha=0.25)
    gaps = np.asarray([values[count] - values[count - 1] for count in candidates])
    axes[1].bar(candidates, gaps, color="#35b779")
    axes[1].set(
        xlabel="Candidate cluster count m",
        ylabel="λₘ₊₁ − λₘ",
        title="Eigengap after each candidate count",
    )
    axes[1].set_xticks(candidates)
    for count, gap in zip(candidates, gaps, strict=True):
        axes[1].text(count, gap, f"{gap:.4f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, float(gaps.max()) * 1.2)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def embedding_3d(output: Path, vectors: NDArray[np.float64], labels: NDArray[np.int32]) -> None:
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    assert isinstance(axis, Axes3D)
    colors = ("#31688e", "#888888", "#d95f02")
    for label, color in enumerate(colors):
        members = labels == label
        axis.plot(
            vectors[members, 1],
            vectors[members, 2],
            vectors[members, 3],
            marker=".",
            linestyle="None",
            markersize=2,
            color=color,
            alpha=0.65,
        )
    axis.set(
        xlabel="q₂",
        ylabel="q₃",
        zlabel="q₄",
        title="3D Laplacian invariant subspace colored by Fiedler domains",
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
