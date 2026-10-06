from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def clustered_matrix_figure(
    output: Path,
    distance: NDArray[np.float32],
    weights: csr_matrix,
    labels: NDArray[np.int32],
) -> None:
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    group_ids, counts = np.unique(sorted_labels, return_counts=True)
    limits = np.cumsum(counts)
    centers = (np.r_[0, limits[:-1]] + limits - 1) / 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    distance_image = axes[0].imshow(
        distance[np.ix_(order, order)],
        cmap="viridis_r",
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    fig.colorbar(distance_image, ax=axes[0], shrink=0.76, label="Euclidean snow-feature distance")
    axes[0].set_title("E-2: all-pairs feature distance D")
    axes[1].spy(weights[order, :][:, order], markersize=0.32, color="#204a87")
    axes[1].set_title("E-3: kNN weighted graph W (nonzero edges)")
    for axis in axes:
        for boundary in limits[:-1] - 0.5:
            axis.axhline(boundary, color="#d95f02", linewidth=0.9)
            axis.axvline(boundary, color="#d95f02", linewidth=0.9)
        axis.set_xticks(
            centers,
            [
                f"C{group_id + 1}\n(n={count})"
                for group_id, count in zip(group_ids, counts, strict=True)
            ],
        )
        axis.set_yticks(centers, [f"C{group_id + 1}" for group_id in group_ids])
        axis.set_xlabel("Cells grouped by spectral cluster; original grid order within each group")
        axis.set_ylabel("Same cell order")
    fig.suptitle(
        "Same cluster-based row/column order in both matrices; orange lines mark block boundaries"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
