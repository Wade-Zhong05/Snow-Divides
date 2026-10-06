from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True, slots=True)
class ConnectivitySummary:
    vertices: int
    components: int
    edges: int
    degree_min: int
    degree_median: float
    degree_mean: float
    degree_max: int
    clustering_mean: float
    clustering_median: float
    clustering_zero_fraction: float
    clustering_q25: float
    clustering_q75: float


def graph_degrees_and_clustering(
    weights: csr_matrix,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    adjacency = weights.copy().astype(np.int32)
    adjacency.data.fill(1)
    degrees = np.diff(adjacency.indptr).astype(np.int64)
    two_steps = adjacency @ adjacency
    closed_walks = np.asarray(adjacency.multiply(two_steps).sum(axis=1)).ravel()
    denominator = degrees * (degrees - 1)
    clustering = np.divide(
        closed_walks,
        denominator,
        out=np.zeros(degrees.size, dtype=np.float64),
        where=denominator > 0,
    )
    return degrees, clustering


def analyze_connectivity(weights: csr_matrix, output: Path) -> ConnectivitySummary:
    degrees, clustering = graph_degrees_and_clustering(weights)
    component_count, _ = connected_components(weights, directed=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    degree_bins = np.arange(degrees.min(), degrees.max() + 2) - 0.5
    axes[0].hist(degrees, bins=degree_bins, color="#31688e", edgecolor="white", linewidth=0.2)
    axes[0].axvline(np.median(degrees), color="#d95f02", linestyle="--", label="median")
    axes[0].set(
        xlabel="Unweighted graph degree (neighbors)",
        ylabel="Number of vertices",
        title="Degree distribution",
    )
    axes[0].legend()
    axes[1].hist(clustering, bins=np.linspace(0, 1, 41), color="#35b779")
    axes[1].axvline(np.median(clustering), color="#d95f02", linestyle="--", label="median")
    axes[1].set(
        xlabel="Local clustering coefficient (unweighted)",
        ylabel="Number of vertices",
        title="Local clustering coefficient distribution",
        xlim=(0, 1),
    )
    axes[1].legend()
    figure.suptitle("E-4 combinatorial connectivity of the reference kNN graph")
    figure.savefig(output / "degree_and_clustering.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    summary = ConnectivitySummary(
        vertices=int(degrees.size),
        components=int(component_count),
        edges=int(weights.nnz // 2),
        degree_min=int(degrees.min()),
        degree_median=float(np.median(degrees)),
        degree_mean=float(degrees.mean()),
        degree_max=int(degrees.max()),
        clustering_mean=float(clustering.mean()),
        clustering_median=float(np.median(clustering)),
        clustering_zero_fraction=float(np.mean(clustering == 0)),
        clustering_q25=float(np.quantile(clustering, 0.25)),
        clustering_q75=float(np.quantile(clustering, 0.75)),
    )
    (output / "connectivity_summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n")
    np.savez_compressed(output / "node_connectivity.npz", degree=degrees, clustering=clustering)
    return summary


def main(run: Path, output: Path) -> None:
    """Export E-4 connectivity diagnostics from a completed pipeline run."""
    if output.exists():
        raise typer.BadParameter(f"Output already exists: {output}")
    manifest = json.loads((run / "manifest.json").read_text())
    k = int(manifest["reference_k"])
    weights = load_npz(run / "e3_knn" / f"graph_k{k}.npz").tocsr()
    output.mkdir(parents=True)
    result = analyze_connectivity(weights, output)
    print(f"Saved E-4 to {output}: {result.vertices} vertices, {result.components} component(s)")


if __name__ == "__main__":
    typer.run(main)
