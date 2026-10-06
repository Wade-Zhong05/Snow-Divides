from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from scipy.sparse import load_npz

from snow_divides.e5_laplacian.interactive import embedding_3d_html
from snow_divides.e5_laplacian.matrices import clustered_matrix_figure


def main(run: Path, output: Path) -> None:
    """Export interactive spectral and reordered matrix views from a completed pipeline run."""
    if output.exists():
        raise typer.BadParameter(f"Output already exists: {output}")
    manifest = json.loads((run / "manifest.json").read_text())
    k = int(manifest["reference_k"])
    cluster_count = int(manifest["reference_clusters"])
    with np.load(run / "e1_features" / "features.npz") as features:
        selected = features["selected_pixel_index"]
        latitude = features["latitude"]
        longitude = features["longitude"]
    with np.load(run / "e5_laplacian" / "spectral_results.npz") as spectrum:
        vectors = spectrum["eigenvectors"]
        counts = spectrum["cluster_counts"]
        labels = spectrum["cluster_labels"][int(np.flatnonzero(counts == cluster_count)[0])]
        bands = spectrum["fiedler_bands"]
        fiedler_labels = spectrum["fiedler_labels"][int(np.flatnonzero(bands == 0)[0])]
    distance = np.load(run / "e2_interactions" / "distance_matrix.npy", mmap_mode="r")
    weights = load_npz(run / "e3_knn" / f"graph_k{k}.npz").tocsr()
    output.mkdir(parents=True)
    embedding_3d_html(
        output / "05_embedding_3d_interactive.html",
        vectors,
        fiedler_labels,
        labels,
        selected,
        latitude,
        longitude,
    )
    clustered_matrix_figure(output / "06_clustered_matrices.png", distance, weights, labels)
    (output / "views_manifest.json").write_text(
        json.dumps(
            {
                "source_run": str(run.resolve()),
                "reference_k": k,
                "reference_clusters": cluster_count,
                "cell_count": int(selected.size),
                "cluster_sizes": np.bincount(labels, minlength=cluster_count).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved views to {output}")


if __name__ == "__main__":
    typer.run(main)
