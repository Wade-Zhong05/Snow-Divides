from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import save_npz
from sklearn.metrics import adjusted_rand_score

from snow_divides.e1_features.data_io import load_cube
from snow_divides.e1_features.histogram import (
    HistogramConfig,
    eligible_pixels,
    extract,
    feature_matrix,
)
from snow_divides.e2_interactions.pairwise import all_pair_distances
from snow_divides.e3_knn.graph import SnowGraph, graph_at_k, neighbors_from_matrix
from snow_divides.e3_knn.plots import connectivity_scan
from snow_divides.e4_connectivity.report import analyze_connectivity
from snow_divides.e5_laplacian.report import analyze_spectrum
from snow_divides.e5_laplacian.spectral import smallest_eigenpairs, spectral_clusters


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    first_year: int
    last_year: int
    histogram: HistogramConfig
    minimum_snow_days: int
    k_values: tuple[int, ...]
    reference_k: int
    cluster_counts: tuple[int, ...]
    reference_clusters: int
    fiedler_bands: tuple[float, ...]
    eigen_count: int


@dataclass(frozen=True, slots=True)
class KScanRow:
    k: int
    components: int
    largest_component_fraction: float
    edges: int
    fiedler_value: float | None
    cluster_ari_to_reference: float | None


def run_pipeline(config: PipelineConfig, output: Path) -> None:
    """Run implemented Part II stages into one output root."""
    started = time.perf_counter()
    project = Path(__file__).resolve().parent.parent
    for stage in ("e1_features", "e2_interactions", "e3_knn", "e4_connectivity", "e5_laplacian"):
        (output / stage).mkdir(parents=True)
    cube = load_cube(
        project / "data" / "snow depth",
        date(config.first_year, 8, 1),
        date(config.last_year + 1, 7, 31),
    )
    print("E-1: extracting time × signed-gradient features", flush=True)
    hist = extract(cube, config.histogram)
    selected = np.flatnonzero(eligible_pixels(hist, config.minimum_snow_days))
    if selected.size <= max(config.k_values) + 1:
        raise RuntimeError("Too few eligible grid cells for the requested maximum k")
    pooled_counts = hist.counts.sum(axis=0, dtype=np.int32)
    pooled_snow = hist.snow_days.sum(axis=0, dtype=np.int32)
    pooled_valid = hist.valid_days.sum(axis=0, dtype=np.int32)
    features = feature_matrix(
        pooled_counts[:, selected],
        pooled_snow[:, selected],
        pooled_valid[:, selected],
        config.histogram,
    )
    np.savez_compressed(
        output / "e1_features" / "features.npz",
        pooled_features=features,
        selected_pixel_index=selected,
        pooled_histogram_counts=pooled_counts[:, selected],
        pooled_snow_days=pooled_snow[:, selected],
        pooled_valid_days=pooled_valid[:, selected],
        latitude=cube.latitude,
        longitude=cube.longitude,
        gradient_angle_edges_degrees=hist.angle_edges_degrees,
    )
    print(f"E-2: all-to-all distance matrix for {selected.size} cells", flush=True)
    matrix = all_pair_distances(features)
    np.save(output / "e2_interactions" / "distance_matrix.npy", matrix)
    print("E-3: kNN graphs and connectivity scan", flush=True)
    neighbors = neighbors_from_matrix(matrix, max(config.k_values))
    graphs: dict[int, SnowGraph] = {}
    rows: list[KScanRow] = []
    labels_by_k: dict[int, np.ndarray[tuple[int], np.dtype[np.int32]]] = {}
    for k in config.k_values:
        graph = graph_at_k(neighbors, k)
        graphs[k] = graph
        save_npz(output / "e3_knn" / f"graph_k{k}.npz", graph.weights)
        fiedler_value = None
        if graph.components == 1:
            spectrum = smallest_eigenpairs(graph.weights, config.reference_clusters)
            fiedler_value = float(spectrum.eigenvalues[1])
            labels_by_k[k] = spectral_clusters(spectrum.eigenvectors, config.reference_clusters)
        rows.append(
            KScanRow(
                k,
                graph.components,
                graph.largest_fraction,
                graph.weights.nnz // 2,
                fiedler_value,
                None,
            )
        )
        print(f"  k={k}: {graph.components} components, {graph.weights.nnz // 2} edges", flush=True)
    reference = graphs[config.reference_k]
    if reference.components != 1:
        raise RuntimeError("Reference k graph is disconnected; choose a connected k")
    reference_labels = labels_by_k[config.reference_k]
    rows = [
        KScanRow(
            row.k,
            row.components,
            row.largest_component_fraction,
            row.edges,
            row.fiedler_value,
            adjusted_rand_score(reference_labels, labels_by_k[row.k])
            if row.k in labels_by_k
            else None,
        )
        for row in rows
    ]
    connectivity_scan(
        output / "e3_knn" / "connectivity_and_fiedler.png",
        np.asarray(config.k_values, dtype=np.int32),
        np.asarray([row.components for row in rows], dtype=np.int32),
        np.asarray([np.nan if row.fiedler_value is None else row.fiedler_value for row in rows]),
    )
    (output / "e3_knn" / "k_scan.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2) + "\n"
    )
    print(f"E-4: degree and local clustering distributions for k={config.reference_k}", flush=True)
    connectivity = analyze_connectivity(reference.weights, output / "e4_connectivity")
    print(f"E-5: {config.eigen_count} eigenpairs and geographic partitions", flush=True)
    spectral = analyze_spectrum(
        reference.weights,
        matrix,
        selected,
        cube.latitude,
        cube.longitude,
        output / "e5_laplacian",
        config.cluster_counts,
        config.fiedler_bands,
        config.eigen_count,
        config.reference_clusters,
    )
    summary = {
        "configuration": asdict(config),
        "input_date_count": len(cube.dates),
        "missing_dates": cube.missing_dates,
        "source_sha256": dict(cube.source_hashes),
        "grid_shape": list(cube.depth.shape[1:]),
        "eligible_cells": int(selected.size),
        "feature_shape": list(features.shape),
        "e2_matrix_shape": list(matrix.shape),
        "e2_metric": "Euclidean on Hellinger-style gradient histogram + occurrence embedding",
        "reference_k": config.reference_k,
        "reference_clusters": config.reference_clusters,
        "k_scan": [asdict(row) for row in rows],
        "connectivity": asdict(connectivity),
        "spectrum": asdict(spectral),
        "stage_status": {
            "E-1": "completed",
            "E-2": "completed",
            "E-3": "completed",
            "E-4": "completed for the connected reference graph",
            "E-5": "completed",
            "E-6": "optional, not run",
            "E-7": "see docs/experiment-log.md",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved all outputs to {output}", flush=True)
