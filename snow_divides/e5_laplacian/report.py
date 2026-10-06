from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from snow_divides.e5_laplacian.interactive import embedding_3d_html
from snow_divides.e5_laplacian.matrices import clustered_matrix_figure
from snow_divides.e5_laplacian.plots import cluster_maps, embedding_3d, fiedler_maps, spectrum_plot
from snow_divides.e5_laplacian.spectral import (
    fiedler_partition,
    smallest_eigenpairs,
    spectral_clusters,
)


@dataclass(frozen=True, slots=True)
class SpectralSummary:
    eigenvalues: tuple[float, ...]
    gap_after_count: tuple[tuple[int, float], ...]
    cluster_sizes: tuple[tuple[int, tuple[int, ...]], ...]
    fiedler_band_thresholds: tuple[tuple[float, float], ...]


def analyze_spectrum(
    weights: csr_matrix,
    distance: NDArray[np.float32],
    selected: NDArray[np.int64],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
    output: Path,
    cluster_counts: tuple[int, ...],
    bands: tuple[float, ...],
    eigen_count: int,
    reference_clusters: int,
) -> SpectralSummary:
    """Persist E-5 spectrum, candidate partitions, Fiedler cuts, and geographic maps."""
    spectrum = smallest_eigenpairs(weights, eigen_count)
    partitions = {
        count: spectral_clusters(spectrum.eigenvectors, count) for count in cluster_counts
    }
    fiedler = spectrum.eigenvectors[:, 1]
    band_labels = {band: fiedler_partition(fiedler, band) for band in bands}
    np.savez_compressed(
        output / "spectral_results.npz",
        eigenvalues=spectrum.eigenvalues,
        eigenvectors=spectrum.eigenvectors,
        cluster_counts=np.asarray(cluster_counts, dtype=np.int32),
        cluster_labels=np.stack([partitions[count] for count in cluster_counts]),
        fiedler_bands=np.asarray(bands),
        fiedler_labels=np.stack([band_labels[band] for band in bands]),
    )
    spectrum_plot(output / "01_laplacian_spectrum.png", spectrum.eigenvalues, cluster_counts)
    cluster_maps(output / "02_cluster_maps.png", partitions, selected, latitude, longitude)
    cluster_maps(
        output / "02_reference_clusters.png",
        {reference_clusters: partitions[reference_clusters]},
        selected,
        latitude,
        longitude,
    )
    fiedler_maps(output / "03_fiedler_maps.png", fiedler, bands, selected, latitude, longitude)
    embedding_3d(output / "04_embedding_3d.png", spectrum.eigenvectors, band_labels[bands[0]])
    embedding_3d_html(
        output / "05_embedding_3d_interactive.html",
        spectrum.eigenvectors,
        band_labels[0.0],
        partitions[reference_clusters],
        selected,
        latitude,
        longitude,
    )
    clustered_matrix_figure(
        output / "06_clustered_matrices.png", distance, weights, partitions[reference_clusters]
    )
    return SpectralSummary(
        tuple(float(value) for value in spectrum.eigenvalues),
        tuple(
            (count, float(spectrum.eigenvalues[count] - spectrum.eigenvalues[count - 1]))
            for count in cluster_counts
        ),
        tuple(
            (count, tuple(int(size) for size in np.bincount(labels, minlength=count)))
            for count, labels in partitions.items()
        ),
        tuple(
            (band, 0.0 if band == 0 else float(np.quantile(np.abs(fiedler), band)))
            for band in bands
        ),
    )
