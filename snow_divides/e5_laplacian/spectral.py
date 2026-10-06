from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


@dataclass(frozen=True, slots=True)
class Spectrum:
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]


def smallest_eigenpairs(weights: csr_matrix, count: int) -> Spectrum:
    """Solve the normalized graph Laplacian with a fixed initial vector."""
    normalized = laplacian(weights, normed=True)
    values, vectors = eigsh(normalized, k=count, which="SM", v0=np.ones(weights.indptr.size - 1))
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return Spectrum(values, vectors)


def spectral_clusters(vectors: NDArray[np.float64], cluster_count: int) -> NDArray[np.int32]:
    """Cluster row-normalized first-m Laplacian eigenvectors with fixed-seed KMeans."""
    embedding = vectors[:, :cluster_count]
    lengths = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = np.divide(embedding, lengths, out=np.zeros_like(embedding), where=lengths > 0)
    labels = KMeans(n_clusters=cluster_count, n_init="auto", random_state=0).fit_predict(normalized)
    return labels.astype(np.int32)


def fiedler_partition(vector: NDArray[np.float64], band_quantile: float) -> NDArray[np.int32]:
    """Use the sign nodal cut; a positive band marks near-zero vertices as uncertain."""
    threshold = 0.0 if band_quantile == 0 else float(np.quantile(np.abs(vector), band_quantile))
    labels = np.full(vector.size, 1, dtype=np.int32)
    labels[vector < -threshold] = 0
    labels[vector > threshold] = 2
    return labels
