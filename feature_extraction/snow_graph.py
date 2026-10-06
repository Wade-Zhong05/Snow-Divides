"""Build snow-feature kNN graphs and measure their geographic interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True, slots=True)
class NeighborSearch:
    indices: NDArray[np.int32]
    distances: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class GraphResult:
    weights: csr_matrix
    labels: NDArray[np.int32]
    eigenvalues: NDArray[np.float64]
    components: int
    largest_component_fraction: float


def find_neighbors(features: NDArray[np.float32], maximum_k: int) -> NeighborSearch:
    """Search solely in snow-feature space; geographic coordinates are absent."""
    model = NearestNeighbors(n_neighbors=maximum_k + 1, metric="euclidean", n_jobs=-1)
    distances, indices = model.fit(features).kneighbors(features)
    nonself = indices != np.arange(features.shape[0])[:, None]
    return NeighborSearch(
        indices[nonself].reshape(features.shape[0], maximum_k).astype(np.int32),
        distances[nonself].reshape(features.shape[0], maximum_k).astype(np.float32),
    )


def build_graph(neighbors: NeighborSearch, k: int, cluster_count: int) -> GraphResult:
    """Union kNN with self-tuning Gaussian weights; partition normalized-Laplacian vectors."""
    nodes = neighbors.indices.shape[0]
    targets = neighbors.indices[:, :k]
    distances = neighbors.distances[:, :k]
    local_scale = np.maximum(distances[:, -1], 1e-6)
    source = np.repeat(np.arange(nodes), k)
    target = targets.reshape(-1)
    width = np.sqrt(local_scale[source] * local_scale[target])
    similarity = np.exp(-np.square(distances.reshape(-1) / width))
    directed = coo_matrix((similarity, (source, target)), shape=(nodes, nodes)).tocsr()
    weights = directed.maximum(directed.T).tocsr()
    weights.setdiag(0)
    weights.eliminate_zeros()
    components, component_labels = connected_components(weights, directed=False)
    largest = np.bincount(component_labels).max() / nodes
    matrix = laplacian(weights, normed=True)
    eigenvalues, vectors = eigsh(matrix, k=cluster_count + 1, which="SM", v0=np.ones(nodes))
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    embedding = vectors[:, order[:cluster_count]]
    row_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = np.divide(embedding, row_norm, out=np.zeros_like(embedding), where=row_norm > 0)
    labels = KMeans(n_clusters=cluster_count, n_init="auto", random_state=0).fit_predict(embedding)
    return GraphResult(
        weights,
        labels.astype(np.int32),
        eigenvalues,
        int(components),
        float(largest),
    )


def edge_lengths_km(
    weights: csr_matrix,
    pixel_indices: NDArray[np.int64],
    latitude: NDArray[np.float64],
    longitude: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Geographic distances are measured after the snow graph has been built."""
    row, col = weights.nonzero()
    upper = row < col
    width = longitude.size
    source_pixels = pixel_indices[row[upper]]
    target_pixels = pixel_indices[col[upper]]
    lat1 = np.deg2rad(latitude[source_pixels // width])
    lat2 = np.deg2rad(latitude[target_pixels // width])
    dlat = lat2 - lat1
    dlon = np.deg2rad(longitude[target_pixels % width] - longitude[source_pixels % width])
    central = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(central, 0, 1)))


def border_counts(
    labels: NDArray[np.int32], shape: tuple[int, int]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Count disagreements on the original four-neighbor grid, independently of graph edges."""
    grid = labels.reshape(shape)
    disagreement = np.zeros(shape, dtype=np.int32)
    support = np.zeros(shape, dtype=np.int32)
    for first, second in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
        valid = (first >= 0) & (second >= 0)
        different = valid & (first != second)
        if first.shape[0] == shape[0]:
            disagreement[:, :-1] += different
            disagreement[:, 1:] += different
            support[:, :-1] += valid
            support[:, 1:] += valid
        else:
            disagreement[:-1, :] += different
            disagreement[1:, :] += different
            support[:-1, :] += valid
            support[1:, :] += valid
    return disagreement, support
