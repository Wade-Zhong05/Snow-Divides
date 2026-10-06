from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True, slots=True)
class NeighborTable:
    indices: NDArray[np.int32]
    distances: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SnowGraph:
    weights: csr_matrix
    components: int
    largest_fraction: float


def neighbors_from_matrix(distance_matrix: NDArray[np.float32], maximum_k: int) -> NeighborTable:
    """Rank each E-2 matrix row, excluding its diagonal, to form E-3 kNN candidates."""
    available = distance_matrix.copy()
    np.fill_diagonal(available, np.inf)
    indices = np.argsort(available, axis=1, kind="stable")[:, :maximum_k]
    distances = np.take_along_axis(available, indices, axis=1)
    return NeighborTable(indices.astype(np.int32), distances.astype(np.float32))


def graph_at_k(neighbors: NeighborTable, k: int) -> SnowGraph:
    """Symmetrize the first k directed candidates by union and self-tuned weights."""
    nodes = neighbors.indices.shape[0]
    targets = neighbors.indices[:, :k]
    distances = neighbors.distances[:, :k]
    scale = np.maximum(distances[:, -1], 1e-6)
    source = np.repeat(np.arange(nodes), k)
    target = targets.reshape(-1)
    width = np.sqrt(scale[source] * scale[target])
    similarity = np.exp(-np.square(distances.reshape(-1) / width))
    directed = coo_matrix((similarity, (source, target)), shape=(nodes, nodes)).tocsr()
    weights = directed.maximum(directed.T).tocsr()
    weights.setdiag(0)
    weights.eliminate_zeros()
    components, membership = connected_components(weights, directed=False)
    largest = np.bincount(membership).max() / nodes
    return SnowGraph(weights, int(components), float(largest))
