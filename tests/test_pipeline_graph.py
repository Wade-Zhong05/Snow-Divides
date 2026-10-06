from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from snow_divides.e2_interactions.pairwise import all_pair_distances
from snow_divides.e3_knn.graph import graph_at_k, neighbors_from_matrix
from snow_divides.e4_connectivity.report import graph_degrees_and_clustering
from snow_divides.e5_laplacian.spectral import fiedler_partition, smallest_eigenpairs


def test_pairwise_matrix_precedes_knn_and_changes_connectivity() -> None:
    # Given: two close pairs of grid-cell features separated by a larger feature distance.
    features = np.array([[0, 0], [0, 1], [4, 0], [4, 1]], dtype=np.float32)

    # When: E-2 forms every pair distance and E-3 takes one or two neighbors per row.
    distances = all_pair_distances(features)
    neighbors = neighbors_from_matrix(distances, 2)
    sparse = graph_at_k(neighbors, 1)
    connected = graph_at_k(neighbors, 2)

    # Then: the E-2 matrix is complete, while k changes the sparse graph's connectivity.
    np.testing.assert_allclose(distances[0], [0, 1, 4, np.sqrt(17)], atol=1e-6)
    assert distances.shape == (4, 4)
    assert sparse.components == 2
    assert connected.components == 1
    assert (connected.weights - connected.weights.T).nnz == 0
    assert np.all(connected.weights.diagonal() == 0)


def test_fiedler_sign_cut_separates_connected_pairs() -> None:
    # Given: a connected k=2 graph built from the same four feature vectors.
    features = np.array([[0, 0], [0, 1], [4, 0], [4, 1]], dtype=np.float32)
    graph = graph_at_k(neighbors_from_matrix(all_pair_distances(features), 2), 2)

    # When: E-5 computes the normalized-Laplacian spectrum and Fiedler nodal signs.
    spectrum = smallest_eigenpairs(graph.weights, 3)
    labels = fiedler_partition(spectrum.eigenvectors[:, 1], 0)

    # Then: zero is the first eigenvalue and each close pair shares one Fiedler side.
    assert abs(spectrum.eigenvalues[0]) < 1e-5
    assert spectrum.eigenvalues[1] > 0
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_local_clustering_counts_closed_triangles_without_edge_weights() -> None:
    # Given: a triangle with a one-edge tail; the positive edge weights differ.
    weights = csr_matrix(
        np.array(
            [[0, 2, 4, 0], [2, 0, 3, 0], [4, 3, 0, 5], [0, 0, 5, 0]],
            dtype=np.float64,
        )
    )

    # When: E-4 measures combinatorial degrees and local clustering.
    degree, clustering = graph_degrees_and_clustering(weights)

    # Then: the tail lowers only the third triangle vertex's coefficient.
    np.testing.assert_array_equal(degree, [2, 2, 3, 1])
    np.testing.assert_allclose(clustering, [1, 1, 1 / 3, 0])
