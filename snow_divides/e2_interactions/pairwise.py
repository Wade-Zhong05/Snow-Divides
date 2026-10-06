from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances


def all_pair_distances(features: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the weighted feature distance for every ordered pair of eligible cells."""
    distances = pairwise_distances(features, metric="euclidean", n_jobs=-1)
    distances = distances.astype(np.float32, copy=False)
    np.fill_diagonal(distances, 0)
    return distances
