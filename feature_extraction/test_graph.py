from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from data_io import SnowCube
from snow_graph import border_counts
from time_histogram import HistogramConfig, extract


def test_histogram_ignores_missing_day_and_snow_free_zero_change() -> None:
    # Given: one snow year, a missing day, and a second snow-free date pair.
    dates = tuple(date(2000, 8, 1) + timedelta(days=offset) for offset in range(365))
    depth = np.zeros((365, 1, 1), dtype=np.float32)
    depth[1:3, 0, 0] = (1, 2)
    depth[3] = np.nan
    depth[4:6, 0, 0] = (3, 4)
    cube = SnowCube(depth, dates, np.array([35.0]), np.array([90.0]), (), ())
    config = HistogramConfig(2, 9, 1.0, 0.0, 0.5)

    # When: the signed daily-gradient histogram is extracted.
    result = extract(cube, config)

    # Then: three +1 and one -4 cm/day snow-active pairs remain; later zeros add nothing.
    assert result.counts.sum() == 4
    assert result.counts[0, 0, 0, 6] == 3
    assert result.counts[0, 0, 0, 0] == 1
    assert result.snow_days[0, 0, 0] == 4
    assert result.valid_days[0, 0, 0] == 15


def test_border_counts_uses_original_grid_neighbors() -> None:
    # Given: two same-label neighbors, one different-label neighbor, one excluded cell.
    labels = np.array([0, 0, 1, -1], dtype=np.int32)

    # When: four-neighbor border counts are measured after clustering.
    disagreement, support = border_counts(labels, (2, 2))

    # Then: only valid adjacent pairs contribute; the single different pair is counted twice.
    np.testing.assert_array_equal(disagreement, np.array([[1, 0], [1, 0]]))
    np.testing.assert_array_equal(support, np.array([[2, 1], [1, 0]]))
