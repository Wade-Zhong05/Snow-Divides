"""Temporal HOG of a snow-depth curve (HW1 optional question 1-b), numpy only.

A pixel's depth curve s(t) over one snow year is treated as a shape in the
(day, depth) plane.  The orientation of each day is the slope angle
theta = arctan(s'(t) / c), so snowfall (rising) and melt (falling) point in
opposite directions, and steep vs gentle change fall in different bins.  Each day
votes |s'(t)| into its orientation bin; votes are summed over half-month cells,
and blocks of consecutive cells are L2-Hys normalised (Dalal & Triggs 2005).
The result describes *when* and *how fast* snow comes and goes, not how much there is.
"""
import numpy as np

SLOPE_BINS = ["fast melt", "melt", "slow melt", "slow accum.", "accum.", "fast accum."]


def _smooth(s, sigma=2.0):
    """Gaussian smoothing along time; the series ends are extended by repetition."""
    r = int(round(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    T = s.shape[-1]
    return sum(w * s[..., np.clip(np.arange(T) + o, 0, T - 1)] for o, w in zip(range(-r, r + 1), k))


def cell_histograms(s, n_cells=24, n_orient=6, c=0.3, sigma=2.0):
    """s: (..., 365) daily depth (cm) -> (..., n_cells, n_orient).

    Bins split (-90, 90) deg evenly.  With n_orient = 6 and c = 0.3 cm/day the bin
    boundaries are at 0, +-0.08 and +-0.3 cm/day (fast melt ... fast accumulation).
    """
    s = _smooth(np.asarray(s, float), sigma)
    T = s.shape[-1]
    rate = 0.5 * (s[..., np.minimum(np.arange(T) + 1, T - 1)] - s[..., np.maximum(np.arange(T) - 1, 0)])
    mag = np.abs(rate)
    pos = np.clip((np.arctan2(rate, c) + np.pi / 2) / np.pi * n_orient - 0.5, 0, n_orient - 1)
    b0 = np.floor(pos).astype(int)
    w1 = pos - b0
    b1 = np.minimum(b0 + 1, n_orient - 1)
    cell = (np.arange(T) * n_cells) // T
    H = np.zeros(s.shape[:-1] + (n_cells, n_orient))
    for b in range(n_orient):
        v = mag * ((b0 == b) * (1 - w1) + (b1 == b) * w1)
        for j in range(n_cells):
            H[..., j, b] = v[..., cell == j].sum(-1)
    return H


def block_normalize(H, block=2, clip=0.2, eps=1.0):
    """Blocks of `block` consecutive cells, L2-Hys.  The last block wraps from late July
    to early August (both snow-free summer).  eps (cm of depth change) keeps nearly
    snow-free blocks near zero instead of inflating them to unit length."""
    B = np.concatenate([np.roll(H, -i, axis=-2) for i in range(block)], axis=-1)
    B = B / np.sqrt((B ** 2).sum(-1, keepdims=True) + eps ** 2)
    B = np.minimum(B, clip)
    return B / np.sqrt((B ** 2).sum(-1, keepdims=True) + eps ** 2)


def _as_distribution(v):
    return v / np.maximum(v.sum(-1, keepdims=True), 1e-12)


def yearly_hog(curves, n_cells=24, **kw):
    """curves (n_years, n, 365) -> per-year features (n_years, n, d), each summing to 1,
    and a mask (n_years, n) of pixel-years whose depth changed at all."""
    B = block_normalize(cell_histograms(curves, n_cells=n_cells, **kw))
    F = B.reshape(B.shape[:-2] + (-1,))
    return _as_distribution(F), F.sum(-1) > 1e-6


def temporal_hog(curves, n_cells=24, **kw):
    """curves (n_years, n, 365) -> (n, d): block-normalised HOG averaged over years,
    normalised to sum 1 (a distribution).  d = n_cells * 2 * 6 = 288 for half-month cells."""
    B = block_normalize(cell_histograms(curves, n_cells=n_cells, **kw))
    return _as_distribution(B.mean(0).reshape(B.shape[1], -1))


def raw_curve(curves, n_cells=24):
    """Baseline descriptor: mean depth per cell (climatology), normalised to sum 1."""
    clim = curves.mean(0)
    cell = (np.arange(clim.shape[-1]) * n_cells) // clim.shape[-1]
    return _as_distribution(np.stack([clim[:, cell == j].mean(1) for j in range(n_cells)], 1))
