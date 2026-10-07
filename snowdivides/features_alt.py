"""Alternative features of a snow-depth signal, for the feature comparison
(experiment/compare_features.py).  Every function maps curves (n_years, n, 365),
day 0 = 1 Aug, to a feature matrix (n, d).

Families
  baseline        raw_curve (in features.py): mean depth per half-month, no gradient
  single lag      count_histogram: teammate's version, each day one vote
                  lag_hog: magnitude-weighted slope HOG at a chosen time lag
  multi-scale     multiscale_hog: lag_hog at several lags, concatenated
  signal          haar_scalogram (wavelet energy), fourier (annual harmonics),
                  phenology (hand-crafted season statistics)
"""
import numpy as np

from .features import _as_distribution, block_normalize


def lag_rate(s, lag):
    """Rate of change over `lag` days, in cm/day: (s[t + a] - s[t - b]) / lag with
    a = ceil(lag / 2), b = floor(lag / 2) (lag 1 = forward difference)."""
    T = s.shape[-1]
    t = np.arange(T)
    a, b = (lag + 1) // 2, lag // 2
    return (s[..., np.minimum(t + a, T - 1)] - s[..., np.maximum(t - b, 0)]) / lag


def _slope_bins(rate, n_bins, c):
    """Fractional bin position of the slope angle arctan(rate / c) over (-90, 90) deg."""
    return (np.arctan2(rate, c) + np.pi / 2) / np.pi * n_bins - 0.5


def count_histogram(curves, lag=1, n_windows=12, n_bins=9, c=0.3):
    """Teammate's feature: each day with snow is one vote for its slope bin (hard
    binning, 9 bins of 20 deg, zero change included), monthly windows, and each window
    normalised to a conditional distribution over bins.  Averaged over years."""
    s = np.asarray(curves, float)
    rate = lag_rate(s, lag)
    T = s.shape[-1]
    snowy = (s > 0) | (s[..., np.minimum(np.arange(T) + 1, T - 1)] > 0)
    b = np.clip(np.round(_slope_bins(rate, n_bins, c)).astype(int), 0, n_bins - 1)
    win = (np.arange(T) * n_windows) // T
    H = np.zeros(s.shape[:-1] + (n_windows, n_bins))
    for w in range(n_windows):
        sw, bw = snowy[..., win == w], b[..., win == w]
        for k in range(n_bins):
            H[..., w, k] = (sw & (bw == k)).sum(-1)
    H = H / np.maximum(H.sum(-1, keepdims=True), 1)            # conditional on the window
    return _as_distribution(H.mean(0).reshape(s.shape[1], -1))


def _gaussian_filter_windows(X, sigma, axis):
    """Gaussian smoothing along `axis` with edge ('nearest') padding, truncated at 4 sigma
    (the defaults of scipy.ndimage.gaussian_filter1d, which the teammate's code uses)."""
    if sigma <= 0:
        return X
    r = int(4 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    X = np.moveaxis(X, axis, -1)
    n = X.shape[-1]
    out = sum(w * X[..., np.clip(np.arange(n) + o, 0, n - 1)] for o, w in zip(range(-r, r + 1), k))
    return np.moveaxis(out, -1, axis)


def teammate_features(curves, occurrence_weight=0.5, n_windows=24, n_bins=9, scale=1.0, smoothing=0.5):
    """Re-implementation of the hf102 branch feature (snow_divides/e1_features/histogram.py):
    lag-1 daily change d(t) - d(t-1) on days with snow on either day, never across the
    1-Aug snow-year boundary; each day is one count in 9 hard bins of
    atan(change / 1 cm/day); counts pooled over years per half-month window, Gaussian-
    smoothed across windows (sigma 0.5), turned into per-window distributions and
    square-rooted (Hellinger embedding), weighted by sqrt((1 - w) / (2 W)); concatenated
    with the per-window share of snow days (depth > 0) weighted by sqrt(w / W).
    Euclidean distance.  Windows here are 24 equal parts of the snow year instead of
    exact calendar half-months."""
    s = np.asarray(curves, float)
    Y, n, T = s.shape
    cur, prev = s[..., 1:], s[..., :-1]                      # day 0 = 1 Aug is never differenced
    active = (cur > 0) | (prev > 0)
    b = np.clip(np.floor((np.arctan2(cur - prev, scale) + np.pi / 2) / np.pi * n_bins).astype(int), 0, n_bins - 1)
    win_all = (np.arange(T) * n_windows) // T
    win = win_all[1:]
    counts = np.zeros((n_windows, n, n_bins))
    for w in range(n_windows):
        aw, bw = active[..., win == w], b[..., win == w]
        for k in range(n_bins):
            counts[w, :, k] = (aw & (bw == k)).sum(axis=(0, 2))
    raw = _gaussian_filter_windows(counts, smoothing, axis=0)
    tot = raw.sum(-1, keepdims=True)
    prob = np.divide(raw, tot, out=np.zeros_like(raw), where=tot > 0)
    grad_part = np.sqrt(prob).transpose(1, 0, 2).reshape(n, -1) * np.sqrt((1 - occurrence_weight) / (2 * n_windows))
    snow = np.stack([(s[..., win_all == w] > 0).sum(axis=(0, 2)) for w in range(n_windows)], 1)
    valid = np.stack([np.full(n, Y * np.sum(win_all == w)) for w in range(n_windows)], 1)
    occ = snow / valid * np.sqrt(occurrence_weight / n_windows)
    return np.concatenate([grad_part, occ], axis=1)


def snow_occurrence(curves, n_windows=24):
    """Baseline: share of days with snow (depth > 0) in each half-month window, all years pooled."""
    s = np.asarray(curves, float)
    win = (np.arange(s.shape[-1]) * n_windows) // s.shape[-1]
    return np.stack([(s[..., win == w] > 0).mean(axis=(0, 2)) for w in range(n_windows)], 1)


def _soft_histograms(rate, n_cells=24, n_orient=6, c=0.3):
    """Magnitude-weighted soft-binned slope histograms, (..., n_cells, n_orient)."""
    mag = np.abs(rate)
    pos = np.clip(_slope_bins(rate, n_orient, c), 0, n_orient - 1)
    b0 = np.floor(pos).astype(int)
    w1 = pos - b0
    b1 = np.minimum(b0 + 1, n_orient - 1)
    T = rate.shape[-1]
    cell = (np.arange(T) * n_cells) // T
    H = np.zeros(rate.shape[:-1] + (n_cells, n_orient))
    for b in range(n_orient):
        v = mag * ((b0 == b) * (1 - w1) + (b1 == b) * w1)
        for j in range(n_cells):
            H[..., j, b] = v[..., cell == j].sum(-1)
    return H


def lag_hog(curves, lag=1, n_cells=24):
    """Temporal HOG (as in features.temporal_hog) but with the rate taken over `lag`
    days and no extra smoothing: a difference over lag days already averages out
    changes shorter than the lag.  d = 288."""
    rate = lag_rate(np.asarray(curves, float), lag)
    B = block_normalize(_soft_histograms(rate, n_cells))
    return _as_distribution(B.mean(0).reshape(rate.shape[1], -1))


def multiscale_hog(curves, lags=(1, 3, 7, 15, 31), weights=None, n_cells=24):
    """lag_hog at several time lags, each normalised, concatenated (d = 288 x len(lags)).
    weights: relative weight of each lag's block (default: equal)."""
    weights = np.ones(len(lags)) if weights is None else np.asarray(weights, float)
    return _as_distribution(np.concatenate([w * lag_hog(curves, L, n_cells) for L, w in zip(lags, weights)], axis=1))


def yearly_multiscale_hog(curves, lags=(1, 7, 15, 31, 61), n_cells=24):
    """Per-year version of multiscale_hog: (n_years, n, d) features, each summing to 1,
    and a (n_years, n) mask of pixel-years whose depth changed at all."""
    s = np.asarray(curves, float)
    blocks = []
    for L in lags:
        B = block_normalize(_soft_histograms(lag_rate(s, L), n_cells))
        blocks.append(_as_distribution(B.reshape(B.shape[:-2] + (-1,))))
    F = np.concatenate(blocks, axis=-1)
    changed = np.abs(np.diff(s, axis=-1)).sum(-1) > 1e-6
    return _as_distribution(F), changed


def haar_scalogram(curves, scales=(2, 4, 8, 16, 32, 64), n_windows=12):
    """Haar wavelet energy map.  At scale L the detail coefficient is
    d_L(t) = mean(s[t : t + L/2]) - mean(s[t - L/2 : t]), a gradient of the signal
    smoothed by a box of L/2 days.  Its energy d^2 is summed per month, per scale and
    per sign (accumulation vs melt): a time x scale x sign distribution (d = 144)."""
    s = np.asarray(curves, float)
    T = s.shape[-1]
    t = np.arange(T)
    win = (t * n_windows) // T
    out = []
    for L in scales:
        h = L // 2
        P = np.pad(s, [(0, 0)] * (s.ndim - 1) + [(h, h)], mode="edge")
        cs = np.concatenate([np.zeros(s.shape[:-1] + (1,)), np.cumsum(P, axis=-1)], axis=-1)
        p = t + h
        d = (cs[..., p + h] - 2 * cs[..., p] + cs[..., p - h]) / h        # right mean - left mean
        e_pos, e_neg = np.where(d > 0, d ** 2, 0), np.where(d < 0, d ** 2, 0)
        out.append(np.stack([np.stack([e[..., win == w].sum(-1) for w in range(n_windows)], -1)
                             for e in (e_pos, e_neg)], -1))             # (Y, n, windows, 2)
    E = np.stack(out, -2)                                                # (Y, n, windows, scales, 2)
    E = E / np.maximum(E.sum(axis=(-1, -2, -3), keepdims=True), 1e-12)  # each year sums to 1
    return _as_distribution(E.mean(0).reshape(s.shape[1], -1))


def fourier(curves, n_harmonics=6):
    """Annual harmonics 1..n of each year's curve, divided by the year's mean depth
    (scale-free), averaged over years as complex numbers, as (real, imag) pairs.
    Harmonic 1 amplitude = strength of the seasonal cycle, its phase = timing of the
    snow peak; higher harmonics = how sharp the onset and melt are.  d = 2n (Euclidean)."""
    Fh = np.fft.rfft(np.asarray(curves, float), axis=-1)
    dc = Fh[..., :1].real
    Z = np.where(dc > 1e-6, Fh[..., 1:n_harmonics + 1] / np.maximum(dc, 1e-6), np.nan)
    Z = np.nanmean(Z.real, 0) + 1j * np.nanmean(Z.imag, 0)
    return np.concatenate([Z.real, Z.imag], axis=1)


PHENOLOGY_NAMES = ["onset day", "melt-out day", "snow days", "peak day", "snow episodes",
                   "accumulation days", "melt days", "flash share"]


def phenology(curves, thr=0.5):
    """Hand-crafted season statistics per year, averaged over years and z-scored
    (Euclidean distance): onset, melt-out, number of snow days, peak day, number of
    separate snow episodes, onset-to-peak and peak-to-melt-out durations, and the share
    of accumulation that falls on days gaining > 1 cm ('flashiness')."""
    s = np.asarray(curves, float)
    on = s > thr
    has = on.any(-1)
    onset = on.argmax(-1).astype(float)
    melt = (s.shape[-1] - 1 - on[..., ::-1].argmax(-1)).astype(float)
    peak = s.argmax(-1).astype(float)
    episodes = (on[..., 1:] & ~on[..., :-1]).sum(-1) + on[..., 0]
    dd = np.diff(s, axis=-1)
    pos = np.clip(dd, 0, None)
    flash = (pos * (dd > 1)).sum(-1) / np.maximum(pos.sum(-1), 1e-6)
    X = np.stack([onset, melt, on.sum(-1), peak, episodes, peak - onset, melt - peak, flash], -1).astype(float)
    X[~has] = np.nan
    X = np.nanmean(X, 0)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
