"""Exact t-SNE (van der Maaten & Hinton 2008) on a precomputed distance matrix, numpy only.

t-SNE is the 'soft fall-off' neighbourhood of HW 1.3: instead of keeping the k nearest
neighbours, every point i gets a Gaussian conditional distribution over all others,
p(j|i) ~ exp(-D_ij^2 / 2 sigma_i^2), with sigma_i tuned so that the distribution has a
given perplexity (an effective number of neighbours).  The 2-D map is then fitted so that
a heavy-tailed (Student-t) neighbourhood distribution among the map points matches P.
O(n^2) memory and time per iteration: fine for a few thousand points.
"""
import numpy as np


def conditional_affinities(D, perplexity, tol=1e-5, max_iter=60):
    """Row-stochastic P(j|i) from distances D, with per-row Gaussian widths found by
    bisection so that exp(entropy of row i) = perplexity."""
    n = len(D)
    D2 = D.astype(float) ** 2
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        d = np.delete(D2[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0 / max(np.median(d), 1e-12)
        for _ in range(max_iter):
            e = np.exp(-(d - d.min()) * beta)
            s = e.sum()
            p = e / s
            H = beta * np.sum(p * (d - d.min())) + np.log(s)
            if abs(H - target) < tol:
                break
            if H > target:                       # too flat: narrow the Gaussian
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    return P


def tsne(D, perplexity=30.0, n_iter=1000, init=None, seed=0, exaggeration=12.0, exag_iter=250,
         learning_rate=None):
    """2-D t-SNE embedding of the points with distance matrix D.
    init: optional (n, 2) starting layout (rescaled to a small standard deviation).
    Returns (Y, final KL divergence)."""
    n = len(D)
    P = conditional_affinities(D, perplexity)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12).astype(np.float32)          # single precision: about 2x faster
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, 2)) if init is None else np.asarray(init, float).copy()
    Y = ((Y - Y.mean(0)) / Y.std(0) * 1e-4).astype(np.float32)
    lr = learning_rate or max(n / exaggeration, 50.0)      # Belkina et al. (2019) heuristic
    update, gains = np.zeros_like(Y), np.ones_like(Y)
    for it in range(n_iter):
        exag = exaggeration if it < exag_iter else 1.0
        momentum = 0.5 if it < exag_iter else 0.8
        sq = (Y ** 2).sum(1)
        num = 1.0 / (1.0 + np.maximum(sq[:, None] + sq[None, :] - 2 * Y @ Y.T, 0.0))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        M = (exag * P - Q) * num
        grad = 4.0 * (M.sum(1)[:, None] * Y - M @ Y)
        gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8).clip(0.01)
        update = momentum * update - lr * gains * grad
        Y = Y + update
        Y -= Y.mean(0)
    kl = float(np.sum(P.astype(float) * np.log(P / Q)))
    return Y.astype(float), kl
