"""Graph construction and analysis (E-2 to E-6), numpy only; dense matrices are fine
for a few thousand vertices."""
import itertools

import numpy as np


# ---------------------------------------------------------------- E-2 pairwise matrices

def js_distance(P):
    """sqrt of the Jensen-Shannon divergence (base 2) between rows of P (distributions).
    JS(p, q) = H((p + q) / 2) - (H(p) + H(q)) / 2; its square root is a metric in [0, 1]."""
    xlogx = lambda X: X * np.log2(np.where(X > 0, X, 1.0))
    H = -xlogx(P).sum(1)
    D = np.empty((len(P), len(P)))
    for i, p in enumerate(P):
        D[i] = -xlogx(0.5 * (p + P)).sum(1) - 0.5 * (H[i] + H)
    np.fill_diagonal(D, 0.0)
    return np.sqrt(np.maximum(D, 0.0))


def euclidean_distance(X):
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def hellinger_distance(P):
    """||sqrt p - sqrt q|| / sqrt 2: a metric in [0, 1] like sqrt-JS, but one matrix product."""
    return euclidean_distance(np.sqrt(P)) / np.sqrt(2)


# ---------------------------------------------------------------- E-3 kNN graph

def knn_graph(D, k):
    """Sort each row of D, keep the k nearest -> directed kNN lists; symmetrise by union
    (an edge if either vertex is among the other's k nearest).  Returns (nbrs, A)."""
    n = len(D)
    nbrs = np.argsort(D + np.diag(np.full(n, np.inf)), axis=1)[:, :k]
    A = np.zeros((n, n), bool)
    A[np.repeat(np.arange(n), k), nbrs.ravel()] = True
    return nbrs, A | A.T


def components(A):
    """Connected-component label of every vertex (iterative depth-first search)."""
    label = np.full(len(A), -1)
    c = 0
    for s in range(len(A)):
        if label[s] >= 0:
            continue
        label[s], stack = c, [s]
        while stack:
            u = stack.pop()
            new = np.flatnonzero(A[u] & (label < 0))
            label[new] = c
            stack.extend(new)
        c += 1
    return label


# ---------------------------------------------------------------- E-4 combinatorial measures

def degree_clustering(A):
    """Degree d_i and local clustering coefficient t_i / C(d_i, 2), where
    t_i = ((A^2 o A) 1)_i / 2 is the number of triangles through i (HW 1.2-h)."""
    Af = A.astype(float)
    deg = Af.sum(1)
    tri = ((Af @ Af) * Af).sum(1) / 2
    pairs = deg * (deg - 1) / 2
    return deg, np.divide(tri, pairs, out=np.zeros_like(tri), where=pairs > 0)


# ---------------------------------------------------------------- E-5 spectral analysis

def gaussian_weights(D, A, k_scale=7):
    """w_ij = exp(-d_ij^2 / (s_i s_j)) on the edges of A, s_i = distance to the k_scale-th
    neighbour (self-tuning scale, Zelnik-Manor & Perona 2004)."""
    s = np.sort(D, axis=1)[:, k_scale]
    if (s == 0).any():
        raise ValueError("duplicate feature vectors: contract feature-equivalent vertices first")
    return np.where(A, np.exp(-D ** 2 / np.outer(s, s)), 0.0)


def normalized_laplacian(W):
    r = 1.0 / np.sqrt(W.sum(1))
    return np.eye(len(W)) - r[:, None] * W * r[None, :]


def nodal_domains(q, A):
    """Strong nodal domains of q: connected components of the subgraphs induced by
    {q > 0} and {q < 0}.  Returns (domain label per vertex, sign of each domain)."""
    lab, signs = np.full(len(q), -1), []
    for s in (1, -1):
        idx = np.flatnonzero(s * q > 0)
        sub = components(A[np.ix_(idx, idx)])
        lab[idx] = sub + len(signs)
        signs += [s] * (sub.max() + 1)
    return lab, np.array(signs)


def most_isolated_eigenvalue(w, first=1, last=25):
    """Index i in [first, last) maximising min(w[i] - w[i-1], w[i+1] - w[i]):
    the nonzero eigenvalue best separated from its neighbours."""
    gaps = [min(w[i] - w[i - 1], w[i + 1] - w[i]) for i in range(first, last)]
    return first + int(np.argmax(gaps)), gaps


# ---------------------------------------------------------------- E-6 clustering

def kmeans(X, K, n_init=20, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    best, best_lab = np.inf, None
    for _ in range(n_init):
        C = X[rng.choice(len(X), K, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(K)])
            if np.allclose(newC, C):
                break
            C = newC
        cost = ((X - C[lab]) ** 2).sum()
        if cost < best:
            best, best_lab = cost, lab
    return best_lab


def spectral_clusters(Q, K):
    """Ng, Jordan & Weiss (2002): rows of the K lowest eigenvectors, unit-normalised, k-means."""
    U = Q[:, :K] / np.linalg.norm(Q[:, :K], axis=1, keepdims=True)
    return kmeans(U, K)


def match_labels(lab, ref):
    """Permute cluster ids of `lab` to agree best with `ref`; returns (relabelled, agreement)."""
    K = int(max(lab.max(), ref.max())) + 1
    C = np.array([[np.sum((lab == a) & (ref == b)) for b in range(K)] for a in range(K)])
    best = max(itertools.permutations(range(K)), key=lambda p: sum(C[a, p[a]] for a in range(K)))
    return np.array(best)[lab], sum(C[a, best[a]] for a in range(K)) / len(lab)


# ---------------------------------------------------------------- spatial graph

def grid_edges(r, c):
    """Edges between vertices that are 8-neighbours on the map grid (rNN with r = 1 cell)."""
    pos = {(a, b): i for i, (a, b) in enumerate(zip(r, c))}
    E = [(i, pos[(a + da, b + db)]) for i, (a, b) in enumerate(zip(r, c))
         for da, db in ((0, 1), (1, -1), (1, 0), (1, 1)) if (a + da, b + db) in pos]
    return np.array(E)


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def ari(a, b):
    """Adjusted Rand index between two labelings (1 = identical, about 0 = unrelated)."""
    C = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(C, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2
    s, sa, sb = comb(C).sum(), comb(C.sum(1)).sum(), comb(C.sum(0)).sum()
    exp = sa * sb / comb(len(a))
    return float((s - exp) / ((sa + sb) / 2 - exp))
