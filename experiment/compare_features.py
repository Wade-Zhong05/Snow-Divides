"""Feature comparison: same vertices, same graph pipeline, different features.

    python -m experiment.compare_features      # ~10 min -> results/compare.npz, experiment/feature_comparison.json

For each feature: kNN graph (k = 21) -> self-tuning Gaussian weights -> normalized
Laplacian -> K = 3 spectral clustering.  The graph is built five times per feature:
on all 14 snow years, on odd / even years (split-half reliability), and on the years
before / from 2008/09 (the SSM/I -> SSMIS sensor change).

Scores (no ground truth exists, so several independent checks):
  glacier  - Yao et al. (2012) groups V+IV, II+VI+VII, I land in 3 different regions
             (recovered) and share of the 74 glaciers in their group's region (purity)
  contig   - share of map-neighbour pairs in the same region (chance ~0.36); the
             features contain no location, so this measures geographic signal
  climate  - |corr(q2, longitude)| among vertices at 4400-5000 m (signal not due to elevation)
  elev_eta - share of elevation variance explained by the 3 regions (lower = less elevation-driven)
  split    - ARI between partitions built on two disjoint halves of the years (7 / 7),
             mean and sd over 5 random splits (robustness to noise)
  split_nn - share of each vertex's 21 nearest neighbours that the two half-year graphs
             have in common (reliability of the graph itself, no clustering involved)
  era, era_nn - the same between pre-2008 (8 years) and post-2008 (6 years); era below
             split hints at sensitivity to the sensor change (or a real decadal change)
  vs_main  - ARI with the partition of the main feature (report)
  k_c, cc  - smallest k with a connected kNN graph; median clustering coefficient at k = 21
"""
import json
import time

import numpy as np

from experiment.run_experiment import (FIRST, LAST, MIN_SNOW_DAYS, contiguity, contract, glacier_check,
                                       name_regions, vertex_edges, vertex_mean)
from snowdivides import data, features
from snowdivides import features_alt as fa
from snowdivides import graph as g
from snowdivides.data import ROOT

K = 21
YEARS = np.arange(LAST - FIRST)                      # 14 snow years, index 8 = 2008/09
N_SPLITS = 5                                         # random 7 / 7 year splits for reliability
_rng = np.random.default_rng(0)
SPLITS = [np.sort(_rng.permutation(YEARS)[:7]) for _ in range(N_SPLITS)]
SUBSETS = {"all": YEARS, "pre": YEARS[:8], "post": YEARS[8:]}
for _i, _a in enumerate(SPLITS):
    SUBSETS[f"s{_i}a"], SUBSETS[f"s{_i}b"] = _a, np.setdiff1d(YEARS, _a)

# name -> (family, function of curves, distance, one-line principle)
METHODS = {
    "raw depth (no gradient)": ("baseline", features.raw_curve, "hellinger",
                                "share of the year's snow depth in each half-month"),
    "snow occurrence only": ("baseline", fa.snow_occurrence, "euclidean",
                             "share of days with snow in each half-month (no gradient)"),
    "teammate hf102 (default)": ("single lag", fa.teammate_features, "euclidean",
                                 "day counts per daily-change bin per half-month (Hellinger) + snow-day share, 50/50"),
    "teammate hf102 (gradient only)": ("single lag", lambda x: fa.teammate_features(x, occurrence_weight=0.0),
                                       "euclidean", "the same without the snow-day share"),
    "HOG, lag 1": ("single lag", lambda x: fa.lag_hog(x, 1), "hellinger",
                   "daily change, each day votes |change|; half-month cells, block-normalised"),
    "HOG, smoothed (main)": ("single lag", features.temporal_hog, "hellinger",
                             "as above after 2-day Gaussian smoothing (the report's feature)"),
    "HOG, lag 3": ("single lag", lambda x: fa.lag_hog(x, 3), "hellinger", "change over 3 days"),
    "HOG, lag 7": ("single lag", lambda x: fa.lag_hog(x, 7), "hellinger", "change over 1 week"),
    "HOG, lag 15": ("single lag", lambda x: fa.lag_hog(x, 15), "hellinger", "change over 2 weeks"),
    "HOG, lag 31": ("single lag", lambda x: fa.lag_hog(x, 31), "hellinger", "change over 1 month"),
    "HOG, lag 61": ("single lag", lambda x: fa.lag_hog(x, 61), "hellinger", "change over 2 months"),
    "multi-scale HOG (1-31)": ("multi-scale", fa.multiscale_hog, "hellinger",
                               "lags 1, 3, 7, 15, 31 concatenated"),
    "multi-scale HOG (7-61)": ("multi-scale", lambda x: fa.multiscale_hog(x, (7, 15, 31, 61)), "hellinger",
                               "lags 7, 15, 31, 61 concatenated"),
    "multi-scale HOG (1, 7-61)": ("multi-scale", lambda x: fa.multiscale_hog(x, (1, 7, 15, 31, 61)), "hellinger",
                                  "lags 1, 7, 15, 31, 61, equal weights"),
    "multi-scale HOG (1 half, 7-61)": ("multi-scale",
                                       lambda x: fa.multiscale_hog(x, (1, 7, 15, 31, 61), (0.5, 1, 1, 1, 1)),
                                       "hellinger", "lags 1, 7, 15, 31, 61, the 1-day block at half weight"),
    "Haar wavelet energy": ("signal", fa.haar_scalogram, "hellinger",
                            "energy of box-smoothed gradients at 2-64-day scales, per month and sign"),
    "Fourier harmonics": ("signal", fa.fourier, "euclidean",
                          "annual harmonics 1-6 relative to mean depth (strength, timing, sharpness)"),
    "phenology statistics": ("signal", fa.phenology, "euclidean",
                             "onset, melt-out, duration, peak day, episodes, flashiness (z-scored)"),
}
MAIN = "HOG, smoothed (main)"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def weights(D, A, k_scale=7):
    """Self-tuning Gaussian weights; a zero scale (exact duplicates) falls back to the smallest positive one."""
    s = np.sort(D, axis=1)[:, k_scale]
    s = np.where(s > 0, s, s[s > 0].min())
    return np.where(A, np.exp(-D ** 2 / np.outer(s, s)), 0.0)


def build(F, metric, vlon):
    D = g.hellinger_distance(F) if metric == "hellinger" else g.euclidean_distance(F)
    nbrs, A = g.knn_graph(D, K)
    w, Q = np.linalg.eigh(g.normalized_laplacian(weights(D, A)))
    return D, A, w, Q, name_regions(g.spectral_clusters(Q, 3), vlon), nbrs


def nn_overlap(a, b):
    """Mean share of common entries between two kNN lists per vertex."""
    return float(np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b)]))


def main(methods=None, out_name="feature_comparison"):
    """methods: subset of METHODS names (default all); results go to experiment/<out_name>.json."""
    t0 = time.time()
    todo = {k: METHODS[k] for k in (methods or METHODS)}
    cube, lat, lon = data.snow_years(FIRST, LAST)
    snow_days = (cube > 0.5).sum(1).mean(0)
    r, c = np.nonzero(snow_days >= MIN_SNOW_DAYS)
    curves_px = cube[:, :, r, c].transpose(0, 2, 1)
    plat, plon = lat[r], lon[c]
    rep, pix2v = contract(curves_px)
    curves = curves_px[:, rep]
    n = len(rep)
    vlon = vertex_mean(plon, pix2v)
    elev_px = data.elevation(plat, plon)
    velev = np.array([np.nanmean(e) if np.isfinite(e).any() else np.nan
                      for e in (elev_px[pix2v == v] for v in range(n))])
    E = vertex_edges(g.grid_edges(r, c), pix2v)
    glaciers = data.glaciers()
    band = np.isfinite(velev) & (velev >= 4400) & (velev <= 5000)
    ok = np.isfinite(velev)
    log(f"{n} vertices, {len(E)} map-neighbour pairs")

    rows, labels = {}, {}
    for name, (family, fn, metric, principle) in todo.items():
        t1 = time.time()
        parts, nn = {}, {}
        for sub, idx in SUBSETS.items():
            F = fn(curves[idx])
            res = build(F, metric, vlon)
            parts[sub], nn[sub] = res[4], res[5]
            if sub == "all":
                D, A, w, Q, lab, _ = res
                d = F.shape[1]
        split = [g.ari(parts[f"s{i}a"], parts[f"s{i}b"]) for i in range(N_SPLITS)]
        split_nn = [nn_overlap(nn[f"s{i}a"], nn[f"s{i}b"]) for i in range(N_SPLITS)]
        check, _ = glacier_check(lab[pix2v], plat, plon, glaciers)
        k_c = next(k for k in range(1, 31) if g.components(g.knn_graph(D, k)[1]).max() == 0)
        _, cc = g.degree_clustering(A)
        q2 = Q[:, 1]
        eta = sum(np.sum(ok & (lab == j)) * (np.nanmean(velev[lab == j]) - np.nanmean(velev)) ** 2
                  for j in range(3)) / (np.nanvar(velev) * ok.sum())
        rows[name] = dict(family=family, principle=principle, d=int(d), metric=metric,
                          glacier_recovered=bool(check["recovered"]), glacier_purity=float(check["purity"]),
                          glacier_by_region=check["by_region"],
                          contiguity=contiguity(lab, E, 50)[0],
                          climate=float(abs(np.corrcoef(q2[band], vlon[band])[0, 1])),
                          elev_eta=float(eta),
                          split=float(np.mean(split)), split_sd=float(np.std(split)),
                          split_nn=float(np.mean(split_nn)),
                          era=g.ari(parts["pre"], parts["post"]), era_nn=nn_overlap(nn["pre"], nn["post"]),
                          k_c=int(k_c), cc=float(np.median(cc)),
                          lam=np.round(w[1:5], 4).tolist(),
                          sizes=np.bincount(lab, minlength=3).tolist())
        labels[name] = lab
        log(f"{name:28s} d={d:4d} {json.dumps({k: v for k, v in rows[name].items() if k in ('glacier_recovered', 'glacier_purity', 'contiguity', 'climate', 'elev_eta', 'split', 'split_sd', 'split_nn', 'era', 'era_nn')})} ({time.time() - t1:.0f}s)")
    for name in rows:
        rows[name]["vs_main"] = g.ari(labels[name], labels[MAIN]) if MAIN in labels else None

    (ROOT / "experiment" / f"{out_name}.json").write_text(json.dumps(rows, indent=1))
    if out_name != "feature_comparison":
        log(f"done in {time.time() - t0:.0f}s")
        return
    np.savez_compressed(ROOT / "results" / "compare.npz", r=r, c=c, pix2v=pix2v, lat=lat, lon=lon,
                        shape=np.array(snow_days.shape), curves_example=curves[:, :: 300],
                        names=np.array(list(labels)), labels=np.stack(list(labels.values())))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
