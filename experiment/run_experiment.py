"""HW1 Part II, E-1 to E-6: compute everything and save it.

    python -m snowdivides.data              # once: download the data (see README)
    python -m experiment.run_experiment     # ~15 min; writes results/experiment.npz, experiment/results.json
    python -m experiment.make_figures       # figures -> experiment/figures/

Pixels: 0.25 deg plateau cells with >= 60 snow days per year (snow years 2000/01-2013/14).
Vertices: pixels with identical 14-year depth series are contracted into one vertex
(feature-equivalent data, HW Part I 1.1.3).  Feature: multi-scale temporal HOG, lags
1, 7, 15, 31, 61 days (d = 1440), chosen by experiment/compare_features.py.
Pairwise matrix: Hellinger distance (sqrt-JS as a check).  kNN graphs for k = 1..64
(the assignment's range is [2^1, 2^6]); main analysis at k = 21.
"""
import json
import time

import networkx as nx
import numpy as np

from snowdivides import data
from snowdivides import features_alt as fa
from snowdivides import graph as g
from snowdivides.data import ROOT

FIRST, LAST = 2000, 2014            # calendar years -> snow years 2000/01 ... 2013/14
MIN_SNOW_DAYS = 60                  # mean days per year with depth > 0.5 cm
LAGS = (1, 7, 15, 31, 61)           # time lags of the multi-scale gradient (days)
K = 21                              # main kNN population
K_SCAN = range(1, 65)               # assignment: k in [2^1, 2^6]
N_EIG = 32                          # assignment: about 2^5 smallest eigenvalues
N_REGIONS = 3
RESULTS = ROOT / "results"
SNOW_YEARS = [f"{y}/{(y + 1) % 100:02d}" for y in range(FIRST, LAST)]
SENSOR_SWITCH = SNOW_YEARS.index("2008/09")    # SSM/I -> SSMIS (DMSP F17) inside this snow year


def feature(curves, n_cells=24):
    return fa.multiscale_hog(curves, LAGS, n_cells=n_cells)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def contract(curves):
    """Group pixels whose depth series are identical in every year and day.
    Returns (rep: one pixel index per vertex, pix2v: vertex index of every pixel)."""
    keys = np.ascontiguousarray(curves.transpose(1, 0, 2).reshape(curves.shape[1], -1))
    _, rep, pix2v = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(rep)                       # keep vertices in pixel order
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    return rep[order], rank[pix2v.ravel()]


def vertex_mean(values, pix2v):
    return np.bincount(pix2v, weights=values) / np.bincount(pix2v)


def vertex_edges(E_pix, pix2v):
    """Map-neighbour pixel pairs -> distinct vertex pairs (drops pairs inside one vertex)."""
    E = np.sort(pix2v[E_pix], axis=1)
    return np.unique(E[E[:, 0] != E[:, 1]], axis=0)


def name_regions(lab, lon):
    """Order K=3 cluster ids west -> east: 0 west, 1 interior, 2 southeast."""
    order = np.argsort([lon[lab == j].mean() for j in range(lab.max() + 1)])
    return np.argsort(order)[lab]


def glacier_check(lab_pix, plat, plon, glaciers):
    """Do the Yao et al. glacier regions fall into the expected three groups?
    westerly = V + IV, transition = II + VI + VII, monsoon = I (III is mixed, left out).
    lab_pix: cluster label of every pixel."""
    gl = np.array([(x["lat"], x["lon"]) for x in glaciers])
    near = g.haversine_km(gl[:, :1], gl[:, 1:2], plat[None], plon[None]).argmin(1)
    groups = {"westerly": {"V", "IV"}, "transition": {"II", "VI", "VII"}, "monsoon": {"I"}}
    major, hits, total = {}, 0, 0
    for name, regs in groups.items():
        cl = lab_pix[near[[i for i, x in enumerate(glaciers) if x["region"] in regs]]]
        major[name] = int(np.bincount(cl).argmax())
        hits += int(np.sum(cl == major[name])); total += len(cl)
    by_region = {R: np.bincount(lab_pix[near[[i for i, x in enumerate(glaciers) if x["region"] == R]]],
                                minlength=lab_pix.max() + 1).tolist() for R in data.REGION_NAMES}
    return dict(recovered=len(set(major.values())) == 3, purity=hits / total, by_region=by_region), near


def contiguity(lab, E, n_shuffle=200, seed=0):
    """Share of map-neighbour vertex pairs in the same cluster, and the same share
    after shuffling the labels (chance level)."""
    rng = np.random.default_rng(seed)
    same = (lab[E[:, 0]] == lab[E[:, 1]]).mean()
    null = [(p[E[:, 0]] == p[E[:, 1]]).mean() for p in (rng.permutation(lab) for _ in range(n_shuffle))]
    return float(same), float(np.mean(null))


def spectrum(D, k):
    _, A = g.knn_graph(D, k)
    W = g.gaussian_weights(D, A)
    w, Q = np.linalg.eigh(g.normalized_laplacian(W))
    return A, W, w, Q


def variant(curves, r, c, plat, plon, ref_pix, glaciers, **kw):
    """E-1 hyperparameter check: rebuild the graph with other cells or another pixel set.
    ref_pix: main-partition label of each of these pixels (-1 if not in the main set)."""
    rep, pix2v = contract(curves)
    F = feature(curves[:, rep], **kw)
    A, _, w, Q = spectrum(g.hellinger_distance(F), K)
    lab = name_regions(g.spectral_clusters(Q, N_REGIONS), vertex_mean(plon, pix2v))[pix2v]
    common = ref_pix >= 0
    _, agree = g.match_labels(lab[common], ref_pix[common])
    check, _ = glacier_check(lab, plat, plon, glaciers)
    return dict(n=int(len(rep)), d=int(F.shape[1]), lam=np.round(w[1:6], 4).tolist(),
                connected=bool(g.components(A).max() == 0), agreement=float(agree),
                contiguity=contiguity(lab, g.grid_edges(r, c), 50)[0],
                glacier_3way=check["recovered"], glacier_purity=check["purity"])


def main():
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    S = {}

    # ------------------------------------------------------------ E-1 data and features
    cube, lat, lon = data.snow_years(FIRST, LAST)
    snow_days = (cube > 0.5).sum(1).mean(0)
    all_r, all_c = np.nonzero(snow_days >= 30)                 # superset, for the threshold check
    curves_all = cube[:, :, all_r, all_c].transpose(0, 2, 1)
    sd_all = snow_days[all_r, all_c]
    keep = sd_all >= MIN_SNOW_DAYS
    r, c = all_r[keep], all_c[keep]                            # pixels
    plat, plon = lat[r], lon[c]
    curves = curves_all[:, keep]
    elev_pix = data.elevation(plat, plon)
    glaciers = data.glaciers()
    rep, pix2v = contract(curves)                              # vertices
    n = len(rep)
    vlat, vlon = vertex_mean(plat, pix2v), vertex_mean(plon, pix2v)
    velev = np.array([np.nanmean(e) if np.isfinite(e).any() else np.nan
                      for e in (elev_pix[pix2v == v] for v in range(n))])
    vsd = vertex_mean(sd_all[keep], pix2v)
    clim = curves[:, rep].mean(0)
    F = feature(curves[:, rep])
    multi = np.flatnonzero(np.bincount(pix2v) > 1)
    dup_cols = sorted({float(lon[c[pix2v == v].min()]) for v in multi})
    log(f"E-1: {cube.shape[2] * cube.shape[3]} cells in window, {len(r)} pixels with >= {MIN_SNOW_DAYS} snow days, "
        f"{len(r) - n} duplicate pixels contracted -> {n} vertices (left columns {dup_cols}), d = {F.shape[1]}")
    S["E1"] = dict(window_cells=int(cube.shape[2] * cube.shape[3]), pixels=int(len(r)), n=n, d=int(F.shape[1]),
                   lags=list(LAGS), contracted=int(len(r) - n), duplicate_columns=dup_cols,
                   snow_years=len(SNOW_YEARS), elev_median=float(np.nanmedian(velev)),
                   snow_days_median=float(np.median(vsd)))

    # ------------------------------------------------------------ E-2 pairwise matrix
    D = g.hellinger_distance(F)
    off = D[np.triu_indices(n, 1)]
    nb_main = g.knn_graph(D, K)[0]
    nb_js = g.knn_graph(g.js_distance(F), K)[0]
    overlap = np.mean([len(set(a) & set(b)) / K for a, b in zip(nb_main, nb_js)])
    log(f"E-2: Hellinger min {off.min():.3f} median {np.median(off):.3f} max {off.max():.3f}; "
        f"21-NN overlap with sqrt-JS {overlap:.2f}")
    S["E2"] = dict(min=float(off.min()), median=float(np.median(off)), max=float(off.max()),
                   q05=float(np.quantile(off, .05)), q95=float(np.quantile(off, .95)), js_overlap=float(overlap))

    # ------------------------------------------------------------ E-3 kNN graphs, components
    comp_labels = {k: g.components(g.knn_graph(D, k)[1]) for k in K_SCAN}
    ncomp = np.array([comp_labels[k].max() + 1 for k in K_SCAN])
    k_c = int(K_SCAN[int(np.argmax(ncomp == 1))])
    nbrs, A = g.knn_graph(D, K)
    edges = np.argwhere(np.triu(A, 1))
    geo = g.haversine_km(vlat[edges[:, 0]], vlon[edges[:, 0]], vlat[edges[:, 1]], vlon[edges[:, 1]])
    small = {}
    for k in K_SCAN:
        if k >= k_c or k < 2:
            continue
        sizes = np.bincount(comp_labels[k])
        m = comp_labels[k] != sizes.argmax()
        small[k] = dict(n_comp=int(len(sizes) - 1), n_vertices=int(m.sum()), sizes=sorted(sizes[sizes != sizes.max()].tolist()),
                        lon=[float(vlon[m].min()), float(vlon[m].max())],
                        share_east_90=float(np.mean(vlon[m] > 90)),
                        elev_median=float(np.nanmedian(velev[m])), snow_days_median=float(np.median(vsd[m])))
    log(f"E-3: components k=1..64 {ncomp.tolist()}, connected from k_c = {k_c}; {len(edges)} edges at k={K}; small {small}")
    S["E3"] = dict(ncomp=ncomp.tolist(), k_c=k_c, n_edges=int(len(edges)), small_components=small,
                   mutual_share=float(np.mean((nbrs[nbrs] == np.arange(n)[:, None, None]).any(-1))),
                   edge_km_median=float(np.median(geo)), edge_km_over_1000=float(np.mean(geo > 1000)))

    # ------------------------------------------------------------ E-4 degree and clustering
    deg, cc = g.degree_clustering(A)
    baselines = {}
    rng = np.random.default_rng(0)
    for dim in (2, 3, 4, 6, 10):
        db, cb = g.degree_clustering(g.knn_graph(g.euclidean_distance(rng.random((n, dim))), K)[1])
        baselines[dim] = dict(cc_median=float(np.median(cb)), deg_max=int(db.max()), deg_mean=float(db.mean()))
    x = np.sort(deg); ccdf = 1 - np.arange(n) / n; tail = x >= np.median(x)
    y = np.log(ccdf[tail])
    r2 = lambda X: float(1 - np.var(y - np.polyval(np.polyfit(X, y, 1), X)) / np.var(y))
    log(f"E-4: degree {deg.min():.0f}/{np.median(deg):.0f}/{deg.mean():.1f}/{deg.max():.0f}; CC median {np.median(cc):.3f}; {baselines}")
    S["E4"] = dict(deg_min=int(deg.min()), deg_median=float(np.median(deg)), deg_mean=float(deg.mean()),
                   deg_max=int(deg.max()), deg_skew=float(((deg - deg.mean()) ** 3).mean() / deg.std() ** 3),
                   tail_r2_exponential=r2(x[tail]), tail_r2_powerlaw=r2(np.log(x[tail])),
                   cc_median=float(np.median(cc)), cc_mean=float(cc.mean()),
                   cc_q05=float(np.quantile(cc, .05)), cc_q95=float(np.quantile(cc, .95)),
                   uniform_baselines=baselines, random_graph_cc=float(deg.mean() / (n - 1)))

    # ------------------------------------------------------------ E-5 Laplacian spectrum
    W = g.gaussian_weights(D, A)
    w, Q = np.linalg.eigh(g.normalized_laplacian(W))
    lam2_k = {k: float(np.linalg.eigvalsh(g.normalized_laplacian(g.gaussian_weights(D, g.knn_graph(D, k)[1])))[1])
              for k in range(k_c, K_SCAN[-1] + 1)}
    k_iso, gaps = g.most_isolated_eigenvalue(w, 1, N_EIG)
    domains = {i: g.nodal_domains(Q[:, i], A) for i in sorted({1, 2, 3, k_iso})}
    sgn = np.sign(np.corrcoef(Q[:, 1], vlon)[0, 1])         # report q2 with its sign as computed
    ok = np.isfinite(velev)
    band = ok & (velev >= 4400) & (velev <= 5000)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    q = Q[:, 1]
    dom_stats = []
    dl, ds = domains[k_iso]
    for d in range(dl.max() + 1):
        m = dl == d
        dom_stats.append(dict(sign=int(ds[d]), n=int(m.sum()), lon=float(vlon[m].mean()), lat=float(vlat[m].mean()),
                              elev=float(np.nanmean(velev[m])), snow_days=float(vsd[m].mean())))
    log(f"E-5: lambda_1..8 {np.round(w[:8], 4).tolist()}; most isolated = lambda_{k_iso + 1}; domains {dom_stats}")
    S["E5"] = dict(lam=np.round(w[:N_EIG], 5).tolist(), k_iso=k_iso + 1, iso_gap=float(gaps[k_iso - 1]),
                   gaps=np.round(gaps, 5).tolist(), lam2_k={str(k): v for k, v in lam2_k.items()},
                   q2_corr=dict(lon=corr(q, vlon), lat=corr(q, vlat), elev=corr(q[ok], velev[ok]),
                                snow_days=corr(q, vsd), lon_in_band=corr(q[band], vlon[band]),
                                elev_in_band=corr(q[band], velev[band]), band_n=int(band.sum())),
                   q2_positive_is_east=bool(sgn > 0),
                   domains={f"q{i + 1}": int(domains[i][0].max() + 1) for i in domains}, iso_domains=dom_stats)

    # ------------------------------------------------------------ E-6 clustering, space, years
    lab = name_regions(g.spectral_clusters(Q, N_REGIONS), vlon)
    k_robust = {}
    for k in (8, 16, 32, 64):
        _, _, _, Qk = spectrum(D, k)
        k_robust[str(k)] = g.ari(name_regions(g.spectral_clusters(Qk, N_REGIONS), vlon), lab)
    check, g_near = glacier_check(lab[pix2v], plat, plon, glaciers)
    E = vertex_edges(g.grid_edges(r, c), pix2v)
    same, null = contiguity(lab, E)
    dE = D[E[:, 0], E[:, 1]]
    cross = lab[E[:, 0]] != lab[E[:, 1]]
    top = dE >= np.quantile(dE, 0.9)
    long_ = geo > 1000
    regions = []
    for j in range(N_REGIONS):
        m = lab == j
        on = clim[m] > 0.3 * clim[m].max(1, keepdims=True)
        regions.append(dict(n=int(m.sum()), lon=float(vlon[m].mean()), lat=float(vlat[m].mean()),
                            elev=float(np.nanmean(velev[m])), snow_days=float(vsd[m].mean()),
                            start=float(np.median(on.argmax(1))), end=float(np.median(364 - on[:, ::-1].argmax(1))),
                            peak_cm=float(clim[m].mean(0).max()), peak_day=int(clim[m].mean(0).argmax())))
    log(f"E-6: regions {regions}; ARI vs k=21 for k=8/16/32/64: {k_robust}")
    log(f"     glacier check {check}; contiguity {same:.3f} vs shuffled {null:.3f}")

    G = nx.Graph()
    G.add_weighted_edges_from((int(i), int(j), float(W[i, j])) for i, j in edges)
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=0), key=len, reverse=True)
    louv = np.empty(n, int)
    for j, cm in enumerate(comms):
        louv[list(cm)] = j
    louv_purity = sum(np.bincount(lab[louv == j]).max() for j in range(len(comms))) / n
    log(f"     Louvain: {len(comms)} communities, purity w.r.t. K=3 regions {louv_purity:.2f}")

    # one graph per snow year: stable cores vs shifting belts; sensor check
    Fy, active = fa.yearly_multiscale_hog(curves[:, rep], LAGS)
    labs = np.full((len(SNOW_YEARS), n), -1)
    agree_y, lam2_y = [], []
    for i in range(len(SNOW_YEARS)):
        act = active[i]
        _, _, wy, Qy = spectrum(g.hellinger_distance(Fy[i][act]), K)
        ly, a = g.match_labels(g.spectral_clusters(Qy, N_REGIONS), lab[act])
        labs[i, act] = ly
        agree_y.append(float(a)); lam2_y.append(float(wy[1]))
    kept = (labs == lab).sum(0)
    snow_free = (~active).sum(1)
    log(f"     per-year agreement {np.round(agree_y, 2).tolist()}; snow-free {snow_free.tolist()}")

    # border sharpness: gradient of the unit-variance 3-D embedding painted on the map
    shape = snow_days.shape
    g2 = np.zeros(len(r))
    for i in (1, 2, 3):
        G_ = np.full(shape, np.nan); G_[r, c] = (Q[:, i] / Q[:, i].std())[pix2v]
        gr, gc = np.gradient(G_)
        g2 += np.nan_to_num((gr ** 2 + gc ** 2)[r, c])
    grad = vertex_mean(np.sqrt(g2), pix2v)
    touch = [set() for _ in range(n)]
    for i, j in E:
        touch[i].add(lab[j]); touch[j].add(lab[i])
    names = ["west", "interior", "southeast"]
    borders = {"not on a border": np.array([t <= {lab[i]} for i, t in enumerate(touch)])}
    for a_ in range(3):
        for b_ in range(a_ + 1, 3):
            borders[f"{names[a_]} | {names[b_]}"] = np.array(
                [(lab[i] == a_ and b_ in t) or (lab[i] == b_ and a_ in t) for i, t in enumerate(touch)])
    border_stats = {k_: dict(n=int(m.sum()), grad=float(np.median(grad[m])), switching=float(np.mean(kept[m] <= 7)),
                             cc=float(np.median(cc[m]))) for k_, m in borders.items()}
    log(f"     borders {border_stats}")

    S["E6"] = dict(regions=regions, k_robust=k_robust, glacier=check, contiguity=same, contiguity_null=null,
                   edge_dist_within=float(np.median(dE[~cross])), edge_dist_across=float(np.median(dE[cross])),
                   top10_on_border=float(cross[top].mean()), border_share=float(cross.mean()),
                   long_edges_same_region=float(np.mean(lab[edges[long_, 0]] == lab[edges[long_, 1]])),
                   louvain_n=len(comms), louvain_sizes=[len(cm) for cm in comms], louvain_purity=float(louv_purity),
                   per_year_agreement=agree_y, per_year_lam2=lam2_y, snow_free=snow_free.tolist(),
                   core=float(np.mean(kept >= 12)), transition=float(np.mean(kept <= 7)),
                   core_by_region=[float(np.mean(kept[lab == j] >= 12)) for j in range(3)],
                   borders=border_stats,
                   sensor=dict(snow_free_before=float(snow_free[:SENSOR_SWITCH].mean()),
                               snow_free_after=float(snow_free[SENSOR_SWITCH:].mean()),
                               lam2_before=float(np.mean(lam2_y[:SENSOR_SWITCH])),
                               lam2_after=float(np.mean(lam2_y[SENSOR_SWITCH:]))))

    # ------------------------------------------------------------ E-1 hyperparameter check
    lab_pix = lab[pix2v]
    V = {}
    for nc in (12, 48):
        V[f"{365 // nc}-day cells"] = variant(curves, r, c, plat, plon, lab_pix, glaciers, n_cells=nc)
    ref_all = np.full(len(all_r), -1); ref_all[keep] = lab_pix
    for th in (30, 90):
        m = sd_all >= th
        V[f">= {th} snow days"] = variant(curves_all[:, m], all_r[m], all_c[m], lat[all_r[m]], lon[all_c[m]],
                                          ref_all[m], glaciers)
    V["main"] = dict(n=n, d=int(F.shape[1]), lam=np.round(w[1:6], 4).tolist(), connected=True, agreement=1.0,
                     contiguity=same, glacier_3way=check["recovered"], glacier_purity=check["purity"])
    for k_, v in V.items():
        log(f"     variant {k_}: {v}")
    S["E1"]["variants"] = V

    np.savez_compressed(RESULTS / "experiment.npz", r=r, c=c, pix2v=pix2v, lat=lat, lon=lon, plat=plat, plon=plon,
                        vlat=vlat, vlon=vlon, velev=velev, vsd=vsd, snow_days_grid=snow_days, clim=clim, F=F,
                        D=D.astype(np.float32), off_sample=off[::50], ncomp=ncomp, comp2=comp_labels[2],
                        comp3=comp_labels[3], deg=deg, cc=cc, w=w[:40], Q=Q[:, :10], k_iso=k_iso,
                        dom_iso=domains[k_iso][0], dom_iso_sign=domains[k_iso][1], dom3=domains[2][0],
                        dom4=domains[3][0], lam2_k=np.array(list(lam2_k.items())), lab=lab, louv=louv, labs=labs,
                        kept=kept, E=E, dE=dE, g_near=g_near)
    (ROOT / "experiment" / "results.json").write_text(json.dumps(S, indent=1))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
