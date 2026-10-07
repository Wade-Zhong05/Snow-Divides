"""t-SNE of the snow-season features (Hellinger distances of the main run).

    python -m experiment.run_tsne        # ~15 min; needs results/experiment.npz

Runs: perplexity 5, 30, 100 from a PCA start, and perplexity 30 from a random start
(to see how much the picture depends on the start).  Scores for each map:
  nn_keep    share of each vertex's 21 nearest neighbours in the 2-D map that are also among
             its 21 nearest neighbours in the 1440-D feature space
  same_reg   share of those 2-D neighbours that are in the vertex's own K=3 region
  kmeans_ari ARI between k-means (K=3) on the 2-D map and the spectral regions
The 2-D Laplacian eigenmap (q2, q3) is scored the same way, for comparison.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from snowdivides import data
from snowdivides import graph as g
from snowdivides.data import ROOT
from snowdivides.tsne import tsne

K = 21
REGION_COLORS = ["#2a78d6", "#1baf7a", "#eb6834"]
REGION_NAMES = ["West", "Interior", "Southeast"]
INK2, GRID = "#52514e", "#e4e3df"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titlelocation": "left",
                     "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def knn_sets(X_or_D, precomputed=False):
    D = X_or_D if precomputed else g.euclidean_distance(X_or_D)
    return g.knn_graph(D, K)[0]


def score(Y, nb_feat, lab):
    nb = knn_sets(Y)
    keep = np.mean([len(set(a) & set(b)) / K for a, b in zip(nb, nb_feat)])
    same = np.mean(lab[nb] == lab[:, None])
    km = g.kmeans(Y / Y.std(0), 3, n_init=10)
    return dict(nn_keep=float(keep), same_reg=float(same), kmeans_ari=g.ari(km, lab))


def main():
    t0 = time.time()
    z = np.load(ROOT / "results" / "experiment.npz")
    D, F, lab, Q = z["D"].astype(float), z["F"], z["lab"], z["Q"]
    vlon, velev, vsd, kept, pix2v, g_near = (z[k] for k in ("vlon", "velev", "vsd", "kept", "pix2v", "g_near"))
    nb_feat = knn_sets(D, precomputed=True)

    X = np.sqrt(F) - np.sqrt(F).mean(0)                       # Hellinger distance = Euclidean on sqrt(F) / sqrt 2
    U, S_, _ = np.linalg.svd(X, full_matrices=False)
    pca = U[:, :2] * S_[:2]

    runs = {"perplexity 30 (PCA start)": dict(perplexity=30, init=pca),
            "perplexity 5 (PCA start)": dict(perplexity=5, init=pca),
            "perplexity 100 (PCA start)": dict(perplexity=100, init=pca),
            "perplexity 30 (random start)": dict(perplexity=30, init=None, seed=1)}
    maps, out = {}, {}
    for name, kw in runs.items():
        t1 = time.time()
        Y, kl = tsne(D, **kw)
        maps[name] = Y
        out[name] = dict(kl=kl, **score(Y, nb_feat, lab))
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {out[name]} ({time.time() - t1:.0f}s)", flush=True)
    eig = Q[:, 1:3] / Q[:, 1:3].std(0)
    out["Laplacian eigenmap (q2, q3)"] = score(eig, nb_feat, lab)
    out["PCA of sqrt(F)"] = score(pca, nb_feat, lab)
    a, b = maps["perplexity 30 (PCA start)"], maps["perplexity 30 (random start)"]
    out["start_agreement_nn"] = float(np.mean([len(set(x) & set(y)) / K for x, y in zip(knn_sets(a), knn_sets(b))]))
    print("eigenmap", out["Laplacian eigenmap (q2, q3)"], "PCA", out["PCA of sqrt(F)"],
          "PCA vs random start, 21-NN overlap", round(out["start_agreement_nn"], 3))
    (ROOT / "experiment" / "tsne.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "results" / "tsne.npz", **{k.replace(" ", "_"): v for k, v in maps.items()})

    # ---------------------------------------------------------------- figure
    gl = data.glaciers()
    gv = pix2v[g_near]                                         # glacier -> vertex
    main_map = maps["perplexity 30 (PCA start)"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    panels = [(main_map, "region", "(a) t-SNE, perplexity 30: K=3 regions, ▲ glaciers"),
              (main_map, vlon, "(b) same map: longitude (°E)"),
              (main_map, velev, "(c) same map: elevation (m)"),
              (main_map, kept, "(d) same map: years (of 14) in its region"),
              (maps["perplexity 5 (PCA start)"], "region", "(e) perplexity 5"),
              (maps["perplexity 100 (PCA start)"], "region", "(f) perplexity 100"),
              (maps["perplexity 30 (random start)"], "region", "(g) perplexity 30, random start"),
              (eig, "region", "(h) Laplacian eigenmap (q2, q3), for comparison")]
    for ax, (Y, col, title) in zip(axes.ravel(), panels):
        if isinstance(col, str):
            for j in range(3):
                m = lab == j
                ax.scatter(Y[m, 0], Y[m, 1], s=2.5, c=REGION_COLORS[j], label=REGION_NAMES[j], rasterized=True)
            if Y is main_map:
                for x, gi in zip(gl, gv):
                    ax.scatter(Y[gi, 0], Y[gi, 1], s=26, marker="^", facecolor="white", edgecolor="black", lw=0.7, zorder=3)
                ax.legend(frameon=False, fontsize=8, markerscale=4, loc="best")
        else:
            ok = np.isfinite(col)
            sc = ax.scatter(Y[ok, 0], Y[ok, 1], s=2.5, c=col[ok], cmap=SEQ, rasterized=True)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.outline.set_visible(False)
        key = {"(e)": "perplexity 5 (PCA start)", "(f)": "perplexity 100 (PCA start)",
               "(g)": "perplexity 30 (random start)", "(h)": "Laplacian eigenmap (q2, q3)"}.get(title[:3])
        if title[:3] == "(a)":
            key = "perplexity 30 (PCA start)"
        if key:
            s_ = out[key]
            title += f"\nneighbours kept {s_['nn_keep']:.2f} · same region {s_['same_reg']:.2f} · k-means ARI {s_['kmeans_ari']:.2f}"
        ax.set_title(title, fontsize=8.8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.savefig(ROOT / "experiment" / "figures" / "tsne.png")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
