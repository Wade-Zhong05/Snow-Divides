"""Figures for the HW1 Part II report, from results/experiment.npz and experiment/results.json.

    python -m experiment.make_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from snowdivides import data, features
from snowdivides import features_alt as fa
from snowdivides.data import ROOT

FIG = ROOT / "experiment" / "figures"
INK, INK2, GRID, EMPTY = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
REGION_COLORS = ["#2a78d6", "#1baf7a", "#eb6834"]           # west, interior, southeast
REGION_NAMES = ["West", "Interior", "Southeast"]
POS, NEG, OTHER = "#2a78d6", "#eb6834", "#b8b7b2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ0 = LinearSegmentedColormap.from_list("seq0", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MONTHS = (["Aug", "Oct", "Dec", "Feb", "Apr", "Jun"], [0, 61, 122, 184, 243, 304])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

z = np.load(ROOT / "results" / "experiment.npz")
S = json.loads((ROOT / "experiment" / "results.json").read_text())
r, c, lat, lon, pix2v = (z[k] for k in ("r", "c", "lat", "lon", "pix2v"))
plat, plon = z["vlat"], z["vlon"]              # vertex positions (mean of the contracted pixels)
lab, n = z["lab"], len(z["lab"])
SHAPE = z["snow_days_grid"].shape
EXT = [lon[0] - .125, lon[-1] + .125, lat[-1] - .125, lat[0] + .125]
GL = data.glaciers()


def paint(values, fill=np.nan):
    """Per-vertex values -> map grid (every pixel of a contracted vertex gets its value)."""
    G = np.full(SHAPE, fill, float)
    G[r, c] = np.asarray(values, float)[pix2v]
    return G


def basemap(ax, title=None):
    ax.imshow(np.where(np.isnan(paint(np.zeros(n))), 1.0, np.nan), extent=EXT, cmap=ListedColormap([EMPTY]))
    ax.set_xlim(EXT[:2]); ax.set_ylim(EXT[2:])
    ax.set_xticks(range(75, 106, 5)); ax.set_yticks(range(26, 41, 2))
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title)


def categorical_map(ax, values, colors, title=None):
    basemap(ax, title)
    G = paint(values.astype(float))
    ax.imshow(np.ma.masked_invalid(G), extent=EXT, cmap=ListedColormap(colors), vmin=-0.5,
              vmax=len(colors) - 0.5, interpolation="nearest")


def glacier_marks(ax, labels=True):
    gl = np.array([(x["lon"], x["lat"]) for x in GL])
    ax.scatter(gl[:, 0], gl[:, 1], s=16, marker="^", facecolor="white", edgecolor=INK, linewidth=0.8, zorder=3)
    if labels:
        for R in data.REGION_NAMES:
            p = np.array([(x["lon"], x["lat"]) for x in GL if x["region"] == R]).mean(0)
            ax.annotate(R, p, xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


def legend_squares(ax, names, colors, **kw):
    for nm, col in zip(names, colors):
        ax.scatter([], [], c=col, marker="s", s=30, label=nm)
    ax.legend(frameon=False, fontsize=8, **kw)


def month_axis(ax, scale=1.0):
    ax.set_xticks(np.array(MONTHS[1]) * scale, MONTHS[0])


# ------------------------------------------------------------------ E-1
def fig_e1():
    D = z["D"]
    idx = [np.flatnonzero(lab == j)[np.argmin(D[np.ix_(lab == j, lab == j)].mean(1))] for j in range(3)]
    clim = z["clim"]
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.45, wspace=0.42)
    ax = fig.add_subplot(gs[0, :2])
    basemap(ax, f"(a) {len(r)} pixels → {n} vertices (≥ 60 snow days/yr); ▲ glaciers of Yao et al. (2012)")
    im = ax.imshow(np.ma.masked_invalid(paint(z["vsd"])), extent=EXT, cmap=SEQ, vmin=60, vmax=365,
                   interpolation="nearest")
    glacier_marks(ax)
    for j, i in enumerate(idx):
        ax.scatter(plon[i], plat[i], s=70, marker="o", facecolor=REGION_COLORS[j], edgecolor="white", lw=1.5, zorder=5)
        ax.annotate("ABC"[j], (plon[i], plat[i]), xytext=(6, -10), textcoords="offset points", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    for j, i in enumerate(idx):
        ax.plot(clim[i], color=REGION_COLORS[j], lw=1.8, label=f"pixel {'ABC'[j]} ({plon[i]:.1f}°E, {plat[i]:.1f}°N)")
    month_axis(ax); ax.set_ylabel("depth (cm), mean of 14 years")
    ax.set_title("(b) Attribute: daily snow depth")
    ax.legend([f"cell {c_}" for c_ in "ABC"], frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)

    lags = S["E1"]["lags"]
    for j, i in enumerate(idx):
        ax = fig.add_subplot(gs[1, j])
        # one (6 slope bins x 24 half-months) block per lag, each scaled to its own maximum
        # (computed on the 14-year mean curve, for display)
        blocks = []
        for L in lags:
            H = fa._soft_histograms(fa.lag_rate(clim[i][None].astype(float), L)).squeeze(0)   # (24, 6)
            blocks.append(H.T / max(H.max(), 1e-9))
        ax.imshow(np.vstack(blocks), aspect="auto", origin="lower", cmap=SEQ0,
                  extent=[0, 24, -0.5, 6 * len(lags) - 0.5], interpolation="nearest")
        for b in range(1, len(lags)):
            ax.axhline(6 * b - 0.5, color="white", lw=1.5)
        ax.set_yticks([6 * b + 2.5 for b in range(len(lags))], [f"lag {L} d" for L in lags], fontsize=7.5)
        ax.set_xticks(np.array(MONTHS[1]) * 24 / 365, MONTHS[0])
        ax.set_title(f"({'cde'[j]}) Multi-scale HOG of cell {'ABC'[j]}")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(FIG / "e1_data_features.png")
    plt.close(fig)


# ------------------------------------------------------------------ E-2
def fig_e2():
    D = z["D"]
    order = np.lexsort((plon, lab))
    sub = order[::3]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.35))
    im = ax[0].imshow(D[np.ix_(sub, sub)], cmap=SEQ.reversed(), vmin=0, vmax=np.quantile(D, 0.98), interpolation="nearest")
    b = np.searchsorted(lab[sub], [1, 2])
    for x in b:
        ax[0].axhline(x - 0.5, color="white", lw=1.2); ax[0].axvline(x - 0.5, color="white", lw=1.2)
    ticks = (np.r_[0, b] + np.r_[b, len(sub)]) / 2
    ax[0].set_xticks(ticks, REGION_NAMES); ax[0].set_yticks(ticks, REGION_NAMES)
    ax[0].set_title("(a) Hellinger matrix (every 3rd vertex), sorted by K=3 region")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.04); cb.outline.set_visible(False); cb.set_label("Hellinger distance")
    off = z["off_sample"]
    ax[1].hist(off, bins=80, color="#3987e5", edgecolor="white", linewidth=0.3)
    e2 = S["E2"]
    ax[1].set_title(f"(b) All pairs: median {e2['median']:.2f}, 5–95% {e2['q05']:.2f}–{e2['q95']:.2f}")
    ax[1].set_xlabel("Hellinger distance"); ax[1].set_ylabel("pairs (1 in 50 sampled)")
    fig.savefig(FIG / "e2_pairwise.png")
    plt.close(fig)


# ------------------------------------------------------------------ E-3
def fig_e3():
    nc, kc = np.array(S["E3"]["ncomp"]), S["E3"]["k_c"]
    ks = np.arange(1, len(nc) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    ax[0].plot(ks, nc, color="#2a78d6", lw=2, marker="o", ms=3)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    ax[0].set_yscale("log"); ax[0].set_xlabel("k"); ax[0].set_ylabel("connected components")
    ax[0].annotate(f"connected from k = {kc}", (kc, 1), xytext=(12, 25), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", color=INK2), fontsize=8)
    ax[0].axvline(21, color=INK2, ls=":", lw=1)
    ax[0].annotate("main k = 21", (21, 1), xytext=(4, 40), textcoords="offset points", fontsize=8, color=INK2)
    ax[0].set_title("(a) Components of the kNN graph, k = 1 … 64")
    ax[0].grid(axis="y", color=GRID, lw=0.6)
    ax[1].axis("off")
    ax[2].remove()
    ax[1].remove()
    mp = fig.add_subplot(1, 3, (2, 3))
    basemap(mp, "(b) Vertices outside the giant component for k < k_c")
    for kk, col, size, mk in [(2, "#2a78d6", 26, "s"), (3, "#eb6834", 60, "D")]:
        comp = z[f"comp{kk}"]
        giant = np.bincount(comp).argmax()
        m = comp != giant
        if not m.any():
            continue
        mp.scatter(plon[m], plat[m], s=size, c=col, marker=mk, edgecolor="white", lw=0.5, zorder=3,
                   label=f"k = {kk}: {m.sum()} vertices in {len(np.unique(comp)) - 1} small component(s)")
    mp.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(FIG / "e3_components.png")
    plt.close(fig)


# ------------------------------------------------------------------ E-4
def fig_e4():
    deg, cc = z["deg"], z["cc"]
    e4 = S["E4"]
    fig = plt.figure(figsize=(15, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(deg, bins=np.arange(deg.min() - 0.5, deg.max() + 1.5, 2), color="#3987e5", edgecolor="white", lw=0.3)
    ax.set_xlabel("degree"); ax.set_ylabel("pixels")
    ax.set_title(f"(a) Degree: min {e4['deg_min']}, median {e4['deg_median']:.0f}, max {e4['deg_max']}")

    ax = fig.add_subplot(gs[0, 1])
    x = np.sort(deg)
    ccdf = 1 - np.arange(len(x)) / len(x)
    ax.semilogy(x, ccdf, color="#2a78d6", lw=2, label="data")
    tail = x >= np.median(x)
    slope, icpt = np.polyfit(x[tail], np.log(ccdf[tail]), 1)
    ax.semilogy(x[tail], np.exp(icpt + slope * x[tail]), "--", color=INK2, lw=1.2,
                label=f"exponential tail, rate {-slope:.3f}")
    ax.set_xlabel("degree d"); ax.set_ylabel("P(degree ≥ d)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Degree CCDF (log-linear): straight tail = exponential")
    ax.grid(color=GRID, lw=0.6)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(cc, bins=50, color="#3987e5", edgecolor="white", lw=0.3, label="snow graph")
    cols = {2: "#eb6834", 3: "#1baf7a", 4: "#4a3aa7", 6: "#e87ba4", 10: INK2}
    for dim, b in e4["uniform_baselines"].items():
        ax.axvline(b["cc_median"], color=cols[int(dim)], lw=1.5, ls="--", label=f"uniform points in {dim}-D")
    ax.set_xlabel("local clustering coefficient"); ax.set_ylabel("pixels"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"(c) Clustering: median {e4['cc_median']:.2f} (random graph ≈ {e4['random_graph_cc']:.3f})")

    bottom = gs[1, :].subgridspec(1, 2, wspace=0.12)
    for j, (vals, name, vmin, vmax) in enumerate([(deg, "degree", deg.min(), np.quantile(deg, 0.98)),
                                                   (cc, "local clustering coefficient", np.quantile(cc, .02), np.quantile(cc, .98))]):
        ax = fig.add_subplot(bottom[0, j])
        basemap(ax, f"({'de'[j]}) {name.capitalize()} on the map")
        im = ax.imshow(np.ma.masked_invalid(paint(vals)), extent=EXT, cmap=SEQ, vmin=vmin, vmax=vmax, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.outline.set_visible(False)
    fig.savefig(FIG / "e4_degree_clustering.png")
    plt.close(fig)


# ------------------------------------------------------------------ E-5
def fig_e5():
    w, Q = z["w"], z["Q"]
    e5 = S["E5"]
    ki = int(z["k_iso"])
    dom, dsign = z["dom_iso"], z["dom_iso_sign"]
    sizes = np.bincount(dom)
    big = np.argsort(sizes)[::-1][:2]
    dcol = np.full(n, 2.0)
    for rank, d in enumerate(big):
        dcol[dom == d] = 0 if dsign[d] > 0 else 1
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    kk = np.arange(1, 33)
    ax.plot(kk, w[:32], "o", color="#86b6ef", ms=5)
    ax.plot(ki + 1, w[ki], "o", color="#eb6834", ms=8, label=f"λ{ki + 1} = {w[ki]:.4f}: best separated")
    ax.set_xlabel("index i"); ax.set_ylabel("λ_i of normalized Laplacian"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) 32 smallest eigenvalues (k = 21)"); ax.grid(color=GRID, lw=0.6)

    ax = fig.add_subplot(gs[0, 1])
    l2 = z["lam2_k"]
    ax.plot(l2[:, 0], l2[:, 1], color="#2a78d6", lw=2, marker="o", ms=3)
    ax.axvline(21, color=INK2, ls=":", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
    ax.set_xlabel("k"); ax.set_ylabel("Fiedler value λ2"); ax.grid(color=GRID, lw=0.6)
    ax.set_title(f"(b) λ2(k) for k_c = {S['E3']['k_c']} ≤ k ≤ 64 (dotted: main k = 21)")

    ax = fig.add_subplot(gs[0, 2], projection="3d")
    cmap = [POS, NEG, OTHER]
    dims = [ki] + [d for d in (1, 2, 3) if d != ki][:2]          # q_k and two other eigenvectors
    for v in (2, 0, 1):
        m = dcol == v
        ax.scatter(Q[m, dims[0]], Q[m, dims[1]], Q[m, dims[2]], s=3, c=cmap[v], depthshade=False)
    for setlim, d_ in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), dims):
        setlim(*np.quantile(Q[:, d_], [0.01, 0.99]))
    ax.set_xlabel(f"q{dims[0] + 1}", labelpad=-6); ax.set_ylabel(f"q{dims[1] + 1}", labelpad=-6)
    ax.set_zlabel(f"q{dims[2] + 1}", labelpad=-6)
    ax.tick_params(labelsize=0, length=0)
    ax.view_init(elev=22, azim=-58)
    ax.set_title(f"(c) Embedding (q{dims[0] + 1}, q{dims[1] + 1}, q{dims[2] + 1}), central 98%;\n"
                 f"colour = nodal domains of q{ki + 1}", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 0])
    categorical_map(ax, dcol, cmap, f"(d) Nodal domains of q{ki + 1} on the map")
    glacier_marks(ax, labels=False)
    legend_squares(ax, [f"q{ki + 1} > 0 ({np.sum(dcol == 0)})", f"q{ki + 1} < 0 ({np.sum(dcol == 1)})"]
                   + ([f"small domains ({np.sum(dcol == 2)})"] if np.any(dcol == 2) else []), cmap, loc="lower left")
    # q2 and q3 for comparison: their nodal domains cross to give the three regions of E-6
    for j, qi in enumerate((1, 2)):
        ax = fig.add_subplot(gs[1, j + 1])
        q = Q[:, qi]
        categorical_map(ax, (q < 0).astype(float), [POS, NEG], f"({'ef'[j]}) Nodal domains of q{qi + 1}, for comparison")
        glacier_marks(ax, labels=False)
        legend_squares(ax, [f"q{qi + 1} > 0 ({np.sum(q > 0)})", f"q{qi + 1} < 0 ({np.sum(q < 0)})"], [POS, NEG],
                       loc="lower left")
    fig.savefig(FIG / "e5_spectrum.png")
    plt.close(fig)


# ------------------------------------------------------------------ E-6
def fig_e6():
    e6 = S["E6"]
    clim, kept = z["clim"], z["kept"]
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.28, wspace=0.32)
    ax = fig.add_subplot(gs[0, :2])
    categorical_map(ax, lab.astype(float), REGION_COLORS,
                    "(a) Spectral clustering K = 3 with Yao et al. (2012) glacier regions I–VII")
    glacier_marks(ax)
    legend_squares(ax, [f"{nm} ({np.sum(lab == j)} px)" for j, nm in enumerate(REGION_NAMES)], REGION_COLORS,
                   loc="lower left")

    ax = fig.add_subplot(gs[0, 2])
    for j in range(3):
        ax.plot(clim[lab == j].mean(0), color=REGION_COLORS[j], lw=2, label=REGION_NAMES[j])
    month_axis(ax); ax.set_ylabel("mean depth (cm)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("(b) Mean snow year of each region"); ax.grid(axis="y", color=GRID, lw=0.6)

    ax = fig.add_subplot(gs[1, 0])
    basemap(ax, "(c) Map neighbours, dark = large feature jump")
    E, dE = z["E"], z["dE"]
    v = np.clip(dE / np.quantile(dE, 0.98), 0, 1)
    for k in np.argsort(dE):
        if v[k] < 0.3:
            continue
        i, j = E[k]
        ax.plot([plon[i], plon[j]], [plat[i], plat[j]], color=plt.cm.Greys(0.25 + 0.75 * v[k]), lw=0.4 + 1.6 * v[k])

    ax = fig.add_subplot(gs[1, 1])
    basemap(ax, "(d) Years (of 14) a vertex stays in its region")
    im = ax.imshow(np.ma.masked_invalid(paint(kept)), extent=EXT, cmap=SEQ, vmin=0, vmax=14, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.outline.set_visible(False)

    years = np.arange(len(e6["snow_free"]))
    sub = gs[1, 2].subgridspec(2, 1, hspace=0.15)
    top_ax = fig.add_subplot(sub[0])
    top_ax.bar(years, e6["snow_free"], color="#86b6ef")
    top_ax.set_ylabel("snow-free vertices")
    top_ax.set_title("(e) Sensor check, one graph per snow year")
    low_ax = fig.add_subplot(sub[1], sharex=top_ax)
    low_ax.plot(years, e6["per_year_lam2"], color="#2a78d6", marker="o", ms=4, lw=1.5)
    low_ax.set_ylabel("per-year λ2")
    low_ax.set_xticks(years[::2], [f"{2000 + y}/{(y + 1) % 100:02d}" for y in years[::2]], rotation=45, fontsize=7)
    plt.setp(top_ax.get_xticklabels(), visible=False)
    for a in (top_ax, low_ax):
        a.axvline(7.5, color=INK2, ls="--", lw=1)
        a.grid(axis="y", color=GRID, lw=0.6)
    top_ax.annotate("SSM/I → SSMIS", (7.5, max(e6["snow_free"])), xytext=(-75, -4), textcoords="offset points",
                    fontsize=8, color=INK2)
    fig.savefig(FIG / "e6_regions.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_e1, fig_e2, fig_e3, fig_e4, fig_e5, fig_e6):
        f()
        print("saved", f.__name__)
