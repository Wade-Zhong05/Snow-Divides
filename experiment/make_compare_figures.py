"""Figures for the feature comparison (needs results/compare.npz and experiment/feature_comparison.json).

    python -m experiment.make_compare_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from snowdivides import features_alt as fa
from snowdivides import graph as g
from snowdivides.data import ROOT

FIG = ROOT / "experiment" / "figures"
INK, INK2, GRID, EMPTY = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
REGION_COLORS = ["#2a78d6", "#1baf7a", "#eb6834"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
MAIN = "HOG, smoothed (main)"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})

z = np.load(ROOT / "results" / "compare.npz")
R = json.loads((ROOT / "experiment" / "feature_comparison.json").read_text())
r, c, pix2v, lat, lon = (z[k] for k in ("r", "c", "pix2v", "lat", "lon"))
SHAPE = tuple(z["shape"])
EXT = [lon[0] - .125, lon[-1] + .125, lat[-1] - .125, lat[0] + .125]
names = list(z["names"])
labels = dict(zip(names, z["labels"]))


def fig_maps():
    ref = labels[MAIN]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.3 * rows))
    for ax, name in zip(axes.ravel(), names):
        lab, _ = g.match_labels(labels[name], ref)             # same colour = best-matching region
        G = np.full(SHAPE, np.nan)
        G[r, c] = lab[pix2v]
        ax.imshow(np.where(np.isnan(G), 1.0, np.nan), extent=EXT, cmap=ListedColormap([EMPTY]))
        ax.imshow(np.ma.masked_invalid(G), extent=EXT, cmap=ListedColormap(REGION_COLORS), vmin=-0.5, vmax=2.5,
                  interpolation="nearest")
        m = R[name]
        ok = "✓" if m["glacier_recovered"] else "✗"
        ax.set_title(f"{name}\nglaciers {ok} {m['glacier_purity']:.2f} · stable {m['split_nn']:.2f} · "
                     f"ARI vs main {m['vs_main']:.2f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("K = 3 regions from each feature (colours matched to the main feature's west / interior / southeast)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "compare_maps.png")
    plt.close(fig)


def fig_lags():
    lags = [1, 3, 7, 15, 31, 61]
    keys = [f"HOG, lag {L}" for L in lags]
    metrics = [("climate", "climate signal |r(q2, lon)| at 4400–5000 m"),
               ("contiguity", "map contiguity"),
               ("split_nn", "kNN overlap between half-year graphs"),
               ("split", "ARI between half-year partitions (mean of 5)"),
               ("era_nn", "kNN overlap before vs after 2008/09")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25))
    for (key, label), col in zip(metrics, SERIES):
        y = [R[k][key] for k in keys]
        ax[0].plot(lags, y, color=col, lw=2, marker="o", ms=4, label=label)
        if key == "split":
            sd = [R[k]["split_sd"] for k in keys]
            ax[0].fill_between(lags, np.array(y) - sd, np.array(y) + sd, color=col, alpha=0.12, lw=0)
    ax[0].set_xscale("log"); ax[0].set_xticks(lags, [str(L) for L in lags])
    ax[0].set_xlabel("time lag of the gradient (days)"); ax[0].set_ylim(0, 1)
    ax[0].grid(color=GRID, lw=0.6); ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[0].set_title("(a) Single-lag HOG: scores vs time lag (band = ± sd over 5 splits)")

    curves = z["curves_example"]                 # (years, a few vertices, 365)
    s = curves[5, 3]
    t = np.arange(365)
    ax[1].remove()
    sub = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.25)[0, 1].subgridspec(4, 1, hspace=0.12)
    rows = [("depth (cm)", s, INK2)] + [(f"rate over {L} d", fa.lag_rate(s[None, None], L)[0, 0], col)
                                        for L, col in zip((1, 7, 31), SERIES)]
    for i, (lab, y, col) in enumerate(rows):
        a = fig.add_subplot(sub[i])
        a.plot(t, y, color=col, lw=1.1)
        a.axhline(0, color=GRID, lw=0.8)
        a.set_ylabel(lab, fontsize=8)
        a.set_xticks([0, 61, 122, 184, 243, 304], ["Aug", "Oct", "Dec", "Feb", "Apr", "Jun"] if i == 3 else [])
        a.tick_params(labelsize=7)
        if i == 0:
            a.set_title("(b) One cell, one snow year: what each lag 'sees' (rates in cm/day)")
    fig.savefig(FIG / "compare_lags.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_maps()
    fig_lags()
    print("saved compare_maps.png, compare_lags.png")
