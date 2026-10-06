#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "typer>=0.15", "scikit-learn>=1.6"]
# ///
# How to run: cd feature_extraction && uv sync --locked && uv run python sweep_graph.py --help
"""Automate time-window and gradient-bin sensitivity runs without geographic edges."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.metrics import adjusted_rand_score


def main(
    window_options: str = "2,1",
    gradient_bin_options: str = "9,7",
    k_value: int = 21,
    first_year: int = 2000,
    last_year: int = 2013,
    output: Path | None = None,
) -> None:
    """Run every histogram combination and compare labelings on shared geographic cells."""
    try:
        windows = list(dict.fromkeys(int(value) for value in window_options.split(",")))
        bins = list(dict.fromkeys(int(value) for value in gradient_bin_options.split(",")))
    except ValueError as error:
        raise typer.BadParameter("Options must contain comma-separated integers") from error
    if not windows or not bins:
        raise typer.BadParameter("Each option list must contain at least one value")
    base = Path(__file__).resolve().parent
    out = output or base / "outputs" / f"graph-sweep-{datetime.now():%Y%m%dT%H%M%S}"
    if out.exists():
        raise typer.BadParameter(f"Output already exists: {out}")
    out.mkdir(parents=True)
    baseline = None
    score_grid = np.zeros((len(bins), len(windows)))
    runs = []
    for column, window_count in enumerate(windows):
        for row, gradient_bins in enumerate(bins):
            run_dir = out / f"windows-{window_count}_gradient-bins-{gradient_bins}"
            print(
                f"Running windows/month={window_count}, gradient bins={gradient_bins}", flush=True
            )
            command = [
                sys.executable,
                str(base / "run_graph.py"),
                "--first-year",
                str(first_year),
                "--last-year",
                str(last_year),
                "--windows-per-month",
                str(window_count),
                "--gradient-bins",
                str(gradient_bins),
                "--k-values",
                str(k_value),
                "--reference-k",
                str(k_value),
                "--no-annual",
                "--output",
                str(run_dir),
            ]
            subprocess.run(command, cwd=base, check=True)
            summary = json.loads((run_dir / "summary.json").read_text())
            with np.load(run_dir / "snow_graph_results.npz") as arrays:
                labels = arrays["reference_labels"].ravel().copy()
            if baseline is None:
                baseline = labels
            shared = (baseline >= 0) & (labels >= 0)
            score = adjusted_rand_score(baseline[shared], labels[shared])
            score_grid[row, column] = score
            runs.append(
                {
                    "windows_per_month": window_count,
                    "gradient_bins": gradient_bins,
                    "output": str(run_dir),
                    "eligible_pixels": summary["eligible_pixels"],
                    "components": summary["k_scan"][0]["components"],
                    "shared_pixels_with_baseline": int(shared.sum()),
                    "adjusted_rand_index_vs_first": score,
                }
            )
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.imshow(score_grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(windows)), [str(12 * value) for value in windows])
    axis.set_yticks(np.arange(len(bins)), [str(value) for value in bins])
    axis.set(
        xlabel="Time windows per snow year",
        ylabel="Signed-gradient bins",
        title="Cluster agreement with first histogram setting (ARI)",
    )
    for row in range(len(bins)):
        for column in range(len(windows)):
            axis.text(
                column,
                row,
                f"{score_grid[row, column]:.2f}",
                ha="center",
                va="center",
                color="white",
            )
    fig.colorbar(image, ax=axis, label="Adjusted Rand index (1 = same partition)")
    fig.savefig(out / "histogram_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (out / "sweep_summary.json").write_text(
        json.dumps(
            {
                "first_setting_is_baseline": {
                    "windows_per_month": windows[0],
                    "gradient_bins": bins[0],
                },
                "fixed_k": k_value,
                "annual_partitions": False,
                "comparison": (
                    "Adjusted Rand index on grid cells eligible in both runs; "
                    "cluster IDs are arbitrary"
                ),
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Saved sensitivity sweep to {out}", flush=True)


if __name__ == "__main__":
    typer.run(main)
