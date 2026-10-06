#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0", "scipy>=1.14", "scikit-learn>=1.6", "matplotlib>=3.9",
#   "plotly>=6.0", "typer>=0.15"
# ]
# ///
# How to run: uv sync --locked && uv run python pipeline.py --help

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from snow_divides.e1_features.histogram import HistogramConfig
from snow_divides.workflow import PipelineConfig, run_pipeline


def main(
    first_year: int = 2000,
    last_year: int = 2013,
    windows_per_month: int = 2,
    gradient_bins: int = 9,
    gradient_scale_cm_day: float = 1.0,
    time_smoothing_bins: float = 0.5,
    occurrence_weight: float = 0.5,
    minimum_snow_days: int = 30,
    k_values: str = "2,3,4,8,10,16,21,26,32,64",
    reference_k: int = 21,
    cluster_counts: str = "2,3,4,5,6",
    reference_clusters: int = 3,
    fiedler_bands: str = "0,0.1,0.2",
    eigen_count: int = 32,
    output: Path | None = None,
) -> None:
    """Run the implemented homework stages with one parameter set and output root."""
    if not 1979 <= first_year <= last_year <= 2013:
        raise typer.BadParameter("Snow years must lie in 1979/80–2013/14")
    if windows_per_month not in (1, 2, 4):
        raise typer.BadParameter("windows-per-month must be 1, 2, or 4")
    if gradient_bins < 3 or gradient_bins % 2 == 0:
        raise typer.BadParameter("gradient-bins must be odd and at least 3")
    if gradient_scale_cm_day <= 0 or time_smoothing_bins < 0:
        raise typer.BadParameter("gradient scale must be positive; smoothing nonnegative")
    if not 0 <= occurrence_weight <= 1 or minimum_snow_days < 0:
        raise typer.BadParameter("occurrence weight must be in [0,1]; snow days nonnegative")
    try:
        ks = tuple(sorted({int(value) for value in k_values.split(",")}))
        clusters = tuple(sorted({int(value) for value in cluster_counts.split(",")}))
        bands = tuple(sorted({float(value) for value in fiedler_bands.split(",")}))
    except ValueError as error:
        raise typer.BadParameter(
            "k-values, cluster-counts and fiedler-bands must be CSV numbers"
        ) from error
    if not ks or ks[0] < 2 or reference_k not in ks:
        raise typer.BadParameter("k-values must include reference-k and be at least 2")
    if not clusters or clusters[0] < 2 or reference_clusters not in clusters:
        raise typer.BadParameter("cluster-counts must include reference-clusters and be at least 2")
    if eigen_count < 4 or clusters[-1] >= eigen_count:
        raise typer.BadParameter(
            "eigen-count must exceed candidate cluster counts and be at least 4"
        )
    if not bands or bands[0] != 0 or bands[-1] >= 1:
        raise typer.BadParameter("fiedler-bands must start at 0 and stay below 1")
    project = Path(__file__).resolve().parent
    destination = output or Path("outputs") / f"part2-{datetime.now():%Y%m%dT%H%M%S}"
    destination = destination if destination.is_absolute() else project / destination
    if destination.exists():
        raise typer.BadParameter(f"Output already exists: {destination}")
    config = PipelineConfig(
        first_year,
        last_year,
        HistogramConfig(
            windows_per_month,
            gradient_bins,
            gradient_scale_cm_day,
            time_smoothing_bins,
            occurrence_weight,
        ),
        minimum_snow_days,
        ks,
        reference_k,
        clusters,
        reference_clusters,
        bands,
        eigen_count,
    )
    run_pipeline(config, destination)


if __name__ == "__main__":
    typer.run(main)
