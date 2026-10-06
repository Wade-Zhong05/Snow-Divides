#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0", "matplotlib>=3.9", "typer>=0.15", "scipy>=1.14", "scikit-learn>=1.6"
# ]
# ///
# How to run: cd feature_extraction && uv sync --locked && uv run python run_graph.py --help
"""Run snow-only graph experiments with geographic interpretation and annual boundaries."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
import typer
from scipy.sparse import save_npz

from data_io import load_cube
from graph_plots import boundary_map, diagnostics, feature_and_region, occurrence_profiles
from snow_graph import border_counts, build_graph, edge_lengths_km, find_neighbors
from time_histogram import HistogramConfig, eligible_pixels, extract, feature_matrix


def main(
    first_year: int = 2000,
    last_year: int = 2013,
    windows_per_month: int = 2,
    gradient_bins: int = 9,
    gradient_scale_cm_day: float = 1.0,
    time_smoothing_bins: float = 0.5,
    occurrence_weight: float = 0.5,
    minimum_snow_days: int = 30,
    k_values: str = "10,21,26",
    reference_k: int = 21,
    annual: bool = True,
    output: Path | None = None,
) -> None:
    """Export histogram parameters, weighted graphs, maps, and reproducible summaries."""
    if not 1979 <= first_year <= last_year <= 2013:
        raise typer.BadParameter("Snow years must lie from 1979/80 through 2013/14")
    if windows_per_month not in (1, 2, 4):
        raise typer.BadParameter("windows-per-month must be 1, 2, or 4")
    if gradient_bins < 3 or gradient_bins % 2 == 0:
        raise typer.BadParameter("gradient-bins must be an odd integer of at least 3")
    if gradient_scale_cm_day <= 0 or time_smoothing_bins < 0:
        raise typer.BadParameter("gradient scale must be positive and smoothing nonnegative")
    if not 0 <= occurrence_weight <= 1 or minimum_snow_days < 0:
        raise typer.BadParameter("occurrence weight must be in [0,1]; snow days nonnegative")
    try:
        ks = sorted({int(part) for part in k_values.split(",")})
    except ValueError as error:
        raise typer.BadParameter("k-values must be comma-separated integers") from error
    if not ks or min(ks) < 2 or reference_k not in ks:
        raise typer.BadParameter("k-values must include reference-k and all be at least 2")
    base = Path(__file__).resolve().parent
    out = output or base / "outputs" / f"graph-{datetime.now():%Y%m%dT%H%M%S}"
    if out.exists():
        raise typer.BadParameter(f"Output already exists: {out}")
    config = HistogramConfig(
        windows_per_month,
        gradient_bins,
        gradient_scale_cm_day,
        time_smoothing_bins,
        occurrence_weight,
    )
    began = time.perf_counter()
    cube = load_cube(
        base.parent / "data" / "snow depth", date(first_year, 8, 1), date(last_year + 1, 7, 31)
    )
    print("Extracting time × signed-gradient histograms", flush=True)
    hist = extract(cube, config)
    selected = np.flatnonzero(eligible_pixels(hist, minimum_snow_days))
    if selected.size <= max(ks) + 1:
        raise typer.BadParameter(
            "Too few eligible pixels for requested k; lower snow threshold or k"
        )
    pooled_counts = hist.counts.sum(axis=0, dtype=np.int32)
    pooled_snow = hist.snow_days.sum(axis=0, dtype=np.int32)
    pooled_valid = hist.valid_days.sum(axis=0, dtype=np.int32)
    features = feature_matrix(
        pooled_counts[:, selected], pooled_snow[:, selected], pooled_valid[:, selected], config
    )
    neighbors = find_neighbors(features, max(ks))
    out.mkdir(parents=True)
    graph_rows = []
    reference = None
    for k in ks:
        print(f"Building snow-feature graph: k={k}", flush=True)
        graph = build_graph(neighbors, k, cluster_count=3)
        graph_rows.append(
            {
                "k": k,
                "components": graph.components,
                "largest_component_fraction": graph.largest_component_fraction,
                "eigenvalues": graph.eigenvalues.tolist(),
                "cluster_sizes": np.bincount(graph.labels, minlength=3).tolist(),
                "edges": graph.weights.nnz // 2,
            }
        )
        if k == reference_k:
            reference = graph
    if reference is None:
        raise RuntimeError("reference graph was not constructed")
    save_npz(out / "reference_graph_csr.npz", reference.weights)
    grid_shape = (cube.latitude.size, cube.longitude.size)
    full_labels = np.full(cube.latitude.size * cube.longitude.size, -1, dtype=np.int32)
    full_labels[selected] = reference.labels
    lengths = edge_lengths_km(reference.weights, selected, cube.latitude, cube.longitude)
    annual_labels = np.full((hist.years.size, full_labels.size), -1, dtype=np.int32)
    boundary_numerator = np.zeros(grid_shape, dtype=np.int32)
    boundary_denominator = np.zeros(grid_shape, dtype=np.int32)
    annual_rows = []
    if annual:
        (out / "annual_graphs").mkdir()
        for year_index, year in enumerate(hist.years):
            observed = hist.valid_days[year_index].sum(axis=0) >= 300
            snowy = hist.snow_days[year_index].sum(axis=0) >= 10
            annual_selected = selected[observed[selected] & snowy[selected]]
            if annual_selected.size <= reference_k + 1:
                raise typer.BadParameter(f"Too few annual eligible pixels for {year}")
            year_features = feature_matrix(
                hist.counts[year_index, :, annual_selected].transpose(1, 0, 2),
                hist.snow_days[year_index, :, annual_selected].T.astype(np.int32),
                hist.valid_days[year_index, :, annual_selected].T.astype(np.int32),
                config,
            )
            year_neighbors = find_neighbors(year_features, reference_k)
            year_graph = build_graph(year_neighbors, reference_k, cluster_count=3)
            save_npz(out / "annual_graphs" / f"snow_year_{year}.npz", year_graph.weights)
            annual_labels[year_index, annual_selected] = year_graph.labels
            different, support = border_counts(annual_labels[year_index], grid_shape)
            boundary_numerator += different
            boundary_denominator += support
            annual_rows.append(
                {
                    "snow_year": int(year),
                    "eligible_pixels": int(annual_selected.size),
                    "components": year_graph.components,
                    "largest_component_fraction": year_graph.largest_component_fraction,
                    "cluster_sizes": np.bincount(year_graph.labels, minlength=3).tolist(),
                }
            )
            print(f"Annual partition {year}/{year + 1}: {annual_selected.size} pixels", flush=True)
    feature_and_region(
        out / "01_snow_features_and_regions.png",
        reference.labels,
        selected,
        pooled_counts,
        cube.latitude,
        cube.longitude,
        hist.angle_edges_degrees,
    )
    occurrence_profiles(
        out / "04_cluster_occurrence_profiles.png",
        reference.labels,
        selected,
        pooled_counts,
        pooled_snow,
        pooled_valid,
    )
    diagnostics(
        out / "02_graph_diagnostics.png",
        np.asarray(ks, dtype=np.int32),
        np.asarray([row["components"] for row in graph_rows], dtype=np.int32),
        np.asarray([row["largest_component_fraction"] for row in graph_rows]),
        lengths,
    )
    if annual:
        boundary_map(
            out / "03_annual_boundary_frequency.png",
            boundary_numerator,
            boundary_denominator,
            cube.latitude,
            cube.longitude,
            hist.years.size,
        )
    np.savez_compressed(
        out / "snow_graph_results.npz",
        selected_pixel_index=selected,
        pooled_features=features,
        pooled_histogram_counts=pooled_counts[:, selected],
        pooled_snow_days=pooled_snow[:, selected],
        pooled_valid_days=pooled_valid[:, selected],
        annual_snow_days=hist.snow_days.sum(axis=1)[:, selected],
        latitude=cube.latitude,
        longitude=cube.longitude,
        gradient_angle_edges_degrees=hist.angle_edges_degrees,
        reference_labels=full_labels.reshape(grid_shape),
        annual_labels=annual_labels.reshape((hist.years.size, *grid_shape)),
        boundary_disagreement_count=boundary_numerator,
        boundary_comparison_count=boundary_denominator,
        edge_length_km=lengths,
    )
    summary = {
        "snow_years": [first_year, last_year],
        "histogram": asdict(config),
        "time_axis": (
            "Aug-Jul; each month split into equal day-index groups; leap February included"
        ),
        "gradient_axis": (
            "atan((depth_today-depth_yesterday)/(gradient_scale_cm_day)); signed degrees"
        ),
        "gradient_selection": (
            "Both dates valid; at least one depth >0; no Aug-1 cross-year or missing-day bridge"
        ),
        "normalization": (
            "Per-window snow-active gradient histogram; occurrence=snow days/valid days"
        ),
        "distance": (
            "Euclidean: sqrt((1-weight)*mean(0.5*||sqrt(p_i)-sqrt(p_j)||^2)"
            "+weight*mean((occ_i-occ_j)^2)); empty windows use zero histogram vectors"
        ),
        "graph": (
            "Euclidean kNN on snow features, union symmetrized, self-tuning Gaussian, "
            "3 normalized-Laplacian clusters"
        ),
        "geography": (
            "Only post-graph maps, haversine edge distances, and original 4-neighbor "
            "boundary comparisons; no radius constraint"
        ),
        "eligibility": (
            f"At least {min(10, hist.years.size)} years with >=300 valid days; "
            f"mean >= {minimum_snow_days} "
            "snow days/year; annual >=10 snow days"
        ),
        "rectangle": "73-105 E, 26-40 N; no Tibetan Plateau polygon or DEM/glacier mask",
        "eligible_pixels": int(selected.size),
        "reference_k": reference_k,
        "k_scan": graph_rows,
        "edge_length_km_quantiles": np.quantile(lengths, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
        "annual": annual_rows,
        "missing_dates": cube.missing_dates,
        "sources_sha256": dict(cube.source_hashes),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": round(time.perf_counter() - began, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved experiment to {out}", flush=True)


if __name__ == "__main__":
    typer.run(main)
