#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "typer>=0.15"]
# ///
# How to run: cd feature_extraction && uv sync && uv run python run.py --help
"""Explore temporal/spatial snow gradients and export per-pixel monthly HOG profiles."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import matplotlib
import numpy as np
import typer

from data_io import load_cube
from gradients import BINS, TEMPORAL_SCALE, differences, oriented_features
from plots import distributions, joint_distributions, seasonal_features


def main(start: str = "2000-08-01", end: str = "2014-07-31", output: Path | None = None) -> None:
    """Use the primary TXT series, preserving unverified supplementary dates as gaps."""
    began = time.perf_counter()
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as error:
        raise typer.BadParameter("Use ISO dates: YYYY-MM-DD") from error
    if first >= last:
        raise typer.BadParameter("The end date must be later than the start date")
    if first < date(1979, 1, 1) or last > date(2014, 12, 31):
        raise typer.BadParameter("Use primary-product dates within 1979–2014")
    base = Path(__file__).resolve().parent
    out = output or base / "outputs" / datetime.now().strftime("%Y%m%dT%H%M%S")
    if out.exists():
        raise typer.BadParameter(f"Output already exists; choose a new directory: {out}")
    cube = load_cube(base.parent / "data" / "snow depth", first, last)
    g = differences(cube)
    if not g.time_active.any() or not g.space_active.any():
        raise typer.BadParameter("No snow-active gradients available in this date range")
    print("Computing monthly oriented histograms", flush=True)
    f = oriented_features(cube, g)
    out.mkdir(parents=True)
    period = f"{start} to {end}"
    stats = distributions(g, f, out, period)
    tails = joint_distributions(g, out, period)
    seasonal_features(f, out, period)
    # Each row follows the latitude/longitude mesh flattened in C order.
    np.savez_compressed(
        out / "hog_features.npz",
        **asdict(f),
        temporal_features=f.temporal_l1.transpose(1, 2, 0, 3).reshape(-1, 12 * BINS),
        spatial_features=f.spatial_l1.transpose(1, 2, 0, 3).reshape(-1, 12 * BINS),
        latitude=cube.latitude,
        longitude=cube.longitude,
        temporal_angle_edges=np.linspace(-90, 90, BINS + 1),
        spatial_angle_edges=np.linspace(-180, 180, BINS + 1),
    )
    summary = {
        "start": start,
        "end": end,
        "date_count": len(cube.dates),
        "present_date_count": len(cube.dates) - len(cube.missing_dates),
        "missing_dates": cube.missing_dates,
        "shape": list(cube.depth.shape),
        "invalid_depth_fraction": float(np.mean(~np.isfinite(cube.depth))),
        "sources_sha256": dict(cube.source_hashes),
        "region": "73–105 E, 26–40 N rectangle, no plateau polygon or 30-day filter",
        "missing_policy": "NaN; no interpolation; unverified leap-day supplements excluded",
        "gradient_rules": "Backward 1-day time difference; centered 2-neighbor space difference",
        "spatial_units": "cm/km, 111.2 km/degree spherical approximation; north positive",
        "snow_active": "Any valid stencil depth >0; descriptive selection, not detection accuracy",
        "hog_bins": BINS,
        "temporal_reference_cm_per_day": TEMPORAL_SCALE,
        "hog_normalization": "Per-pixel calendar-month L1; no spatial cell/block L2-Hys",
        "hog_weights": "Temporal absolute cm/day change; spatial cm/km magnitude; hard bins",
        "quantile_order": [0, 0.005, 0.25, 0.5, 0.75, 0.995, 1],
        "distribution_stats": stats,
        "joint_clipped_fractions": tails,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "elapsed_seconds": round(time.perf_counter() - began, 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved figures, features and summary to {out}", flush=True)


if __name__ == "__main__":
    typer.run(main)
