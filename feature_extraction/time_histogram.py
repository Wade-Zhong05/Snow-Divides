"""Snow-year time × signed daily-gradient histograms for geographic grid cells."""

from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from data_io import SnowCube


@dataclass(frozen=True, slots=True)
class HistogramConfig:
    windows_per_month: int
    gradient_bins: int
    gradient_scale_cm_day: float
    time_smoothing_bins: float
    occurrence_weight: float


@dataclass(frozen=True, slots=True)
class SnowHistograms:
    counts: NDArray[np.int32]
    snow_days: NDArray[np.int16]
    valid_days: NDArray[np.int16]
    years: NDArray[np.int32]
    angle_edges_degrees: NDArray[np.float64]


def extract(cube: SnowCube, config: HistogramConfig) -> SnowHistograms:
    """Count valid snow-active differences; never bridge a missing day or snow-year boundary."""
    years = np.arange(cube.dates[0].year, cube.dates[-1].year, dtype=np.int32)
    pixels = cube.latitude.size * cube.longitude.size
    windows = 12 * config.windows_per_month
    counts = np.zeros((years.size, windows, pixels, config.gradient_bins), dtype=np.int32)
    snow_days = np.zeros((years.size, windows, pixels), dtype=np.int16)
    valid_days = np.zeros_like(snow_days)
    depth = cube.depth.reshape(len(cube.dates), pixels)
    for index, day in enumerate(cube.dates):
        year = day.year if day.month >= 8 else day.year - 1
        year_index = year - int(years[0])
        month_index = (day.month - 8) % 12
        days_in_month = calendar.monthrange(day.year, day.month)[1]
        part = (day.day - 1) * config.windows_per_month // days_in_month
        window = month_index * config.windows_per_month + part
        current = depth[index]
        valid = np.isfinite(current)
        valid_days[year_index, window] += valid
        snow_days[year_index, window] += valid & (current > 0)
        if index == 0 or (day.month == 8 and day.day == 1):
            continue
        previous = depth[index - 1]
        active = valid & np.isfinite(previous) & ((current > 0) | (previous > 0))
        pixel_indices = np.flatnonzero(active)
        gradient = current[pixel_indices] - previous[pixel_indices]
        angle = np.arctan2(gradient, config.gradient_scale_cm_day)
        bins = np.clip(
            np.floor((angle + np.pi / 2) / np.pi * config.gradient_bins).astype(np.int32),
            0,
            config.gradient_bins - 1,
        )
        np.add.at(counts[year_index, window], (pixel_indices, bins), 1)
    return SnowHistograms(
        counts,
        snow_days,
        valid_days,
        years,
        np.rad2deg(np.linspace(-np.pi / 2, np.pi / 2, config.gradient_bins + 1)),
    )


def eligible_pixels(histograms: SnowHistograms, minimum_snow_days: int) -> NDArray[np.bool_]:
    """Require up to 10 adequately observed years and mean annual snow days."""
    annual_valid = histograms.valid_days.sum(axis=1)
    annual_snow = histograms.snow_days.sum(axis=1)
    observed = annual_valid >= 300
    years = observed.sum(axis=0)
    mean_snow = np.divide(
        (annual_snow * observed).sum(axis=0),
        years,
        out=np.zeros(annual_snow.shape[1], dtype=np.float64),
        where=years > 0,
    )
    return (years >= min(10, histograms.years.size)) & (mean_snow >= minimum_snow_days)


def feature_matrix(
    counts: NDArray[np.int32],
    snow_days: NDArray[np.int32],
    valid_days: NDArray[np.int32],
    config: HistogramConfig,
) -> NDArray[np.float32]:
    """Embed per-window Hellinger histograms and snow occurrence in Euclidean space."""
    raw = counts.astype(np.float32)
    if config.time_smoothing_bins > 0:
        raw = gaussian_filter1d(raw, config.time_smoothing_bins, axis=0, mode="nearest")
    totals = raw.sum(axis=-1, keepdims=True)
    probability = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)
    windows = raw.shape[0]
    gradient_part = np.sqrt(probability).transpose(1, 0, 2).reshape(counts.shape[-2], -1)
    gradient_part *= np.sqrt((1 - config.occurrence_weight) / (2 * windows))
    occurrence = np.divide(
        snow_days,
        valid_days,
        out=np.zeros(snow_days.shape, dtype=np.float32),
        where=valid_days > 0,
    ).T
    occurrence *= np.sqrt(config.occurrence_weight / windows)
    return np.concatenate((gradient_part, occurrence), axis=1)
