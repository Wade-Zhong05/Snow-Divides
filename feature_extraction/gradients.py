"""Finite differences and signed, magnitude-weighted orientation histograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from data_io import SnowCube

BINS: Final = 18
TEMPORAL_SCALE: Final = 1.0  # cm/day, the reference slope for arctan


@dataclass(frozen=True, slots=True)
class Gradients:
    temporal: NDArray[np.float32]
    east: NDArray[np.float32]
    north: NDArray[np.float32]
    spatial: NDArray[np.float32]
    time_active: NDArray[np.bool_]
    space_active: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Features:
    temporal_raw: NDArray[np.float64]
    spatial_raw: NDArray[np.float64]
    temporal_l1: NDArray[np.float64]
    spatial_l1: NDArray[np.float64]
    temporal_counts: NDArray[np.int64]
    spatial_counts: NDArray[np.int64]


def differences(cube: SnowCube) -> Gradients:
    """Temporal backward difference; centered spatial derivatives on interior pixels."""
    z = cube.depth
    temporal = np.full_like(z, np.nan)
    temporal[1:] = z[1:] - z[:-1]  # Dates include gaps, so missing days cannot be bridged.
    east = np.full_like(z, np.nan)
    north = np.full_like(z, np.nan)
    dx = 111.2 * 0.25 * np.cos(np.deg2rad(cube.latitude))
    dy = 111.2 * 0.25
    east[:, :, 1:-1] = (z[:, :, 2:] - z[:, :, :-2]) / (2 * dx[None, :, None])
    north[:, 1:-1, :] = (z[:, :-2, :] - z[:, 2:, :]) / (2 * dy)
    east[~np.isfinite(z)] = np.nan
    north[~np.isfinite(z)] = np.nan
    spatial = np.hypot(east, north)
    time_active = np.zeros_like(z, dtype=np.bool_)
    time_active[1:] = ((z[1:] > 0) | (z[:-1] > 0)) & np.isfinite(temporal[1:])
    space_active = np.zeros_like(z, dtype=np.bool_)
    space_active[:, 1:-1, 1:-1] = (
        (z[:, 1:-1, 1:-1] > 0)
        | (z[:, 1:-1, :-2] > 0)
        | (z[:, 1:-1, 2:] > 0)
        | (z[:, :-2, 1:-1] > 0)
        | (z[:, 2:, 1:-1] > 0)
    )
    space_active &= np.isfinite(spatial)
    return Gradients(temporal, east, north, spatial, time_active, space_active)


def normalized(raw: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize each pixel/month over angles; unsupported/flat rows remain zero."""
    total = raw.sum(axis=-1, keepdims=True)
    return np.divide(raw, total, out=np.zeros_like(raw), where=total > 0)


def oriented_features(cube: SnowCube, gradients: Gradients) -> Features:
    """Accumulate hard angle bins per pixel/calendar-month, pooling available years."""
    shape = (12, *cube.depth.shape[1:], BINS)
    temporal = np.zeros(shape, dtype=np.float64)
    spatial = np.zeros(shape, dtype=np.float64)
    tc = np.zeros(shape[:-1], dtype=np.int64)
    sc = np.zeros_like(tc)
    rr, cc = np.indices(cube.depth.shape[1:])
    for i, day in enumerate(cube.dates):
        month = day.month - 1
        dt = gradients.temporal[i]
        magnitude = gradients.spatial[i]
        valid_t = gradients.time_active[i]
        valid_s = gradients.space_active[i]
        tc[month] += valid_t
        sc[month] += valid_s
        angle_t = np.arctan2(dt[valid_t], TEMPORAL_SCALE)
        bin_t = np.minimum(((angle_t + np.pi / 2) / np.pi * BINS).astype(int), BINS - 1)
        angle_s = np.arctan2(gradients.north[i][valid_s], gradients.east[i][valid_s])
        bin_s = (((angle_s + np.pi) / (2 * np.pi) * BINS).astype(int)) % BINS
        np.add.at(temporal[month], (rr[valid_t], cc[valid_t], bin_t), np.abs(dt[valid_t]))
        np.add.at(spatial[month], (rr[valid_s], cc[valid_s], bin_s), magnitude[valid_s])
    return Features(temporal, spatial, normalized(temporal), normalized(spatial), tc, sc)
