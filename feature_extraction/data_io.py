"""Read the header-bearing 1979–2014 primary product without extracting archives."""

from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SnowCube:
    depth: NDArray[np.float32]
    dates: tuple[date, ...]
    latitude: NDArray[np.float64]
    longitude: NDArray[np.float64]
    missing_dates: tuple[str, ...]
    source_hashes: tuple[tuple[str, str], ...]


def read_grid(raw: bytes) -> NDArray[np.float32]:
    """Require the documented grid; missing or unexpected headers must fail visibly."""
    lines = raw.decode("ascii").splitlines()
    expected = {
        "ncols": 321.0,
        "nrows": 161.0,
        "xllcenter": 60.0,
        "yllcenter": 15.0,
        "cellsize": 0.25,
        "nodata_value": -1.0,
    }
    header = {
        parts[0].lower(): float(parts[1]) for line in lines[:6] if len(parts := line.split()) == 2
    }
    if header != expected:
        message = f"Unsupported raster header: {header}"
        raise ValueError(message)
    grid = np.fromstring("\n".join(lines[6:]), sep=" ", dtype=np.float32)
    if grid.size != 161 * 321 or not np.all(np.isfinite(grid)):
        message = "Invalid grid size or nonfinite stored values"
        raise ValueError(message)
    grid = grid.reshape(161, 321)
    grid[grid < 0] = np.nan
    return grid


def load_cube(root: Path, start: date, end: date) -> SnowCube:
    """Load the README rectangle; preserve absent dates as NaN, without interpolation."""
    if not date(1979, 1, 1) <= start <= end <= date(2014, 12, 31):
        message = "Supported primary-product dates: 1979-01-01 through 2014-12-31"
        raise ValueError(message)
    dates = tuple(start + timedelta(days=i) for i in range((end - start).days + 1))
    latitude = np.arange(55.0, 14.99, -0.25)
    longitude = np.arange(60.0, 140.01, 0.25)
    rows = np.flatnonzero((latitude >= 26) & (latitude <= 40))
    cols = np.flatnonzero((longitude >= 73) & (longitude <= 105))
    cube = np.full((len(dates), rows.size, cols.size), np.nan, dtype=np.float32)
    found: set[date] = set()
    hashes: list[tuple[str, str]] = []
    for year in range(start.year, end.year + 1):
        path = root / f"snowdepth-{year}.tar.gz"
        with path.open("rb") as source:
            hashes.append((path.name, hashlib.file_digest(source, "sha256").hexdigest()))
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".txt"):
                    continue
                stem = Path(member.name).stem
                if len(stem) != 7 or not stem.isdecimal() or int(stem[:4]) != year:
                    message = f"Unexpected daily filename: {member.name}"
                    raise ValueError(message)
                day = date(year, 1, 1) + timedelta(days=int(stem[4:]) - 1)
                if day.year != year or int(stem[4:]) < 1:
                    message = f"Invalid day of year: {stem}"
                    raise ValueError(message)
                if not start <= day <= end:
                    continue
                if day in found:
                    message = f"Duplicate date: {day}"
                    raise ValueError(message)
                stream = archive.extractfile(member)
                if stream is None:
                    message = f"Unreadable member: {member.name}"
                    raise ValueError(message)
                with stream:
                    grid = read_grid(stream.read())
                cube[(day - start).days] = grid[np.ix_(rows, cols)]
                found.add(day)
        print(f"Read {year}: {len(found)} daily grids so far", flush=True)
    return SnowCube(
        cube,
        dates,
        latitude[rows],
        longitude[cols],
        tuple(str(day) for day in dates if day not in found),
        tuple(hashes),
    )
