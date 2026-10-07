"""Download and read the three TPDC datasets listed in README.md.

    python -m snowdivides.data          # fetch everything the experiment needs (~95 MB)

Snow depth: one ASCII grid per day, 161 x 321 cells at 0.25 deg, row 0 = 55 N,
column 0 = 60 E (cell centres), cm, NODATA = -1.  Files from 2017 on have no header.
DEM: band 41 of AM.tif in the snowmelt-onset product (uncompressed, planar uint16).
Glaciers: Table S4 (sheet t4) of the Yao et al. (2012) supplement.
"""
import json
import re
import struct
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNOW_DIR, DEM_DIR, GLACIER_XLSX = DATA / "snow_depth", DATA / "dem", DATA / "yao2012.xlsx"

TPDC = "https://data.tpdc.ac.cn"
SNOW_DEPTH_ID = "df40346a-0202-4ed2-bb07-b65dfcda9368"
MELT_ONSET_ID = "01be1b50-d9b6-4189-8aa0-9e005514b6d1"
GLACIER_ID = "439b01bd-1799-4171-b9ed-16e82ccc43df"

NROWS, NCOLS, CELL = 161, 321, 0.25
LAT = 55.0 - CELL * np.arange(NROWS)
LON = 60.0 + CELL * np.arange(NCOLS)
PLATEAU = dict(lat=(26.0, 40.0), lon=(73.0, 105.0))


# ---------------------------------------------------------------- download

def _get_json(path):
    req = urllib.request.Request(TPDC + path, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=60))["data"] or []


def _files(metadata_id):
    """All files of a TPDC dataset, {name: file id} (walks the folder tree)."""
    out, todo = {}, _get_json(f"/file/file/getRootFileDataList?metadataId={metadata_id}")
    while todo:
        item = todo.pop()
        if item["type"] == "dir":
            todo += _get_json(f"/file/file/getFileDataList?parentId={item['id']}")
        else:
            out[item["name"]] = item["id"]
    return out


def _download(file_id, out):
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{TPDC}/file/file/batchDownloadByFileId?fileId={file_id}",
                                 method="POST", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out.write_bytes(r.read())
    time.sleep(0.5)
    return out


def download(years=range(2000, 2015)):
    snow = _files(SNOW_DEPTH_ID)
    for y in years:
        _download(snow[f"snowdepth-{y}.tar.gz"], SNOW_DIR / f"snowdepth-{y}.tar.gz")
    z = _download(_files(MELT_ONSET_ID)["Snowmelt_PassiveMicrowave_HMA_DOY.zip"], DEM_DIR / "melt_onset.zip")
    zipfile.ZipFile(z).extract("AM.tif", DEM_DIR)
    xlsx = [n for n in _files(GLACIER_ID) if n.endswith(".xlsx")][0]
    _download(_files(GLACIER_ID)[xlsx], GLACIER_XLSX)


# ---------------------------------------------------------------- snow depth

def _parse(text):
    lines = text.splitlines()
    if lines and lines[0].split()[0].lower() == "ncols":
        lines = lines[6:]
    a = np.array(" ".join(lines).split(), dtype=np.float32).reshape(NROWS, NCOLS)
    a[a < 0] = np.nan
    return a


def load_year(year):
    """(day-of-year array, float32 grids (n_days, 161, 321)) for one calendar year."""
    t = tarfile.open(SNOW_DIR / f"snowdepth-{year}.tar.gz")
    members = {Path(m.name).stem: m for m in t.getmembers() if m.isfile()}
    keys = sorted(k for k in members if k.isdigit() and k.startswith(str(year)))
    return (np.array([int(k[4:]) for k in keys]),
            np.stack([_parse(t.extractfile(members[k]).read().decode()) for k in keys]))


def snow_years(first, last):
    """Snow years first/first+1 ... last-1/last (1 Aug - 31 Jul) on the plateau window.

    Each snow year joins two consecutive calendar years (days 213-365 of one, 1-212 of
    the next), so no artificial jump appears on 1 January.  Returns
    (cube (n_years, 365, rows, cols), lat, lon); day 0 of every year is 1 Aug.
    """
    r = (LAT >= PLATEAU["lat"][0]) & (LAT <= PLATEAU["lat"][1])
    c = (LON >= PLATEAU["lon"][0]) & (LON <= PLATEAU["lon"][1])
    prev, out = None, []
    for y in range(first, last + 1):
        g = load_year(y)[1][:365][:, r][:, :, c]
        if prev is not None:
            out.append(np.concatenate([prev[212:], g[:212]]))
        prev = g
    return np.stack(out), LAT[r], LON[c]


# ---------------------------------------------------------------- elevation

def elevation(lat_q, lon_q):
    """DEM (m) at the given points by nearest neighbour; NaN outside the DEM (east of ~103.2 E)."""
    buf = (DEM_DIR / "AM.tif").read_bytes()
    bo = "<" if buf[:2] == b"II" else ">"
    off = struct.unpack(bo + "I", buf[4:8])[0]
    size, code, tags = {3: 2, 4: 4, 12: 8}, {3: "H", 4: "I", 12: "d"}, {}
    for i in range(struct.unpack(bo + "H", buf[off:off + 2])[0]):
        e = buf[off + 2 + 12 * i: off + 14 + 12 * i]
        tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
        if typ not in size:
            continue
        n = size[typ] * cnt
        data = e[8:8 + n] if n <= 4 else buf[struct.unpack(bo + "I", e[8:12])[0]:][:n]
        tags[tag] = struct.unpack(bo + code[typ] * cnt, data)
    w, h = tags[256][0], tags[257][0]
    strips = range(40 * h, 41 * h)                      # planar, one strip per row: band 41 = strips 40h..41h-1
    raw = b"".join(buf[tags[273][s]:tags[273][s] + tags[279][s]] for s in strips)
    dem = np.frombuffer(raw, dtype=bo + "u2").reshape(h, w)
    dx, dy = tags[33550][:2]
    x0, y0 = tags[33922][3:5]
    lon = x0 + dx * (np.arange(w) + 0.5)
    lat = y0 - dy * (np.arange(h) + 0.5)
    lat_q, lon_q = np.asarray(lat_q), np.asarray(lon_q)
    z = dem[np.abs(lat[:, None] - lat_q).argmin(0), np.abs(lon[:, None] - lon_q).argmin(0)].astype(float)
    z[(lon_q > lon[-1] + dx / 2) | (z == 0)] = np.nan
    return z


# ---------------------------------------------------------------- glaciers

REGION_NAMES = {"I": "SE Tibet", "II": "Central Himalaya (Nyainqentanglha)", "III": "Central Himalaya",
                "IV": "West Himalaya", "V": "Pamir / Karakoram / W Kunlun", "VI": "Tanggula (inner TP)",
                "VII": "Qilian"}


def glaciers():
    """Yao et al. (2012) Table S4: list of dicts with region, name, lat, lon, rate (length change, m/yr)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(GLACIER_XLSX)
    strings = ["".join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    dms = lambda s: (lambda d: int(d[0]) + int(d[1]) / 60)(re.findall(r"\d+", s))
    out, region = [], ""
    for row in ET.fromstring(z.read("xl/worksheets/sheet5.xml")).iter("{%s}row" % ns["m"]):
        cells = {}
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            if v is not None:
                cells[re.match(r"[A-Z]+", c.get("r")).group()] = strings[int(v.text)] if c.get("t") == "s" else v.text
        try:
            lat, lon = dms(cells["E"]), dms(cells["F"])
            y0, y1 = map(int, re.findall(r"\d{4}", cells["K"])[:2])
            rate = float(cells["L"]) / (y1 - y0)
        except (KeyError, ValueError, IndexError):
            continue
        region = cells.get("A", "").strip() or region
        out.append(dict(region=region, name=cells.get("D", "").strip(), lat=lat, lon=lon, rate=rate))
    return out


if __name__ == "__main__":
    download()
    print("data ready in", DATA)
