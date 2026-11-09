"""Split a LAS file into square tiles on an integer grid, reading it in chunks."""
import json
import os

import numpy as np

from . import las

DEFAULT_CHUNK = 1_000_000


def tile_units(tile_size, scale):
    """The tile edge in scaled-integer units; it must be a whole number of them."""
    try:
        units = round(tile_size / scale)
    except (OverflowError, ValueError):
        raise ValueError(f"tile size {tile_size} is not a whole multiple of scale {scale}") from None
    if units < 1 or abs(units * scale - tile_size) > 1e-6 * scale:
        raise ValueError(f"tile size {tile_size} is not a whole multiple of scale {scale}")
    return units


def scan(path, header, chunk_points):
    """Lowest and highest x and y as plain integers, read from the points and not from the header."""
    lo = [None, None]
    hi = [None, None]
    for chunk in las.iter_chunks(path, header, chunk_points):
        for k, name in enumerate(("x", "y")):
            a, b = int(chunk[name].min()), int(chunk[name].max())
            lo[k] = a if lo[k] is None else min(lo[k], a)
            hi[k] = b if hi[k] is None else max(hi[k], b)
    return lo, hi


def cells(chunk, origin, units, rows):
    """Tile column and row of every point. The subtraction is done in 64 bits: two int32 values
    that are valid LAS coordinates can be further apart than an int32 holds."""
    ix = (chunk["x"].astype(np.int64) - origin[0]) // units[0]
    iy = (chunk["y"].astype(np.int64) - origin[1]) // units[1]
    return ix, iy, ix * rows + iy


def tile_name(ix, iy):
    return f"tile_{ix}_{iy}.las"


class Accumulator:
    """Count, per-return counts and integer bounds of one tile, updated as chunks arrive."""

    def __init__(self):
        self.count = 0
        self.by_return = np.zeros(5, dtype=np.int64)
        self.lo = [None] * 3
        self.hi = [None] * 3

    def add(self, records):
        self.count += len(records)
        self.by_return += las.return_counts(records["flags"])
        for k, name in enumerate("xyz"):
            a, b = int(records[name].min()), int(records[name].max())
            self.lo[k] = a if self.lo[k] is None else min(self.lo[k], a)
            self.hi[k] = b if self.hi[k] is None else max(self.hi[k], b)

    def bounds(self, scale, offset):
        out = []
        for k in range(3):
            out += [las.world(self.hi[k], scale[k], offset[k]), las.world(self.lo[k], scale[k], offset[k])]
        return tuple(out)


def tile_file(src, out_dir, tile_size, chunk_points=DEFAULT_CHUNK):
    """Write one LAS file per occupied tile into out_dir and a manifest.json beside them.

    Tile (i, j) holds the points with i*T <= x - x0 < (i+1)*T and j*T <= y - y0 < (j+1)*T, in
    scaled-integer units, where x0 and y0 are the lowest x and y found in the file. Records are
    copied byte for byte; nothing is requantized. The VLR block (CRS and other metadata) is
    copied into every tile.
    """
    if chunk_points < 1:
        raise ValueError("chunk must be at least 1 point")
    header, vlr = las.read_header(src)
    units = (tile_units(tile_size, header.scale[0]), tile_units(tile_size, header.scale[1]))
    if os.path.isdir(out_dir) and any(
            n == "manifest.json" or (n.startswith("tile_") and n.endswith(".las")) for n in os.listdir(out_dir)):
        raise ValueError(f"{out_dir} already holds tiles; use an empty directory")
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = scan(src, header, chunk_points)
    manifest = {
        "source_points": header.point_count, "point_format": header.point_format,
        "tile_size": tile_size, "scale": list(header.scale), "offset": list(header.offset),
        "tiles": [],
    }
    if lo[0] is None:
        _write_manifest(out_dir, manifest, lo)
        return manifest
    rows = (hi[1] - lo[1]) // units[1] + 1
    if ((hi[0] - lo[0]) // units[0] + 1) * rows >= 2**62:
        raise ValueError("tile size is too small for the extent of this file")
    acc = {}
    for chunk in las.iter_chunks(src, header, chunk_points):
        ix, iy, key = cells(chunk, lo, units, rows)
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        keys, starts = np.unique(sorted_key, return_index=True)
        ends = list(starts[1:]) + [len(chunk)]
        for k, s, e in zip(keys, starts, ends):
            records = chunk[order[s:e]]
            path = os.path.join(out_dir, tile_name(int(k) // rows, int(k) % rows))
            if int(k) not in acc:
                acc[int(k)] = Accumulator()
                with open(path, "wb") as f:
                    f.write(bytes(las.HEADER_SIZE))
                    f.write(vlr)
            with open(path, "ab") as f:
                f.write(records.tobytes())
            acc[int(k)].add(records)
    for k in sorted(acc):
        a = acc[k]
        name = tile_name(k // rows, k % rows)
        h = las.Header(header.file_source_id, header.global_encoding, header.guid, header.system_id,
                       header.software, header.creation_doy, header.creation_year, header.point_offset,
                       header.n_vlr, header.point_format, header.record_length, a.count,
                       tuple(int(c) for c in a.by_return), header.scale, header.offset,
                       a.bounds(header.scale, header.offset))
        with open(os.path.join(out_dir, name), "r+b") as f:
            f.write(las.pack_header(h))
        manifest["tiles"].append({"file": name, "ix": k // rows, "iy": k % rows, "points": a.count})
    _write_manifest(out_dir, manifest, lo)
    return manifest


def _write_manifest(out_dir, manifest, lo):
    manifest["origin_int"] = [None, None] if lo[0] is None else [lo[0], lo[1]]
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
