"""Check a directory of tiles against the file it was cut from."""
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import las, tile

CHUNK = 100_000
MASK = (1 << 64) - 1


@dataclass
class Report:
    problems: list = field(default_factory=list)
    tiles: int = 0
    points: int = 0

    @property
    def ok(self):
        return not self.problems


def weights(length, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, MASK, size=length, dtype=np.uint64, endpoint=True) | np.uint64(1)


def mix(z):
    """splitmix64 finalizer. Without it the digest would be linear in the record bytes, and two
    changes that cancel in the sum (a byte swapped between records) would go unseen."""
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(0xBF58476D1CE4E5B9)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def row_hashes(records, seed):
    length = records.dtype.itemsize
    rows = np.frombuffer(records.tobytes(), dtype=np.uint8).reshape(len(records), length)
    return mix(rows.astype(np.uint64) @ weights(length, seed))


def digest(records):
    """Two 64-bit sums of per-record hashes. Equal for equal multisets of records, in any order."""
    return tuple(int(row_hashes(records, seed).sum(dtype=np.uint64)) for seed in (1, 2))


def add(a, b):
    return ((a[0] + b[0]) & MASK, (a[1] + b[1]) & MASK)


def check(src, out_dir, chunk_points=CHUNK):
    """Return a Report. An empty list of problems means every point of src is in exactly one
    tile, in the right cell, and every tile header matches the points it holds."""
    report = Report()
    header, vlr = las.read_header(src)
    try:
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        report.problems.append(f"manifest.json: {e}")
        return report

    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        report.problems.append("manifest.json has no list of tiles")
        return report
    for t in manifest["tiles"]:
        if not isinstance(t, dict) or not {"file", "ix", "iy", "points"} <= set(t):
            report.problems.append(f"manifest tile entry {t!r} lacks file, ix, iy or points")
            return report
    if manifest.get("source_points") != header.point_count:
        report.problems.append(f"manifest says {manifest.get('source_points')} source points, file has {header.point_count}")
    for key, want in (("point_format", header.point_format), ("scale", list(header.scale)), ("offset", list(header.offset))):
        if manifest.get(key) != want:
            report.problems.append(f"manifest {key} {manifest.get(key)} differs from the source {want}")
    try:
        units = (tile.tile_units(manifest["tile_size"], header.scale[0]),
                 tile.tile_units(manifest["tile_size"], header.scale[1]))
    except (KeyError, ValueError) as e:
        report.problems.append(f"manifest tile size: {e}")
        return report
    origin, _ = tile.scan(src, header, chunk_points)
    if manifest.get("origin_int") != ([None, None] if origin[0] is None else origin):
        report.problems.append(f"manifest origin {manifest.get('origin_int')} differs from the lowest x, y in the source {origin}")
        return report

    listed = {t["file"] for t in manifest["tiles"]}
    present = {n for n in os.listdir(out_dir) if n.startswith("tile_") and n.endswith(".las")}
    for name in sorted(present - listed):
        report.problems.append(f"{name} is not in the manifest")
    for name in sorted(listed - present):
        report.problems.append(f"{name} is in the manifest but missing")

    if header.point_count and not manifest["tiles"]:
        report.problems.append("the source has points and the manifest lists no tiles")

    src_digest = (0, 0)
    for chunk in las.iter_chunks(src, header, chunk_points):
        src_digest = add(src_digest, digest(chunk))

    tile_digest = (0, 0)
    for t in manifest["tiles"]:
        if t["file"] not in present:
            continue
        path = os.path.join(out_dir, t["file"])
        try:
            th, tvlr = las.read_header(path)
        except ValueError as e:
            report.problems.append(str(e))
            continue
        report.tiles += 1
        if th.point_count == 0:
            report.problems.append(f"{t['file']}: holds no points")
            continue
        for key in ("point_format", "scale", "offset", "n_vlr", "point_offset"):
            if getattr(th, key) != getattr(header, key):
                report.problems.append(f"{t['file']}: {key} {getattr(th, key)} differs from the source {getattr(header, key)}")
        if tvlr != vlr:
            report.problems.append(f"{t['file']}: VLR block differs from the source")
        acc = tile.Accumulator()
        wrong_cell = 0
        for chunk in las.iter_chunks(path, th, chunk_points):
            ix, iy, _ = tile.cells(chunk, origin, units, 1)
            wrong_cell += int(((ix != t["ix"]) | (iy != t["iy"])).sum())
            acc.add(chunk)
            tile_digest = add(tile_digest, digest(chunk))
        report.points += acc.count
        if wrong_cell:
            report.problems.append(f"{t['file']}: {wrong_cell} points lie outside tile ({t['ix']}, {t['iy']})")
        if th.point_count != t["points"]:
            report.problems.append(f"{t['file']}: header counts {th.point_count} points, manifest {t['points']}")
        if th.by_return != tuple(int(c) for c in acc.by_return):
            report.problems.append(f"{t['file']}: header per-return counts {th.by_return} differ from the points")
        if th.bounds != acc.bounds(th.scale, th.offset):
            report.problems.append(f"{t['file']}: header bounds differ from the points")

    if report.points != header.point_count:
        report.problems.append(f"tiles hold {report.points} points, the source has {header.point_count}")
    if tile_digest != src_digest:
        report.problems.append("the records in the tiles are not the records in the source (digest differs)")
    return report
