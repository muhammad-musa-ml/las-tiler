"""A seeded synthetic scan: sloped ground, a box-shaped building and a pipe, in UTM-sized coordinates."""
import numpy as np

from . import las

SCALE = (0.001, 0.001, 0.001)
OFFSET = (500000.0, 4500000.0, 0.0)
CHUNK = 500_000


def scene(n, rng, extent):
    """n world-frame points (metres, relative to the scene corner) as three float arrays."""
    kind = rng.choice(3, size=n, p=[0.7, 0.2, 0.1])
    x = rng.random(n) * extent
    y = rng.random(n) * extent
    z = 0.03 * x + rng.normal(0, 0.02, n)  # ground, rising 3 cm per metre
    box = kind == 1
    wall = box & (rng.random(n) < 0.6)
    bx0, by0, bw = 0.3 * extent, 0.3 * extent, 0.25 * extent
    x[box] = bx0 + rng.random(box.sum()) * bw
    y[box] = by0 + rng.random(box.sum()) * bw
    z[box] = 8.0 + 0.03 * x[box]  # roof
    side = rng.integers(0, 4, wall.sum())
    along = rng.random(wall.sum()) * bw
    x[wall] = np.where(side == 0, bx0, np.where(side == 1, bx0 + bw, bx0 + along))
    y[wall] = np.where(side == 2, by0, np.where(side == 3, by0 + bw, by0 + along))
    z[wall] = rng.random(wall.sum()) * 8.0
    pipe = kind == 2
    angle = rng.random(pipe.sum()) * 2 * np.pi
    x[pipe] = 0.8 * extent + 0.25 * np.cos(angle)
    y[pipe] = rng.random(pipe.sum()) * extent
    z[pipe] = 3.0 + 0.25 * np.sin(angle)
    return x, y, z


def to_records(x, y, z, rng, point_format):
    n = len(x)
    rec = np.zeros(n, dtype=las.record_dtype(point_format))
    rec["x"] = np.round(x / SCALE[0]).astype(np.int32)
    rec["y"] = np.round(y / SCALE[1]).astype(np.int32)
    rec["z"] = np.round(z / SCALE[2]).astype(np.int32)
    rec["intensity"] = rng.integers(0, 4096, n)
    numbers = rng.integers(1, 4, n)
    rec["flags"] = numbers | (3 << 3)
    rec["classification"] = np.where(z < 0.5 + 0.03 * x, 2, 1)
    if "gps_time" in rec.dtype.names:
        rec["gps_time"] = np.sort(rng.random(n)) * 600.0
    return rec


def write_scene(path, n_points, seed, extent=400.0, point_format=1):
    """Write n_points points to a LAS 1.2 file in chunks, so the file may be larger than memory."""
    header = las.Header(
        0, 0, bytes(16), b"las-tiler synth".ljust(32, b"\0"), b"las-tiler".ljust(32, b"\0"),
        1, 2026, las.HEADER_SIZE, 0, point_format, las.RECORD_LENGTHS[point_format], 0,
        (0,) * 5, SCALE, OFFSET, (0.0,) * 6,
    )
    streams = np.random.SeedSequence(seed).spawn((n_points + CHUNK - 1) // CHUNK)
    count, by_return = 0, np.zeros(5, dtype=np.int64)
    lo = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(bytes(las.HEADER_SIZE))
        for i, stream in enumerate(streams):
            rng = np.random.default_rng(stream)
            m = min(CHUNK, n_points - i * CHUNK)
            rec = to_records(*scene(m, rng, extent), rng, point_format)
            f.write(rec.tobytes())
            count += m
            by_return += las.return_counts(rec["flags"])
            for k, name in enumerate("xyz"):
                lo[k] = min(lo[k], rec[name].min())
                hi[k] = max(hi[k], rec[name].max())
        if count:
            bounds = []
            for k in range(3):
                bounds += [las.world(hi[k], SCALE[k], OFFSET[k]), las.world(lo[k], SCALE[k], OFFSET[k])]
        else:
            bounds = [0.0] * 6
        header.point_count = count
        header.by_return = tuple(int(c) for c in by_return)
        header.bounds = tuple(bounds)
        f.seek(0)
        f.write(las.pack_header(header))
    return header
