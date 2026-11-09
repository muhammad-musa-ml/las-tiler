import numpy as np

from las_tiler import las


def make_header(point_format=1, scale=(0.001, 0.001, 0.001), offset=(0.0, 0.0, 0.0), n_vlr=0):
    return las.Header(
        0, 0, bytes(16), b"tests".ljust(32, b"\0"), b"las-tiler tests".ljust(32, b"\0"),
        1, 2026, las.HEADER_SIZE, n_vlr, point_format, las.RECORD_LENGTHS[point_format], 0,
        (0,) * 5, scale, offset, (0.0,) * 6,
    )


def make_records(point_format, x, y, z, flags=None, seed=0):
    """Records with the given integer coordinates and random bytes in every other field."""
    n = len(x)
    dtype = las.record_dtype(point_format)
    rec = np.zeros(n, dtype=dtype)
    rng = np.random.default_rng(seed)
    for name in dtype.names:
        if name in ("x", "y", "z"):
            continue
        if dtype[name].kind == "f":
            rec[name] = rng.random(n) * 1e5
        else:
            info = np.iinfo(dtype[name])
            rec[name] = rng.integers(info.min, info.max, n, endpoint=True)
    rec["x"], rec["y"], rec["z"] = x, y, z
    if flags is None:
        flags = rng.integers(1, 6, n) | (5 << 3)
    rec["flags"] = flags
    return rec


def write(path, records, point_format=1, scale=(0.001, 0.001, 0.001), offset=(0.0, 0.0, 0.0), vlr=b""):
    header = make_header(point_format, scale, offset, n_vlr=1 if vlr else 0)
    las.write_las(path, header, vlr, records)
    return path
