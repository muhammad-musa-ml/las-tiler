"""The parts of the LAS 1.2 format that tiling needs: header, point records, chunked reads."""
import os
import struct
from dataclasses import dataclass

import numpy as np

HEADER_FORMAT = "<4sHH16sBB32s32sHHHIIBHI5I3d3d6d"
HEADER_SIZE = 227
RECORD_LENGTHS = {0: 20, 1: 28, 2: 26, 3: 34}


@dataclass
class Header:
    file_source_id: int
    global_encoding: int
    guid: bytes
    system_id: bytes
    software: bytes
    creation_doy: int
    creation_year: int
    point_offset: int
    n_vlr: int
    point_format: int
    record_length: int
    point_count: int
    by_return: tuple
    scale: tuple
    offset: tuple
    bounds: tuple  # max x, min x, max y, min y, max z, min z


def record_dtype(point_format):
    fields = [
        ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("intensity", "<u2"),
        ("flags", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
        ("user_data", "u1"), ("point_source_id", "<u2"),
    ]
    if point_format in (1, 3):
        fields.append(("gps_time", "<f8"))
    if point_format in (2, 3):
        fields += [("red", "<u2"), ("green", "<u2"), ("blue", "<u2")]
    return np.dtype(fields)


def pack_header(h):
    return struct.pack(
        HEADER_FORMAT, b"LASF", h.file_source_id, h.global_encoding, h.guid, 1, 2,
        h.system_id, h.software, h.creation_doy, h.creation_year, HEADER_SIZE,
        h.point_offset, h.n_vlr, h.point_format, h.record_length, h.point_count,
        *h.by_return, *h.scale, *h.offset, *h.bounds,
    )


def read_header(path):
    """Return (header, vlr_bytes). Refuses anything it cannot read exactly."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise ValueError(f"{path}: shorter than a LAS 1.2 header")
        v = struct.unpack(HEADER_FORMAT, raw)
        if v[0] != b"LASF":
            raise ValueError(f"{path}: not a LAS file")
        if (v[4], v[5]) != (1, 2):
            raise ValueError(f"{path}: LAS {v[4]}.{v[5]}; only 1.2 is read")
        if v[10] != HEADER_SIZE:
            raise ValueError(f"{path}: header size {v[10]}, expected {HEADER_SIZE}")
        h = Header(v[1], v[2], v[3], v[6], v[7], v[8], v[9], v[11], v[12], v[13],
                   v[14], v[15], tuple(v[16:21]), tuple(v[21:24]), tuple(v[24:27]),
                   tuple(v[27:33]))
        if h.point_format not in RECORD_LENGTHS:
            raise ValueError(f"{path}: point format {h.point_format} is not read")
        if h.record_length != RECORD_LENGTHS[h.point_format]:
            raise ValueError(f"{path}: record length {h.record_length} does not match format {h.point_format}")
        if h.point_offset < HEADER_SIZE:
            raise ValueError(f"{path}: point data starts inside the header")
        if min(h.scale) <= 0:
            raise ValueError(f"{path}: scale must be positive")
        expected = h.point_offset + h.point_count * h.record_length
        if size != expected:
            raise ValueError(f"{path}: {size} bytes on disk, header says {expected}")
        vlr = f.read(h.point_offset - HEADER_SIZE)
    return h, vlr


def iter_chunks(path, header, chunk_points):
    """Yield the point records in arrays of at most chunk_points."""
    dtype = record_dtype(header.point_format)
    left = header.point_count
    with open(path, "rb") as f:
        f.seek(header.point_offset)
        while left:
            n = min(chunk_points, left)
            chunk = np.fromfile(f, dtype=dtype, count=n)
            if len(chunk) != n:
                raise ValueError(f"{path}: ended before {header.point_count} points")
            left -= n
            yield chunk


def return_counts(flags):
    """Points per return number 1 to 5, as the legacy header field counts them."""
    numbers = (flags & 7).astype(np.int64)
    return np.bincount(numbers, minlength=8)[1:6]


def world(ints, scale, offset):
    return float(ints) * scale + offset


def summarize(records, scale, offset):
    """Point count, per-return counts and bounds for a record array, bounds as the header stores them."""
    if len(records) == 0:
        return 0, (0,) * 5, (0.0,) * 6
    x, y, z = records["x"], records["y"], records["z"]
    bounds = (
        world(x.max(), scale[0], offset[0]), world(x.min(), scale[0], offset[0]),
        world(y.max(), scale[1], offset[1]), world(y.min(), scale[1], offset[1]),
        world(z.max(), scale[2], offset[2]), world(z.min(), scale[2], offset[2]),
    )
    return len(records), tuple(int(c) for c in return_counts(records["flags"])), bounds


def write_las(path, header, vlr_bytes, records):
    """Write a whole record array as one LAS file; the header is rebuilt from the records."""
    count, by_return, bounds = summarize(records, header.scale, header.offset)
    h = Header(header.file_source_id, header.global_encoding, header.guid, header.system_id,
               header.software, header.creation_doy, header.creation_year,
               HEADER_SIZE + len(vlr_bytes), header.n_vlr, header.point_format,
               header.record_length, count, by_return, header.scale, header.offset, bounds)
    with open(path, "wb") as f:
        f.write(pack_header(h))
        f.write(vlr_bytes)
        f.write(records.tobytes())
    return h
