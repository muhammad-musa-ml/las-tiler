import struct

import numpy as np
import pytest

from helpers import make_records, write
from las_tiler import las


def test_header_is_227_bytes():
    assert struct.calcsize(las.HEADER_FORMAT) == 227


@pytest.mark.parametrize("fmt,length", [(0, 20), (1, 28), (2, 26), (3, 34)])
def test_record_length_matches_the_spec(fmt, length):
    assert las.record_dtype(fmt).itemsize == length == las.RECORD_LENGTHS[fmt]


def test_field_positions_follow_the_spec(tmp_path):
    # Byte positions are the ones in the LAS 1.2 header table, read here with struct alone.
    rec = make_records(1, [5, -3], [7, 9], [1, 2])
    path = write(tmp_path / "a.las", rec, scale=(0.01, 0.02, 0.5), offset=(100.0, 200.0, 300.0))
    raw = path.read_bytes()
    assert raw[0:4] == b"LASF"
    assert struct.unpack_from("<H", raw, 4)[0] == 7  # file source id
    assert struct.unpack_from("<H", raw, 6)[0] == 1  # global encoding
    assert raw[8:24] == bytes(range(16))  # project GUID
    assert raw[24:26] == bytes([1, 2])
    assert raw[26:58].rstrip(b"\0") == b"tests"  # system identifier
    assert raw[58:90].rstrip(b"\0") == b"las-tiler tests"  # generating software
    assert struct.unpack_from("<H", raw, 90)[0] == 1  # creation day of year
    assert struct.unpack_from("<H", raw, 92)[0] == 2026  # creation year
    assert struct.unpack_from("<H", raw, 94)[0] == 227
    assert struct.unpack_from("<I", raw, 96)[0] == 227
    assert struct.unpack_from("<I", raw, 100)[0] == 0  # number of VLRs
    assert raw[104] == 1
    assert struct.unpack_from("<H", raw, 105)[0] == 28
    assert struct.unpack_from("<I", raw, 107)[0] == 2
    assert struct.unpack_from("<3d", raw, 131) == (0.01, 0.02, 0.5)
    assert struct.unpack_from("<3d", raw, 155) == (100.0, 200.0, 300.0)
    assert struct.unpack_from("<6d", raw, 179) == (
        5 * 0.01 + 100.0, -3 * 0.01 + 100.0, 9 * 0.02 + 200.0, 7 * 0.02 + 200.0,
        2 * 0.5 + 300.0, 1 * 0.5 + 300.0,
    )
    assert len(raw) == 227 + 2 * 28


def test_return_counts_ignore_numbers_outside_one_to_five(tmp_path):
    flags = np.array([1, 1, 2, 5, 0, 6, 7, 3], dtype=np.uint8) | (5 << 3)
    rec = make_records(0, np.arange(8), np.arange(8), np.arange(8), flags=flags)
    header, _ = las.read_header(write(tmp_path / "a.las", rec, point_format=0))
    assert header.by_return == (2, 1, 1, 0, 1)


def test_vlr_bytes_come_back_unchanged(tmp_path):
    vlr = bytes(range(60))
    rec = make_records(3, [1, 2], [3, 4], [5, 6], seed=3)
    path = write(tmp_path / "a.las", rec, point_format=3, vlr=vlr)
    raw = path.read_bytes()
    assert struct.unpack_from("<I", raw, 96)[0] == 227 + 60
    assert struct.unpack_from("<I", raw, 100)[0] == 1
    assert raw[227:287] == vlr
    header, got = las.read_header(path)
    assert got == vlr
    assert header.point_offset == 227 + 60
    assert header.n_vlr == 1


def test_chunks_cover_every_record_once(tmp_path):
    rec = make_records(2, np.arange(10), np.arange(10) * 2, np.arange(10) * 3, seed=4)
    path = write(tmp_path / "a.las", rec, point_format=2)
    header, _ = las.read_header(path)
    chunks = list(las.iter_chunks(path, header, 3))
    assert [len(c) for c in chunks] == [3, 3, 3, 1]
    assert np.concatenate(chunks).tobytes() == rec.tobytes()


def patched(path, position, value, tmp_path):
    raw = bytearray(path.read_bytes())
    raw[position:position + len(value)] = value
    out = tmp_path / "patched.las"
    out.write_bytes(bytes(raw))
    return out


@pytest.fixture
def good(tmp_path):
    return write(tmp_path / "good.las", make_records(1, [1, 2, 3], [4, 5, 6], [7, 8, 9]))


def test_refuses_a_file_that_is_not_las(good, tmp_path):
    with pytest.raises(ValueError, match="not a LAS file"):
        las.read_header(patched(good, 0, b"LASX", tmp_path))


def test_refuses_other_versions(good, tmp_path):
    with pytest.raises(ValueError, match="only 1.2"):
        las.read_header(patched(good, 25, bytes([4]), tmp_path))


def test_refuses_another_major_version(good, tmp_path):
    with pytest.raises(ValueError, match="only 1.2"):
        las.read_header(patched(good, 24, bytes([2]), tmp_path))


def test_refuses_extra_bytes_per_record_even_when_the_file_size_agrees(tmp_path):
    # Format 0 with a 22-byte record length and 6 extra bytes on disk: the size matches the header, so
    # the record-length check is the only thing between this file and misaligned reads.
    rec = make_records(0, [1, 2, 3], [4, 5, 6], [7, 8, 9])
    path = write(tmp_path / "a.las", rec, point_format=0)
    raw = bytearray(path.read_bytes())
    struct.pack_into("<H", raw, 105, 22)
    out = tmp_path / "extra.las"
    out.write_bytes(bytes(raw) + bytes(6))
    with pytest.raises(ValueError, match="record length"):
        las.read_header(out)


def test_refuses_a_truncated_file(good, tmp_path):
    cut = tmp_path / "cut.las"
    cut.write_bytes(good.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bytes on disk"):
        las.read_header(cut)


def test_refuses_bytes_after_the_last_record(good, tmp_path):
    longer = tmp_path / "longer.las"
    longer.write_bytes(good.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="bytes on disk"):
        las.read_header(longer)


def test_refuses_a_record_length_that_does_not_match_the_format(good, tmp_path):
    with pytest.raises(ValueError, match="record length"):
        las.read_header(patched(good, 105, struct.pack("<H", 29), tmp_path))


def test_refuses_a_compressed_point_format(good, tmp_path):
    # LAZ files set the top bit of the point format byte.
    with pytest.raises(ValueError, match="point format 129"):
        las.read_header(patched(good, 104, bytes([129]), tmp_path))


def test_refuses_a_file_shorter_than_a_header(tmp_path):
    short = tmp_path / "short.las"
    short.write_bytes(b"LASF" + bytes(100))
    with pytest.raises(ValueError, match="shorter than"):
        las.read_header(short)
