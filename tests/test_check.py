import json
import os
import shutil
import struct

import numpy as np
import pytest

from helpers import make_records, write
from las_tiler import check, las, tile

VLR = bytes(range(100, 160))


@pytest.fixture
def cut(tmp_path):
    """A source of 2,000 points with a VLR block, cut into tiles that pass the check."""
    rng = np.random.default_rng(21)
    n = 2_000
    rec = make_records(1, rng.integers(0, 40_000, n), rng.integers(0, 40_000, n),
                       rng.integers(0, 4_000, n), seed=21)
    src = write(tmp_path / "src.las", rec, vlr=VLR)
    out = tmp_path / "out"
    tile.tile_file(src, out, 10.0, chunk_points=700)
    assert check.check(src, out).ok
    return src, out, rec


def tile_files(out):
    return sorted(n for n in os.listdir(out) if n.endswith(".las"))


def rewrite(out, name, records):
    header, vlr = las.read_header(out / name)
    las.write_las(out / name, header, vlr, records)


def records_of(out, name):
    header, _ = las.read_header(out / name)
    return np.concatenate(list(las.iter_chunks(out / name, header, 10_000)))


def problems(src, out):
    report = check.check(src, out)
    assert not report.ok
    return " | ".join(report.problems)


def test_a_correct_cut_passes_and_reports_what_it_checked(cut):
    src, out, rec = cut
    report = check.check(src, out)
    assert report.ok and report.points == len(rec) and report.tiles == len(tile_files(out))
    assert report.tiles > 10


def test_a_dropped_point_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[3]
    rewrite(out, name, records_of(out, name)[1:])
    text = problems(src, out)
    assert "digest differs" in text and "manifest" in text


def test_a_duplicated_point_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[3]
    recs = records_of(out, name)
    rewrite(out, name, np.concatenate([recs, recs[:1]]))
    assert "digest differs" in problems(src, out)


def test_one_changed_byte_is_found_when_every_count_still_agrees(cut):
    src, out, _ = cut
    name = tile_files(out)[2]
    recs = records_of(out, name)
    recs["intensity"][0] ^= 1
    rewrite(out, name, recs)
    text = problems(src, out)
    assert text == "the records in the tiles are not the records in the source (digest differs)"


def test_a_point_in_the_wrong_tile_is_found(cut):
    src, out, _ = cut
    a, b = tile_files(out)[1], tile_files(out)[7]
    ra, rb = records_of(out, a), records_of(out, b)
    rewrite(out, a, ra[1:])
    rewrite(out, b, np.concatenate([rb, ra[:1]]))
    text = problems(src, out)
    assert "lie outside tile" in text


def test_a_wrong_bounds_field_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[0]
    raw = bytearray((out / name).read_bytes())
    struct.pack_into("<d", raw, 179, struct.unpack_from("<d", raw, 179)[0] + 0.001)
    (out / name).write_bytes(bytes(raw))
    assert "header bounds differ" in problems(src, out)


def test_a_wrong_per_return_count_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[0]
    raw = bytearray((out / name).read_bytes())
    struct.pack_into("<I", raw, 111, struct.unpack_from("<I", raw, 111)[0] + 1)
    (out / name).write_bytes(bytes(raw))
    assert "per-return counts" in problems(src, out)


def test_a_changed_vlr_block_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[0]
    raw = bytearray((out / name).read_bytes())
    raw[230] ^= 0xFF
    (out / name).write_bytes(bytes(raw))
    assert "VLR block differs" in problems(src, out)


def test_a_stray_tile_file_is_found(cut):
    src, out, _ = cut
    shutil.copy(out / tile_files(out)[0], out / "tile_99_99.las")
    assert "tile_99_99.las is not in the manifest" in problems(src, out)


def test_a_missing_tile_file_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[0]
    os.remove(out / name)
    assert f"{name} is in the manifest but missing" in problems(src, out)


def test_a_wrong_origin_in_the_manifest_is_found(cut):
    src, out, _ = cut
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["origin_int"][0] += 1
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert "origin" in problems(src, out)


def test_a_manifest_with_no_tiles_does_not_pass_a_source_that_has_points(cut):
    src, out, _ = cut
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["tiles"] = []
    (out / "manifest.json").write_text(json.dumps(manifest))
    for name in tile_files(out):
        os.remove(out / name)
    assert "lists no tiles" in problems(src, out)


def test_a_missing_manifest_is_a_problem_and_not_a_crash(cut):
    src, out, _ = cut
    os.remove(out / "manifest.json")
    assert "manifest.json" in problems(src, out)


def test_a_source_with_no_points_passes_with_no_tiles(tmp_path):
    src = write(tmp_path / "a.las", make_records(1, [], [], []))
    tile.tile_file(src, tmp_path / "out", 1.0)
    report = check.check(src, tmp_path / "out")
    assert report.ok and report.tiles == 0 and report.points == 0


def test_digest_ignores_order_and_sees_a_byte_moved_between_records():
    rec = make_records(1, np.arange(50), np.arange(50), np.arange(50), seed=5)
    assert check.digest(rec) == check.digest(rec[::-1].copy())
    swapped = rec.copy()
    assert swapped["intensity"][0] != swapped["intensity"][1]
    swapped["intensity"][[0, 1]] = swapped["intensity"][[1, 0]]
    assert check.digest(swapped) != check.digest(rec)


def test_digest_arithmetic_wraps_like_64_bit_integers():
    rec = make_records(0, [1, 2, 3], [4, 5, 6], [7, 8, 9], seed=8)
    w = check.weights(20, 1)
    rows = np.frombuffer(rec.tobytes(), dtype=np.uint8).reshape(3, 20)
    plain = [sum(int(b) * int(x) for b, x in zip(row, w)) % 2**64 for row in rows]
    got = rows.astype(np.uint64) @ w
    assert [int(v) for v in got] == plain
