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


def tile_pair(out, same):
    """Two manifest tiles that share a row (same iy) or a column (same ix) and differ in the other."""
    tiles = json.loads((out / "manifest.json").read_text())["tiles"]
    other = "ix" if same == "iy" else "iy"
    for a in tiles:
        for b in tiles:
            if a[same] == b[same] and a[other] != b[other]:
                return a["file"], b["file"]
    raise AssertionError("the fixture has no such pair of tiles")


@pytest.mark.parametrize("same", ["iy", "ix"])
def test_a_point_in_the_wrong_tile_is_found_when_only_one_index_is_wrong(cut, same):
    src, out, _ = cut
    a, b = tile_pair(out, same)
    ra, rb = records_of(out, a), records_of(out, b)
    rewrite(out, a, ra[1:])
    rewrite(out, b, np.concatenate([rb, ra[:1]]))
    assert "1 points lie outside tile" in problems(src, out)


def patch_header(out, name, position, fmt, delta):
    raw = bytearray((out / name).read_bytes())
    struct.pack_into(fmt, raw, position, struct.unpack_from(fmt, raw, position)[0] + delta)
    (out / name).write_bytes(bytes(raw))


@pytest.mark.parametrize("position", [179, 187, 195, 203, 211, 219])
def test_a_wrong_bounds_field_is_found(cut, position):
    src, out, _ = cut
    patch_header(out, tile_files(out)[0], position, "<d", 0.001)
    assert "header bounds differ" in problems(src, out)


@pytest.mark.parametrize("position", [111, 115, 119, 123, 127])
def test_a_wrong_per_return_count_is_found(cut, position):
    src, out, _ = cut
    patch_header(out, tile_files(out)[0], position, "<I", 1)
    assert "per-return counts" in problems(src, out)


@pytest.mark.parametrize("position,label", [(131, "scale"), (155, "offset")])
def test_a_tile_scale_or_offset_that_differs_from_the_source_is_found(cut, position, label):
    src, out, _ = cut
    patch_header(out, tile_files(out)[0], position, "<d", 1.0)
    assert f": {label} (" in problems(src, out)


def test_a_tile_vlr_count_that_differs_from_the_source_is_found(cut):
    src, out, _ = cut
    patch_header(out, tile_files(out)[0], 100, "<I", 1)
    assert ": n_vlr 2 differs from the source 1" in problems(src, out)


def test_a_manifest_scale_that_differs_from_the_source_is_found(cut):
    src, out, _ = cut
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["scale"][0] = 0.002
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert "manifest scale" in problems(src, out)


def test_a_manifest_source_point_count_that_differs_is_found(cut):
    src, out, _ = cut
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["source_points"] += 1
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert "manifest says" in problems(src, out)


def test_a_tile_file_with_no_points_is_found(cut):
    src, out, _ = cut
    name = tile_files(out)[0]
    rewrite(out, name, records_of(out, name)[:0])
    assert f"{name}: holds no points" in problems(src, out)


@pytest.mark.parametrize("content", ["[]", '{"tiles": [{"file": "tile_0_0.las"}]}'])
def test_a_manifest_of_the_wrong_shape_is_a_problem_and_not_a_crash(cut, content):
    src, out, _ = cut
    (out / "manifest.json").write_text(content)
    assert "manifest" in problems(src, out)


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


def plain_row_hash(row, weights):
    """The per-record hash written with Python integers: weighted byte sum, then splitmix64's finalizer."""
    z = sum(int(b) * int(w) for b, w in zip(row, weights)) % 2**64
    z ^= z >> 30
    z = z * 0xBF58476D1CE4E5B9 % 2**64
    z ^= z >> 27
    z = z * 0x94D049BB133111EB % 2**64
    return z ^ (z >> 31)


@pytest.mark.parametrize("seed", [1, 2])
def test_row_hashes_match_a_plain_python_computation(seed):
    rec = make_records(1, np.arange(6) * 100, np.arange(6), np.arange(6), seed=8)
    rows = np.frombuffer(rec.tobytes(), dtype=np.uint8).reshape(6, 28)
    w = check.weights(28, seed)
    assert [int(v) for v in check.row_hashes(rec, seed)] == [plain_row_hash(r, w) for r in rows]


def test_digest_is_two_full_width_sums_of_the_row_hashes():
    rec = make_records(1, np.arange(300) * 7, np.arange(300), np.arange(300), seed=9)
    d = check.digest(rec)
    for seed, got in zip((1, 2), d):
        assert got == sum(int(v) for v in check.row_hashes(rec, seed)) % 2**64
    assert min(d) > 2**40
