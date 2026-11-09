import json
import os

import numpy as np
import pytest

from helpers import make_records, write
from las_tiler import las, tile


def read_tiles(out_dir):
    """{(ix, iy): [record bytes]} for every tile file listed in the manifest."""
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    tiles = {}
    for t in manifest["tiles"]:
        path = os.path.join(out_dir, t["file"])
        header, _ = las.read_header(path)
        records = np.concatenate(list(las.iter_chunks(path, header, 1000)))
        tiles[(t["ix"], t["iy"])] = [r.tobytes() for r in records]
    return tiles


def reference(records, units):
    """The same assignment written with Python integers, one point at a time."""
    x0, y0 = int(records["x"].min()), int(records["y"].min())
    out = {}
    for r in records:
        key = ((int(r["x"]) - x0) // units[0], (int(r["y"]) - y0) // units[1])
        out.setdefault(key, []).append(r.tobytes())
    return out


@pytest.mark.parametrize("fmt", [0, 1, 2, 3])
def test_every_point_lands_where_a_plain_python_loop_puts_it(tmp_path, fmt):
    rng = np.random.default_rng(10 + fmt)
    n = 1_200
    rec = make_records(fmt, rng.integers(-50_000, 50_000, n), rng.integers(-50_000, 50_000, n),
                       rng.integers(-5_000, 5_000, n), seed=fmt)
    src = write(tmp_path / "a.las", rec, point_format=fmt)
    tile.tile_file(src, tmp_path / "out", 10.0, chunk_points=900)
    assert read_tiles(tmp_path / "out") == reference(rec, (10_000, 10_000))


def test_a_point_on_a_tile_edge_goes_to_the_upper_tile(tmp_path):
    x = np.array([5000, 5999, 6000, 6999, 7000])
    rec = make_records(1, x, np.zeros(5, dtype=int), np.zeros(5, dtype=int))
    src = write(tmp_path / "a.las", rec)
    manifest = tile.tile_file(src, tmp_path / "out", 1.0)
    assert [(t["ix"], t["points"]) for t in manifest["tiles"]] == [(0, 2), (1, 2), (2, 1)]
    assert manifest["origin_int"] == [5000, 0]


def test_coordinates_further_apart_than_an_int32_holds(tmp_path):
    # 2e9 - (-2e9) does not fit in an int32; a subtraction done in 32 bits wraps to a negative tile.
    x = np.array([-2_000_000_000, 0, 2_000_000_000])
    rec = make_records(1, x, np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    src = write(tmp_path / "a.las", rec, scale=(1.0, 1.0, 1.0))
    manifest = tile.tile_file(src, tmp_path / "out", 1000.0)
    assert [t["ix"] for t in manifest["tiles"]] == [0, 2_000_000, 4_000_000]


def test_chunk_size_does_not_change_any_output_byte(tmp_path):
    rng = np.random.default_rng(5)
    n = 300
    rec = make_records(3, rng.integers(0, 30_000, n), rng.integers(0, 30_000, n),
                       rng.integers(0, 3_000, n))
    src = write(tmp_path / "a.las", rec, point_format=3)
    outputs = []
    for chunk in (1, 7, 64, 100_000):
        out = tmp_path / f"out{chunk}"
        tile.tile_file(src, out, 5.0, chunk_points=chunk)
        outputs.append({name: (out / name).read_bytes() for name in sorted(os.listdir(out))})
    assert len(outputs[0]) > 10
    assert outputs[0] == outputs[1] == outputs[2] == outputs[3]


def test_each_tile_header_matches_its_own_points(tmp_path):
    rng = np.random.default_rng(6)
    n = 600
    rec = make_records(1, rng.integers(0, 20_000, n), rng.integers(0, 20_000, n),
                       rng.integers(-900, 900, n))
    scale, offset = (0.01, 0.02, 0.5), (100.0, 200.0, 300.0)
    src = write(tmp_path / "a.las", rec, scale=scale, offset=offset)
    manifest = tile.tile_file(src, tmp_path / "out", 100.0, chunk_points=400)
    assert len(manifest["tiles"]) >= 6
    assert sum(t["points"] for t in manifest["tiles"]) == n
    for t in manifest["tiles"]:
        header, _ = las.read_header(tmp_path / "out" / t["file"])
        pts = [r for r in rec if (int(r["x"]) - rec["x"].min()) // 10_000 == t["ix"]
               and (int(r["y"]) - rec["y"].min()) // 5_000 == t["iy"]]
        assert header.point_count == len(pts) == t["points"]
        for k in range(1, 6):
            assert header.by_return[k - 1] == sum(1 for r in pts if int(r["flags"]) & 7 == k)
        for k, name in enumerate("xyz"):
            vals = [int(r[name]) for r in pts]
            assert header.bounds[2 * k] == max(vals) * scale[k] + offset[k]
            assert header.bounds[2 * k + 1] == min(vals) * scale[k] + offset[k]
        assert header.scale == scale and header.offset == offset


def test_the_vlr_block_is_copied_into_every_tile(tmp_path):
    vlr = bytes(range(200, 256)) * 2
    rec = make_records(1, np.arange(0, 5000, 1000), np.zeros(5, dtype=int), np.zeros(5, dtype=int))
    src = write(tmp_path / "a.las", rec, vlr=vlr)
    manifest = tile.tile_file(src, tmp_path / "out", 1.0)
    assert len(manifest["tiles"]) == 5
    for t in manifest["tiles"]:
        header, got = las.read_header(tmp_path / "out" / t["file"])
        assert got == vlr and header.n_vlr == 1 and header.point_offset == 227 + len(vlr)


@pytest.mark.parametrize("size", [0.0005, 1.0004, 0.0, -1.0])
def test_a_tile_size_that_is_not_a_whole_number_of_scale_steps_is_refused(tmp_path, size):
    rec = make_records(1, [0, 5], [0, 5], [0, 5])
    src = write(tmp_path / "a.las", rec)
    with pytest.raises(ValueError, match="multiple of scale"):
        tile.tile_file(src, tmp_path / "out", size)


@pytest.mark.parametrize("size,scale,units", [(50.0, 0.001, 50_000), (0.3, 0.1, 3), (1.0, 1.0, 1), (7.0, 0.01, 700)])
def test_tile_units_accepts_sizes_that_are_whole_multiples_even_when_the_float_product_is_inexact(size, scale, units):
    assert tile.tile_units(size, scale) == units


@pytest.mark.parametrize("size,scale", [(1.0004, 0.001), (0.3001, 0.1), (1000.0000005, 0.001),
                                        (float("inf"), 0.001), (float("nan"), 0.001)])
def test_tile_units_refuses_sizes_that_are_not_whole_multiples(size, scale):
    with pytest.raises(ValueError, match="multiple of scale"):
        tile.tile_units(size, scale)


def test_a_directory_that_already_holds_tiles_is_refused(tmp_path):
    rec = make_records(1, [0, 5000], [0, 0], [0, 0])
    src = write(tmp_path / "a.las", rec)
    tile.tile_file(src, tmp_path / "out", 1.0)
    before = sorted(os.listdir(tmp_path / "out"))
    with pytest.raises(ValueError, match="already holds tiles"):
        tile.tile_file(src, tmp_path / "out", 2.0)
    assert sorted(os.listdir(tmp_path / "out")) == before


@pytest.mark.parametrize("chunk", [0, -5])
def test_a_chunk_size_below_one_is_refused(tmp_path, chunk):
    src = write(tmp_path / "a.las", make_records(1, [1, 2], [1, 2], [1, 2]))
    with pytest.raises(ValueError, match="at least 1"):
        tile.tile_file(src, tmp_path / "out", 1.0, chunk_points=chunk)


class OpenCounter:
    """Stands in for open() inside tile.py and records how many files are open at once."""

    def __init__(self):
        self.now = 0
        self.peak = 0

    def __call__(self, path, mode="r"):
        counter = self
        real = open(path, mode)
        counter.now += 1
        counter.peak = max(counter.peak, counter.now)

        class Handle:
            def __enter__(self):
                return real

            def __exit__(self, *exc):
                real.close()
                counter.now -= 1

        return Handle()


def test_only_one_tile_file_is_open_at_a_time(tmp_path, monkeypatch):
    n = 300
    rec = make_records(0, np.arange(n) * 1000, np.zeros(n, dtype=int), np.zeros(n, dtype=int),
                       seed=9)
    src = write(tmp_path / "a.las", rec, point_format=0)
    counter = OpenCounter()
    monkeypatch.setattr(tile, "open", counter, raising=False)
    manifest = tile.tile_file(src, tmp_path / "out", 1.0, chunk_points=n)
    assert len(manifest["tiles"]) == n
    assert all(t["points"] == 1 for t in manifest["tiles"])
    assert counter.peak == 1


def test_a_truncated_source_is_refused_before_anything_is_written(tmp_path):
    rec = make_records(1, [1, 2, 3], [1, 2, 3], [1, 2, 3])
    src = write(tmp_path / "a.las", rec)
    src.write_bytes(src.read_bytes()[:-5])
    with pytest.raises(ValueError, match="bytes on disk"):
        tile.tile_file(src, tmp_path / "out", 1.0)
    assert not (tmp_path / "out").exists()


def test_a_file_with_no_points_gives_a_manifest_with_no_tiles(tmp_path):
    src = write(tmp_path / "a.las", make_records(1, [], [], []))
    manifest = tile.tile_file(src, tmp_path / "out", 1.0)
    assert manifest["tiles"] == [] and manifest["origin_int"] == [None, None]
    assert os.listdir(tmp_path / "out") == ["manifest.json"]
