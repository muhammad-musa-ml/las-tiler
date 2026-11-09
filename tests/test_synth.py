import hashlib

import numpy as np
import pytest

from las_tiler import las, synth


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_gives_the_same_bytes_and_another_seed_does_not(tmp_path):
    synth.write_scene(tmp_path / "a.las", 5_000, seed=1)
    synth.write_scene(tmp_path / "b.las", 5_000, seed=1)
    synth.write_scene(tmp_path / "c.las", 5_000, seed=2)
    assert digest(tmp_path / "a.las") == digest(tmp_path / "b.las")
    assert digest(tmp_path / "a.las") != digest(tmp_path / "c.las")


def test_header_matches_the_points_across_several_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(synth, "CHUNK", 4_000)
    path = tmp_path / "a.las"
    synth.write_scene(path, 10_500, seed=3)
    header, _ = las.read_header(path)
    records = np.concatenate(list(las.iter_chunks(path, header, 3_000)))
    count, by_return, bounds = las.summarize(records, header.scale, header.offset)
    assert (header.point_count, header.by_return, header.bounds) == (count, by_return, bounds)
    assert count == 10_500
    assert sum(by_return) == 10_500


def test_scene_has_ground_and_raised_points(tmp_path):
    path = tmp_path / "a.las"
    synth.write_scene(path, 20_000, seed=4)
    header, _ = las.read_header(path)
    rec = next(las.iter_chunks(path, header, 20_000))
    assert (rec["classification"] == 2).sum() > 10_000
    assert rec["z"].max() * header.scale[2] > 7.5


def test_an_extent_that_does_not_fit_32_bit_coordinates_is_refused(tmp_path):
    with pytest.raises(ValueError, match="32-bit"):
        synth.write_scene(tmp_path / "a.las", 10, seed=1, extent=2_200_000.0)
    assert not (tmp_path / "a.las").exists()


def test_an_empty_scene_is_a_valid_file(tmp_path):
    path = tmp_path / "a.las"
    synth.write_scene(path, 0, seed=5)
    header, _ = las.read_header(path)
    assert header.point_count == 0
