import os

from las_tiler import cli


def test_synth_tile_check_end_to_end(tmp_path, capsys):
    src, out = str(tmp_path / "a.las"), str(tmp_path / "tiles")
    assert cli.main(["synth", src, "--points", "20000", "--seed", "7", "--extent", "100"]) == 0
    assert cli.main(["tile", src, out, "--tile-size", "25", "--chunk", "6000"]) == 0
    assert cli.main(["check", src, out]) == 0
    text = capsys.readouterr().out
    assert "wrote 20000 points" in text
    assert "checked 16 tiles, 20000 points" in text
    assert text.rstrip().endswith("OK")


def test_check_exits_1_and_names_the_problem_for_a_damaged_tile(tmp_path, capsys):
    src, out = str(tmp_path / "a.las"), str(tmp_path / "tiles")
    cli.main(["synth", src, "--points", "5000", "--seed", "8", "--extent", "100"])
    cli.main(["tile", src, out, "--tile-size", "50"])
    victim = os.path.join(out, sorted(n for n in os.listdir(out) if n.endswith(".las"))[0])
    raw = bytearray(open(victim, "rb").read())
    raw[-1] ^= 1
    open(victim, "wb").write(bytes(raw))
    capsys.readouterr()
    assert cli.main(["check", src, out]) == 1
    text = capsys.readouterr().out
    assert "digest differs" in text and text.rstrip().endswith("FAILED")


def test_a_missing_source_exits_2_for_check_and_for_tile(tmp_path, capsys):
    missing = str(tmp_path / "nosuch.las")
    assert cli.main(["check", missing, str(tmp_path / "t")]) == 2
    assert cli.main(["tile", missing, str(tmp_path / "t"), "--tile-size", "10"]) == 2
    assert capsys.readouterr().err.count("error:") == 2


def test_an_infinite_tile_size_and_a_zero_chunk_exit_2(tmp_path, capsys):
    src = str(tmp_path / "a.las")
    cli.main(["synth", src, "--points", "100", "--seed", "1"])
    assert cli.main(["tile", src, str(tmp_path / "t1"), "--tile-size", "inf"]) == 2
    assert cli.main(["tile", src, str(tmp_path / "t2"), "--tile-size", "10", "--chunk", "0"]) == 2


def test_a_bad_tile_size_exits_2_with_a_message(tmp_path, capsys):
    src = str(tmp_path / "a.las")
    cli.main(["synth", src, "--points", "100", "--seed", "1"])
    capsys.readouterr()
    assert cli.main(["tile", src, str(tmp_path / "t"), "--tile-size", "10.0004"]) == 2
    assert "multiple of scale" in capsys.readouterr().err
