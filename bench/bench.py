"""Time the tiler and measure its memory on a seeded synthetic scan; write results/results.json and results.md.

Run from the repository root: python bench/bench.py
"""
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from las_tiler import check, las, synth, tile  # noqa: E402

POINTS = 5_000_000
SEED = 1
EXTENT = 400.0
TILE_SIZE = 50.0
TIMED_CHUNK = 1_000_000
TIMED_RUNS = 5
MEMORY_CHUNKS = [250_000, 1_000_000, POINTS]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="las_tiler_bench_")
    try:
        src = os.path.join(work, "scan.las")
        t0 = time.perf_counter()
        header = synth.write_scene(src, POINTS, SEED, EXTENT)
        gen_seconds = time.perf_counter() - t0
        size_mb = os.path.getsize(src) / 1e6
        print(f"scan: {header.point_count} points, {size_mb:.1f} MB, generated in {gen_seconds:.1f} s", flush=True)

        runs = []
        for i in range(TIMED_RUNS):
            out = os.path.join(work, f"timed{i}")
            t0 = time.perf_counter()
            manifest = tile.tile_file(src, out, TILE_SIZE, TIMED_CHUNK)
            seconds = time.perf_counter() - t0
            runs.append(seconds)
            print(f"run {i + 1}: {seconds:.2f} s, {POINTS / seconds / 1e6:.2f} million points/s", flush=True)
            if i < TIMED_RUNS - 1:
                shutil.rmtree(out)
        last = os.path.join(work, f"timed{TIMED_RUNS - 1}")

        t0 = time.perf_counter()
        report = check.check(src, last)
        check_seconds = time.perf_counter() - t0
        print(f"check: ok={report.ok}, {report.tiles} tiles, {report.points} points, {check_seconds:.1f} s", flush=True)

        memory = []
        for chunk in MEMORY_CHUNKS:
            out = os.path.join(work, f"mem{chunk}")
            tracemalloc.start()
            tile.tile_file(src, out, TILE_SIZE, chunk)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            shutil.rmtree(out)
            memory.append({"chunk_points": chunk, "peak_traced_mib": round(peak / 2**20, 1)})
            print(f"chunk {chunk}: peak {peak / 2**20:.1f} MiB", flush=True)

        rates = [POINTS / s / 1e6 for s in runs]
        by_size = sorted(manifest["tiles"], key=lambda t: t["points"])
        far = round(EXTENT / synth.SCALE[0])
        on_far_edge = {"x": 0, "y": 0}
        for chunk in las.iter_chunks(src, header, TIMED_CHUNK):
            on_far_edge["x"] += int((chunk["x"] == far).sum())
            on_far_edge["y"] += int((chunk["y"] == far).sum())
        result = {
            "tile_points": {
                "smallest_six": [{"ix": t["ix"], "iy": t["iy"], "points": t["points"]} for t in by_size[:6]],
                "largest": by_size[-1]["points"],
                "origin_int": manifest["origin_int"],
                "far_edge_int": far,
                "points_on_far_edge": on_far_edge,
            },
            "points": POINTS, "seed": SEED, "extent_m": EXTENT, "tile_size_m": TILE_SIZE,
            "point_format": 1, "file_mb": round(size_mb, 1), "tiles": len(manifest["tiles"]),
            "timed_chunk_points": TIMED_CHUNK,
            "tile_seconds": [round(s, 2) for s in runs],
            "tile_million_points_per_second": {"min": round(min(rates), 2), "mean": round(sum(rates) / len(rates), 2), "max": round(max(rates), 2)},
            "check": {"ok": report.ok, "tiles": report.tiles, "points": report.points, "seconds": round(check_seconds, 1)},
            "memory": memory,
            "machine": {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system(), "cpus": os.cpu_count()},
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=1)
    rate = result["tile_million_points_per_second"]
    lines = [
        "# Results",
        "",
        f"Seeded synthetic scan, {POINTS:,} points ({result['file_mb']} MB), point format 1, {result['tiles']} tiles of {TILE_SIZE:g} m.",
        f"Python {result['machine']['python']}, numpy {result['machine']['numpy']}, {result['machine']['system']}, {result['machine']['cpus']} logical CPUs.",
        "",
        f"Tiling, {TIMED_RUNS} runs at {TIMED_CHUNK:,} points per chunk: {', '.join(str(s) for s in result['tile_seconds'])} s",
        f"({rate['min']} to {rate['max']} million points/s, mean {rate['mean']}).",
        "",
        "Smallest six tiles (ix, iy, points): "
        + ", ".join(f"({t['ix']}, {t['iy']}, {t['points']})" for t in result["tile_points"]["smallest_six"])
        + f"; largest {result['tile_points']['largest']:,} points.",
        f"Grid origin {result['tile_points']['origin_int']} in scaled-integer units. Points with x or y equal to {result['tile_points']['far_edge_int']} "
        f"(the far edge of the {EXTENT:g} m square): x {result['tile_points']['points_on_far_edge']['x']}, y {result['tile_points']['points_on_far_edge']['y']}.",
        "",
        f"Check on the last run's tiles: ok={report.ok}, {report.tiles} tiles, {report.points:,} points, {result['check']['seconds']} s.",
        "",
        "| points per chunk | peak traced memory (MiB) |",
        "|---|---|",
    ]
    lines += [f"| {m['chunk_points']:,} | {m['peak_traced_mib']} |" for m in memory]
    lines.append("")
    with open(os.path.join(results_dir, "results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
