# las-tiler

I wrote a small tool that cuts a LAS 1.2 point cloud into square tiles without loading the whole file, and a checker that compares the tiles with the file they came from. I wanted to see what the data-pipeline side of a LiDAR product involves. I have never run it on a real scan: every number below comes from a seeded synthetic one.

## What it does

- `tile` reads the points in chunks, assigns each point to a tile and appends it to that tile's file. One tile file is open at a time, so the number of tiles does not run into the open-file limit. Records are copied byte for byte, with no rescaling or rounding, and the VLR block (where LAS files keep their coordinate reference system) is copied into every tile.
- Tile (i, j) holds the points with `i*T <= x - x0 < (i+1)*T`, and the same for y, in scaled-integer units. `x0` and `y0` are the lowest x and y found in the points; the header's bounds are not trusted for this. Floor division puts a point on a tile edge in the upper tile. The subtraction is done on 64-bit integers, so two valid int32 coordinates that are further apart than an int32 can hold do not wrap.
- `check` reads the source and every tile and reports a problem if a point lies outside its tile's cell, if a tile header's count, per-return counts or bounds differ from its points, if the VLR block or the point format, scale or offset differ from the source, if a tile holds no points, if a tile file is missing from the manifest or listed and absent, if the source has points and the manifest lists no tiles, or if the tiles together do not hold the same records as the source. The last comparison is a sum of hashed records, so it does not depend on order and needs no sorting.
- `synth` writes a seeded synthetic scan: sloped ground, a box-shaped building and a pipe, at UTM-sized coordinates.

It reads LAS 1.2, point formats 0 to 3. It refuses any other version, any file whose size does not match its header, and an output directory that already holds tiles. `check` exits 1 when it finds a problem; every command exits 2 when its input cannot be read or is refused.

## Run it

```
python -m pytest -q                       # 94 tests
python -m las_tiler synth scan.las --points 1000000 --seed 1
python -m las_tiler tile scan.las tiles --tile-size 50
python -m las_tiler check scan.las tiles
python bench/bench.py                     # rewrites results/results.json and results.md
python bench/mutants.py                   # writes results/mutants.md; slow, it runs the suite once per mutant
```

It needs numpy, and pytest for the tests. I ran it on Python 3.13.5 with numpy 2.3.3 on Windows only. `--tile-size` is in the file's coordinate units and must be a whole multiple of the file's scale.

## Results

From `results/results.md`, written by `bench/bench.py` on the code of the commit "Refuse unusable input, exit 2 for it, and pin more of the header and the check"; the only later change to code the benchmark runs is an argument check at the top of `iter_chunks`. The scan has 5,000,000 points (140.0 MB, point format 1) in a 400 m square, cut into 50 m tiles.

- Five timed runs at 1,000,000 points per chunk took 6.48 to 7.65 seconds (0.65 to 0.77 million points per second, mean 0.72). Each run is the scan pass plus the cut.
- The cut produced 69 tiles. Sixty-four cover the square; the other five hold 1, 1, 2, 2 and 3 points. The scan has 3 points with x and 6 points with y at 400000 scaled units, the far edge of the 400 m square, and the grid starts at 0, so those points fall in a ninth row or column (`results/results.md` lists the five tiles). The largest tile holds 190,472 points.
- `check` passed on the last run's tiles: 69 tiles and 5,000,000 points, in 10.2 seconds.
- Peak traced memory, from `tracemalloc`, was 24.0 MiB at 250,000 points per chunk, 95.7 MiB at 1,000,000 and 448.2 MiB when the whole file was read as one chunk.

`bench/mutants.py` breaks one guard at a time in a temporary copy of the code, runs the whole suite and records which test fails first. It writes `results/mutants.md`; I have not committed a complete run of it against this version of the code.

## What this does not show

- The scan is synthetic. The benchmark scan is point format 1 with return numbers 1 to 3; real scans have more noise, outliers, returns and projections than this covers.
- I did not open the tiles in laspy, PDAL or CloudCompare. The file layout follows my reading of the LAS 1.2 header table, and the tests read the header fields back from the raw bytes with `struct` at the positions in that table, not with another reader.
- The comparison of records in `check` is probabilistic: two 64-bit sums of hashed records, so a collision is possible in principle.
- The timings are five runs on one Windows machine and say nothing about other disks or larger files. Memory is what `tracemalloc` traced, which is not the process's total memory.
- It reads LAS 1.2 only. LAS 1.4, compressed LAZ files and extra bytes per record are refused.
- Tile edges are anchored at the lowest x and y of each file, so tiles from two files line up only if those lowest values differ by a whole number of tiles.
- `synth` seeds one random stream per 500,000-point chunk, so the same seed gives the same file only for the same point count, extent and chunk size.

## License

MIT, see `LICENSE`.
