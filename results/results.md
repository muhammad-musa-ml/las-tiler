# Results

Seeded synthetic scan, 5,000,000 points (140.0 MB), point format 1, 69 tiles of 50 m.
Python 3.13.5, numpy 2.3.3, Windows, 24 logical CPUs.

Tiling, 5 runs at 1,000,000 points per chunk: 6.88, 6.71, 6.48, 7.05, 7.65 s
(0.65 to 0.77 million points/s, mean 0.72).

Smallest six tiles (ix, iy, points): (3, 8, 1), (5, 8, 1), (1, 8, 2), (6, 8, 2), (8, 4, 3), (0, 2, 54236); largest 190,472 points.
Grid origin [0, 0] in scaled-integer units. Points with x or y equal to 400000 (the far edge of the 400 m square): x 3, y 6.

Check on the last run's tiles: ok=True, 69 tiles, 5,000,000 points, 10.2 s.

| points per chunk | peak traced memory (MiB) |
|---|---|
| 250,000 | 24.0 |
| 1,000,000 | 95.7 |
| 5,000,000 | 448.2 |
