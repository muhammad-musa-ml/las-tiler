"""Break one guard at a time in a copy of the code and record which test notices.

Run from the repository root: python bench/mutants.py
The repository itself is never edited; every mutant runs in a temporary copy, and the run ends
by checking that no file of the repository changed. A mutant whose anchor text is not found
exactly once is reported UNRUN and is not counted as killed; a suite that hangs is reported
HUNG and is not counted as killed either.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# (what is broken, file, text to replace, replacement)
MUTANTS = [
    ("tile cell subtraction done in int32", "las_tiler/tile.py",
     'ix = (chunk["x"].astype(np.int64) - origin[0]) // units[0]',
     'ix = (chunk["x"] - origin[0]) // units[0]'),
    ("a point on a tile edge goes to the lower tile", "las_tiler/tile.py",
     'ix = (chunk["x"].astype(np.int64) - origin[0]) // units[0]',
     'ix = (chunk["x"].astype(np.int64) - origin[0] - 1) // units[0]'),
    ("sort inside a chunk is not stable", "las_tiler/tile.py",
     'order = np.argsort(key, kind="stable")', 'order = np.argsort(key)'),
    ("VLR block not copied into tiles", "las_tiler/tile.py", 'f.write(vlr)', 'pass'),
    ("each chunk overwrites the tile file", "las_tiler/tile.py",
     'with open(path, "ab") as f:', 'with open(path, "wb") as f:'),
    ("tile size need not be a multiple of the scale", "las_tiler/tile.py",
     'if units < 1 or abs(units * scale - tile_size) > 1e-6 * scale:', 'if units < 1:'),
    ("tile size tolerance scales with the tile size, not the scale", "las_tiler/tile.py",
     'abs(units * scale - tile_size) > 1e-6 * scale:', 'abs(units * scale - tile_size) > 1e-6 * max(1.0, tile_size):'),
    ("a non-finite tile size is not turned into an error", "las_tiler/tile.py",
     'except (OverflowError, ValueError):', 'except ZeroDivisionError:'),
    ("a directory that already holds tiles is accepted", "las_tiler/tile.py",
     'if os.path.isdir(out_dir) and any(', 'if False and any('),
    ("a chunk size below one is accepted", "las_tiler/tile.py", 'if chunk_points < 1:', 'if False:'),
    ("a chunk size below one is accepted by the chunk reader", "las_tiler/las.py", 'if chunk_points < 1:', 'if False:'),
    ("OSError is not turned into exit 2", "las_tiler/cli.py",
     'except (ValueError, OSError) as e:', 'except ValueError as e:'),
    ("an extent beyond 32-bit coordinates is accepted", "las_tiler/synth.py",
     'if extent / SCALE[0] >= 2**31:', 'if False:'),
    ("the major version is not compared", "las_tiler/las.py",
     'if (v[4], v[5]) != (1, 2):', 'if v[5] != 2:'),
    ("per-return counts of a tile are not accumulated", "las_tiler/tile.py",
     'self.by_return += las.return_counts(records["flags"])', 'self.by_return += 0'),
    ("file size is not compared with the header", "las_tiler/las.py", 'if size != expected:', 'if False:'),
    ("return number read from four bits", "las_tiler/las.py",
     'numbers = (flags & 7).astype(np.int64)', 'numbers = (flags & 15).astype(np.int64)'),
    ("record length is not compared with the format", "las_tiler/las.py",
     'if h.record_length != RECORD_LENGTHS[h.point_format]:', 'if False:'),
    ("digest is a plain sum with no mixing", "las_tiler/check.py",
     'return mix(rows.astype(np.uint64) @ weights(length, seed))', 'return rows.astype(np.uint64) @ weights(length, seed)'),
    ("points outside their cell are not counted", "las_tiler/check.py",
     'wrong_cell += int(((ix != t["ix"]) | (iy != t["iy"])).sum())', 'wrong_cell += 0'),
    ("only the column of a point is compared with its tile", "las_tiler/check.py",
     'wrong_cell += int(((ix != t["ix"]) | (iy != t["iy"])).sum())', 'wrong_cell += int((ix != t["ix"]).sum())'),
    ("only the row of a point is compared with its tile", "las_tiler/check.py",
     'wrong_cell += int(((ix != t["ix"]) | (iy != t["iy"])).sum())', 'wrong_cell += int((iy != t["iy"]).sum())'),
    ("digest cut to 32 bits", "las_tiler/check.py",
     'return tuple(int(row_hashes(records, seed).sum(dtype=np.uint64)) for seed in (1, 2))',
     'return tuple(int(row_hashes(records, seed).sum(dtype=np.uint64)) & 0xFFFFFFFF for seed in (1, 2))'),
    ("tile scale, offset and VLR count are not compared with the source", "las_tiler/check.py",
     'for key in ("point_format", "scale", "offset", "n_vlr", "point_offset"):', 'for key in ():'),
    ("manifest format, scale and offset are not compared with the source", "las_tiler/check.py",
     'for key, want in (("point_format", header.point_format), ("scale", list(header.scale)), ("offset", list(header.offset))):',
     'for key, want in ():'),
    ("manifest source point count is not compared", "las_tiler/check.py",
     'if manifest.get("source_points") != header.point_count:', 'if False:'),
    ("a tile with no points is not flagged", "las_tiler/check.py", 'if th.point_count == 0:', 'if False:'),
    ("a manifest that is not an object crashes", "las_tiler/check.py",
     'if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):',
     'if not isinstance(manifest.get("tiles"), list):'),
    ("digest comparison skipped", "las_tiler/check.py", 'if tile_digest != src_digest:', 'if False:'),
    ("VLR comparison skipped", "las_tiler/check.py", 'if tvlr != vlr:', 'if False:'),
    ("header bounds comparison skipped", "las_tiler/check.py",
     'if th.bounds != acc.bounds(th.scale, th.offset):', 'if False:'),
    ("a source with points and no tiles is not flagged", "las_tiler/check.py",
     'if header.point_count and not manifest["tiles"]:', 'if False:'),
    ("tile files missing from the manifest are not flagged", "las_tiler/check.py",
     'for name in sorted(present - listed):', 'for name in []:'),
]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def tree_md5(root):
    """md5 of every file under root by relative path, leaving out git data and caches."""
    out = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache")]
        for name in files:
            path = os.path.join(folder, name)
            out[os.path.relpath(path, root)] = md5(path)
    return out


def run_tests(cwd):
    """('pass' | 'fail' | 'hung', first failing test, output)."""
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                           cwd=cwd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return "hung", "(suite stopped after 240 s)", ""
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", p.stdout, re.M)
    return ("pass" if p.returncode == 0 else "fail"), failed[0] if failed else "", p.stdout


def try_mutant(copy, rel, old, new):
    """Break one guard in the copy, run the suite, restore the file; returns (verdict, detail, md5s)."""
    path = os.path.join(copy, rel)
    with open(path, newline="") as f:
        original = f.read()
    if original.count(old) != 1:
        return "UNRUN", f"anchor occurs {original.count(old)} times in {rel}", None
    before = md5(path)
    with open(path, "w", newline="") as f:
        f.write(original.replace(old, new))
    mutated = md5(path)
    if mutated == before:
        with open(path, "w", newline="") as f:
            f.write(original)
        return "UNRUN", f"the replacement did not change {rel}", None
    status, failed, _ = run_tests(copy)
    with open(path, "w", newline="") as f:
        f.write(original)
    after = md5(path)
    if after != before:
        sys.exit(f"the copy of {rel} was not restored (md5 {before} before, {after} after)")
    verdict = {"fail": "killed", "pass": "SURVIVED", "hung": "HUNG"}[status]
    return verdict, failed, (before, mutated, after)


def main():
    rows = []
    repo_before = tree_md5(ROOT)
    with tempfile.TemporaryDirectory(prefix="las_tiler_mut_") as tmp:
        copy = os.path.join(tmp, "copy")
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "results"))
        status, _, out = run_tests(copy)
        baseline = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        print("baseline:", status, baseline, flush=True)
        if status != "pass":
            sys.exit("baseline does not pass; no mutant result would mean anything")
        for name, rel, old, new in MUTANTS:
            start = time.monotonic()
            verdict, detail, sums = try_mutant(copy, rel, old, new)
            seconds = time.monotonic() - start
            rows.append((name, rel, verdict, detail.split("::")[-1]))
            md5s = f" md5 before {sums[0]} mutated {sums[1]} restored {sums[2]}" if sums else ""
            print(f"{verdict}: {name} -> {detail} [{seconds:.0f} s]{md5s}", flush=True)
    repo_after = tree_md5(ROOT)
    if repo_after != repo_before:
        sys.exit("a file of the repository changed during the run")
    print(f"repository unchanged: {len(repo_after)} files, md5 compared before and after", flush=True)

    count = {v: sum(1 for r in rows if r[2] == v) for v in ("killed", "SURVIVED", "HUNG", "UNRUN")}
    total = f"{count['killed']} of {len(rows)} killed; {count['SURVIVED']} survived; {count['HUNG']} hung; {count['UNRUN']} not run."
    lines = ["# Mutants", "",
             "Each row breaks one guard in a temporary copy of the code and runs the whole suite with -x.",
             f"Baseline before any change: {baseline}.", "",
             "| what was broken | file | result | first failing test |", "|---|---|---|---|"]
    lines += [f"| {n} | {r} | {v} | {t} |" for n, r, v, t in rows]
    lines += ["", total, ""]
    print(total, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "mutants.md"), "w") as f:
        f.write("\n".join(lines))
    sys.exit(0 if count["killed"] == len(rows) else 1)


if __name__ == "__main__":
    main()
