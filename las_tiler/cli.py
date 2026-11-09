"""Command line: python -m las_tiler synth | tile | check."""
import argparse
import sys

from . import check, synth, tile


def build_parser():
    p = argparse.ArgumentParser(prog="las_tiler")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="write a seeded synthetic scan")
    s.add_argument("out")
    s.add_argument("--points", type=int, default=1_000_000)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--extent", type=float, default=400.0, help="side of the scene in metres")
    t = sub.add_parser("tile", help="cut a LAS 1.2 file into square tiles")
    t.add_argument("src")
    t.add_argument("out_dir")
    t.add_argument("--tile-size", type=float, required=True, help="edge in metres; a whole multiple of the scale")
    t.add_argument("--chunk", type=int, default=tile.DEFAULT_CHUNK, help="points read at a time")
    c = sub.add_parser("check", help="check a directory of tiles against its source")
    c.add_argument("src")
    c.add_argument("out_dir")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "synth":
            h = synth.write_scene(args.out, args.points, args.seed, args.extent)
            print(f"wrote {h.point_count} points to {args.out}")
        elif args.command == "tile":
            m = tile.tile_file(args.src, args.out_dir, args.tile_size, args.chunk)
            print(f"wrote {len(m['tiles'])} tiles holding {sum(t['points'] for t in m['tiles'])} points to {args.out_dir}")
        else:
            report = check.check(args.src, args.out_dir)
            print(f"checked {report.tiles} tiles, {report.points} points")
            for problem in report.problems:
                print(f"PROBLEM: {problem}")
            print("OK" if report.ok else "FAILED")
            return 0 if report.ok else 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
