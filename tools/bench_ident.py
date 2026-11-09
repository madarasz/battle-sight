"""Benchmark the operative names on photos with a <name>.ident.json.

    uv run tools/bench_ident.py [PATH ...] [--config data/board.toml]
                                [--gallery-root data/teams] [--json FILE|-]
                                [--baseline FILE] [--write-baseline FILE] [-q]

PATH: photos or folders (default tests/images). Only photos with a
<name>.ident.json count. <name>.empty.* is the empty-board reference of
<name>.*, not a photo of its own.

Prints one row per photo, the total, and one line per wrong name, unknown
name, wrong base size and missed base.

Exit codes: 0 done, 1 worse than --baseline, 2 bad input or a baseline of
another gallery. Plan: specs/ident-benchmark.md. Docs: docs/cli.md, docs/testing.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from battlesight import ident_bench
from battlesight.config import DEFAULT_CONFIG, BoardConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTOS = ROOT / "tests" / "images"


def git_version() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return rev + ("+dirty" if dirty else "")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help=f"photos or folders (default {DEFAULT_PHOTOS.relative_to(ROOT)})")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board config")
    p.add_argument("--gallery-root", default=None, metavar="DIR", help="folder with one folder per team (default data/teams)")
    p.add_argument("--json", default=None, metavar="FILE|-", help="also write the report here (- = stdout, table to stderr)")
    p.add_argument("--baseline", default=None, metavar="FILE", help="compare with a baseline, exit 1 when worse")
    p.add_argument("--write-baseline", default=None, metavar="FILE", help="write the metrics as a new baseline")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress lines")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text_out = sys.stderr if args.json == "-" else sys.stdout

    def say(*lines: str) -> None:
        for line in lines:
            print(line, file=text_out)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    try:
        cfg = BoardConfig.load(args.config)
        baseline = None
        if args.baseline:
            try:
                baseline = json.loads(Path(args.baseline).read_text())
            except (OSError, ValueError) as exc:
                raise ValueError(f"baseline {args.baseline}: {exc}") from None
        photos = ident_bench.find_photos([Path(p) for p in args.paths] or [DEFAULT_PHOTOS])
        if not photos:
            raise ValueError("no photo with <name>.ident.json")
        scores, gallery = ident_bench.run(photos, cfg, args.gallery_root, log)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    git = git_version()
    rep = ident_bench.report(scores, ident_bench.ident_values(cfg), gallery, git)
    say(ident_bench.format_table(scores))
    details = ident_bench.format_details(scores)
    if details:
        say("", details)
    text = json.dumps(rep, indent=2)
    if args.json == "-":
        print(text)
    elif args.json:
        Path(args.json).write_text(text + "\n")

    code = 0
    if baseline is not None:
        try:
            if ident_bench.same_gallery(rep, baseline):
                cmp = ident_bench.compare(rep, baseline)
                say("", f"baseline {args.baseline} (git {baseline.get('git')}):", *cmp.lines())
                code = 0 if cmp.ok else 1
            else:
                print(
                    f"error: baseline {args.baseline} was made with another gallery ({baseline['gallery']['files']} files, now {gallery['files']}): the scores do not compare. Write the baseline again.",
                    file=sys.stderr,
                )
                code = 2
        except KeyError as exc:
            print(f"error: baseline {args.baseline} lacks {exc}", file=sys.stderr)
            return 2
    if args.write_baseline:
        base = ident_bench.report(scores, ident_bench.ident_values(cfg), gallery, git, detail=False)
        Path(args.write_baseline).write_text(json.dumps(base, indent=2) + "\n")
        say(f"wrote baseline {args.write_baseline}")
    return code


if __name__ == "__main__":
    sys.exit(main())
