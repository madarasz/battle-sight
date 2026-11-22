"""Remove duplicate gallery photos (the same frame saved more than once).

    uv run tools/dedupe_gallery.py [--gallery-root data/teams] [--dry-run]

Per operative folder, photos that `ident_bench._is_copy` calls the same frame
are duplicates: the oldest stays, the others go (with their background file).
The embeddings cache of a changed team is deleted: it is rebuilt at the next load.
Exit codes: 0 done, 2 bad input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from battlesight import ident as idt
from battlesight.ident_bench import _half, _is_copy

ROOT = Path(__file__).resolve().parents[1]


def duplicates(folder: Path) -> list[tuple[Path, Path]]:
    """(duplicate, the older photo it copies) for the photos of one operative folder."""
    kept: list[tuple[Path, object]] = []
    out: list[tuple[Path, Path]] = []
    for f in sorted(folder.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        small = _half(img)
        twin = next((k for k, t in kept if t.shape == small.shape and _is_copy(small, t)), None)  # type: ignore[attr-defined]
        if twin is None:
            kept.append((f, small))
        else:
            out.append((f, twin))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gallery-root", default=str(ROOT / "data" / "teams"))
    ap.add_argument("--dry-run", action="store_true", help="list the duplicates, delete nothing")
    args = ap.parse_args()
    root = Path(args.gallery_root)
    if not root.is_dir():
        print(f"error: {root}: not a folder", file=sys.stderr)
        return 2
    total = 0
    for team in idt.list_teams(root):
        changed = False
        for folder in sorted(p for p in (root / team).iterdir() if p.is_dir() and not p.name.startswith(".")):
            for dup, twin in duplicates(folder):
                print(f"{team}/{folder.name}: {dup.name} = {twin.name}")
                total += 1
                if not args.dry_run:
                    dup.unlink()
                    idt.background_path(dup).unlink(missing_ok=True)
                    changed = True
        if changed:
            (root / team / idt.CACHE_FILE).unlink(missing_ok=True)
    print(f"{total} duplicate photos" + (" found" if args.dry_run else " removed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
