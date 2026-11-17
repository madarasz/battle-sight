"""Benchmark the light terrain search on photos with a <name>.light.json.

    uv run tools/bench_light.py [PATH ...] [--config data/board.toml]
                                [--pieces data/terrain] [-v]

PATH: photos or folders (default tests/images). Only photos with a
<name>.light.json count (ground truth: `pieces` = key, centre and angle of
each light terrain piece on the board; `not_pieces` = board points with no
piece: bases, markers, prints, walls, bare board).

Two searches, as in the live app, with the `[light]` values of the config:

- "Locate terrain" (`lightfit.locate_at`, all piece files of the terrain):
  a click at the centre of each piece and 4 clicks 7 mm from it along the
  piece axes. Right = the true key, centre within `TOL_MM`. A click on a
  `not_pieces` point that gives a piece is a false find.
- "Identify terrains" (`lightfit.locate`, one piece file): the search
  square of the card size plus `[light] margin_mm` around the true centre
  (the live app uses the card rectangle of the layout). Each `not_pieces`
  point with each piece file: a confirmed pose there is a false find.

Prints one row per photo and the total. Exit codes: 0 done, 2 bad input.
Docs: docs/testing.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from battlesight import bench, lightfit
from battlesight import terrain as ter
from battlesight.config import DEFAULT_CONFIG, BoardConfig
from battlesight.pipeline import BASES_MARGIN_MM, Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTOS = ROOT / "tests" / "images"
TOL_MM = 5.0  # a found centre within this of the true centre is right
OFF_MM = 7.0  # the clicks off the centre, along the piece axes


def truth_file(photo: Path) -> Path:
    return photo.with_name(photo.stem + ".light.json")


@lru_cache(maxsize=2)
def top_down(photo: str, config: str) -> np.ndarray | None:
    """The top-down bases image of the photo (as the live app searches)."""
    frame = cv2.imread(photo)
    result = Pipeline(BoardConfig.load(config), "bases").analyze(frame, until="bases")
    return result.scratch.get("bases_image")


def run_job(job: tuple) -> tuple:
    """One click or one card search. Returns (job, key or confirmed, centre or None)."""
    kind, photo, config, pieces_dir, key, point = job
    cfg = BoardConfig.load(config)
    image = top_down(photo, config)
    pieces, _ = ter.load_pieces(Path(pieces_dir))
    origin = (-BASES_MARGIN_MM, -BASES_MARGIN_MM)
    if kind == "click":
        res = lightfit.locate_at(image, cfg.bases_px_per_mm, origin, {k: p.outline for k, p in pieces.items()}, point, cfg.light_min_score, cfg.light_min_margin)
        return job, res.key, None if res.pose is None else res.pose.center_mm, res.scores
    half = max(cfg.light_size(key)[:2]) / 2 + cfg.light_margin_mm
    rect = (point[0] - half, point[1] - half, point[0] + half, point[1] + half)
    pose = lightfit.locate(image, cfg.bases_px_per_mm, origin, pieces[key].outline, rect, point, 0.0, min_score=cfg.light_min_score)
    return job, pose.confirmed, pose.center_mm, {key: pose.score}


def jobs_for(photo: Path, truth: dict, config: str, pieces_dir: str, keys: list[str]) -> list[tuple]:
    out = []
    for p in truth["pieces"]:
        c, a = p["center_mm"], math.radians(p.get("angle_deg", 0.0))
        ux, uy = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
        for dx, dy in ((0, 0), (OFF_MM, 0), (-OFF_MM, 0), (0, OFF_MM), (0, -OFF_MM)):
            at = (c[0] + dx * ux[0] + dy * uy[0], c[1] + dx * ux[1] + dy * uy[1])
            out.append(("click", str(photo), config, pieces_dir, p["key"], at))
        out.append(("card", str(photo), config, pieces_dir, p["key"], tuple(c)))
    for q in truth["not_pieces"]:
        out.append(("click", str(photo), config, pieces_dir, None, tuple(q)))
        out += [("card", str(photo), config, pieces_dir, k, tuple(q)) for k in keys]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help=f"photos or folders (default {DEFAULT_PHOTOS.relative_to(ROOT)})")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board config")
    p.add_argument("--pieces", default=str(ROOT / "data" / "terrain"), metavar="DIR", help="folder with one folder of piece files per terrain")
    p.add_argument("-v", "--verbose", action="store_true", help="one line per miss, wrong key and false find")
    args = p.parse_args(argv)

    photos = [f for f in bench.find_photos(args.paths or [DEFAULT_PHOTOS]) if truth_file(f).exists()]
    if not photos:
        print("no photo with a <name>.light.json", file=sys.stderr)
        return 2
    jobs, truths = [], {}
    for photo in photos:
        try:
            truth = json.loads(truth_file(photo).read_text())
            pieces_dir = Path(args.pieces) / truth["terrain"]
            keys = sorted(ter.load_pieces(pieces_dir)[0])
        except (OSError, ValueError, KeyError) as e:
            print(f"{truth_file(photo)}: {e}", file=sys.stderr)
            return 2
        truths[str(photo)] = truth
        jobs += jobs_for(photo, truth, args.config, str(pieces_dir), keys)

    t0 = time.time()
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        results = list(ex.map(run_job, jobs, chunksize=4))
    rows = {str(f): dict(clicks=0, right=0, wrong=0, missed=0, false_clicks=0, nots=0, cards=0, found=0, false_cards=0, card_tries=0) for f in photos}
    for (kind, photo, _, _, key, point), got, center, scores in results:
        r = rows[photo]
        near = center is not None and key is not None and math.dist(center, _true_center(truths[photo], key, point)) < TOL_MM
        name = Path(photo).name
        best = " ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items()))
        if kind == "click" and key is None:
            r["nots"] += 1
            if got is not None:
                r["false_clicks"] += 1
                args.verbose and print(f"  false click  {name} ({point[0]:.0f}, {point[1]:.0f}): {got}  [{best}]")
        elif kind == "click":
            r["clicks"] += 1
            if got == key and near:
                r["right"] += 1
            elif got is None:
                r["missed"] += 1
                args.verbose and print(f"  missed click {name} {key} ({point[0]:.0f}, {point[1]:.0f})  [{best}]")
            else:
                r["wrong"] += 1
                args.verbose and print(f"  wrong click  {name} {key} ({point[0]:.0f}, {point[1]:.0f}): {got}  [{best}]")
        elif _is_piece(truths[photo], key, point):
            r["cards"] += 1
            r["found"] += bool(got) and near
            if not (got and near):
                args.verbose and print(f"  missed card  {name} {key}  [{best}]")
        else:
            r["card_tries"] += 1
            r["false_cards"] += bool(got)
            if got:
                args.verbose and print(f"  false card   {name} {key} ({point[0]:.0f}, {point[1]:.0f})  [{best}]")

    head = f"{'photo':<34} {'pieces':>6} {'clicks right':>12} {'wrong':>5} {'missed':>6} {'false':>7} {'cards found':>11} {'false':>7}"
    print(head)
    print("-" * len(head))
    total = {k: 0 for k in next(iter(rows.values()))}
    for photo, r in rows.items():
        for k, v in r.items():
            total[k] += v
        print(_row(Path(photo).name, r, len(truths[photo]["pieces"])))
    print("-" * len(head))
    print(_row("total", total, sum(len(t["pieces"]) for t in truths.values())))
    print(f"{time.time() - t0:.0f} s")
    return 0


def _row(name: str, r: dict, pieces: int) -> str:
    return f"{name:<34} {pieces:>6} {r['right']:>5} of {r['clicks']:<4} {r['wrong']:>5} {r['missed']:>6} {r['false_clicks']:>2} of {r['nots']:<2} {r['found']:>4} of {r['cards']:<4} {r['false_cards']:>2} of {r['card_tries']:<3}"


def _is_piece(truth: dict, key: str, point) -> bool:
    return any(p["key"] == key and tuple(p["center_mm"]) == tuple(point) for p in truth["pieces"])


def _true_center(truth: dict, key: str, point) -> tuple[float, float]:
    """The centre of the piece of that key nearest the point."""
    return min((tuple(p["center_mm"]) for p in truth["pieces"] if p["key"] == key), key=lambda c: math.dist(c, point), default=(math.inf, math.inf))


if __name__ == "__main__":
    sys.exit(main())
