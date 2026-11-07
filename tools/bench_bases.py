"""Benchmark the bases stage on photos with ground truth.

    uv run tools/bench_bases.py [PATH ...] [--config data/board.toml]
                                [--mode auto|single|reference]
                                [--set KEY=VALUE ...] [--sweep KEY=V1,V2,... ...]
                                [--out out/bench] [--json FILE|-]
                                [--baseline FILE] [--write-baseline FILE]
                                [--no-images] [-q]

PATH: photos or folders (default data/snapshots/tests). Only photos with a
<name>.bases.json count. <name>.empty.* is the empty-board reference of
<name>.*, not a photo of its own.

Prints one row per photo, the total, the target, and one line per miss,
wrong size and false base. Writes <name>.bench.jpg (debug image) and
bench.json to --out. KEY is a [bases] key of the board config
(diameters_mm, min_coverage, ...). In --sweep, values are split by ","; for
diameters_mm split the alternatives by ";" (diameters_mm=32,40;25,28,32,40,50).

Exit codes: 0 done, 1 worse than --baseline, 2 bad input.
Plan: specs/bases-benchmark.md. Docs: docs/cli.md, docs/testing.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from battlesight import bench, overlay
from battlesight import rectify as rect
from battlesight.camera import CameraSettings
from battlesight.config import DEFAULT_CONFIG, BoardConfig
from battlesight.pipeline import Pipeline, bases_image, changed_fraction, detect_bases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHOTOS = ROOT / "data" / "snapshots" / "tests"
BASES_FIELDS = bench.BASES_FIELDS
LIST_KEYS = {"diameters_mm"}
SLOW_KEYS = {"px_per_mm"}  # a new value needs a new bases image

ORANGE = (0, 140, 255)
GREY = (170, 170, 170)


class BadInput(Exception):
    pass


# --- options -----------------------------------------------------------------


def parse_value(key: str, text: str):
    if key not in BASES_FIELDS:
        raise BadInput(f"unknown [bases] key {key!r}, known: {', '.join(sorted(BASES_FIELDS))}")
    try:
        if key in LIST_KEYS:
            vals = tuple(sorted(float(v) for v in text.split(",") if v.strip()))
            if not vals or min(vals) <= 0:
                raise ValueError
            return vals
        return float(text)
    except ValueError:
        raise BadInput(f"bad value {text!r} for {key}") from None


def _split(item: str) -> tuple[str, str]:
    key, sep, text = item.partition("=")
    if not sep:
        raise BadInput(f"{item!r}: use KEY=VALUE")
    return key.strip(), text.strip()


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, text = _split(item)
        out[key] = parse_value(key, text)
    return out


def parse_sweep(items: list[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    for item in items:
        key, text = _split(item)
        alts = text.split(";") if key in LIST_KEYS else text.split(",")
        out[key] = [parse_value(key, a) for a in alts if a.strip()]
        if not out[key]:
            raise BadInput(f"{item!r}: no values")
    return out


def apply(cfg: BoardConfig, overrides: dict) -> BoardConfig:
    return dataclasses.replace(cfg, **{BASES_FIELDS[k]: v for k, v in overrides.items()})


def json_overrides(overrides: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()}


def git_version() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return rev + ("+dirty" if dirty else "")


# --- photos ------------------------------------------------------------------


@dataclass
class Prepared:
    """One photo after corners and rectify. The bases images are kept per
    `bases.px_per_mm`, so a sweep does not run the first stages again."""

    photo: Path
    truth: bench.GroundTruth
    frame: np.ndarray
    view: rect.BoardView
    rectified: np.ndarray  # top-down image at rectify.px_per_mm, for the debug image
    ref_frame: np.ndarray | None = None
    ref_view: rect.BoardView | None = None
    images: dict[float, tuple[np.ndarray, np.ndarray | None]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.photo.stem

    def bases_images(self, cfg: BoardConfig) -> tuple[np.ndarray, np.ndarray | None]:
        key = cfg.bases_px_per_mm
        if key not in self.images:
            ref = None if self.ref_frame is None else bases_image(self.ref_frame, self.ref_view, cfg)
            self.images[key] = (bases_image(self.frame, self.view, cfg), ref)
        return self.images[key]


def _rectify(path: Path, cfg: BoardConfig, what: str):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)  # applies EXIF orientation
    if frame is None:
        raise BadInput(f"cannot read {path}")
    r = Pipeline(cfg, "rectify", CameraSettings()).analyze(frame, source=str(path))
    if not r.ok:
        raise BadInput(f"{what} {path.name}: {r.summary()}")
    return frame, r.scratch["view"], r.scratch["rectified"]


def prepare(paths: list[Path], cfg: BoardConfig, mode: str, log) -> list[Prepared]:
    photos = [p for p in bench.find_photos(paths) if bench.truth_path(p).is_file()]
    out = []
    for photo in photos:
        ref = None if mode == "single" else bench.reference_path(photo)
        if mode == "reference" and ref is None:
            continue
        try:
            truth = bench.load_truth(bench.truth_path(photo))
        except (ValueError, OSError) as exc:
            raise BadInput(str(exc)) from None
        log(f"prepare {photo.name}" + (f" with {ref.name}" if ref else ""))
        frame, view, rectified = _rectify(photo, cfg, "photo")
        p = Prepared(photo, truth, frame, view, rectified)
        if ref is not None:
            p.ref_frame, p.ref_view, _ = _rectify(ref, cfg, "reference")
        out.append(p)
    if not out:
        raise BadInput("no photo with ground truth (<name>.bases.json)" + (" and a reference" if mode == "reference" else ""))
    return out


def run(prepared: list[Prepared], cfg: BoardConfig) -> list[bench.PhotoScore]:
    scores = []
    for p in prepared:
        img, ref = p.bases_images(cfg)
        t0 = time.perf_counter()
        found, fg = detect_bases(img, cfg, ref)
        ms = (time.perf_counter() - t0) * 1000.0
        mode = "single_frame" if ref is None else "reference"
        scores.append(bench.score(p.name, p.truth, [c.to_dict() for c in found], mode, changed_fraction(fg, cfg), ms))
    return scores


# --- debug image -------------------------------------------------------------


def _label(img: np.ndarray, text: str, org: tuple[int, int], color, s: float) -> None:
    thick = max(1, round(1.5 * s))
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, thick)
    org = (int(org[0] - tw / 2), int(org[1]))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, overlay.BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, thick, cv2.LINE_AA)


def draw_debug(p: Prepared, s: bench.PhotoScore, cfg: BoardConfig) -> np.ndarray:
    """Top-down image with matches, wrong sizes, false bases, misses, decoys."""
    img = p.rectified.copy()
    k = overlay._scale(img)
    thick = max(1, round(2 * k))
    ppm = cfg.rectify_px_per_mm

    def px(c) -> tuple[int, int]:
        q = rect.mm_to_rect(cfg, np.asarray(c, dtype=np.float64).reshape(1, 2))[0]
        return int(round(q[0])), int(round(q[1]))

    def above(c, d) -> tuple[int, int]:
        x, y = px(c)
        return x, int(y - (0.5 * d + 2.0) * ppm)

    def below(c, d) -> tuple[int, int]:
        x, y = px(c)
        return x, int(y + (0.5 * d + 6.0) * ppm)

    for dc in s.decoys:
        overlay.draw_dashed_circle(img, px(dc.center_mm), 0.5 * dc.diameter_mm * ppm, GREY, 1, dashes=48, duty=0.3)
    for t in s.missed:
        d = 30.0 if t.diameter_mm is None else t.diameter_mm
        overlay.draw_dashed_circle(img, px(t.center_mm), 0.5 * d * ppm, overlay.YELLOW, thick)
        _label(img, f"miss {t.note}"[:40], below(t.center_mm, d), overlay.YELLOW, k)
    for m in s.matches:
        f = m.found
        d = float(f["diameter_mm"])
        color = ORANGE if m.size_ok is False else overlay.GREEN
        cv2.circle(img, px(f["center_mm"]), int(round(0.5 * d * ppm)), color, thick, cv2.LINE_AA)
        if m.size_ok is False:
            cv2.line(img, px(f["center_mm"]), px(m.truth.center_mm), color, 1, cv2.LINE_AA)
            _label(img, f"{d:g}/{m.truth.diameter_mm:g}", above(f["center_mm"], d), color, k)
        else:
            _label(img, f"{d:g}", above(f["center_mm"], d), color, k)
    for f, kind in s.false:
        d = float(f["diameter_mm"])
        cv2.circle(img, px(f["center_mm"]), int(round(0.5 * d * ppm)), overlay.RED, thick, cv2.LINE_AA)
        _label(img, f"{d:g} {kind} {f.get('coverage', 0):.2f}", above(f["center_mm"], d), overlay.RED, k)

    legend = [
        (overlay.GREEN, "match, right size"),
        (ORANGE, "match, wrong size (found/true)"),
        (overlay.RED, "false base (kind, coverage)"),
        (overlay.YELLOW, "missed base (dashed)"),
        (GREY, "not a base (dotted)"),
    ]
    lh = int(22 * k)
    x0, y0 = int(10 * k), img.shape[0] - lh * len(legend) - int(10 * k)
    cv2.rectangle(img, (0, y0 - lh), (int(330 * k), img.shape[0]), overlay.BLACK, -1)
    for i, (color, text) in enumerate(legend):
        y = y0 + i * lh
        cv2.circle(img, (x0 + int(8 * k), y - int(5 * k)), int(6 * k), color, thick, cv2.LINE_AA)
        cv2.putText(img, text, (x0 + int(24 * k), y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, overlay.WHITE, 1, cv2.LINE_AA)
    overlay.draw_text_banner(img, bench.format_total_line(s), overlay.GREEN if s.done_ok else overlay.RED)
    return img


# --- sweep -------------------------------------------------------------------


def _key(t: dict) -> tuple:
    m = t["metrics"]
    return (-m["f1"], m["false"])


def sweep_table(rows: list[dict], holdout: list[str]) -> str:
    head = f"{'overrides':<44}{'truth':>6}{'found':>6}{'match':>6}{'false':>6}{'wrong':>6}{'recall':>8}{'prec':>6}{'f1':>6}{'err_max':>8}{'done':>6}"
    if holdout:
        head += f"{'hold_f1':>8}"
    lines = [head]
    for r in rows:
        m = r["tuned"]["metrics"]
        err = m["center_err_mm"] or {}
        ov = " ".join(f"{k}={','.join(f'{x:g}' for x in v) if isinstance(v, list) else f'{v:g}'}" for k, v in r["overrides"].items())
        line = (
            f"{ov:<44}{m['truth']:>6}{m['found']:>6}{m['matched']:>6}{m['false']:>6}{m['wrong_size']:>6}"
            f"{m['recall']:>8.2f}{m['precision']:>6.2f}{m['f1']:>6.2f}{bench._f(err.get('max')):>8}{m['done']:>3}/{m['photos']:<2}"
        )
        if holdout:
            line += f"{r['holdout']['metrics']['f1']:>8.2f}" if r.get("holdout") else f"{'-':>8}"
        lines.append(line)
    return "\n".join(lines)


# --- main --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help=f"photos or folders (default {DEFAULT_PHOTOS.relative_to(ROOT)})")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board config")
    p.add_argument("--mode", choices=("auto", "single", "reference"), default="auto", help="auto: reference mode when <name>.empty.* exists")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a [bases] key")
    p.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2", help="run all combinations of these [bases] values")
    p.add_argument("--out", default="out/bench", help="folder for <name>.bench.jpg and bench.json")
    p.add_argument("--json", default=None, metavar="FILE|-", help="also write bench.json here (- = stdout, table to stderr)")
    p.add_argument("--baseline", default=None, metavar="FILE", help="compare with a baseline, exit 1 when worse")
    p.add_argument("--write-baseline", default=None, metavar="FILE", help="write the metrics as a new baseline")
    p.add_argument("--no-images", action="store_true", help="do not write debug images")
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
        overrides = parse_set(args.set)
        grid = parse_sweep(args.sweep)
        baseline = None
        if args.baseline:
            try:
                baseline = json.loads(Path(args.baseline).read_text())
            except (OSError, ValueError) as exc:
                raise BadInput(f"baseline {args.baseline}: {exc}") from None
        if SLOW_KEYS & set(grid):
            log("warning: a px_per_mm sweep makes a new bases image for each value (slow)")
        prepared = prepare([Path(p) for p in args.paths] or [DEFAULT_PHOTOS], cfg, args.mode, log)
    except (BadInput, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    holdout = list((baseline or {}).get("holdout", []))
    if args.write_baseline and Path(args.write_baseline).is_file() and not holdout:
        try:
            holdout = list(json.loads(Path(args.write_baseline).read_text()).get("holdout", []))
        except ValueError:
            pass

    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())] if grid else [{}]
    rows = []
    for combo in combos:
        ov = {**overrides, **combo}
        run_cfg = apply(cfg, ov)
        if len(combos) > 1:
            log(f"run {json_overrides(ov)}")
        scores = run(prepared, run_cfg)
        tuned = [s for s in scores if s.name not in holdout]
        held = [s for s in scores if s.name in holdout]
        rows.append(
            {
                "overrides": json_overrides(ov),
                "cfg": run_cfg,
                "scores": scores,
                "total": bench.total(scores),
                "tuned": bench.total(tuned) if tuned else bench.total(scores),
                "holdout": bench.total(held) if held else None,
            }
        )
    rows.sort(key=lambda r: _key(r["tuned"]))
    best = rows[0]
    scores, run_cfg = best["scores"], best["cfg"]

    rep = bench.report(scores, bench.bases_values(run_cfg), best["overrides"], git_version())
    if holdout:
        rep["holdout"] = holdout
    if len(rows) > 1:
        rep["sweep"] = [{"overrides": r["overrides"], "total": r["total"], "holdout": r["holdout"]} for r in rows]
        say("sweep, best first" + (" (ranked without the holdout photos)" if holdout else "") + ":", sweep_table(rows, holdout), "", f"best: {best['overrides']}")
    say(bench.format_table(scores))
    details = bench.format_details(scores)
    if details:
        say("", details)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rep, indent=2)
    (out_dir / "bench.json").write_text(text + "\n")
    if args.json == "-":
        print(text)
    elif args.json:
        Path(args.json).write_text(text + "\n")
    if not args.no_images:
        by_name = {p.name: p for p in prepared}
        for s in scores:
            cv2.imwrite(str(out_dir / f"{s.name}.bench.jpg"), draw_debug(by_name[s.name], s, run_cfg), [cv2.IMWRITE_JPEG_QUALITY, 90])
    log(f"wrote {out_dir}/bench.json" + ("" if args.no_images else f" and {len(scores)} debug images"))

    code = 0
    if baseline is not None:
        try:
            cmp = bench.compare(rep, baseline)
        except KeyError as exc:
            print(f"error: baseline {args.baseline} lacks {exc}", file=sys.stderr)
            return 2
        say("", f"baseline {args.baseline} (git {baseline.get('git')}):", *cmp.lines())
        code = 0 if cmp.ok else 1
    if args.write_baseline:
        base = bench.report(scores, bench.bases_values(run_cfg), best["overrides"], rep["git"], detail=False)
        if holdout:
            base["holdout"] = holdout
        Path(args.write_baseline).write_text(json.dumps(base, indent=2) + "\n")
        say(f"wrote baseline {args.write_baseline}")
    return code


if __name__ == "__main__":
    sys.exit(main())
