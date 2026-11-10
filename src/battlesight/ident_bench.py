"""Benchmark of the operative names on photos with `<name>.ident.json`.

Scores split the losses by cause: a base that detection missed or sized wrong
is not a naming error. `compare` is the "not worse" check against a baseline.
A baseline holds only for the gallery it was made with (`gallery_state`).

Used by `tools/bench_ident.py` and `tests/test_photos.py`.
Docs: docs/cli.md, docs/testing.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import bench
from . import ident as idt
from .config import BoardConfig
from .pipeline import Pipeline

SCHEMA = 1
DONE_ACCURACY = 0.9  # plan priority 4: 9 of 10 right on the first try


class NoGallery(ValueError):
    """A team of a photo has no gallery folder or no gallery photos."""


def truth_path(photo: Path) -> Path:
    return photo.with_name(f"{photo.stem}.ident.json")


def gallery_state(root: str | Path, teams: Iterable[str]) -> dict[str, Any]:
    """Teams, file count and a content hash of the gallery folders. The
    embedding cache and hidden files do not count."""
    root = Path(root)
    teams = sorted(set(teams))
    h = hashlib.sha256()
    n = 0
    for team in teams:
        d = root / team
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            rel = f.relative_to(d)
            if f.suffix == ".npz" or any(part.startswith(".") for part in rel.parts):
                continue
            h.update(f"{team}/{rel.as_posix()}\n".encode())
            h.update(f.read_bytes())
            n += 1
    return {"teams": teams, "files": n, "fingerprint": h.hexdigest()[:16]}


# --- score of one photo ---------------------------------------------------------


@dataclass
class PhotoScore:
    name: str
    mode: str
    truth: int
    right: list[float] = field(default_factory=list)  # score of each right name
    wrong: list[tuple[str, str, bool]] = field(default_factory=list)  # (true, got, sure)
    unknown: list[str] = field(default_factory=list)  # true labels of bases named nothing
    wrong_size: list[tuple[str, float, float]] = field(default_factory=list)  # (true, found mm, true mm)
    missed: list[str] = field(default_factory=list)  # no found base within tol_mm
    ms: float = 0.0
    hidden: int = 0  # gallery photos not used: copies of a base of this photo (`copies_of`)

    @property
    def accuracy(self) -> float:
        return len(self.right) / self.truth if self.truth else 1.0

    @property
    def done_ok(self) -> bool:
        return self.accuracy >= DONE_ACCURACY

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "mode": self.mode, "metrics": metrics([self])}
        if detail:
            d["wrong"] = [{"true": t, "got": g, "sure": s} for t, g, s in self.wrong]
            d["unknown"] = list(self.unknown)
            d["wrong_size"] = [{"true": t, "found_mm": f, "true_mm": m} for t, f, m in self.wrong_size]
            d["missed"] = list(self.missed)
        return d


def score(
    name: str,
    truth: idt.IdentTruth,
    found: Sequence[tuple[Sequence[float], float]],
    idents: Sequence[idt.Ident],
    sizes: dict[str, float | None],
    mode: str = "",
    ms: float = 0.0,
) -> PhotoScore:
    """Score one photo. `found` holds (centre mm, diameter mm) of each found
    base, `sizes` the base size of each operative. A true base takes the
    nearest free found base within `tol_mm`, as in `ident.score_truth`."""
    s = PhotoScore(name, mode, len(truth.bases), ms=ms)
    centers = np.array([c for c, _ in found], float).reshape(-1, 2)
    used: set[int] = set()
    for c, label in truth.bases:
        i = -1
        if len(centers):
            dist = np.linalg.norm(centers - c, axis=1)
            dist[list(used)] = np.inf
            i = int(np.argmin(dist))
        if i < 0 or dist[i] > truth.tol_mm:
            s.missed.append(label)
            continue
        used.add(i)
        x = idents[i] if i < len(idents) else None
        got = None if x is None else x.label
        true_mm = sizes.get(label)
        if got == label:
            s.right.append(float(x.score))
        elif true_mm is not None and found[i][1] != true_mm:
            s.wrong_size.append((label, float(found[i][1]), float(true_mm)))
        elif got is None:
            s.unknown.append(label)
        else:
            s.wrong.append((label, got, bool(x.sure)))
    return s


# --- metrics -----------------------------------------------------------------


def metrics(scores: Sequence[PhotoScore]) -> dict[str, Any]:
    """Counts summed over the photos, then the ratios. `naming_accuracy` counts
    only the bases that reached the naming step: found, with the right size."""
    truth = sum(s.truth for s in scores)
    right = sum(len(s.right) for s in scores)
    wrong = sum(len(s.wrong) for s in scores)
    unknown = sum(len(s.unknown) for s in scores)
    named = right + wrong + unknown
    values = [v for s in scores for v in s.right]
    return {
        "truth": truth,
        "right": right,
        "wrong": wrong,
        "sure_wrong": sum(sure for s in scores for _, _, sure in s.wrong),
        "unknown": unknown,
        "wrong_size": sum(len(s.wrong_size) for s in scores),
        "missed": sum(len(s.missed) for s in scores),
        "accuracy": round(right / truth, 4) if truth else 1.0,
        "naming_accuracy": round(right / named, 4) if named else 1.0,
        "score_right_mean": round(float(np.mean(values)), 4) if values else None,
        "done_ok": all(s.done_ok for s in scores),
        "done": sum(s.done_ok for s in scores),
        "photos": len(scores),
        "hidden": sum(s.hidden for s in scores),
        "ms": round(sum(s.ms for s in scores), 1),
    }


def ident_values(cfg: BoardConfig) -> dict[str, Any]:
    return dataclasses.asdict(idt.IdentParams.from_config(cfg))


def report(scores: Sequence[PhotoScore], config: dict[str, Any], gallery: dict[str, Any], git: str | None = None, detail: bool = True) -> dict[str, Any]:
    """Report of a run. `detail=False` is the baseline form: no name lists."""
    return {
        "schema": SCHEMA,
        "config": config,
        "gallery": gallery,
        "git": git,
        "photos": [s.to_dict(detail) for s in scores],
        "total": {"metrics": metrics(scores)},
    }


# --- baseline ------------------------------------------------------------------


def same_gallery(run: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return run["gallery"]["fingerprint"] == baseline["gallery"]["fingerprint"]


def _shared_total(photos: dict[str, dict[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    right = sum(photos[n]["right"] for n in names)
    named = right + sum(photos[n]["wrong"] + photos[n]["unknown"] for n in names)
    return {"right": right, "naming_accuracy": round(right / named, 4) if named else 1.0, "sure_wrong": sum(photos[n]["sure_wrong"] for n in names)}


def compare(run: dict[str, Any], baseline: dict[str, Any]) -> bench.Comparison:
    """Compare a `report()` with a baseline of the same gallery. The totals
    use only the photos in both, so a new photo cannot change the result."""
    c = bench.Comparison()
    run_photos = {p["name"]: p["metrics"] for p in run["photos"]}
    base_photos = {p["name"]: p["metrics"] for p in baseline["photos"]}
    shared = [n for n in base_photos if n in run_photos]
    rt, bt = _shared_total(run_photos, shared), _shared_total(base_photos, shared)
    rules = [  # (label, higher is better)
        ("right", True),
        ("naming_accuracy", True),
        ("sure_wrong", False),
    ]
    for label, up in rules:
        r, b = rt[label], bt[label]
        if r != b:
            (c.better if (r > b) == up else c.worse).append(f"{label} {b} -> {r}")
    for name, bp in base_photos.items():
        rp = run_photos.get(name)
        if rp is None:
            c.worse.append(f"{name}: in the baseline, missing in the run")
            continue
        if rp["right"] != bp["right"]:
            (c.better if rp["right"] > bp["right"] else c.worse).append(f"{name}: right {bp['right']} -> {rp['right']}")
        if bp["done_ok"] != rp["done_ok"]:
            (c.worse if bp["done_ok"] else c.better).append(f"{name}: done_ok {bp['done_ok']} -> {rp['done_ok']}".replace("True", "true").replace("False", "false"))
    c.new = [n for n in run_photos if n not in base_photos]
    return c


# --- gallery photos cut from a test photo -----------------------------------------

COPY_SHIFT_PX = 6.0  # the base centre of the copy can be this far off, in half-size pixels (3 mm at 4 px per mm)
COPY_STEP = 32  # a pixel differs when a channel is this far off (0..255)
COPY_MAX_CHANGED = 0.003  # part of the centre pixels that differ. A copy gives under 0.001, another photo 0.2 or more.


def _half(crop: np.ndarray) -> np.ndarray:
    return cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2), interpolation=cv2.INTER_AREA)


def gallery_thumbs(gallery: idt.Gallery) -> list[np.ndarray | None]:
    """Half-size image of each gallery photo, for `copies_of`."""
    out: list[np.ndarray | None] = []
    for f in gallery.files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        out.append(None if img is None else _half(img))
    return out


def _changed(a: np.ndarray, b: np.ndarray) -> float:
    """Part of the centre pixels of `a` and `b` that differ. The border is left out: it has no data after a move."""
    m = a.shape[0] // 5
    return float((cv2.absdiff(a, b)[m:-m, m:-m].max(axis=2) > COPY_STEP).mean())


def _is_copy(a: np.ndarray, b: np.ndarray) -> bool:
    """Half-size crops `a` and `b` show the same frame. The base centre of a
    copy can be a little off (another fit of the same rim), so `b` is also
    compared after a move by the phase correlation shift."""
    if _changed(a, b) < COPY_MAX_CHANGED:
        return True
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (a, b))
    (dx, dy), _ = cv2.phaseCorrelate(gb, ga, cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F))
    if max(abs(dx), abs(dy)) > COPY_SHIFT_PX:
        return False
    # The measured shift can be half a pixel off: also try the whole pixels around it.
    shifts = {(dx, dy)} | {(float(x), float(y)) for x in (np.floor(dx), np.ceil(dx)) for y in (np.floor(dy), np.ceil(dy))}
    for sx, sy in shifts:
        moved = cv2.warpAffine(b, np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]], np.float32), (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR)
        if _changed(a, moved) < COPY_MAX_CHANGED:
            return True
    return False


def copies_of(crops: Sequence[np.ndarray], thumbs: Sequence[np.ndarray | None]) -> set[int]:
    """Indexes of the gallery photos that show the same frame as one of the
    `crops` (stored crops of the found bases of a test photo). A name fix in
    the live app on a test photo (`--image-source`) puts such a copy in the
    gallery. With it the test only finds the same image again."""
    out: set[int] = set()
    for crop in crops:
        small = _half(crop)
        for i, t in enumerate(thumbs):
            if i not in out and t is not None and t.shape == small.shape and _is_copy(small, t):
                out.add(i)
    return out


# --- run on photos ---------------------------------------------------------------


def find_photos(paths: Iterable[str | Path]) -> list[Path]:
    return [p for p in bench.find_photos(paths) if truth_path(p).is_file()]


def run(photos: Sequence[Path], cfg: BoardConfig, root: str | Path | None = None, log: Callable[[str], None] = lambda _: None) -> tuple[list[PhotoScore], dict[str, Any]]:
    """Run all stages on the photos: (scores, `gallery_state`). Same pipeline
    as the done test: nominal lens, empty-board reference when there is one.
    Gallery photos that are copies of a base of the photo (`copies_of`) are
    not used for that photo; `PhotoScore.hidden` counts them.
    NoGallery when a team has no gallery, ValueError on a bad file or photo."""
    root = Path(idt.DEFAULT_GALLERY_ROOT if root is None else root)
    truths: dict[Path, idt.IdentTruth] = {}
    for p in photos:
        try:
            truths[p] = idt.IdentTruth.load(truth_path(p))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{truth_path(p).name}: {exc!r}") from exc
    teams = {t for tr in truths.values() for t in tr.teams}
    missing = sorted(t for t in teams if not (root / t).is_dir())
    if missing:
        raise NoGallery(f"no gallery for {', '.join(missing)} in {root}")
    galleries: dict[tuple[str, ...], idt.Gallery] = {}
    thumbs: dict[tuple[str, ...], list] = {}
    scores = []
    for p, truth in truths.items():
        key = tuple(sorted(truth.teams))
        if key not in galleries:
            log(f"gallery {', '.join(key)}")
            galleries[key] = idt.Gallery.for_teams(key, cfg, root)
            thumbs[key] = gallery_thumbs(galleries[key])
        gallery = galleries[key]
        if len(gallery) == 0:
            raise NoGallery(f"gallery of {', '.join(key)} has no photos")
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot read {p}")
        ref_path = bench.reference_path(p)
        ref = None if ref_path is None else cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        log(f"run {p.name}" + (f" with {ref_path.name}" if ref_path else ""))
        pipe = Pipeline(cfg, reference=ref, gallery=gallery)
        t0 = time.perf_counter()
        r = pipe.analyze(frame, source=str(p))
        ms = (time.perf_counter() - t0) * 1000.0
        if not r.ok:
            raise ValueError(f"{p.name}: {r.summary()}")
        hidden = copies_of(r.scratch.get("base_crops", []), thumbs[key])
        if hidden:  # name the bases again, without the copies of this photo
            log(f"  {len(hidden)} gallery photos are copies of this photo: not used")
            gallery.hidden = hidden
            try:
                r = pipe.analyze(frame, source=str(p))
            finally:
                gallery.hidden = set()
        found = [(b.center_mm, b.diameter_mm) for b in r.scratch["bases"]]
        sizes = {op.label: op.base_mm for op in gallery.operatives}
        s = score(p.stem, truth, found, r.scratch.get("idents", []), sizes, r.stages["bases"].data["mode"], ms)
        s.hidden = len(hidden)
        scores.append(s)
    return scores, gallery_state(root, teams)


# --- text output -----------------------------------------------------------------

_HEAD = f"{'photo':<22}{'mode':<8}{'truth':>6}{'right':>6}{'wrong':>6}{'sure':>6}{'unk':>5}{'size':>5}{'miss':>5}{'acc':>7}{'naming':>8}{'score':>7}{'done':>6}{'ms':>7}"


def _row(name: str, mode: str, m: dict[str, Any], done: str, ms: str) -> str:
    score_txt = "-" if m["score_right_mean"] is None else f"{m['score_right_mean']:.2f}"
    return (
        f"{name:<22}{mode:<8}{m['truth']:>6}{m['right']:>6}{m['wrong']:>6}{m['sure_wrong']:>6}{m['unknown']:>5}{m['wrong_size']:>5}{m['missed']:>5}"
        f"{m['accuracy']:>7.2f}{m['naming_accuracy']:>8.2f}{score_txt:>7}{done:>6}{ms:>7}"
    )


def format_table(scores: Sequence[PhotoScore]) -> str:
    lines = [_HEAD]
    for s in scores:
        lines.append(_row(s.name, bench.MODE_SHORT.get(s.mode, s.mode), metrics([s]), "yes" if s.done_ok else "no", f"{s.ms:.0f}"))
    t = metrics(scores)
    lines.append(_row("TOTAL", "", t, f"{t['done']}/{t['photos']}", ""))
    lines.append(f"target: accuracy {DONE_ACCURACY:.2f} on each photo. sure = wrong names with no unsure mark. naming = right / (right + wrong + unk).")
    if t["hidden"]:
        lines.append(f"{t['hidden']} gallery photos not used: copies of a base of a test photo (" + ", ".join(f"{s.name} {s.hidden}" for s in scores if s.hidden) + ").")
    return "\n".join(lines)


def format_details(scores: Sequence[PhotoScore]) -> str:
    lines = []
    for s in scores:
        lines += [f"{s.name}: WRONG {t} named {g}" + (" (sure)" if sure else "") for t, g, sure in s.wrong]
        lines += [f"{s.name}: UNKNOWN {t}" for t in s.unknown]
        lines += [f"{s.name}: SIZE {t} found {f:g} mm, true {m:g} mm" for t, f, m in s.wrong_size]
        lines += [f"{s.name}: MISS {t}" for t in s.missed]
    return "\n".join(lines)
