"""Bases benchmark: ground truth, matching, metrics, baseline compare.

Plan: `specs/bases-benchmark.md`. Pure functions, no window. The photo tests
(`tests/test_photos.py`), `tools/bench_bases.py` and `tools/annotate_bases.py`
use the same code.

Ground truth lives next to each test photo in `<name>.bases.json` (schema 2,
schema 1 files still load). A detection is one entry of the bases stage
`data["bases"]` (a `Circle.to_dict()`): only `center_mm`, `diameter_mm`
and `coverage` are used here.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import BoardConfig

TRUTH_SCHEMA = 2
BENCH_SCHEMA = 1
BASE_CENTER_TOL_MM = 2.0  # plan priority 3: centre error under 2 mm
ESTIMATE_TOL_MM = 8.0  # match radius of an estimated (not clicked) base
UNKNOWN_SIZE_INSIDE_MM = 20.0  # "inside a true base" radius when its size is unknown
CENTER_ERR_SLACK_MM = 0.5  # baseline rule: largest clear-rim centre error may grow this much

RIMS = ("clear", "hidden")
SOURCES = ("clicked", "detector", "estimate")
KINDS = ("teleporter", "token", "print", "barricade", "wall", "model", "other")

SUFFIXES = (".jpg", ".jpeg", ".png")
REFERENCE_TAG = ".empty"  # <name>.empty.jpg is the empty-board reference of <name>.jpg

TARGET = {"missed": 0, "false": 0, "recall": 1.0, "precision": 1.0, "f1": 1.0, "err_max": BASE_CENTER_TOL_MM}


# --- photo files -------------------------------------------------------------


def is_reference(path: Path) -> bool:
    """True for `<name>.empty.<ext>`: an empty-board reference, not a photo of its own."""
    return Path(path).stem.endswith(REFERENCE_TAG)


def is_photo(path: Path) -> bool:
    path = Path(path)
    return path.suffix.lower() in SUFFIXES and not is_reference(path)


def find_photos(paths: Iterable[str | Path]) -> list[Path]:
    """Photos in the given files and folders (not recursive), sorted, no references."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += [q for q in p.iterdir() if is_photo(q)]
        elif is_photo(p):
            out.append(p)
    return sorted(set(out))


def reference_path(photo: Path) -> Path | None:
    """`<name>.empty.<ext>` next to the photo, or None."""
    for p in sorted(photo.parent.glob(f"{photo.stem}{REFERENCE_TAG}.*")):
        if p.suffix.lower() in SUFFIXES:
            return p
    return None


def truth_path(photo: Path) -> Path:
    return photo.with_name(f"{photo.stem}.bases.json")


# --- ground truth ------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    """One true base (`bases[]`)."""

    center_mm: tuple[float, float]
    diameter_mm: float | None  # None: size not known, any class matches
    rim: str  # "clear" | "hidden"
    tol_mm: float = BASE_CENTER_TOL_MM
    source: str = "clicked"
    note: str = ""
    points_mm: tuple[tuple[float, float], ...] | None = None  # clicked rim points
    label: str = ""  # operative, edit state of the annotate tool (saved in <name>.ident.json, not here)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "center_mm": [round(float(v), 2) for v in self.center_mm],
            "diameter_mm": _num(self.diameter_mm),
            "rim": self.rim,
        }
        if self.tol_mm != BASE_CENTER_TOL_MM:
            d["tol_mm"] = _num(self.tol_mm)
        d["source"] = self.source
        d["note"] = self.note
        if self.points_mm:
            d["points_mm"] = [[round(float(x), 2), round(float(y), 2)] for x, y in self.points_mm]
        return d


@dataclass(frozen=True)
class Decoy:
    """One known round thing that is not a base (`not_bases[]`)."""

    center_mm: tuple[float, float]
    diameter_mm: float
    kind: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_mm": [round(float(v), 2) for v in self.center_mm],
            "diameter_mm": _num(self.diameter_mm),
            "kind": self.kind,
            "note": self.note,
        }


@dataclass
class GroundTruth:
    bases: list[Truth] = field(default_factory=list)
    not_bases: list[Decoy] = field(default_factory=list)
    comment: str = ""
    roster_mm: tuple[float, ...] | None = None  # base sizes of the teams in the game, None = not known

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": TRUTH_SCHEMA}
        if self.comment:
            out["comment"] = self.comment
        if self.roster_mm:
            out["roster_mm"] = [_num(v) for v in self.roster_mm]
        out["bases"] = [t.to_dict() for t in self.bases]
        out["not_bases"] = [d.to_dict() for d in self.not_bases]
        return out


def _num(v: float | None) -> float | int | None:
    """JSON number: 40 stays 40, not 40.0."""
    if v is None:
        return None
    v = float(v)
    return int(v) if v.is_integer() else round(v, 2)


def _pair(v: Any, what: str) -> tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"{what}: center_mm must be [x, y]")
    return float(v[0]), float(v[1])


def parse_truth(raw: dict[str, Any]) -> GroundTruth:
    """Ground truth from a parsed `<name>.bases.json`. A file without `schema`
    is version 1: a clear rim gets `source: "detector"` (a checked detector
    fit), a hidden rim `source: "estimate"` (a guess), no `not_bases`."""
    schema = int(raw.get("schema", 1))
    if schema not in (1, 2):
        raise ValueError(f"unknown ground truth schema {schema}")
    bases = []
    for i, b in enumerate(raw.get("bases", [])):
        what = f"bases[{i}]"
        rim = b.get("rim")
        if rim not in RIMS:
            raise ValueError(f"{what}: rim {rim!r}, must be one of {RIMS}")
        v1_source = "detector" if rim == "clear" else "estimate"
        source = b.get("source", "detector") if schema == 2 else v1_source
        if source not in SOURCES:
            raise ValueError(f"{what}: source {source!r}, must be one of {SOURCES}")
        d = b.get("diameter_mm")
        pts = b.get("points_mm")
        bases.append(
            Truth(
                center_mm=_pair(b.get("center_mm"), what),
                diameter_mm=None if d is None else float(d),
                rim=rim,
                tol_mm=float(b.get("tol_mm", BASE_CENTER_TOL_MM)),
                source=source,
                note=str(b.get("note", "")),
                points_mm=None if not pts else tuple(_pair(p, what) for p in pts),
            )
        )
    decoys = []
    for i, n in enumerate(raw.get("not_bases", []) if schema == 2 else []):
        what = f"not_bases[{i}]"
        kind = n.get("kind")
        if kind not in KINDS:
            raise ValueError(f"{what}: kind {kind!r}, must be one of {KINDS}")
        if n.get("diameter_mm") is None:
            raise ValueError(f"{what}: diameter_mm is required")
        decoys.append(Decoy(_pair(n.get("center_mm"), what), float(n["diameter_mm"]), kind, str(n.get("note", ""))))
    roster = raw.get("roster_mm")
    if roster is not None and (not isinstance(roster, list) or not roster or any(not isinstance(v, (int, float)) or v <= 0 for v in roster)):
        raise ValueError("roster_mm must be a list of base diameters in mm")
    return GroundTruth(bases, decoys, str(raw.get("comment", "")), None if roster is None else tuple(sorted(float(v) for v in roster)))


def load_truth(path: str | Path) -> GroundTruth:
    path = Path(path)
    try:
        return parse_truth(json.loads(path.read_text()))
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def dump_truth(gt: GroundTruth) -> str:
    """JSON text with one base per line, so a diff shows which base changed."""
    d = gt.to_dict()
    lines = ["{", f'  "schema": {d["schema"]},']
    if "comment" in d:
        lines.append(f'  "comment": {json.dumps(d["comment"])},')
    if "roster_mm" in d:
        lines.append(f'  "roster_mm": {json.dumps(d["roster_mm"])},')
    for key in ("bases", "not_bases"):
        items = d[key]
        end = "," if key == "bases" else ""
        if not items:
            lines.append(f'  "{key}": []{end}')
            continue
        lines.append(f'  "{key}": [')
        lines += [f"    {json.dumps(it)}" + ("," if i < len(items) - 1 else "") for i, it in enumerate(items)]
        lines.append(f"  ]{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def save_truth(path: str | Path, gt: GroundTruth) -> None:
    Path(path).write_text(dump_truth(gt))


# --- matching ----------------------------------------------------------------


@dataclass
class Match:
    truth: Truth
    found: dict
    error_mm: float  # centre distance
    size_ok: bool | None  # None when truth.diameter_mm is None

    def to_dict(self) -> dict[str, Any]:
        return {"truth": self.truth.to_dict(), "found": self.found, "error_mm": round(self.error_mm, 3), "size_ok": self.size_ok}


def _dist(a, b) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def false_kind(det: dict, truth: Sequence[Truth], decoys: Sequence[Decoy] = ()) -> str:
    """Category of a false detection: the kind of the nearest decoy within
    half its diameter, else `inside_base` when the centre lies inside a true
    base, else `other`."""
    c = det["center_mm"]
    near = [(_dist(c, d.center_mm), d.kind) for d in decoys if _dist(c, d.center_mm) <= 0.5 * d.diameter_mm]
    if near:
        return min(near)[1]
    for t in truth:
        r = UNKNOWN_SIZE_INSIDE_MM if t.diameter_mm is None else 0.5 * t.diameter_mm
        if _dist(c, t.center_mm) < r:
            return "inside_base"
    return "other"


def match(truth: Sequence[Truth], found: Sequence[dict], decoys: Sequence[Decoy] = ()) -> tuple[list[Match], list[Truth], list[tuple[dict, str]]]:
    """One-to-one global greedy match of true bases and detections.

    All pairs within `truth.tol_mm`, nearest first (ties: truth order). A pair
    whose truth or detection is taken is skipped. Returns (matches in truth
    order, missed truths, false detections with their category)."""
    pairs = []
    for i, t in enumerate(truth):
        for j, f in enumerate(found):
            d = _dist(t.center_mm, f["center_mm"])
            if d <= t.tol_mm:
                pairs.append((d, i, j))
    pairs.sort()
    t_used: dict[int, tuple[int, float]] = {}
    f_used: set[int] = set()
    for d, i, j in pairs:
        if i in t_used or j in f_used:
            continue
        t_used[i] = (j, d)
        f_used.add(j)
    matches = []
    for i in sorted(t_used):
        j, d = t_used[i]
        t, f = truth[i], found[j]
        size_ok = None if t.diameter_mm is None else abs(float(f["diameter_mm"]) - t.diameter_mm) < 1e-6
        matches.append(Match(t, f, d, size_ok))
    missed = [t for i, t in enumerate(truth) if i not in t_used]
    false = [(f, false_kind(f, truth, decoys)) for j, f in enumerate(found) if j not in f_used]
    return matches, missed, false


# --- metrics -----------------------------------------------------------------


@dataclass
class PhotoScore:
    name: str
    mode: str  # "single_frame" | "reference"
    changed_fraction: float | None
    matches: list[Match]
    missed: list[Truth]
    false: list[tuple[dict, str]]  # (detection, category)
    ms: float  # bases stage run time
    decoys: list[Decoy] = field(default_factory=list)

    @property
    def done_ok(self) -> bool:
        """Priority 3 done test for this photo: nothing missed, no wrong size,
        no false base, every clear-rim match within 2 mm."""
        return (
            not self.missed
            and not self.false
            and all(m.size_ok is not False for m in self.matches)
            and all(m.error_mm <= BASE_CENTER_TOL_MM for m in self.matches if m.truth.rim == "clear")
        )

    def metrics(self, rim: str | None = None) -> dict[str, Any]:
        return metrics([self], rim)

    def to_dict(self, detail: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "mode": self.mode,
            "changed_fraction": self.changed_fraction,
            "metrics": self.metrics(),
            "by_rim": {rim: self.metrics(rim) for rim in RIMS},
        }
        if detail:
            d["matches"] = [m.to_dict() for m in self.matches]
            d["missed"] = [t.to_dict() for t in self.missed]
            d["false"] = [{"found": f, "kind": k} for f, k in self.false]
        return d


def score(name: str, gt: GroundTruth, found: Sequence[dict], mode: str, changed_fraction: float | None = None, ms: float = 0.0) -> PhotoScore:
    matches, missed, false = match(gt.bases, found, gt.not_bases)
    return PhotoScore(name, mode, changed_fraction, matches, missed, false, ms, list(gt.not_bases))


def _ratio(a: int, b: int) -> float:
    return 1.0 if b == 0 else a / b


def _err_stats(errs: list[float]) -> dict[str, float] | None:
    if not errs:
        return None
    a = np.asarray(errs)
    return {"median": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3), "max": round(float(a.max()), 3)}


def metrics(scores: Sequence[PhotoScore], rim: str | None = None) -> dict[str, Any]:
    """Counts summed over the photos, then the ratios (section 5.3 of the plan).
    With `rim` only the true bases with that rim count; detections that match
    no base have no rim, so `found`, `false` and `precision` are left out."""
    matches = [m for s in scores for m in s.matches if rim is None or m.truth.rim == rim]
    missed = [t for s in scores for t in s.missed if rim is None or t.rim == rim]
    n_truth, n_match = len(matches) + len(missed), len(matches)
    size_ok = sum(m.size_ok is True for m in matches)
    wrong = sum(m.size_ok is False for m in matches)
    errs = [m.error_mm for m in matches if m.truth.rim == "clear" and m.truth.source in ("clicked", "detector")]
    recall = _ratio(n_match, n_truth)
    out: dict[str, Any] = {"truth": n_truth}
    if rim is None:
        false = [k for s in scores for _, k in s.false]
        n_found = n_match + len(false)
        precision = _ratio(n_match, n_found)
        out["found"] = n_found
    out |= {"matched": n_match, "size_ok": size_ok, "wrong_size": wrong, "missed": len(missed)}
    if rim is None:
        out |= {"false": len(false), "false_by_kind": dict(sorted(Counter(false).items()))}
    out["recall"] = round(recall, 4)
    if rim is None:
        out["precision"] = round(precision, 4)
        out["f1"] = round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 4)
    out["size_accuracy"] = round(_ratio(size_ok, size_ok + wrong), 4)
    out["center_err_mm"] = _err_stats(errs)
    if rim is None:
        out["done_ok"] = all(s.done_ok for s in scores)
        out["done"] = sum(s.done_ok for s in scores)
        out["photos"] = len(scores)
        out["ms"] = round(sum(s.ms for s in scores), 1)
    return out


def total(scores: Sequence[PhotoScore]) -> dict[str, Any]:
    m = metrics(scores)
    modes = sorted({s.mode for s in scores})
    return {
        "metrics": m,
        "by_rim": {rim: metrics(scores, rim) for rim in RIMS},
        "by_mode": {mode: metrics([s for s in scores if s.mode == mode]) for mode in modes},
        "false_by_kind": m["false_by_kind"],
    }


def report(
    scores: Sequence[PhotoScore],
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    git: str | None = None,
    detail: bool = True,
) -> dict[str, Any]:
    """`bench.json` content (section 6.4). `detail=False` gives the baseline
    form: no match, miss and false lists."""
    return {
        "schema": BENCH_SCHEMA,
        "config": config,
        "overrides": overrides or {},
        "git": git,
        "photos": [s.to_dict(detail) for s in scores],
        "total": total(scores),
    }


# [bases] config key -> BoardConfig field, for reports and --set.
BASES_FIELDS = {f.name[len("bases_") :]: f.name for f in dataclasses.fields(BoardConfig) if f.name.startswith("bases_")}


def bases_values(cfg: BoardConfig) -> dict[str, Any]:
    """All [bases] values of the config, JSON-safe, for the report."""
    out = {}
    for key, name in sorted(BASES_FIELDS.items()):
        v = getattr(cfg, name)
        out[key] = list(v) if isinstance(v, tuple) else v
    return out


# --- baseline ----------------------------------------------------------------


@dataclass
class Comparison:
    worse: list[str] = field(default_factory=list)
    better: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)  # photos not in the baseline

    @property
    def ok(self) -> bool:
        return not self.worse

    def lines(self) -> list[str]:
        out = [f"new photo, not in baseline: {n}" for n in self.new]
        out += [f"WORSE: {w}" for w in self.worse]
        out += [f"better: {b}" for b in self.better]
        if self.worse:
            out.append("worse than baseline")
        elif self.better:
            out.append("better than baseline: run with --write-baseline to keep it")
        else:
            out.append("same as baseline")
        return out


def compare(run: dict[str, Any], baseline: dict[str, Any]) -> Comparison:
    """Compare a `report()` with a baseline (section 7.2 of the plan)."""
    c = Comparison()
    rt, bt = run["total"], baseline["total"]
    rules = [  # (label, run value, baseline value, higher is better)
        ("matched", rt["metrics"]["matched"], bt["metrics"]["matched"], True),
        ("matched clear rims", rt["by_rim"]["clear"]["matched"], bt["by_rim"]["clear"]["matched"], True),
        ("false", rt["metrics"]["false"], bt["metrics"]["false"], False),
        ("wrong_size", rt["metrics"]["wrong_size"], bt["metrics"]["wrong_size"], False),
    ]
    for label, r, b, up in rules:
        if r == b:
            continue
        (c.better if (r > b) == up else c.worse).append(f"{label} {b} -> {r}")
    re_, be = rt["metrics"]["center_err_mm"], bt["metrics"]["center_err_mm"]
    if re_ is not None and be is not None:
        if re_["max"] > be["max"] + CENTER_ERR_SLACK_MM:
            c.worse.append(f"center_err_mm.max {be['max']:.2f} -> {re_['max']:.2f} (limit +{CENTER_ERR_SLACK_MM} mm)")
        elif re_["max"] < be["max"] - 0.1:
            c.better.append(f"center_err_mm.max {be['max']:.2f} -> {re_['max']:.2f}")
    run_photos = {p["name"]: p for p in run["photos"]}
    base_photos = {p["name"]: p for p in baseline["photos"]}
    for name, bp in base_photos.items():
        rp = run_photos.get(name)
        if rp is None:
            c.worse.append(f"{name}: in the baseline, missing in the run")
            continue
        for label, up in (("matched", True), ("false", False)):  # a gain on one photo must not hide a loss on another
            r, b = rp["metrics"][label], bp["metrics"][label]
            if r != b:
                (c.better if (r > b) == up else c.worse).append(f"{name}: {label} {b} -> {r}")
        if bp["metrics"]["done_ok"] and not rp["metrics"]["done_ok"]:
            c.worse.append(f"{name}: done_ok true -> false")
        elif rp["metrics"]["done_ok"] and not bp["metrics"]["done_ok"]:
            c.better.append(f"{name}: done_ok false -> true")
    c.new = [n for n in run_photos if n not in base_photos]
    return c


# --- text output -------------------------------------------------------------

MODE_SHORT = {"single_frame": "single", "reference": "ref"}
_HEAD = (
    f"{'photo':<22}{'mode':<8}{'truth':>5}{'found':>6}{'match':>6}{'size_ok':>8}{'miss':>5}{'false':>6}"
    f"{'recall':>8}{'prec':>6}{'f1':>6}{'err_med':>9}{'err_max':>8}{'done':>6}{'ms':>6}"
)


def _f(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _row(name: str, mode: str, m: dict[str, Any], done: str, ms: str) -> str:
    err = m["center_err_mm"] or {}
    return (
        f"{name:<22}{mode:<8}{m['truth']:>5}{m['found']:>6}{m['matched']:>6}{m['size_ok']:>8}{m['missed']:>5}{m['false']:>6}"
        f"{m['recall']:>8.2f}{m['precision']:>6.2f}{m['f1']:>6.2f}{_f(err.get('median')):>9}{_f(err.get('max')):>8}{done:>6}{ms:>6}"
    )


def format_table(scores: Sequence[PhotoScore]) -> str:
    """One row per photo, the total, the target, false bases by kind, recall by rim."""
    lines = [_HEAD]
    for s in scores:
        lines.append(_row(s.name, MODE_SHORT.get(s.mode, s.mode), s.metrics(), "yes" if s.done_ok else "no", f"{s.ms:.0f}"))
    t = total(scores)
    m = t["metrics"]
    lines.append(_row("TOTAL", "", m, f"{m['done']}/{m['photos']}", ""))
    tg = TARGET
    lines.append(
        f"{'target':<22}{'':<8}{'':>5}{'':>6}{'':>6}{'':>8}{tg['missed']:>5}{tg['false']:>6}"
        f"{tg['recall']:>8.2f}{tg['precision']:>6.2f}{tg['f1']:>6.2f}{'-':>9}{tg['err_max']:>8.2f}{f'{len(scores)}/{len(scores)}':>6}"
    )
    kinds = ", ".join(f"{k} {n}" for k, n in t["false_by_kind"].items()) or "none"
    lines.append(f"false by kind: {kinds}")
    rims = ", ".join(f"{rim} rims: recall {r['recall']:.2f} ({r['matched']}/{r['truth']})" for rim, r in t["by_rim"].items())
    lines.append(rims)
    if len(t["by_mode"]) > 1:
        lines.append(", ".join(f"{MODE_SHORT.get(k, k)}: f1 {v['f1']:.2f}, false {v['false']}" for k, v in t["by_mode"].items()))
    return "\n".join(lines)


def _size(d: float | None) -> str:
    return "?" if d is None else f"{d:g}"


def format_details(scores: Sequence[PhotoScore]) -> str:
    """One line per miss, wrong size and false base."""
    lines = []
    for s in scores:
        for t in s.missed:
            x, y = t.center_mm
            lines.append(f"{s.name:<18}MISS   {_size(t.diameter_mm)} mm {t.rim} at ({x:.1f}, {y:.1f})  {t.note}")
        for m in s.matches:
            if m.size_ok is False:
                x, y = m.truth.center_mm
                lines.append(
                    f"{s.name:<18}SIZE   found {m.found['diameter_mm']:g} mm, true {_size(m.truth.diameter_mm)} at ({x:.1f}, {y:.1f}), "
                    f"{m.error_mm:.1f} mm off  {m.truth.note}"
                )
        for f, kind in s.false:
            x, y = f["center_mm"]
            note = ""
            if kind not in ("inside_base", "other"):
                near = min(s.decoys, key=lambda d: _dist(d.center_mm, f["center_mm"]), default=None)
                note = f" ({near.note})" if near is not None and near.note else ""
            lines.append(f"{s.name:<18}FALSE  {f['diameter_mm']:g} mm at ({x:.1f}, {y:.1f}) cov {f.get('coverage', 0):.2f}  {kind}{note}")
    return "\n".join(lines)


def format_total_line(s: PhotoScore) -> str:
    """Short result line of one photo, for the debug image banner."""
    m = s.metrics()
    return (
        f"{s.name} {MODE_SHORT.get(s.mode, s.mode)}: {m['matched']}/{m['truth']} found, {m['wrong_size']} wrong size, "
        f"{m['false']} false, f1 {m['f1']:.2f}" + (", DONE" if s.done_ok else "")
    )
