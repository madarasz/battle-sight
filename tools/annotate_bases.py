"""Click the ground truth of the bases on a test photo.

    uv run tools/annotate_bases.py tests/images/<name>.jpg [--config data/board.toml] [--prefill] [--team NAME ...]

Shows the top-down bases image (bases.px_per_mm, nominal lens). Writes
<name>.bases.json (schema 2) next to the photo, with --team also the operative
names in <name>.ident.json, and <name>.expected.json when it is missing. Keys:

    left click     add a point on the bottom rim of a base (5 or more is best)
    Enter          fit a circle to the points, snap to the nearest size class
                   (fewer than 3 points: an estimate at the click, tol_mm 8)
    1..9           with points: fit with size class 1..9 of bases.diameters_mm
                   without points: set the size of the entry under the mouse
    0              size unknown (null)
    Backspace      drop the points
    h              rim clear <-> hidden (entry under the mouse, else the last one)
    t              note for the entry under the mouse (type it in the terminal)
    x              not_bases entry: of the points (3 or more), else of the base or
                   proposal under the mouse; asks its kind in the terminal
    n              --team: assign an operative to the base under the mouse; the
                   terminal lists the choices, push 1..9 in this window
                   (0 no name, n next page, Esc cancel); with --prefill each
                   base starts with the best guess and the guess is key 1
    a              --prefill: accept the proposal (cyan dashed) under the mouse
    p              --prefill: show all proposals, also the ones on a saved entry
    d              delete the entry or proposal under the mouse
    u              undo
    + / -, z / Z, mouse wheel   zoom in / out at the mouse (1x, 2x, 4x, 8x of the screen fit)
    arrows, right-drag   pan
    s              save
    q, Esc         quit (asks to save when there are changes)

Plan: specs/bases-benchmark.md section 8. Docs: docs/cli.md, docs/testing.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from battlesight import bench
from battlesight import ident as idt
from battlesight.bench import Decoy, GroundTruth, Truth

MENU_SIZE = 9  # operatives on one page of the assign list (keys 1..9)
CLICK_TOL_MM = bench.BASE_CENTER_TOL_MM  # tol_mm of a clicked base
PICK_MIN_MM = 12.0  # an entry is "under the mouse" within its radius, at least this
COVER_MIN_MM = 3.0  # a proposal this near a saved base (or within its tol_mm) is hidden
AUTO = "auto"  # size choice: nearest class of the free fit


def rim_coverage_deg(points: np.ndarray, center: np.ndarray) -> float:
    """Angle around the centre that the points span: 360 minus the largest gap."""
    a = np.sort(np.degrees(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])) % 360.0)
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - gaps.max())


def fit_points(points) -> tuple[np.ndarray, float]:
    """Free circle fit: (centre mm, diameter mm)."""
    from battlesight.circles import _fit_circle

    c, r = _fit_circle(np.asarray(points, dtype=np.float64))
    return c, 2.0 * r


def _circle_key(center_mm, diameter_mm: float) -> tuple[float, float, float]:
    return round(float(center_mm[0]), 2), round(float(center_mm[1]), 2), float(diameter_mm)


class Annotator:
    """Ground truth edit state. No window calls, so tests can drive it."""

    def __init__(self, gt: GroundTruth, classes: tuple[float, ...], proposals: list[dict] | None = None, operatives: list[idt.Operative] | None = None, suggest=None):
        self.gt = GroundTruth(list(gt.bases), list(gt.not_bases), gt.comment, gt.roster_mm)
        self.classes = tuple(classes)
        self.proposals = list(proposals or [])
        self.operatives = list(operatives or [])
        self.assign_base: int | None = None  # base of the open assign list
        self.assign_page = 0
        self.unmatched_names: list[tuple[np.ndarray, str]] = []  # names in the file with no base here
        self.suggest = suggest  # [(centre mm, diameter mm)] -> [idt.Ident], or None
        self.suggestions: dict[tuple, idt.Ident] = {}  # never saved: only the order and the hints
        self.points: list[tuple[float, float]] = []
        self.history: list[tuple] = []
        self.dirty = False
        self.show_all_proposals = False  # else hide proposals on a saved entry

    def covered(self, p: dict) -> bool:
        """True when a saved base or not_bases entry is already at the proposal."""
        c = p["center_mm"]
        if any(bench._dist(c, t.center_mm) <= max(t.tol_mm, COVER_MIN_MM) for t in self.gt.bases):
            return True
        return any(bench._dist(c, d.center_mm) <= 0.5 * d.diameter_mm for d in self.gt.not_bases)

    def visible_proposals(self) -> list[int]:
        """Indices of the proposals to show and pick."""
        return [i for i, p in enumerate(self.proposals) if self.show_all_proposals or not self.covered(p)]

    # --- undo ----------------------------------------------------------------

    def _push(self) -> None:
        self.history.append((list(self.gt.bases), list(self.gt.not_bases), list(self.proposals), list(self.points), self.dirty))

    def undo(self) -> bool:
        if not self.history:
            return False
        bases, decoys, props, pts, dirty = self.history.pop()
        self.gt.bases, self.gt.not_bases, self.proposals, self.points, self.dirty = bases, decoys, props, pts, dirty
        return True

    def _changed(self) -> None:
        self.dirty = True

    # --- points and fit ------------------------------------------------------

    def add_point(self, p_mm) -> None:
        self._push()
        self.points.append((float(p_mm[0]), float(p_mm[1])))

    def clear_points(self) -> None:
        if self.points:
            self._push()
            self.points = []

    def preview(self) -> tuple[np.ndarray, float, float] | None:
        """(centre, free diameter, nearest class) of the current points, or None."""
        if len(self.points) < 3:
            return None
        c, d = fit_points(self.points)
        return c, d, self._nearest_class(d)

    def _nearest_class(self, d: float) -> float:
        return min(self.classes, key=lambda k: abs(k - d))

    def _size(self, choice, fit_d: float | None) -> float | None:
        if choice is None:
            return None
        if choice == AUTO:
            return None if fit_d is None else self._nearest_class(fit_d)
        return self.classes[int(choice)]

    def commit(self, choice=AUTO) -> Truth | None:
        """Make a base of the points. 3 or more: clicked, free fit, size from
        `choice` (AUTO, a class index, or None for unknown). Fewer: an
        estimate at the mean of the clicks, `tol_mm` 8."""
        if not self.points:
            return None
        pts = np.asarray(self.points)
        self._push()
        if len(pts) >= 3:
            c, d = fit_points(pts)
            rim = "clear" if rim_coverage_deg(pts, c) >= 180.0 else "hidden"
            t = Truth((float(c[0]), float(c[1])), self._size(choice, d), rim, CLICK_TOL_MM, "clicked", "", tuple(self.points))
        else:
            c = pts.mean(axis=0)
            t = Truth((float(c[0]), float(c[1])), self._size(choice, None), "hidden", bench.ESTIMATE_TOL_MM, "estimate", "")
        self.gt.bases.append(t)
        self.points = []
        self._changed()
        return t

    # --- entries under the mouse --------------------------------------------

    def pick(self, at_mm, proposals: bool = False) -> tuple[str, int] | None:
        """("base" | "decoy" | "proposal", index) of the nearest entry under
        the mouse, or None."""
        best = None
        groups = [("base", [(t.center_mm, t.diameter_mm) for t in self.gt.bases]), ("decoy", [(d.center_mm, d.diameter_mm) for d in self.gt.not_bases])]
        if proposals:
            groups = [("proposal", [(i, self.proposals[i]["center_mm"], self.proposals[i]["diameter_mm"]) for i in self.visible_proposals()])]
        else:
            groups = [(kind, [(i, c, d) for i, (c, d) in enumerate(items)]) for kind, items in groups]
        for kind, items in groups:
            for i, c, d in items:
                dist = bench._dist(c, at_mm)
                reach = max(PICK_MIN_MM, 0.5 * (d or 30.0))
                if dist <= reach and (best is None or dist < best[0]):
                    best = (dist, kind, i)
        return None if best is None else (best[1], best[2])

    def _base_at(self, at_mm, fallback_last: bool = True) -> int | None:
        hit = None if at_mm is None else self.pick(at_mm)
        if hit is not None and hit[0] == "base":
            return hit[1]
        if hit is None and fallback_last and self.gt.bases:
            return len(self.gt.bases) - 1
        return None

    def toggle_rim(self, at_mm=None) -> Truth | None:
        i = self._base_at(at_mm)
        if i is None:
            return None
        self._push()
        t = self.gt.bases[i]
        self.gt.bases[i] = replace(t, rim="hidden" if t.rim == "clear" else "clear")
        self._changed()
        return self.gt.bases[i]

    def set_size(self, at_mm, choice: int | None) -> Truth | None:
        """Size class `choice` (index) or None (unknown) for the base under the mouse."""
        i = self._base_at(at_mm)
        if i is None:
            return None
        self._push()
        self.gt.bases[i] = replace(self.gt.bases[i], diameter_mm=self._size(choice, None))
        self._changed()
        return self.gt.bases[i]

    def set_note(self, at_mm, note: str) -> bool:
        hit = self.pick(at_mm) if at_mm is not None else None
        if hit is None and self.gt.bases:
            hit = ("base", len(self.gt.bases) - 1)
        if hit is None:
            return False
        self._push()
        kind, i = hit
        if kind == "base":
            self.gt.bases[i] = replace(self.gt.bases[i], note=note)
        else:
            self.gt.not_bases[i] = replace(self.gt.not_bases[i], note=note)
        self._changed()
        return True

    def to_decoy(self, at_mm, kind: str) -> Decoy | None:
        """Make a `not_bases` entry: of the points (3 or more), else of the
        base under the mouse (moved), else of the proposal under the mouse."""
        if kind not in bench.KINDS:
            raise ValueError(f"kind must be one of {bench.KINDS}")
        if len(self.points) >= 3:
            self._push()
            c, d = fit_points(self.points)
            dec = Decoy((float(c[0]), float(c[1])), round(d, 1), kind)
            self.points = []
        else:
            i = self._base_at(at_mm, fallback_last=False)
            hit = None if i is not None or at_mm is None else self.pick(at_mm, proposals=True)
            if i is not None:
                self._push()
                t = self.gt.bases.pop(i)
                d = t.diameter_mm
                if d is None and t.points_mm:
                    d = round(fit_points(t.points_mm)[1], 1)
                dec = Decoy(t.center_mm, 30.0 if d is None else d, kind, t.note)
            elif hit is not None:
                self._push()
                pr = self.proposals.pop(hit[1])
                dec = Decoy((float(pr["center_mm"][0]), float(pr["center_mm"][1])), float(pr["diameter_mm"]), kind)
            else:
                return None
        self.gt.not_bases.append(dec)
        self._changed()
        return dec

    def accept(self, at_mm) -> Truth | None:
        """Detector proposal under the mouse -> base with `source: detector`."""
        hit = self.pick(at_mm, proposals=True)
        if hit is None:
            return None
        self._push()
        p = self.proposals.pop(hit[1])
        rim = "clear" if p.get("coverage", 1.0) >= 0.5 else "hidden"
        t = Truth((float(p["center_mm"][0]), float(p["center_mm"][1])), float(p["diameter_mm"]), rim, CLICK_TOL_MM, "detector", "")
        t = replace(t, label=self._best_guess(t.center_mm, t.diameter_mm))
        self.gt.bases.append(t)
        self._changed()
        return t

    def delete(self, at_mm) -> bool:
        hit = self.pick(at_mm) or self.pick(at_mm, proposals=True)
        if hit is None:
            return False
        self._push()
        kind, i = hit
        {"base": self.gt.bases, "decoy": self.gt.not_bases, "proposal": self.proposals}[kind].pop(i)
        if kind != "proposal":
            self._changed()
        return True

    # --- operative names -----------------------------------------------------

    def refresh_suggestions(self) -> bool:
        """Run `suggest` on the bases of known size and the open proposals when
        a base has no suggestion yet. True when it ran."""
        if self.suggest is None:
            return False
        bases = [(t.center_mm, t.diameter_mm) for t in self.gt.bases if t.diameter_mm is not None]
        if self.suggestions and all(_circle_key(*b) in self.suggestions for b in bases):
            return False
        items = bases + [(tuple(p["center_mm"]), float(p["diameter_mm"])) for p in self.proposals if not self.covered(p)]
        self.suggestions = {_circle_key(*it): x for it, x in zip(items, self.suggest(items))}
        return True

    def suggestion(self, center_mm, diameter_mm) -> idt.Ident | None:
        return None if diameter_mm is None else self.suggestions.get(_circle_key(center_mm, diameter_mm))

    def _best_guess(self, center_mm, diameter_mm) -> str:
        """The suggested name, or "" when there is none or a base has it."""
        s = self.suggestion(center_mm, diameter_mm)
        if s is None or not s.label or any(t.label == s.label for t in self.gt.bases):
            return ""
        return s.label

    def apply_guesses(self) -> int:
        """Put the best guess on each base with no name. Number of names set."""
        n = 0
        for i, t in enumerate(self.gt.bases):
            label = "" if t.label else self._best_guess(t.center_mm, t.diameter_mm)
            if label:
                self.gt.bases[i] = replace(t, label=label)
                n += 1
        if n:
            self._changed()
        return n

    def assign_choices(self) -> list[idt.Operative]:
        """Operatives that fit the size of the base of the open assign list.
        With a suggestion: the suggested name first, then the best scores."""
        t = self.gt.bases[self.assign_base]
        d = t.diameter_mm
        allowed = [op for op in self.operatives if d is None or op.base_mm is None or op.base_mm == d]
        s = self.suggestion(t.center_mm, d)
        if s is None:
            return allowed
        rank: dict[str, int] = {}
        for label in ([s.label] if s.label else []) + [lab for lab, _ in s.candidates]:
            rank.setdefault(label, len(rank))
        return sorted(allowed, key=lambda op: rank.get(op.label, len(rank)))

    def assign_items(self) -> list[idt.Operative]:
        """The operatives on the current page: key 1 is the first."""
        return self.assign_choices()[self.assign_page * MENU_SIZE : (self.assign_page + 1) * MENU_SIZE]

    def assign_pages(self) -> int:
        return (len(self.assign_choices()) + MENU_SIZE - 1) // MENU_SIZE

    def start_assign(self, at_mm) -> bool:
        """Open the assign list of the base under the mouse. False: no base
        there, or no operative of its size."""
        i = None if at_mm is None else self._base_at(at_mm, fallback_last=False)
        if i is None:
            return False
        self.refresh_suggestions()
        self.assign_base, self.assign_page = i, 0
        if not self.assign_choices():
            self.assign_base = None
            return False
        return True

    def next_assign_page(self) -> None:
        self.assign_page = (self.assign_page + 1) % self.assign_pages()

    def cancel_assign(self) -> None:
        self.assign_base = None

    def choose_assign(self, key: int) -> bool:
        """Key 1..9: name the base. Key 0: clear the name. False: no such
        entry on this page, and the list stays open."""
        if key == 0:
            label = ""
        else:
            items = self.assign_items()
            if key > len(items):
                return False
            label = items[key - 1].label
        self._push()
        self.gt.bases[self.assign_base] = replace(self.gt.bases[self.assign_base], label=label)
        self.assign_base = None
        self._changed()
        return True

    def assign_lines(self) -> list[str]:
        """Text of the assign list for the terminal."""
        t = self.gt.bases[self.assign_base]
        size = "unknown size" if t.diameter_mm is None else f"{t.diameter_mm:g} mm"
        used = {b.label for i, b in enumerate(self.gt.bases) if i != self.assign_base and b.label}
        now = f", now {t.label}" if t.label else ""
        lines = [f"assign base at ({t.center_mm[0]:.1f}, {t.center_mm[1]:.1f}), {size}{now}:"]
        s = self.suggestion(t.center_mm, t.diameter_mm)
        scores = {} if s is None else dict(s.candidates)
        for k, op in enumerate(self.assign_items(), 1):
            line = f"  {k}  {op.label}"
            if op.label in scores:
                line += f"  {scores[op.label]:.2f}"
            if s is not None and op.label == s.label:
                line += "  (best guess)" if s.sure else "  (best guess, unsure)"
            if op.label in used:
                line += "  (used on another base)"
            lines.append(line)
        lines.append("  0  no name")
        if self.assign_pages() > 1:
            lines.append(f"  n  next page ({self.assign_page + 1}/{self.assign_pages()})")
        lines.append("  Esc  cancel")
        return lines

    def load_names(self, truth: idt.IdentTruth) -> None:
        """Put the names of a `<name>.ident.json` on the nearest free base
        within its `tol_mm`. A name with no base stays in `unmatched_names`
        and is saved again, so hand-made entries are never lost."""
        for c, label in truth.bases:
            free = [(bench._dist(c, t.center_mm), i) for i, t in enumerate(self.gt.bases) if not t.label]
            dist, i = min(free, default=(np.inf, -1))
            if dist <= truth.tol_mm:
                self.gt.bases[i] = replace(self.gt.bases[i], label=label)
            else:
                self.unmatched_names.append((c, label))

    def names_truth(self, teams: list[str], tol_mm: float) -> idt.IdentTruth:
        named = [(np.array(t.center_mm, float), t.label) for t in self.gt.bases if t.label]
        return idt.IdentTruth(list(teams), named + self.unmatched_names, tol_mm)

    def save(self, path: str | Path) -> None:
        bench.save_truth(path, self.gt)
        self.dirty = False


# --- window ------------------------------------------------------------------

KEYS_UP = {63232, 65362, 2490368}
KEYS_DOWN = {63233, 65364, 2621440}
KEYS_LEFT = {63234, 65361, 2424832}
KEYS_RIGHT = {63235, 65363, 2555904}
KEYS_ENTER = {10, 13}
KEYS_BACKSPACE = {8, 127}
ZOOMS = (1, 2, 4, 8)


def ident_path(photo: Path) -> Path:
    return photo.with_name(f"{photo.stem}.ident.json")


def expected_path(photo: Path) -> Path:
    return photo.with_name(f"{photo.stem}.expected.json")


def write_expected(photo: Path, frame: np.ndarray, cfg, camera) -> tuple[Path | None, str]:
    """Write the first regression report of a photo, with the same pipeline and
    reference as tests/test_photos.py. Never replaces a file: the tests compare
    later runs with it. (path or None, message)."""
    import cv2

    from battlesight.pipeline import Pipeline

    path = expected_path(photo)
    if path.is_file():
        return None, ""
    ref_path = bench.reference_path(photo)
    ref = None if ref_path is None else cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    r = Pipeline(cfg, "bases", camera, ref).analyze(frame, source=photo.as_posix())
    if not r.ok:
        return None, f"{path.name} not written: {r.summary()}"
    path.write_text(json.dumps(r.to_dict(), indent=2) + "\n")
    return path, r.summary()


def load_operatives(teams: list[str], root: Path = idt.DEFAULT_GALLERY_ROOT) -> list[idt.Operative]:
    """All operatives of the teams, in the order of `teams`. ValueError for a
    team with no gallery."""
    known = idt.list_teams(root)
    missing = [t for t in teams if t not in known]
    if missing:
        raise ValueError(f"no gallery for {', '.join(missing)} in {root} (teams: {', '.join(known) or 'none'})")
    return [op for t in teams for op in idt.team_operatives(t, Path(root) / t)]


def load_names_file(path: Path, teams: list[str]) -> idt.IdentTruth | None:
    """The existing `<name>.ident.json`, or None. ValueError when it has names
    of a team outside `teams` or cannot be read."""
    if not path.is_file():
        return None
    try:
        truth = idt.IdentTruth.load(path)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: {exc!r}") from exc
    other = sorted({idt.split_label(label)[0] for _, label in truth.bases} - set(teams))
    if other:
        raise ValueError(f"{path.name} has names of {', '.join(other)}: add to --team")
    return truth


def make_suggester(img: np.ndarray, reference: np.ndarray | None, cfg, gallery: idt.Gallery):
    """The `suggest` function of `Annotator`: runs the identification of the
    app on circles of the bases image."""
    from battlesight.circles import Circle
    from battlesight.pipeline import base_crops

    def suggest(items: list[tuple]) -> list[idt.Ident]:
        found = [Circle(np.array(c, float), float(d), float(d), 1.0, 0.0, 0.0) for c, d in items]
        crops = base_crops(img, found, cfg)
        backgrounds = [] if reference is None else base_crops(reference, found, cfg)
        return idt.identify(crops, [c.diameter_mm for c in found], gallery, backgrounds, [c.center_mm for c in found])

    return suggest


def run_window(photo: Path, cfg, prefill: bool, display: tuple[int, int], teams: list[str] | None = None) -> int:
    import cv2

    from battlesight import overlay
    from battlesight.camera import CameraSettings
    from battlesight.pipeline import BASES_MARGIN_MM, Pipeline, bases_image, detect_bases

    frame = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"error: cannot read {photo}", file=sys.stderr)
        return 2
    cam = CameraSettings()
    r = Pipeline(cfg, "rectify", cam).analyze(frame, source=str(photo))
    if not r.ok:
        print(f"error: {photo.name}: {r.summary()}", file=sys.stderr)
        return 2
    img = bases_image(frame, r.scratch["view"], cfg)
    ppm, origin = cfg.bases_px_per_mm, BASES_MARGIN_MM
    out_path = bench.truth_path(photo)
    names_path = ident_path(photo)
    try:
        gt = bench.load_truth(out_path) if out_path.is_file() else GroundTruth()
        operatives = load_operatives(teams) if teams else []
        names = load_names_file(names_path, teams) if teams else None
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    names_tol = 10.0 if names is None else names.tol_mm
    proposals = []
    ref = None
    if prefill:
        ref_path = bench.reference_path(photo)
        ref_frame = None if ref_path is None else cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if ref_frame is not None:
            rr = Pipeline(cfg, "rectify", cam).analyze(ref_frame, source=str(ref_path))
            if rr.ok:
                ref = bases_image(ref_frame, rr.scratch["view"], cfg)
        found, _ = detect_bases(img, cfg, ref, sizes=gt.roster_mm)
        proposals = [c.to_dict() for c in found]
        print(f"{len(proposals)} detector proposals" + (f" (reference {ref_path.name})" if ref is not None else ""))
    suggest = None
    if prefill and teams:
        gallery = idt.Gallery.for_teams(teams, cfg)
        if len(gallery) == 0:
            print("no operative suggestions: the gallery has no photos")
        else:
            suggest = make_suggester(img, ref, cfg, gallery)
    ann = Annotator(gt, cfg.bases_diameters_mm, proposals, operatives, suggest)
    ann.refresh_suggestions()
    if names is not None:
        ann.load_names(names)
        for c, label in ann.unmatched_names:
            print(f"warning: {label} at ({c[0]:.1f}, {c[1]:.1f}) has no base here: kept in {names_path.name}")
    if suggest is not None:
        print(f"best guess set on {ann.apply_guesses()} bases: check each name, fix with n")
    print(__doc__.split("\n\n")[3])
    print("classes: " + ", ".join(f"{i + 1}={d:g}" for i, d in enumerate(ann.classes)))

    def save_all() -> None:
        ann.save(out_path)
        print(f"wrote {out_path}")
        result = ann.names_truth(teams or [], names_tol)
        if teams and (result.bases or names_path.is_file()):
            result.save(names_path)
            print(f"wrote {names_path}")
        path, message = write_expected(photo, frame, cfg, cam)
        if path is not None:
            print(f"wrote {path}: first report, tests compare later runs with it. Check it: {message}")
        elif message:
            print(f"warning: {message}")

    H, W = img.shape[:2]
    fit = min(display[0] / W, display[1] / H)
    state = {"zoom": 0, "vx": 0.0, "vy": 0.0, "mouse": None, "drag": None}

    def scale() -> float:
        return fit * ZOOMS[state["zoom"]]

    def clamp() -> None:
        s = scale()
        state["vx"] = float(np.clip(state["vx"], 0, max(0.0, W - display[0] / s)))
        state["vy"] = float(np.clip(state["vy"], 0, max(0.0, H - display[1] / s)))

    def to_mm(x: float, y: float) -> tuple[float, float]:
        s = scale()
        return (x / s + state["vx"]) / ppm - origin, (y / s + state["vy"]) / ppm - origin

    def to_disp(p_mm) -> tuple[int, int]:
        s = scale()
        return int(round(((p_mm[0] + origin) * ppm - state["vx"]) * s)), int(round(((p_mm[1] + origin) * ppm - state["vy"]) * s))

    def zoom(step: int) -> None:
        z = int(np.clip(state["zoom"] + step, 0, len(ZOOMS) - 1))
        if z == state["zoom"]:
            return
        mx, my = state["mouse"] or (display[0] / 2, display[1] / 2)
        s0 = scale()
        ix, iy = mx / s0 + state["vx"], my / s0 + state["vy"]
        state["zoom"] = z
        s1 = scale()
        state["vx"], state["vy"] = ix - mx / s1, iy - my / s1
        clamp()

    def on_mouse(event, x, y, flags, _):
        state["mouse"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            ann.add_point(to_mm(x, y))
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom(+1 if cv2.getMouseWheelDelta(flags) > 0 else -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drag"] = (x, y, state["vx"], state["vy"])
        elif event == cv2.EVENT_RBUTTONUP:
            state["drag"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            x0, y0, vx, vy = state["drag"]
            s = scale()
            state["vx"], state["vy"] = vx - (x - x0) / s, vy - (y - y0) / s
            clamp()

    def render() -> np.ndarray:
        s = scale()
        w, h = int(min(W, display[0] / s)), int(min(H, display[1] / s))
        x0, y0 = int(state["vx"]), int(state["vy"])
        crop = img[y0 : y0 + h, x0 : x0 + w]
        view = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_NEAREST if s >= 1 else cv2.INTER_AREA)
        state["vx"], state["vy"] = float(x0), float(y0)  # integer crop: keep mm mapping exact
        rr = lambda d: max(1, int(round(0.5 * d * ppm * s)))  # noqa: E731
        def hint(center_mm, d) -> None:
            s = ann.suggestion(center_mm, d)
            if s is not None and s.label:
                cv2.putText(view, f"{idt.split_label(s.label)[1]}?", to_disp((center_mm[0] - 0.5 * d, center_mm[1] + 0.5 * d + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay.CYAN, 1, cv2.LINE_AA)

        for i in ann.visible_proposals():
            p = ann.proposals[i]
            overlay.draw_dashed_circle(view, to_disp(p["center_mm"]), rr(p["diameter_mm"]), overlay.CYAN, 1, dashes=32)
            hint(p["center_mm"], p["diameter_mm"])
        for dc in ann.gt.not_bases:
            overlay.draw_dashed_circle(view, to_disp(dc.center_mm), rr(dc.diameter_mm), (170, 170, 170), 1, dashes=48, duty=0.3)
            cv2.putText(view, dc.kind, to_disp((dc.center_mm[0], dc.center_mm[1] + 0.5 * dc.diameter_mm + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        for t in ann.gt.bases:
            color = overlay.GREEN if t.rim == "clear" else overlay.YELLOW
            d = 30.0 if t.diameter_mm is None else t.diameter_mm
            if t.source == "estimate":
                overlay.draw_dashed_circle(view, to_disp(t.center_mm), rr(d), color, 2)
            else:
                cv2.circle(view, to_disp(t.center_mm), rr(d), color, 2 if t.source == "clicked" else 1, cv2.LINE_AA)
            cv2.drawMarker(view, to_disp(t.center_mm), color, cv2.MARKER_CROSS, 10, 1)
            for q in t.points_mm or ():
                cv2.circle(view, to_disp(q), 2, color, -1)
            label = f"{'?' if t.diameter_mm is None else f'{t.diameter_mm:g}'} {t.source[0]} {t.note}"[:30]
            cv2.putText(view, label, to_disp((t.center_mm[0] - 0.5 * d, t.center_mm[1] - 0.5 * d - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            if t.label:
                cv2.putText(view, idt.split_label(t.label)[1], to_disp((t.center_mm[0] - 0.5 * d, t.center_mm[1] + 0.5 * d + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, overlay.WHITE, 1, cv2.LINE_AA)
            else:
                hint(t.center_mm, t.diameter_mm)
        if ann.assign_base is not None:
            t = ann.gt.bases[ann.assign_base]
            cv2.circle(view, to_disp(t.center_mm), rr(30.0 if t.diameter_mm is None else t.diameter_mm) + 3, (255, 0, 255), 3, cv2.LINE_AA)
        for q in ann.points:
            cv2.circle(view, to_disp(q), 3, overlay.RED, -1, cv2.LINE_AA)
        pv = ann.preview()
        fit_txt = ""
        if pv is not None:
            c, d, k = pv
            cv2.circle(view, to_disp(c), rr(d), (255, 0, 255), 1, cv2.LINE_AA)
            fit_txt = f" | fit {d:.1f} mm -> {k:g}"
        canvas = np.zeros((display[1], display[0], 3), np.uint8)
        canvas[: view.shape[0], : view.shape[1]] = view[: display[1], : display[0]]
        m = state["mouse"]
        pos = "" if m is None else "mouse ({:.1f}, {:.1f}) mm".format(*to_mm(*m))
        n_clear = sum(t.rim == "clear" for t in ann.gt.bases)
        named_txt = f" | named {sum(bool(t.label) for t in ann.gt.bases)}/{len(ann.gt.bases)}" if ann.operatives else ""
        text = (
            f"{photo.stem} {ZOOMS[state['zoom']]}x | {pos} | bases {len(ann.gt.bases)} ({n_clear} clear), not {len(ann.gt.not_bases)}"
            f" | proposals {len(ann.visible_proposals())}/{len(ann.proposals)} | points {len(ann.points)}{fit_txt}{named_txt}" + (" | UNSAVED" if ann.dirty else "")
        )
        overlay.draw_text_banner(canvas, text, overlay.YELLOW if ann.dirty else overlay.GREEN)
        if ann.assign_base is not None:
            overlay.draw_text_banner(canvas, f"assign operative: see the terminal, push 1-{len(ann.assign_items())} here, 0 no name, Esc cancel", overlay.CYAN, row=1)
        return canvas

    win = f"annotate {photo.name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    pan = 0.15
    while True:
        cv2.imshow(win, render())
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        ch = chr(key & 0xFF) if key < 256 else ""
        at = None if state["mouse"] is None else to_mm(*state["mouse"])
        if ann.assign_base is not None:  # the assign list takes all keys until it closes
            if ch and ch in "0123456789":
                i = ann.assign_base
                if ann.choose_assign(int(ch)):
                    print(f"name: {ann.gt.bases[i].label or '(none)'}")
                else:
                    print(f"no operative on key {ch}")
            elif ch == "n":
                ann.next_assign_page()
                print("\n".join(ann.assign_lines()))
            elif key == 27:
                ann.cancel_assign()
                print("assign cancelled")
            continue
        if key in KEYS_ENTER:
            t = ann.commit()
            if t is not None:
                print(f"base {t.source} {t.diameter_mm} mm at ({t.center_mm[0]:.1f}, {t.center_mm[1]:.1f}) rim {t.rim}")
        elif ch.isdigit():
            choice = None if ch == "0" else int(ch) - 1
            if choice is not None and choice >= len(ann.classes):
                continue
            if ann.points:
                ann.commit(choice)
            else:
                ann.set_size(at, choice)
        elif key in KEYS_BACKSPACE:
            ann.clear_points()
        elif ch == "h":
            ann.toggle_rim(at)
        elif ch == "t":
            ann.set_note(at, input("note: ").strip())
        elif ch == "x":
            kind = input(f"kind ({', '.join(bench.KINDS)}): ").strip()
            try:
                if ann.to_decoy(at, kind) is None:
                    print("no base or proposal under the mouse, and fewer than 3 points")
            except ValueError as exc:
                print(exc)
        elif ch == "n":
            if not ann.operatives:
                print("start with --team to assign operatives")
            elif ann.start_assign(at):
                print("\n".join(ann.assign_lines()))
            else:
                print("no base under the mouse, or no operative of its size")
        elif ch == "a":
            if ann.accept(at) is None:
                print("no proposal under the mouse")
        elif ch == "p":
            ann.show_all_proposals = not ann.show_all_proposals
            print("show all proposals" if ann.show_all_proposals else "hide proposals on saved entries")
        elif ch == "d":
            ann.delete(at)
        elif ch == "u":
            ann.undo()
        elif ch and ch in "+=z":
            zoom(+1)
        elif ch and ch in "-_Z":
            zoom(-1)
        elif key in KEYS_UP | KEYS_DOWN | KEYS_LEFT | KEYS_RIGHT:
            step = pan * display[0] / scale()
            state["vx"] += step * ((key in KEYS_RIGHT) - (key in KEYS_LEFT))
            state["vy"] += step * ((key in KEYS_DOWN) - (key in KEYS_UP))
            clamp()
        elif ch == "s":
            save_all()
        elif ch == "q" or key == 27:
            if ann.dirty and input("save changes? [y/N] ").strip().lower().startswith("y"):
                save_all()
            break
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    from battlesight.config import DEFAULT_CONFIG, BoardConfig

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo", help="test photo; writes <name>.bases.json next to it")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="board config")
    p.add_argument("--prefill", action="store_true", help="show the detections as proposals (a = accept)")
    p.add_argument("--team", nargs="+", default=None, metavar="NAME", help="teams on the photo, folders in data/teams: turns on the n key (assign an operative), writes <name>.ident.json")
    p.add_argument("--display-width", type=int, default=1600, help="window width in pixels")
    p.add_argument("--display-height", type=int, default=950, help="window height in pixels")
    args = p.parse_args(argv)
    try:
        cfg = BoardConfig.load(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return run_window(Path(args.photo), cfg, args.prefill, (args.display_width, args.display_height), args.team)


if __name__ == "__main__":
    sys.exit(main())
