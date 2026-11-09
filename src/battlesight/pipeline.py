"""Single-frame analysis pipeline. No camera, no window, no Qt.

A frame goes through an ordered list of stages. Each stage reads the frame and
the results of earlier stages, and returns a `StageResult`. When a stage fails,
the stages after it are skipped, because they depend on it.

Add a priority as a new stage:

    @stage("rectify", draw=_draw_rectify)
    def _rectify(ctx: Context) -> StageResult:
        corners = ctx.results["corners"].data["board_corners_px"]
        ...

The `image-pipeline` subcommand, the tests, and a later FastAPI front end all call
`analyze` and `annotate`. Everything in `StageResult.data` must be JSON-safe
(plain numbers, strings, lists, dicts).
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .calib import TagDetector, TagObs, board_corners_px, lens_for_frame
from .camera import CameraSettings
from .config import CORNER_NAMES, BoardConfig
from . import circles
from . import ident as idt
from . import rectify as rect

SCHEMA_VERSION = 1


@dataclass
class StageResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "skipped": self.skipped, "message": self.message, "ms": round(self.ms, 1), "data": self.data}


@dataclass
class Context:
    """State shared by the stages of one frame.

    `cache` keeps expensive objects (detectors, models) between frames when the
    caller passes the same `Pipeline`. `scratch` keeps non-JSON objects (TagObs,
    homographies) for later stages and for drawing.
    """

    frame: np.ndarray
    cfg: BoardConfig
    cache: dict[str, Any]
    camera: CameraSettings = field(default_factory=CameraSettings)  # lens for the rectify stage
    results: dict[str, StageResult] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BasesSearch:
    """Limits of one bases search (the enroll mode).

    `area_px` is a rectangle of the camera frame (x0, y0, x1, y1): base centres
    must lie in it. `diameters_mm` replaces the known base sizes. `relaxed` is
    the second search: fixed radius, loose thresholds, no reference, few
    results. It can be wrong, so the caller must ask the user."""

    area_px: tuple[int, int, int, int] | None = None
    diameters_mm: tuple[float, ...] | None = None
    relaxed: bool = False


@dataclass
class FrameResult:
    source: str
    width: int
    height: int
    stages: dict[str, StageResult]
    scratch: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.stages.values())

    def summary(self) -> str:
        parts = []
        for name, r in self.stages.items():
            mark = "skip" if r.skipped else ("ok" if r.ok else "FAIL")
            parts.append(f"{name} {mark}" + (f": {r.message}" if r.message else ""))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "source": self.source,
            "size": [self.width, self.height],
            "ok": self.ok,
            "stages": {name: r.to_dict() for name, r in self.stages.items()},
        }


RunFn = Callable[[Context], StageResult]
DrawFn = Callable[[np.ndarray, "FrameResult", BoardConfig], None]
# Extra images of a stage (for example the top-down board), by name.
ViewsFn = Callable[[np.ndarray, "FrameResult", BoardConfig], dict[str, np.ndarray]]


@dataclass(frozen=True)
class Stage:
    name: str
    run: RunFn
    draw: DrawFn | None = None
    views: ViewsFn | None = None


STAGES: dict[str, Stage] = {}  # in run order


def stage(name: str, draw: DrawFn | None = None, views: ViewsFn | None = None) -> Callable[[RunFn], RunFn]:
    """Register a stage. Stages run in the order they are registered.
    `draw` draws on the camera frame, `views` returns extra named images."""

    def deco(fn: RunFn) -> RunFn:
        if name in STAGES:
            raise ValueError(f"stage {name!r} already registered")
        STAGES[name] = Stage(name, fn, draw, views)
        return fn

    return deco


def stage_names() -> list[str]:
    return list(STAGES)


def select_stages(until: str | None = None) -> list[str]:
    """Stage names to run: all stages up to and including `until` (all when None).
    A stage depends on every stage before it."""
    names = stage_names()
    if until is None:
        return names
    if until not in STAGES:
        raise ValueError(f"unknown stage {until!r}, known: {names}")
    return names[: names.index(until) + 1]


class Pipeline:
    """Runs the selected stages on frames. Keeps the cache between frames.

    `reference` is an optional empty-board frame (terrain set up, no
    operatives or tokens) from the same camera and light. With it the bases
    stage looks only where the frame changed.

    `gallery` (ident.Gallery) holds the enrolled operatives. Without it the
    ident stage passes and names nothing."""

    def __init__(
        self,
        cfg: BoardConfig,
        until: str | None = None,
        camera: CameraSettings | None = None,
        reference: np.ndarray | None = None,
        gallery: idt.Gallery | None = None,
    ):
        self.cfg = cfg
        self.camera = camera or CameraSettings()
        self.names = select_stages(until)
        self.cache: dict[str, Any] = {}
        if reference is not None:
            self.set_reference(reference)
        if gallery is not None:
            self.cache["gallery"] = gallery

    @property
    def gallery(self) -> idt.Gallery | None:
        return self.cache.get("gallery")

    def set_gallery(self, gallery: idt.Gallery | None) -> None:
        if gallery is None:
            self.cache.pop("gallery", None)
        else:
            self.cache["gallery"] = gallery

    def set_reference(self, frame: np.ndarray) -> None:
        """Use `frame` as the empty-board reference. The reference finds its own
        tags, so a small camera move between it and later frames does not matter.
        Raises ValueError when its corners or rectify stage fails."""
        r = Pipeline(self.cfg, "rectify", self.camera).analyze(frame, source="reference")
        if not r.ok:
            raise ValueError(f"empty-board reference: {r.summary()}")
        self.cache["bases_reference"] = bases_image(frame, r.scratch["view"], self.cfg)

    def clear_reference(self) -> None:
        self.cache.pop("bases_reference", None)

    @property
    def has_reference(self) -> bool:
        return "bases_reference" in self.cache

    def analyze(self, frame: np.ndarray, source: str = "", until: str | None = None, search: BasesSearch | None = None) -> FrameResult:
        """Run the stages on one frame. `until` stops earlier than the stages
        of the pipeline (the live app runs `bases` only on one key). `search`
        limits the bases stage."""
        ctx = Context(frame, self.cfg, self.cache, self.camera)
        ctx.scratch["bases_search"] = search
        failed = False
        names = self.names if until is None else [n for n in self.names if n in select_stages(until)]
        for name in names:
            if failed:
                ctx.results[name] = StageResult(ok=False, skipped=True, message="earlier stage failed")
                continue
            t0 = time.perf_counter()
            try:
                r = STAGES[name].run(ctx)
            except Exception as exc:  # one bad frame must not stop a batch
                r = StageResult(ok=False, message=f"{type(exc).__name__}: {exc}")
            r.ms = (time.perf_counter() - t0) * 1000.0
            ctx.results[name] = r
            failed = not r.ok
        h, w = frame.shape[:2]
        return FrameResult(source, w, h, ctx.results, ctx.scratch)

    def annotate(self, frame: np.ndarray, result: FrameResult, banner: bool = True) -> np.ndarray:
        """Copy of the frame with every stage's overlay and a status banner."""
        from .overlay import GREEN, RED, draw_text_banner

        out = frame.copy()
        for name, r in result.stages.items():
            draw = STAGES[name].draw
            if draw is not None and not r.skipped:
                draw(out, result, self.cfg)
        if banner:
            draw_text_banner(out, result.summary(), GREEN if result.ok else RED, f"{result.width}x{result.height}")
        return out

    def views(self, frame: np.ndarray, result: FrameResult) -> dict[str, np.ndarray]:
        """Extra images of the stages that passed, by name (for example "rectified")."""
        out: dict[str, np.ndarray] = {}
        for name, r in result.stages.items():
            fn = STAGES[name].views
            if fn is not None and r.ok:
                out.update(fn(frame, result, self.cfg))
        return out


def analyze(
    frame: np.ndarray,
    cfg: BoardConfig,
    until: str | None = None,
    source: str = "",
    camera: CameraSettings | None = None,
    reference: np.ndarray | None = None,
) -> FrameResult:
    """One-shot helper. Use `Pipeline` for many frames."""
    return Pipeline(cfg, until, camera, reference).analyze(frame, source)


def _round(a: np.ndarray, nd: int = 2) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


# --- priority 1: board corners -------------------------------------------------


def _draw_corners(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> None:
    from .overlay import draw_board_corners, draw_tags

    draw_tags(frame, result.scratch.get("tags", {}), cfg)
    draw_board_corners(frame, result.scratch.get("board_corners_px", {}))


@stage("corners", draw=_draw_corners)
def _corners(ctx: Context) -> StageResult:
    detector = ctx.cache.get("tag_detector")
    if detector is None:
        detector = ctx.cache["tag_detector"] = TagDetector(ctx.cfg)
    obs: dict[int, TagObs] = detector.detect(ctx.frame)
    corners = board_corners_px(ctx.cfg, obs)
    ctx.scratch["tags"] = obs
    ctx.scratch["board_corners_px"] = corners
    missing = [i for i in ctx.cfg.tag_ids if i not in obs]
    data = {
        "tags": {
            str(i): {"corner": CORNER_NAMES[ctx.cfg.corner_index(i)], "corners_px": _round(t.corners), "side_px": round(t.side_px, 2)}
            for i, t in sorted(obs.items())
        },
        "missing": missing,
        "board_corners_px": [_round(corners[k]) if k in corners else None for k in range(4)],
    }
    if missing:
        return StageResult(ok=False, message=f"{len(obs)}/4 tags, missing {missing}", data=data)
    return StageResult(ok=True, message="4/4 tags", data=data)


# --- priority 2: undistort and rectify -----------------------------------------


def _sig(a: np.ndarray, digits: int = 9) -> list:
    """Round to significant digits: homography entries span many magnitudes."""
    return [[float(f"{v:.{digits}g}") for v in row] for row in np.asarray(a, dtype=np.float64)]


def _draw_rectify(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> None:
    from .overlay import draw_grid

    view = result.scratch.get("view")
    if view is not None:
        draw_grid(frame, cfg, view.mm_to_px)


def _views_rectify(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> dict[str, np.ndarray]:
    from .overlay import draw_grid

    img = result.scratch.get("rectified")
    if img is None:
        return {}
    img = img.copy()
    draw_grid(img, cfg, lambda p: rect.mm_to_rect(cfg, p))
    return {"rectified": img}


def _edge_dict(e: rect.EdgeFit | None) -> dict | None:
    if e is None:
        return None
    return {"offset_mm": round(e.offset_mm, 2), "angle_deg": round(e.angle_deg, 3), "n": e.n, "rms_mm": round(e.rms_mm, 2)}


@stage("rectify", draw=_draw_rectify, views=_views_rectify)
def _rectify(ctx: Context) -> StageResult:
    cfg = ctx.cfg
    h, w = ctx.frame.shape[:2]
    lens = ctx.cache.get(("lens", w, h))
    if lens is None:
        lens = ctx.cache[("lens", w, h)] = lens_for_frame(ctx.camera, (w, h))
    obs: dict[int, TagObs] = ctx.scratch["tags"]
    view = rect.fit_view(cfg, obs, lens)
    warped = rect.warp_board(ctx.frame, view, cfg)
    ctx.scratch["view"] = view
    ctx.scratch["rectified"] = warped

    edges = rect.measure_edges(warped, cfg)
    angles = rect.corner_angles_deg(edges)
    grid = rect.measure_grid_lines(warped, cfg) if cfg.grid_printed else None
    _, size = rect.rect_transform(cfg)
    data = {
        "lens": lens.to_dict(),
        "H_mm_to_px": _sig(view.H),
        "reproj_rms_mm": round(view.reproj_rms_mm, 3),
        "reproj_max_mm": round(view.reproj_max_mm, 3),
        "camera_mm": _round(view.camera_mm, 1),
        "thickness_mm": cfg.thickness_mm,
        "surface_scale": round(view.surface_scale, 6),
        "board_corners_px": _round(view.mm_to_px(cfg.board_corners_mm())),
        "rectified": {"px_per_mm": cfg.rectify_px_per_mm, "margin_mm": cfg.rectify_margin_mm, "size": list(size)},
        "edges": {name: _edge_dict(e) for name, e in edges.items()},
        "corner_angles_deg": [None if a is None else round(a, 3) for a in angles],
        "grid_lines": None if grid is None else {k: [None if v is None else round(v, 2) for v in vals] for k, vals in grid.items()},
    }
    msg = f"fit rms {view.reproj_rms_mm:.2f} mm, max {view.reproj_max_mm:.2f} mm, {lens.source} lens"
    if view.reproj_max_mm > cfg.max_reproj_mm:
        hint = ""
        if rect.swapped_fits_better(cfg, obs, lens, view):
            hint = "; board width and height look swapped in the config"
        return StageResult(ok=False, message=f"{msg} > {cfg.max_reproj_mm:g} mm{hint}", data=data)
    return StageResult(ok=True, message=msg, data=data)


# --- priority 3: bases -------------------------------------------------------

BASES_MARGIN_MM = 5.0  # table border around the board in the bases image


def bases_config(cfg: BoardConfig) -> BoardConfig:
    """Config of the top-down image for the circle fit: `bases.px_per_mm`, small border."""
    return dataclasses.replace(cfg, rectify_px_per_mm=cfg.bases_px_per_mm, rectify_margin_mm=BASES_MARGIN_MM)


def bases_image(frame: np.ndarray, view: rect.BoardView, cfg: BoardConfig) -> np.ndarray:
    return rect.warp_board(frame, view, bases_config(cfg))


def _board_region(cfg: BoardConfig, shape: tuple[int, ...]) -> np.ndarray:
    """Pixels of the bases image inside the board: where base centres can lie."""
    region = np.zeros(shape[:2], bool)
    (x0, y0), (x1, y1) = np.rint(rect.mm_to_rect(bases_config(cfg), [(0.0, 0.0), (cfg.width_mm, cfg.height_mm)])).astype(int)
    region[y0:y1, x0:x1] = True
    return region


def _draw_bases(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> None:
    from .overlay import draw_bases

    view = result.scratch.get("view")
    if view is not None:
        draw_bases(frame, result.scratch.get("bases", []), view.mm_to_px)


def _views_bases(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> dict[str, np.ndarray]:
    from .overlay import draw_bases

    img = result.scratch.get("rectified")
    if img is None:
        return {}
    img = img.copy()
    draw_bases(img, result.scratch.get("bases", []), lambda p: rect.mm_to_rect(cfg, p))
    return {"bases": img}


def _search_area(area_px: tuple[int, int, int, int], view: rect.BoardView, cfg: BoardConfig, shape: tuple[int, ...]) -> np.ndarray:
    """Pixels of the bases image inside a rectangle of the camera frame."""
    x0, y0, x1, y1 = area_px
    mm = view.px_to_mm(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(rect.mm_to_rect(bases_config(cfg), mm)).astype(np.int32), 1)
    return mask > 0


def _center_in(mask: np.ndarray, c: circles.Circle, cfg: BoardConfig) -> bool:
    x, y = np.rint(rect.mm_to_rect(bases_config(cfg), c.center_mm[None])[0]).astype(int)
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def detect_bases(
    img: np.ndarray,
    cfg: BoardConfig,
    reference: np.ndarray | None,
    search: BasesSearch | None = None,
    area: np.ndarray | None = None,
) -> tuple[list[circles.Circle], np.ndarray | None]:
    """Bases in the bases image `img` (`bases_image`). `reference` is the
    bases image of the empty board, or None. `search` and `area` (bool mask of
    `search.area_px`) limit the search. Returns (circles, foreground mask or
    None). Raises ValueError when the reference has another size."""
    region = _board_region(cfg, img.shape)
    if area is not None:
        region &= area
    fg = None
    if reference is not None:
        if reference.shape != img.shape:
            raise ValueError("empty-board reference has another size: set it again")
        fg = circles.foreground_mask(img, reference, cfg.bases_diff_threshold)
    params = circles.CircleParams.from_config(cfg, BASES_MARGIN_MM)
    if search is not None and search.diameters_mm is not None:
        params = dataclasses.replace(params, diameters_mm=search.diameters_mm)
    relaxed = search is not None and search.relaxed
    if relaxed:
        params = dataclasses.replace(
            params,
            min_coverage=cfg.bases_relaxed_min_coverage,
            min_contrast=cfg.bases_relaxed_min_contrast,
            rim_ratio=cfg.bases_relaxed_rim_ratio,
            fixed_radius=True,
        )
    # An old reference can hide the base: the relaxed search does not use it.
    found = circles.find_circles(img, params, region, None if relaxed else fg)
    if area is not None:  # the fit can move a centre out of the area
        found = [c for c in found if _center_in(area, c, cfg)]
    return (found[: cfg.bases_relaxed_max_found] if relaxed else found), fg


def changed_fraction(fg: np.ndarray | None, cfg: BoardConfig) -> float | None:
    """Part of the board that differs from the empty-board reference."""
    return None if fg is None else round(float(fg[_board_region(cfg, fg.shape)].mean()), 3)


@stage("bases", draw=_draw_bases, views=_views_bases)
def _bases(ctx: Context) -> StageResult:
    cfg = ctx.cfg
    view: rect.BoardView = ctx.scratch["view"]
    img = bases_image(ctx.frame, view, cfg)
    ref = ctx.cache.get("bases_reference")
    try:
        search: BasesSearch | None = ctx.scratch.get("bases_search")
        area = None if search is None or search.area_px is None else _search_area(search.area_px, view, cfg, img.shape)
        found, fg = detect_bases(img, cfg, ref, search, area)
    except ValueError as exc:
        return StageResult(ok=False, message=str(exc))
    ctx.scratch["bases"] = found
    ctx.scratch["bases_image"] = img
    mode = "reference" if ref is not None else "single_frame"
    centers_px = view.mm_to_px(np.array([c.center_mm for c in found])) if found else np.zeros((0, 2))
    data = {
        "mode": mode,
        "px_per_mm": cfg.bases_px_per_mm,
        "diameters_mm": list(cfg.bases_diameters_mm),
        "changed_fraction": changed_fraction(fg, cfg),
        "bases": [{**c.to_dict(), "center_px": _round(p)} for c, p in zip(found, centers_px)],
    }
    sizes = ", ".join(f"{n}x{d:g}" for d, n in sorted(Counter(c.diameter_mm for c in found).items()))
    msg = f"{len(found)} bases" + (f" ({sizes} mm)" if found else "") + ("" if ref is not None else ", no reference")
    return StageResult(ok=True, message=msg, data=data)


# --- priority 4: operative identity -----------------------------------------


def base_crops(img: np.ndarray, found: list[circles.Circle], cfg: BoardConfig) -> list[np.ndarray]:
    """Gallery crops (ident.store_crop) of the found bases from the bases image."""
    bcfg = bases_config(cfg)
    if not found:
        return []
    centers = rect.mm_to_rect(bcfg, np.array([c.center_mm for c in found]))
    return [idt.store_crop(img, p, c.diameter_mm, cfg.bases_px_per_mm) for c, p in zip(found, centers)]


def _draw_ident(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> None:
    from .overlay import draw_idents

    view = result.scratch.get("view")
    if view is not None and result.scratch.get("idents"):
        draw_idents(frame, result.scratch.get("bases", []), result.scratch["idents"], view.mm_to_px)


def _views_ident(frame: np.ndarray, result: FrameResult, cfg: BoardConfig) -> dict[str, np.ndarray]:
    from .overlay import draw_bases, draw_idents

    img = result.scratch.get("rectified")
    if img is None or not result.scratch.get("idents"):
        return {}
    img = img.copy()
    to_px = lambda p: rect.mm_to_rect(cfg, p)  # noqa: E731
    draw_bases(img, result.scratch.get("bases", []), to_px)
    draw_idents(img, result.scratch.get("bases", []), result.scratch["idents"], to_px)
    return {"ident": img}


@stage("ident", draw=_draw_ident, views=_views_ident)
def _ident(ctx: Context) -> StageResult:
    gallery: idt.Gallery | None = ctx.cache.get("gallery")
    found: list[circles.Circle] = ctx.scratch["bases"]
    crops = base_crops(ctx.scratch["bases_image"], found, ctx.cfg)
    ctx.scratch["base_crops"] = crops
    ref = ctx.cache.get("bases_reference")
    ctx.scratch["base_backgrounds"] = [] if ref is None else base_crops(ref, found, ctx.cfg)
    if gallery is None:
        return StageResult(ok=True, message="no gallery", data={"teams": None, "bases": []})
    data: dict[str, Any] = {"teams": sorted(gallery.team_dirs), "model": gallery.embedder.name, "gallery_photos": len(gallery), "bases": []}
    if len(gallery) == 0:
        return StageResult(ok=True, message="gallery has no photos: enroll the operatives", data=data)
    idents = idt.identify(crops, [c.diameter_mm for c in found], gallery, ctx.scratch["base_backgrounds"], [c.center_mm for c in found])
    ctx.scratch["idents"] = idents
    data["bases"] = [{"center_mm": _round(c.center_mm), "diameter_mm": c.diameter_mm, **x.to_dict()} for c, x in zip(found, idents)]
    named = sum(x.label is not None for x in idents)
    unsure = sum(x.label is not None and not x.sure for x in idents)
    data["masked"] = sum(x.masked for x in idents)
    msg = f"{named}/{len(idents)} named" + (f", {unsure} unsure" if unsure else "")
    if gallery.params.mask:
        msg += f", {data['masked']} masked" if gallery.can_mask else ", no mask: gallery photos without background"
    return StageResult(ok=True, message=msg, data=data)
