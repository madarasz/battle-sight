"""Single-frame analysis pipeline. No camera, no window, no Qt.

A frame goes through an ordered list of stages. Each stage reads the frame and
the results of earlier stages, and returns a `StageResult`. When a stage fails,
the stages after it are skipped, because they depend on it.

Add a priority as a new stage:

    @stage("rectify", draw=_draw_rectify)
    def _rectify(ctx: Context) -> StageResult:
        corners = ctx.results["corners"].data["board_corners_px"]
        ...

The `image` subcommand, the tests, and a later FastAPI front end all call
`analyze` and `annotate`. Everything in `StageResult.data` must be JSON-safe
(plain numbers, strings, lists, dicts).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calib import TagDetector, TagObs, board_corners_px
from .config import CORNER_NAMES, BoardConfig

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
    results: dict[str, StageResult] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class Stage:
    name: str
    run: RunFn
    draw: DrawFn | None = None


STAGES: dict[str, Stage] = {}  # in run order


def stage(name: str, draw: DrawFn | None = None) -> Callable[[RunFn], RunFn]:
    """Register a stage. Stages run in the order they are registered."""

    def deco(fn: RunFn) -> RunFn:
        if name in STAGES:
            raise ValueError(f"stage {name!r} already registered")
        STAGES[name] = Stage(name, fn, draw)
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
    """Runs the selected stages on frames. Keeps the cache between frames."""

    def __init__(self, cfg: BoardConfig, until: str | None = None):
        self.cfg = cfg
        self.names = select_stages(until)
        self.cache: dict[str, Any] = {}

    def analyze(self, frame: np.ndarray, source: str = "") -> FrameResult:
        ctx = Context(frame, self.cfg, self.cache)
        failed = False
        for name in self.names:
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

    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Copy of the frame with every stage's overlay and a status banner."""
        from .overlay import GREEN, RED, draw_text_banner

        out = frame.copy()
        for name, r in result.stages.items():
            draw = STAGES[name].draw
            if draw is not None and not r.skipped:
                draw(out, result, self.cfg)
        draw_text_banner(out, result.summary(), GREEN if result.ok else RED, f"{result.width}x{result.height}")
        return out


def analyze(frame: np.ndarray, cfg: BoardConfig, until: str | None = None, source: str = "") -> FrameResult:
    """One-shot helper. Use `Pipeline` for many frames."""
    return Pipeline(cfg, until).analyze(frame, source)


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
