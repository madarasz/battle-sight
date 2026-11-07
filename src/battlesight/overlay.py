"""Drawing helpers. All functions draw in place on a BGR frame."""

from __future__ import annotations

import cv2
import numpy as np

from .calib import DriftReport, DriftStatus, TagObs
from .config import CORNER_NAMES, BoardConfig

GREEN = (80, 220, 80)
YELLOW = (40, 200, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 220, 60)

_STATUS_COLOR = {
    DriftStatus.NO_REFERENCE: YELLOW,
    DriftStatus.OK: GREEN,
    DriftStatus.MOVED: YELLOW,
    DriftStatus.LOST: RED,
}


def _scale(frame: np.ndarray) -> float:
    """Line and font scale relative to a 1080p frame."""
    return max(0.6, frame.shape[1] / 1920.0)


def draw_tags(frame: np.ndarray, obs: dict[int, TagObs], cfg: BoardConfig, report: DriftReport | None = None) -> None:
    s = _scale(frame)
    thick = max(1, round(2 * s))
    for tag_id, tag in obs.items():
        color = GREEN
        if report is not None and tag_id in report.moved:
            color = YELLOW
        pts = tag.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, thick, cv2.LINE_AA)
        c = tuple(int(v) for v in tag.center)
        k = cfg.corner_index(tag_id)
        label = f"{tag_id}" if k is None else f"{tag_id} {CORNER_NAMES[k]}"
        cv2.putText(frame, label, (c[0] + int(8 * s), c[1] - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(frame, label, (c[0] + int(8 * s), c[1] - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, color, thick, cv2.LINE_AA)
        # Tag corner 0 (its own top-left) in a different colour: shows tag orientation.
        cv2.circle(frame, tuple(int(v) for v in tag.corners[0]), max(2, int(4 * s)), CYAN, -1, cv2.LINE_AA)
    if report is not None:
        for tag_id in report.lost:
            # Nothing to draw at the tag itself, it is gone. Show where the reference was.
            pass


def draw_board_corners(frame: np.ndarray, corners: dict[int, np.ndarray], reference: dict[int, np.ndarray] | None = None) -> None:
    s = _scale(frame)
    thick = max(1, round(2 * s))
    r = int(10 * s)
    for k, p in corners.items():
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.line(frame, (x - r, y), (x + r, y), WHITE, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y - r), (x, y + r), WHITE, thick, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, WHITE, 1, cv2.LINE_AA)
    if len(corners) == 4:
        quad = np.array([corners[k] for k in range(4)], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [quad], True, GREEN, 1, cv2.LINE_AA)
    if reference:
        for k, p in reference.items():
            if k in corners:
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(frame, (x, y), r, RED, thick, cv2.LINE_AA)
            cv2.putText(frame, "lost", (x + r, y + r), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, RED, thick, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, report: DriftReport, extra: str = "") -> None:
    draw_text_banner(frame, report.summary(), _STATUS_COLOR[report.status], extra)


def draw_text_banner(frame: np.ndarray, text: str, color: tuple[int, int, int], extra: str = "") -> None:
    s = _scale(frame)
    h = int(44 * s)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), BLACK, -1)
    cv2.rectangle(frame, (0, 0), (int(12 * s), h), color, -1)
    cv2.putText(frame, text, (int(24 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, color, max(1, round(2 * s)), cv2.LINE_AA)
    if extra:
        (tw, _), _ = cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, 1)
        cv2.putText(frame, extra, (frame.shape[1] - tw - int(16 * s), int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, WHITE, 1, cv2.LINE_AA)


def fit_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def grid_polylines_mm(cfg: BoardConfig, step_mm: float = 10.0) -> list[np.ndarray]:
    """Grid lines and board outline as point lists in board mm. Each line is
    sampled every `step_mm`, so it bends with lens distortion when drawn in the
    camera frame."""
    xs, ys = cfg.grid_lines_mm()
    m, W, H = cfg.grid_margin_mm, cfg.width_mm, cfg.height_mm

    def seg(a, b):
        n = max(2, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / step_mm)) + 1)
        return np.linspace(a, b, n)

    lines = [seg((x, m), (x, H - m)) for x in xs] + [seg((m, y), (W - m, y)) for y in ys]
    return lines


def board_outline_mm(cfg: BoardConfig, step_mm: float = 10.0) -> np.ndarray:
    c = cfg.board_corners_mm()
    pts = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step_mm)))
        pts.append(np.linspace(a, b, n, endpoint=False))
    return np.concatenate(pts)


def draw_grid(frame: np.ndarray, cfg: BoardConfig, to_px, color: tuple[int, int, int] = YELLOW, outline: tuple[int, int, int] | None = CYAN) -> None:
    """Draw the board grid (and outline) with `to_px`: board mm (N, 2) -> pixels (N, 2)."""
    s = _scale(frame)
    thick = max(1, round(1.5 * s))
    for line in grid_polylines_mm(cfg):
        pts = np.rint(to_px(line)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, thick, cv2.LINE_AA)
    if outline is not None:
        pts = np.rint(to_px(board_outline_mm(cfg))).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, outline, thick, cv2.LINE_AA)


def draw_bases(frame: np.ndarray, bases, to_px, color: tuple[int, int, int] = GREEN) -> None:
    """Draw found bases (circles.Circle) with `to_px`: board mm (N, 2) -> pixels (N, 2).
    The outline is the base bottom. The label is the diameter class in mm."""
    s = _scale(frame)
    thick = max(1, round(2 * s))
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for c in bases:
        r = 0.5 * c.diameter_mm
        ring = c.center_mm + r * np.c_[np.cos(a), np.sin(a)]
        pts = np.rint(to_px(ring)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, thick, cv2.LINE_AA)
        center = np.rint(to_px(c.center_mm.reshape(1, 2))[0]).astype(int)
        cv2.drawMarker(frame, (int(center[0]), int(center[1])), color, cv2.MARKER_CROSS, int(8 * s), thick, cv2.LINE_AA)
        top = np.rint(to_px((c.center_mm + (0.0, -r - 2.0)).reshape(1, 2))[0]).astype(int)
        label = f"{c.diameter_mm:g}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, thick)
        org = (int(top[0] - tw / 2), int(top[1]))
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, color, thick, cv2.LINE_AA)


def draw_dashed_circle(
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    dashes: int = 24,
    duty: float = 0.5,
) -> None:
    """Dashed circle in pixels: `dashes` arcs, each `duty` of its step long.
    A small duty with many dashes gives a dotted circle."""
    c = (int(round(center[0])), int(round(center[1])))
    r = max(1, int(round(radius)))
    step = 360.0 / dashes
    for k in range(dashes):
        a0 = k * step
        cv2.ellipse(frame, c, (r, r), 0, a0, a0 + duty * step, color, thickness, cv2.LINE_AA)
