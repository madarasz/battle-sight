"""AprilTag detection at the board corners and drift monitoring.

Priority 1 of the plan. Homography and lens calibration come in priority 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import BoardConfig


@dataclass
class TagObs:
    """One detected tag in image pixels."""

    tag_id: int
    corners: np.ndarray  # (4, 2) float64, order TL, TR, BR, BL in the tag's own frame
    center: np.ndarray  # (2,)
    side_px: float  # mean edge length in pixels


class TagDetector:
    """Detect AprilTag 36h11 markers and keep only the four corner tags."""

    def __init__(self, cfg: BoardConfig):
        self.cfg = cfg
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        # Large frames (4K) need larger adaptive threshold windows to cover a tag module.
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 45
        params.adaptiveThreshWinSizeStep = 6
        # Tags are small relative to a 4K frame; allow a small perimeter.
        params.minMarkerPerimeterRate = 0.01
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)
        self._wanted = set(cfg.tag_ids)

    def detect(self, frame: np.ndarray) -> dict[int, TagObs]:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        out: dict[int, TagObs] = {}
        if ids is None:
            return out
        for quad, tag_id in zip(corners, ids.flatten()):
            tag_id = int(tag_id)
            if tag_id not in self._wanted or tag_id in out:
                continue
            pts = quad.reshape(4, 2).astype(np.float64)
            edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            out[tag_id] = TagObs(
                tag_id=tag_id,
                corners=pts,
                center=pts.mean(axis=0),
                side_px=float(edges.mean()),
            )
        return out


def board_corners_px(cfg: BoardConfig, obs: dict[int, TagObs]) -> dict[int, np.ndarray]:
    """Board corner pixel positions derived from the detected tags.

    Tag k lies outside the board, diagonal to board corner k. Its own corner
    nearest the board is corner k + 2 (TL tag corner for the BR board corner,
    and so on). Push that corner away from the tag by the white margin along
    both tag edges. Returns {corner_index: (2,) array}.
    """
    ratio = cfg.tag_margin_mm / cfg.tag_size_mm
    out: dict[int, np.ndarray] = {}
    for tag_id, tag in obs.items():
        k = cfg.corner_index(tag_id)
        if k is None:
            continue
        c = tag.corners
        j = (k + 2) % 4
        e_next = c[(j + 1) % 4] - c[j]
        e_prev = c[(j + 3) % 4] - c[j]
        out[k] = c[j] - (e_next + e_prev) * ratio
    return out


class DriftStatus(Enum):
    NO_REFERENCE = "no reference"
    OK = "ok"
    MOVED = "moved"
    LOST = "lost"


@dataclass
class DriftReport:
    status: DriftStatus
    lost: list[int] = field(default_factory=list)  # tag ids missing too long
    moved: dict[int, float] = field(default_factory=dict)  # tag id -> shift in mm, over tolerance
    shift_mm: dict[int, float] = field(default_factory=dict)  # tag id -> shift in mm, all visible
    visible: int = 0
    settle_progress: float = 0.0  # 0..1 while the reference is locking

    @property
    def ok(self) -> bool:
        return self.status is DriftStatus.OK

    def summary(self) -> str:
        if self.status is DriftStatus.NO_REFERENCE:
            return f"locking reference {int(self.settle_progress * 100):3d}%  ({self.visible}/4 tags)"
        if self.status is DriftStatus.LOST:
            return "TAG LOST: " + ", ".join(str(i) for i in self.lost)
        if self.status is DriftStatus.MOVED:
            worst = max(self.moved.values())
            return "BOARD MOVED: " + ", ".join(f"tag {i} {mm:.1f} mm" for i, mm in self.moved.items()) + f"  (max {worst:.1f} mm)"
        worst = max(self.shift_mm.values()) if self.shift_mm else 0.0
        return f"4 corners OK  (max shift {worst:.2f} mm)"


class DriftMonitor:
    """Compare every frame's tags against a locked reference.

    - The reference locks on its own after `settle_frames` frames in a row with
      all four tags visible and still.
    - A tag counts as lost after `lost_after_frames` consecutive misses.
    - A tag counts as moved when its mean corner shift exceeds the tolerance.
    Pixel shift converts to mm with the tag's own apparent size.
    """

    def __init__(self, cfg: BoardConfig):
        self.cfg = cfg
        self.reference: dict[int, TagObs] | None = None
        self._miss = {tag_id: 0 for tag_id in cfg.tag_ids}
        self._settle: list[dict[int, TagObs]] = []

    def reset(self) -> None:
        self.reference = None
        self._settle.clear()
        for k in self._miss:
            self._miss[k] = 0

    def set_reference(self, obs: dict[int, TagObs]) -> None:
        if not all(tag_id in obs for tag_id in self.cfg.tag_ids):
            raise ValueError("reference needs all four tags visible")
        self.reference = {tag_id: obs[tag_id] for tag_id in self.cfg.tag_ids}
        self._settle.clear()

    def _shift_mm(self, ref: TagObs, now: TagObs) -> float:
        px_per_mm = ref.side_px / self.cfg.tag_size_mm
        return float(np.linalg.norm(now.corners - ref.corners, axis=1).mean() / px_per_mm)

    def _try_auto_lock(self, obs: dict[int, TagObs]) -> float:
        """Collect still, complete frames. Returns settle progress 0..1."""
        if not all(tag_id in obs for tag_id in self.cfg.tag_ids):
            self._settle.clear()
            return 0.0
        if self._settle:
            first = self._settle[0]
            jitter = max(self._shift_mm(first[i], obs[i]) for i in self.cfg.tag_ids)
            if jitter > self.cfg.drift_tolerance_mm * 0.5:
                self._settle.clear()
        self._settle.append(obs)
        n = self.cfg.settle_frames
        if len(self._settle) >= n:
            # Average the corners over the settle window for a steadier reference.
            ref: dict[int, TagObs] = {}
            for tag_id in self.cfg.tag_ids:
                stack = np.stack([f[tag_id].corners for f in self._settle])
                mean = stack.mean(axis=0)
                edges = np.linalg.norm(np.roll(mean, -1, axis=0) - mean, axis=1)
                ref[tag_id] = TagObs(tag_id, mean, mean.mean(axis=0), float(edges.mean()))
            self.reference = ref
            self._settle.clear()
            return 1.0
        return len(self._settle) / n

    def update(self, obs: dict[int, TagObs]) -> DriftReport:
        for tag_id in self.cfg.tag_ids:
            self._miss[tag_id] = 0 if tag_id in obs else self._miss[tag_id] + 1
        lost = [i for i in self.cfg.tag_ids if self._miss[i] >= self.cfg.lost_after_frames]
        visible = sum(1 for i in self.cfg.tag_ids if i in obs)

        if self.reference is None:
            progress = self._try_auto_lock(obs)
            if self.reference is None:
                return DriftReport(DriftStatus.NO_REFERENCE, lost=lost, visible=visible, settle_progress=progress)

        shift = {i: self._shift_mm(self.reference[i], obs[i]) for i in self.cfg.tag_ids if i in obs}
        moved = {i: mm for i, mm in shift.items() if mm > self.cfg.drift_tolerance_mm}
        if lost:
            status = DriftStatus.LOST
        elif moved:
            status = DriftStatus.MOVED
        else:
            status = DriftStatus.OK
        return DriftReport(status, lost=lost, moved=moved, shift_mm=shift, visible=visible, settle_progress=1.0)
