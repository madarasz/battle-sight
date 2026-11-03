"""AprilTag detection at the board corners, drift monitoring, lens model.

Priority 1: tags and drift. Priority 2: `Lens` (intrinsics and distortion,
from a ChArUco calibration in data/camera.toml, or a nominal pinhole) and the ChArUco target that
`battlesight calibrate` uses. The board homography is in `rectify.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import BoardConfig

if TYPE_CHECKING:
    from .camera import CameraSettings


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


# --- lens model (priority 2) ---------------------------------------------------

FULL_FRAME_DIAG_MM = 43.267  # 35 mm film diagonal: reference for "35 mm equivalent" focal length


@dataclass(frozen=True)
class Lens:
    """Pinhole camera matrix and OpenCV distortion coefficients for one image size.

    `source` is "calibrated" (from the [lens] table of the camera file) or "nominal" (focal
    length from the 35 mm equivalent, principal point at the image centre,
    no distortion).
    """

    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (k,) OpenCV order k1 k2 p1 p2 k3 ...
    size: tuple[int, int]  # (width, height) in pixels
    source: str
    rms_px: float | None = None

    @classmethod
    def nominal(cls, size: tuple[int, int], focal_35mm: float) -> Lens:
        w, h = size
        f = focal_35mm / FULL_FRAME_DIAG_MM * float(np.hypot(w, h))
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        return cls(K, np.zeros(5), (int(w), int(h)), "nominal")

    @classmethod
    def from_toml(cls, raw: dict) -> Lens:
        """The [lens] table of data/camera.toml."""
        return cls(
            np.array(raw["camera_matrix"], dtype=np.float64),
            np.array(raw["dist_coeffs"], dtype=np.float64).ravel(),
            (int(raw["image_size"][0]), int(raw["image_size"][1])),
            "calibrated",
            raw.get("rms_px"),
        )

    def to_toml(self) -> dict:
        raw = {
            "image_size": list(self.size),
            "camera_matrix": self.K.tolist(),
            "dist_coeffs": self.dist.tolist(),
        }
        if self.rms_px is not None:
            raw["rms_px"] = float(self.rms_px)
        return raw

    def scaled_to(self, size: tuple[int, int]) -> Lens | None:
        """Same lens for another image size with the same aspect ratio (for
        example 4K calibration, 1080p frames). None when the aspect differs."""
        sx, sy = size[0] / self.size[0], size[1] / self.size[1]
        if abs(sx - sy) > 0.005 * max(sx, sy):
            return None
        S = np.diag([sx, sy, 1.0])
        return Lens(S @ self.K, self.dist.copy(), (int(size[0]), int(size[1])), self.source, self.rms_px)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist) > 1e-9))

    def undistort_px(self, pts: np.ndarray) -> np.ndarray:
        """Image pixels -> pixels of an ideal pinhole camera with the same K."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if not self.has_distortion:
            return pts.reshape(-1, 2).copy()
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def distort_px(self, pts: np.ndarray) -> np.ndarray:
        """Inverse of `undistort_px`: ideal pinhole pixels -> image pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if not self.has_distortion:
            return pts.copy()
        n = np.linalg.solve(self.K, np.c_[pts, np.ones(len(pts))].T).T
        out, _ = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return out.reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "size": list(self.size),
            "fx": round(float(self.K[0, 0]), 2),
            "fy": round(float(self.K[1, 1]), 2),
            "cx": round(float(self.K[0, 2]), 2),
            "cy": round(float(self.K[1, 2]), 2),
            "dist": [round(float(d), 6) for d in self.dist],
            "rms_px": None if self.rms_px is None else round(float(self.rms_px), 3),
        }


def lens_for_frame(camera: CameraSettings, size: tuple[int, int]) -> Lens:
    """The calibrated lens when the camera settings have one that fits the frame
    size (same aspect ratio), otherwise the nominal lens."""
    if camera.lens is not None:
        lens = camera.lens.scaled_to(size)
        if lens is not None:
            return lens
    return Lens.nominal(size, camera.focal_35mm)


# --- ChArUco calibration target (priority 2) ------------------------------------


@dataclass(frozen=True)
class CharucoTarget:
    """Chessboard with ArUco markers in the white squares. Fits on A4 landscape.

    Uses a 5x5 ArUco dictionary, so its markers never pass as corner tags.
    """

    squares: tuple[int, int] = (10, 7)  # (columns, rows)
    square_mm: float = 25.0
    marker_mm: float = 18.0

    def board(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        return cv2.aruco.CharucoBoard(self.squares, self.square_mm, self.marker_mm, dictionary)

    @property
    def corner_count(self) -> int:
        return (self.squares[0] - 1) * (self.squares[1] - 1)


class CharucoFinder:
    """Find the inner chessboard corners of a `CharucoTarget` in a frame."""

    def __init__(self, target: CharucoTarget = CharucoTarget(), min_corners: int = 12):
        self.target = target
        self.min_corners = min_corners
        self._board = target.board()
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(object points (N, 3) in mm, image points (N, 2)), or None when
        fewer than `min_corners` corners are found."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self._detector.detectBoard(gray)
        if ids is None or len(ids) < self.min_corners:
            return None
        obj, img = self._board.matchImagePoints(corners, ids)
        return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64)


def calibrate_lens(views: list[tuple[np.ndarray, np.ndarray]], size: tuple[int, int], fix_k3: bool = True) -> Lens:
    """Calibrate intrinsics and distortion (k1 k2 p1 p2, and k3 unless
    `fix_k3`) from ChArUco views.

    Use 10 or more views: target at different places in the frame, some
    tilted by 20 to 40 deg. The result's `rms_px` is the reprojection error.
    k3 stays 0 by default: phone lenses have mild distortion, and a free k3
    bends the frame corners when the views do not reach them.
    """
    if len(views) < 3:
        raise ValueError(f"need at least 3 views, got {len(views)}")
    obj = [v[0].astype(np.float32) for v in views]
    img = [v[1].astype(np.float32).reshape(-1, 1, 2) for v in views]
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    return Lens(K.astype(np.float64), dist.ravel().astype(np.float64), (int(size[0]), int(size[1])), "calibrated", float(rms))
