"""Board and tag configuration loaded from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "data" / "board.toml"

CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")

# Unit vectors that point from each board corner into the board (x right, y down).
_INWARD = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)


@dataclass(frozen=True)
class BoardConfig:
    name: str
    width_mm: float
    height_mm: float
    tag_ids: tuple[int, int, int, int]
    tag_size_mm: float
    tag_margin_mm: float
    drift_tolerance_mm: float
    lost_after_frames: int
    settle_frames: int

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> BoardConfig:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        board, tags, drift = raw["board"], raw["tags"], raw.get("drift", {})
        ids = tuple(int(i) for i in tags["ids"])
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("tags.ids must hold four distinct tag ids")
        return cls(
            name=str(board.get("name", "board")),
            width_mm=float(board["width_mm"]),
            height_mm=float(board["height_mm"]),
            tag_ids=ids,  # type: ignore[arg-type]
            tag_size_mm=float(tags["size_mm"]),
            tag_margin_mm=float(tags.get("margin_mm", 0.0)),
            drift_tolerance_mm=float(drift.get("tolerance_mm", 2.0)),
            lost_after_frames=int(drift.get("lost_after_frames", 3)),
            settle_frames=int(drift.get("settle_frames", 10)),
        )

    def corner_index(self, tag_id: int) -> int | None:
        """Board corner index (0..3) that this tag marks, or None."""
        try:
            return self.tag_ids.index(tag_id)
        except ValueError:
            return None

    def board_corners_mm(self) -> np.ndarray:
        """Board corners in board coordinates, shape (4, 2), order TL, TR, BR, BL."""
        w, h = self.width_mm, self.height_mm
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    def tag_corners_mm(self) -> dict[int, np.ndarray]:
        """Black-square corners of every tag in board coordinates.

        Returns {tag_id: (4, 2) array} with the tag's own corner order
        TL, TR, BR, BL, as the tag lies upright on the board.
        """
        out: dict[int, np.ndarray] = {}
        m, s = self.tag_margin_mm, self.tag_size_mm
        for k, (tag_id, corner) in enumerate(zip(self.tag_ids, self.board_corners_mm())):
            inward = _INWARD[k]
            near = corner + inward * m
            far = corner + inward * (m + s)
            x0, x1 = sorted((near[0], far[0]))
            y0, y1 = sorted((near[1], far[1]))
            out[tag_id] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        return out
