"""Board and tag configuration loaded from a TOML file."""

from __future__ import annotations

import dataclasses
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
    # Play surface height above the table (the tag cards lie on the table).
    thickness_mm: float = 0.0
    # Grid: lines start `grid_margin_mm` in from each edge. With `grid_sections`
    # (nx, ny) the space inside the margin splits into equal sections (Tomb
    # World 7 x 6). Without it, lines repeat every `grid_pitch_mm` (Volkus 1 in).
    grid_margin_mm: float = 0.0
    grid_sections: tuple[int, int] | None = None
    grid_pitch_mm: float = 25.4
    grid_printed: bool = False  # the grid is printed on the board: rectify checks its lines
    # Rectified top-down image.
    rectify_px_per_mm: float = 2.0
    rectify_margin_mm: float = 50.0
    max_reproj_mm: float = 2.5  # rectify fails when the tag corner fit is worse
    # Bases: known diameters at the base bottom, found by a circle fit in a
    # top-down image at `bases_px_per_mm` (see circles.py for the other values).
    bases_diameters_mm: tuple[float, ...] = (25.0, 28.0, 32.0, 40.0, 50.0)
    bases_px_per_mm: float = 4.0
    bases_diameter_tol_mm: float = 3.0
    bases_min_coverage: float = 0.5
    bases_min_contrast: float = 12.0
    bases_rim_ratio: float = 0.75
    bases_diff_threshold: float = 25.0
    bases_min_foreground: float = 0.25
    bases_reference_min_coverage: float = 0.4  # min_coverage with a reference and known team sizes
    # Second search of the enroll mode, when the normal search finds no base.
    bases_relaxed_min_coverage: float = 0.3
    bases_relaxed_min_contrast: float = 6.0
    bases_relaxed_rim_ratio: float = 0.9
    bases_relaxed_max_found: int = 5
    # Operative identity (ident.py): crop around each base, DINOv2 embedding,
    # nearest operatives in the enrolled gallery.
    ident_model: str = "facebook/dinov2-small"
    ident_crop_pad_mm: float = 25.0
    ident_input_px: int = 224
    ident_rotations: int = 8
    ident_top_k: int = 3
    ident_min_score: float = 0.5
    ident_min_margin: float = 0.03
    ident_mask: bool = False
    ident_mask_threshold: float = 25.0
    ident_mask_min_base: float = 0.25
    ident_mask_max_fraction: float = 0.6
    # Terrain footprints on the board (terrain.py, map files in data/maps/).
    terrain_wall_mm: float = 4.0  # wall thickness
    terrain_pillar_mm: float = 13.0  # pillar square side
    terrain_wall_end_mm: float = 6.5  # half pillar: pillar side across the wall, this along it
    terrain_hatchway_width_mm: float = 22.0  # along the wall
    terrain_hatchway_depth_mm: float = 10.0  # across the wall
    terrain_breach_long_mm: float = 35.0  # breach point width on a 2-grid wall (A1)
    terrain_breach_short_mm: float = 28.0  # breach point width on a 1-grid wall (B2)
    terrain_objective_mm: float = 40.0  # objective marker diameter
    terrain_objective_range_mm: float = 25.4  # dashed circle this far from the marker edge
    terrain_breach_walls: tuple[str, ...] = ("A1", "B2")  # walls with a breach point; the other walls have hatchways
    terrain_no_middle_pillar: tuple[str, ...] = ("A2",)  # A walls with no pillar at the middle
    terrain_teleporter_mm: float = 60.0  # teleporter pad diameter

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> BoardConfig:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        board, tags, drift = raw["board"], raw["tags"], raw.get("drift", {})
        grid, rect, bases = raw.get("grid", {}), raw.get("rectify", {}), raw.get("bases", {})
        ident, terrain = raw.get("ident", {}), raw.get("terrain", {})
        if "camera" in raw:
            raise ValueError("[camera] moved from the board config to data/camera.toml (focal_35mm, lens calibration)")
        d = {f.name: f.default for f in dataclasses.fields(cls)}  # a key that the file leaves out gets the default above
        sections = grid.get("sections")
        if sections is not None and (len(sections) != 2 or min(sections) < 1):
            raise ValueError("grid.sections must hold two positive counts [nx, ny]")
        diameters = tuple(float(v) for v in bases.get("diameters_mm", d["bases_diameters_mm"]))
        if not diameters or min(diameters) <= 0:
            raise ValueError("bases.diameters_mm must hold positive diameters")
        ids = tuple(int(i) for i in tags["ids"])
        if len(ids) != 4 or len(set(ids)) != 4:
            raise ValueError("tags.ids must hold four distinct tag ids")

        def table(prefix: str, raw_table: dict, skip: tuple[str, ...] = ()) -> dict:
            """Fields `<prefix>_<key>` from the TOML table, with the type of the default."""
            out = {}
            for name, default in d.items():
                key = name[len(prefix) + 1 :]
                if name.startswith(prefix + "_") and name not in skip:
                    out[name] = type(default)(raw_table.get(key, default))
            return out

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
            thickness_mm=float(board.get("thickness_mm", d["thickness_mm"])),
            grid_margin_mm=float(grid.get("margin_mm", d["grid_margin_mm"])),
            grid_sections=None if sections is None else (int(sections[0]), int(sections[1])),
            grid_pitch_mm=float(grid.get("pitch_mm", d["grid_pitch_mm"])),
            grid_printed=bool(grid.get("printed", d["grid_printed"])),
            max_reproj_mm=float(rect.get("max_reproj_mm", d["max_reproj_mm"])),
            bases_diameters_mm=tuple(sorted(diameters)),
            **table("rectify", rect),
            **table("bases", bases, skip=("bases_diameters_mm",)),
            **table("ident", ident),
            **table("terrain", terrain),
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

        Each card lies outside the board, diagonal to its corner: the card's
        inner corner touches the board corner, and the tag sits one margin
        further out. Returns {tag_id: (4, 2) array} with the tag's own corner
        order TL, TR, BR, BL, as the tag lies upright next to the board.
        """
        out: dict[int, np.ndarray] = {}
        m, s = self.tag_margin_mm, self.tag_size_mm
        for k, (tag_id, corner) in enumerate(zip(self.tag_ids, self.board_corners_mm())):
            inward = _INWARD[k]
            near = corner - inward * m
            far = corner - inward * (m + s)
            x0, x1 = sorted((near[0], far[0]))
            y0, y1 = sorted((near[1], far[1]))
            out[tag_id] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        return out

    def grid_lines_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Grid line positions: (x of the vertical lines, y of the horizontal lines).

        With sections, the lines include both margin lines. With a pitch, they
        start at the margin and repeat up to the far margin.
        """
        m = self.grid_margin_mm
        out = []
        for i, size in enumerate((self.width_mm, self.height_mm)):
            if self.grid_sections is not None:
                out.append(np.linspace(m, size - m, self.grid_sections[i] + 1))
            else:
                out.append(np.arange(m, size - m + 1e-6, self.grid_pitch_mm))
        return out[0], out[1]
