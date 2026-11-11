"""Operative identity (priority 4): name each found base from a photo gallery.

Identity is separate from detection. The bases stage finds the bases with
geometry. This module looks at the crop around each base and finds the
operative in an enrolled gallery. A new team needs gallery photos only, never
a retrained detector.

1. Crop: a square around the base in the top-down bases image (the same image
   the circle fit uses, `bases.px_per_mm`), base diameter plus
   `ident.crop_pad_mm` on each side. The model body leans out of the base by
   parallax, so the pad must hold it. Pixels outside the inscribed circle are
   set to grey, so a rotated crop looks the same as an upright one.
2. Embedding: DINOv2 (`ident.model`, Hugging Face name) on the crop at
   `ident.input_px`. The feature is the class token and the mean patch token,
   each L2 normalised, joined, and normalised again. No training.
3. Gallery: `data/teams/<team>/<operative>/*.png`, crops saved by the
   enroll command or by a fix in the ident command, and `team.toml` with the
   base size of each operative. Each crop is embedded at `ident.rotations`
   angles, because the models turn on the board. The embeddings are cached in
   `<team>/embeddings.npz`.
4. Match: the score of an operative is the mean cosine similarity of its
   `ident.top_k` best gallery embeddings. Only operatives with the size class
   of the base can match. Each operative is on the board at most once, so a
   one-to-one assignment (Hungarian) picks the labels. A base whose score is
   under `ident.min_score` stays unknown.
5. Fixes: the user clicks a base and picks the right name. The crop goes into
   the gallery, so the next match is better.

Labels are "<team>/<operative>", so two teams can play on one board.
"""

from __future__ import annotations

import json
import re
import time
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from . import circles
from .config import BoardConfig
from .measure import base_at

DEFAULT_GALLERY_ROOT = Path(__file__).resolve().parents[2] / "data" / "teams"
TEAM_FILE = "team.toml"
CACHE_FILE = "embeddings.npz"
# Saved crops hold this much board around the base on each side, so a later
# change of `ident.crop_pad_mm` (up to this value) does not need new photos.
STORE_PAD_MM = 40.0
MASK_GRAY = 127
PREPARE_VERSION = 2  # add 1 when `prepare` changes: the cached embeddings are old then
CROP_SUFFIXES = {".png", ".jpg", ".jpeg"}
BACKGROUND_DIR = "bg"  # subfolder of an operative folder: the gallery scan skips folders
MASK_CLOSE_MM = 3.0  # gaps up to this size in the foreground are closed
MASK_OTHER_MARGIN_MM = 2.0  # removed around another found base
_BAD_NAME = re.compile(r"[/\\:]|^\.|^$")


@dataclass(frozen=True)
class IdentParams:
    model: str = "facebook/dinov2-small"
    crop_pad_mm: float = 25.0
    input_px: int = 224
    rotations: int = 8
    top_k: int = 3
    min_score: float = 0.5
    min_margin: float = 0.03
    store_px_per_mm: float = 4.0  # scale of the saved crops (`bases.px_per_mm`)
    mask: bool = False
    mask_threshold: float = 25.0
    mask_min_base: float = 0.25
    mask_max_fraction: float = 0.6

    @classmethod
    def from_config(cls, cfg: BoardConfig) -> IdentParams:
        return cls(
            model=cfg.ident_model,
            crop_pad_mm=cfg.ident_crop_pad_mm,
            input_px=cfg.ident_input_px,
            rotations=cfg.ident_rotations,
            top_k=cfg.ident_top_k,
            min_score=cfg.ident_min_score,
            min_margin=cfg.ident_min_margin,
            store_px_per_mm=cfg.bases_px_per_mm,
            mask=cfg.ident_mask,
            mask_threshold=cfg.ident_mask_threshold,
            mask_min_base=cfg.ident_mask_min_base,
            mask_max_fraction=cfg.ident_mask_max_fraction,
        )

    def cache_key(self, embedder: str) -> str:
        key = f"{embedder}|pad {self.crop_pad_mm:g}|px {self.input_px}|rot {self.rotations}|store {self.store_px_per_mm:g}|prepare {PREPARE_VERSION}"
        return key + (f"|mask {self.mask_threshold:g} {self.mask_min_base:g} {self.mask_max_fraction:g}" if self.mask else "")


# --- crops --------------------------------------------------------------------


def store_crop(img: np.ndarray, center_px: Sequence[float], diameter_mm: float, px_per_mm: float) -> np.ndarray:
    """Square crop of `img` (top-down bases image at `px_per_mm`) around a base,
    with `STORE_PAD_MM` of board on each side. The table outside the image is grey."""
    side = int(round((diameter_mm + 2 * STORE_PAD_MM) * px_per_mm))
    M = np.array([[1.0, 0.0, side / 2 - center_px[0]], [0.0, 1.0, side / 2 - center_px[1]]])
    return cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(MASK_GRAY,) * 3)


def prepare(crop: np.ndarray, diameter_mm: float, p: IdentParams, angle_deg: float = 0.0) -> np.ndarray:
    """Model input from a stored crop (`store_crop`): the centre square of side
    diameter + 2 * crop_pad_mm, turned by `angle_deg`, at `input_px`, grey
    outside the inscribed circle."""
    side = crop.shape[1]
    stored_mm = diameter_mm + 2 * STORE_PAD_MM
    want_mm = diameter_mm + 2 * min(p.crop_pad_mm, STORE_PAD_MM)
    want_px = side * want_mm / stored_mm
    scale = p.input_px / want_px
    c = (side / 2.0, side / 2.0)
    # warpAffine cannot average pixels (it treats INTER_AREA as linear): turn
    # at the full size, then shrink with resize.
    size = int(round(want_px)) if scale < 1 else p.input_px
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0 if scale < 1 else scale)
    M[:, 2] += (size / 2.0 - c[0], size / 2.0 - c[1])
    out = cv2.warpAffine(crop, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(MASK_GRAY,) * 3)
    if size != p.input_px:
        out = cv2.resize(out, (p.input_px, p.input_px), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[: p.input_px, : p.input_px]
    r = p.input_px / 2.0
    out[(xx + 0.5 - r) ** 2 + (yy + 0.5 - r) ** 2 > r * r] = MASK_GRAY
    return out


def mask_crop(
    crop: np.ndarray,
    background: np.ndarray,
    diameter_mm: float,
    p: IdentParams,
    others: Sequence[tuple[float, float, float]] = (),
) -> np.ndarray | None:
    """`crop` with grey on all pixels that are not this operative, from the
    difference to `background` (the same square of the empty board). `others`
    are the other bases near it: (dx mm, dy mm, diameter mm) from this base.
    None when the difference does not look like one operative (wrong empty
    board, changed light): the caller then uses the full crop."""
    side = crop.shape[1]
    px_per_mm = side / (diameter_mm + 2 * STORE_PAD_MM)
    centre = (side // 2, side // 2)
    radius = int(round(0.5 * diameter_mm * px_per_mm))
    disc = cv2.circle(np.zeros(crop.shape[:2], np.uint8), centre, radius, 1, -1)

    fg = circles.foreground_mask(crop, background, p.mask_threshold).astype(np.uint8)
    if fg[disc > 0].mean() < p.mask_min_base:
        return None
    k = int(round(MASK_CLOSE_MM * px_per_mm)) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    def cut_others(mask: np.ndarray) -> None:
        for dx, dy, d in others:
            at = (int(round(centre[0] + dx * px_per_mm)), int(round(centre[1] + dy * px_per_mm)))
            # The margin also removes the ring that the blur of `foreground_mask` leaves around a base.
            cv2.circle(mask, at, int(round((0.5 * d + MASK_OTHER_MARGIN_MM) * px_per_mm)), 0, -1)
        mask[disc > 0] = 1  # the base is known from the circle fit, also where it looks like the board

    cut_others(fg)
    _, parts = cv2.connectedComponents(fg, connectivity=8)
    keep = (parts == parts[centre[1], centre[0]]).astype(np.uint8)
    outline, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(keep, outline, -1, 1, cv2.FILLED)
    cut_others(keep)  # the fill must not bring back a base inside the outline

    view_mm = diameter_mm + 2 * min(p.crop_pad_mm, STORE_PAD_MM)
    view = cv2.circle(np.zeros(crop.shape[:2], np.uint8), centre, int(round(0.5 * view_mm * px_per_mm)), 1, -1)
    if keep[view > 0].mean() > p.mask_max_fraction:
        return None
    out = crop.copy()
    out[keep == 0] = MASK_GRAY
    return out


def background_path(crop_path: Path) -> Path:
    """Empty-board crop of a gallery photo: same square of the board, from the
    reference in use when the photo was taken. The file may not exist."""
    crop_path = Path(crop_path)
    return crop_path.parent / BACKGROUND_DIR / crop_path.name


def rotation_angles(p: IdentParams) -> list[float]:
    return [360.0 * k / max(1, p.rotations) for k in range(max(1, p.rotations))]


# --- embedders ----------------------------------------------------------------


class Embedder(Protocol):
    name: str

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """BGR uint8 images, all the same size -> (n, D) float32, L2-normalised rows."""
        ...


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class ColorEmbedder:
    """Hue-saturation and lightness histograms. No ML: for tests and as a
    rough fallback (`ident.model = "color-hist"`)."""

    name = "color-hist"

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for img in images:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Leave the grey mask out: it is the same in every crop.
            mask = (np.abs(img.astype(np.int16) - MASK_GRAY).max(axis=2) > 2).astype(np.uint8)
            hs = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).ravel()
            v = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).ravel()
            out.append(np.sqrt(np.concatenate([hs, v])))
        return _normalise(np.array(out)) if out else np.zeros((0, 56), np.float32)


class DinoEmbedder:
    """DINOv2 through Hugging Face `transformers`. Loads on first use (the
    first run downloads the weights), on Apple MPS when available."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model: str = "facebook/dinov2-small", device: str | None = None, batch: int = 32):
        self.name = model
        self._device = device
        self._batch = batch
        self._model = None

    def _load(self):
        import torch
        from transformers import AutoModel

        if self._device is None:
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:  # cached weights first: no network check at the table
            model = AutoModel.from_pretrained(self.name, local_files_only=True)
        except OSError:
            model = AutoModel.from_pretrained(self.name)  # first run: download
        self._model = model.eval().to(self._device)
        return self._model

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        import torch

        if not images:
            return np.zeros((0, 0), np.float32)
        model = self._model or self._load()
        feats = []
        for i in range(0, len(images), self._batch):
            rgb = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images[i : i + self._batch]]).astype(np.float32) / 255.0
            x = torch.from_numpy(((rgb - self.MEAN) / self.STD).transpose(0, 3, 1, 2).copy()).to(self._device)
            with torch.no_grad():
                h = model(pixel_values=x).last_hidden_state
            cls_tok = torch.nn.functional.normalize(h[:, 0], dim=-1)
            patch = torch.nn.functional.normalize(h[:, 1:].mean(dim=1), dim=-1)
            feats.append(torch.cat([cls_tok, patch], dim=-1).cpu().numpy())
        return _normalise(np.concatenate(feats))


def make_embedder(model: str) -> Embedder:
    return ColorEmbedder() if model == ColorEmbedder.name else DinoEmbedder(model)


# --- gallery ------------------------------------------------------------------


class DuplicatePhoto(ValueError):
    """The crop is already a gallery photo of this operative (`path`)."""

    def __init__(self, path: Path):
        super().__init__(f"{path.name} is the same photo")
        self.path = path


def same_photo(a: np.ndarray, b: np.ndarray) -> bool:
    """Two stored crops with the same pixels: the same frame, cut at the same
    place (a name fix done twice on one image). Two frames of a camera are
    never the same, because of the sensor noise."""
    return a.shape == b.shape and bool(np.array_equal(a, b))


def check_name(name: str, what: str = "name") -> str:
    name = name.strip()
    if _BAD_NAME.search(name):
        raise ValueError(f"{what} {name!r}: must not be empty, start with '.', or hold / \\ :")
    return name


def list_teams(root: str | Path = DEFAULT_GALLERY_ROOT) -> list[str]:
    """Team folders in the gallery root: folders with a team.toml or with
    operative folders. Sorted. Empty when the root does not exist."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if (d / TEAM_FILE).exists() or any(q.is_dir() and not q.name.startswith(".") for q in d.iterdir()):
            out.append(d.name)
    return out


def label_of(team: str, operative: str) -> str:
    return f"{team}/{operative}"


def split_label(label: str) -> tuple[str, str]:
    team, _, op = label.partition("/")
    return team, op


@dataclass
class Operative:
    team: str
    name: str
    base_mm: float | None = None
    wounds: int | None = None  # maximum wounds (W), None when team.toml has no value

    @property
    def label(self) -> str:
        return label_of(self.team, self.name)


def check_wounds(value, where: str) -> int:
    """A maximum wounds value: a whole number of 1 or more. Else a ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: wounds must be a whole number of 1 or more, not {value!r}")
    return value


def read_team_file(team_dir: Path) -> dict[str, tuple[float | None, int | None]]:
    """{operative name: (base diameter in mm, maximum wounds)} from
    `team.toml`, None for a value that is not there. A bad wounds value is a
    ValueError."""
    path = team_dir / TEAM_FILE
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    out: dict[str, tuple[float | None, int | None]] = {}
    for name, v in raw.get("operatives", {}).items():
        v = v if isinstance(v, dict) else {}
        base, wounds = v.get("base_mm"), v.get("wounds")
        out[str(name)] = (None if base is None else float(base), None if wounds is None else check_wounds(wounds, f"{path} {name}"))
    return out


def load_team_file(team_dir: Path) -> dict[str, float | None]:
    """{operative name: base diameter in mm or None} from `team.toml`."""
    return {name: base for name, (base, _wounds) in read_team_file(team_dir).items()}


def team_operatives(team: str, team_dir: Path) -> list[Operative]:
    """Operatives of a team, sorted: `team.toml` entries and operative
    folders. Needs no photos and no model."""
    listed = read_team_file(team_dir)
    names = set(listed)
    if team_dir.is_dir():
        names |= {q.name for q in team_dir.iterdir() if q.is_dir() and not q.name.startswith(".")}
    return [Operative(team, name, *listed.get(name, (None, None))) for name in sorted(names)]


def roster_sizes(root: str | Path, teams: Iterable[str]) -> tuple[float, ...] | None:
    """Base sizes of the operatives of the teams, sorted, from the team files
    (no photos, no model). None when there is no operative, or when an
    operative has no known size: then all sizes are possible."""
    sizes = [op.base_mm for t in teams for op in team_operatives(t, Path(root) / t)]
    if not sizes or any(s is None for s in sizes):
        return None
    return tuple(sorted(set(sizes)))  # type: ignore[arg-type]


def save_team_file(team_dir: Path, team: str, operatives: dict[str, float | None], wounds: dict[str, int | None] | None = None) -> None:
    """Write `team.toml`: base size and maximum wounds of each operative."""
    wounds = wounds or {}
    lines = [
        f"# Battle Sight gallery of team {team!r}. One table per operative.",
        "# base_mm: base diameter (one of [bases] diameters_mm). Only bases of this size can get this name.",
        "# wounds: maximum wounds (W) of the operative, for the health bar of the live app.",
        "# Photos: one folder per operative next to this file, written by `battlesight enroll` and by fixes.",
        "",
    ]
    for name in sorted(operatives):
        lines.append(f"[operatives.{json.dumps(name)}]")
        if operatives[name] is not None:
            lines.append(f"base_mm = {operatives[name]:g}")
        if wounds.get(name) is not None:
            lines.append(f"wounds = {wounds[name]:d}")
        lines.append("")
    team_dir.mkdir(parents=True, exist_ok=True)
    (team_dir / TEAM_FILE).write_text("\n".join(lines))


class Gallery:
    """Enrolled crops and their embeddings, for one or more teams.

    `view_emb` has one row per crop and rotation, `view_op` the operative
    index of each row, `view_file` the crop index (for leave-one-out).
    `masked_emb`, `masked_op`, `masked_file` are the same for the masked crops,
    only for the photos with a usable mask, empty when `params.mask` is off.
    A cached photo holds the raw rows, then the masked rows (NaN = no mask)."""

    def __init__(self, team_dirs: Iterable[str | Path], embedder: Embedder, params: IdentParams):
        self.embedder = embedder
        self.params = params
        self.team_dirs = {Path(d).name: Path(d) for d in team_dirs}
        if len(self.team_dirs) == 0:
            raise ValueError("a gallery needs at least one team folder")
        self.operatives: list[Operative] = []
        self.files: list[Path] = []
        self.file_op: list[int] = []
        self.view_emb = np.zeros((0, 0), np.float32)
        self.view_op = np.zeros(0, int)
        self.view_file = np.zeros(0, int)
        self.masked_emb = np.zeros((0, 0), np.float32)
        self.masked_op = np.zeros(0, int)
        self.masked_file = np.zeros(0, int)
        self.hidden: set[int] = set()  # file indexes that `scores` must not use (the benchmark hides copies of a test photo)
        self.reload()

    @classmethod
    def for_teams(cls, teams: Iterable[str], cfg: BoardConfig, root: str | Path = DEFAULT_GALLERY_ROOT, embedder: Embedder | None = None) -> Gallery:
        p = IdentParams.from_config(cfg)
        return cls([Path(root) / check_name(t, "team") for t in teams], embedder or make_embedder(p.model), p)

    # operatives

    def index(self, label: str) -> int | None:
        for i, op in enumerate(self.operatives):
            if op.label == label:
                return i
        return None

    def labels(self) -> list[str]:
        return [op.label for op in self.operatives]

    def counts(self) -> dict[str, int]:
        c = {op.label: 0 for op in self.operatives}
        for i in self.file_op:
            c[self.operatives[i].label] += 1
        return c

    def __len__(self) -> int:
        return len(self.files)

    def ensure_operative(self, team: str, name: str, base_mm: float | None, wounds: int | None = None) -> Operative:
        """The operative, created (and written to team.toml) when new. Raises
        ValueError when it exists with another base size. `wounds` (maximum
        wounds) sets or replaces the value in team.toml."""
        if team not in self.team_dirs:
            raise ValueError(f"team {team!r} is not loaded, loaded: {sorted(self.team_dirs)}")
        name = check_name(name, "operative")
        if wounds is not None:
            check_wounds(wounds, label_of(team, name))
        i = self.index(label_of(team, name))
        if i is not None:
            op = self.operatives[i]
            if base_mm is not None and op.base_mm is not None and op.base_mm != base_mm:
                raise ValueError(f"{op.label} has a {op.base_mm:g} mm base, not {base_mm:g} mm")
            changed = False
            if op.base_mm is None and base_mm is not None:
                op.base_mm, changed = base_mm, True
            if wounds is not None and op.wounds != wounds:
                op.wounds, changed = wounds, True
            if changed:
                self._save_team(team)
            return op
        op = Operative(team, name, base_mm, wounds)
        self.operatives.append(op)
        self._save_team(team)
        return op

    def _save_team(self, team: str) -> None:
        ops = [op for op in self.operatives if op.team == team]
        save_team_file(self.team_dirs[team], team, {op.name: op.base_mm for op in ops}, {op.name: op.wounds for op in ops})

    # loading and the embedding cache

    def reload(self) -> None:
        ops: list[Operative] = []
        files: list[Path] = []
        file_op: list[int] = []
        for team, d in self.team_dirs.items():
            for op in team_operatives(team, d):
                ops.append(op)
                folder = d / op.name
                if folder.is_dir():
                    for f in sorted(folder.iterdir()):
                        if f.suffix.lower() in CROP_SUFFIXES:
                            files.append(f)
                            file_op.append(len(ops) - 1)
        self.operatives, self.files, self.file_op = ops, files, file_op
        self._embed_all()

    def _diameter(self, op: Operative, crop: np.ndarray) -> float:
        if op.base_mm is not None:
            return op.base_mm
        # Unknown size: get it from the crop side at the configured scale.
        return max(1.0, crop.shape[1] / self.params.store_px_per_mm - 2 * STORE_PAD_MM)

    def _embed_crop(self, crop: np.ndarray, op: Operative, background: np.ndarray | None = None) -> np.ndarray:
        d = self._diameter(op, crop)
        angles = rotation_angles(self.params)
        raw = self.embedder.embed([prepare(crop, d, self.params, a) for a in angles])
        if not self.params.mask:
            return raw
        cut = None if background is None else mask_crop(crop, background, d, self.params)
        masked = np.full_like(raw, np.nan) if cut is None else self.embedder.embed([prepare(cut, d, self.params, a) for a in angles])
        return np.concatenate([raw, masked])

    def _saved_background(self, crop_path: Path, crop: np.ndarray) -> np.ndarray | None:
        path = background_path(crop_path)
        if not self.params.mask or not path.is_file():
            return None
        bg = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return bg if bg is not None and bg.shape == crop.shape else None

    def _embed_all(self) -> None:
        key = self.params.cache_key(self.embedder.name)
        per_file: list[np.ndarray | None] = [None] * len(self.files)
        for team, d in self.team_dirs.items():
            cached = _load_cache(d / CACHE_FILE, key)
            for i, f in enumerate(self.files):
                if self.operatives[self.file_op[i]].team != team:
                    continue
                hit = cached.get(_cache_name(d, f))
                if hit is not None and hit[0] == _mtime(f):
                    per_file[i] = hit[1]
        dirty = {self.operatives[self.file_op[i]].team for i, e in enumerate(per_file) if e is None}
        for i, e in enumerate(per_file):
            if e is None:
                crop = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
                if crop is None:
                    raise ValueError(f"{self.files[i]}: cannot read gallery crop")
                per_file[i] = self._embed_crop(crop, self.operatives[self.file_op[i]], self._saved_background(self.files[i], crop))
        self._set_views([e for e in per_file if e is not None])
        for team in dirty:
            self._save_cache(team, key, per_file)

    def _set_views(self, per_file: list[np.ndarray]) -> None:
        n = self._view_count()
        self.view_emb, self.view_op, self.view_file = self._stack_views([(i, e[:n]) for i, e in enumerate(per_file)])
        self.masked_emb, self.masked_op, self.masked_file = self._stack_views([(i, e[n:]) for i, e in enumerate(per_file) if self._has_mask(e)])
        self._per_file = per_file

    def _view_count(self) -> int:
        return len(rotation_angles(self.params))

    def _has_mask(self, emb: np.ndarray) -> bool:
        rows = emb[self._view_count() :]
        return len(rows) > 0 and bool(np.isfinite(rows).all())

    def _stack_views(self, views: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not views:
            return np.zeros((0, 0), np.float32), np.zeros(0, int), np.zeros(0, int)
        emb = np.concatenate([e for _, e in views]).astype(np.float32)
        op = np.concatenate([np.full(len(e), self.file_op[i]) for i, e in views])
        file = np.concatenate([np.full(len(e), i) for i, e in views])
        return emb, op, file

    @property
    def can_mask(self) -> bool:
        """Masked crops can be compared: each operative with photos has a
        masked photo. Else the masked scores of the operatives do not compare."""
        return self.params.mask and len(self.masked_emb) > 0 and set(self.file_op) <= set(self.masked_op.tolist())

    def _save_cache(self, team: str, key: str, per_file: list) -> None:
        d = self.team_dirs[team]
        idx = [i for i, f in enumerate(self.files) if self.operatives[self.file_op[i]].team == team]
        if not idx:
            return
        np.savez(
            d / CACHE_FILE,
            key=np.array(key),
            names=np.array([_cache_name(d, self.files[i]) for i in idx]),
            mtimes=np.array([_mtime(self.files[i]) for i in idx]),
            emb=np.stack([per_file[i] for i in idx]),
        )

    def add(self, label: str, crop: np.ndarray, base_mm: float | None, background: np.ndarray | None = None) -> Path:
        """Save `crop` (`store_crop`) as a new gallery photo of `label` and add
        its embeddings. Creates the operative when new. `background` is the
        same square of the empty board, kept at `background_path`, not embedded.
        Returns the file. DuplicatePhoto when the operative has this photo
        already: nothing is saved."""
        team, name = split_label(label)
        op = self.ensure_operative(team, name, base_mm)
        twin = self.find_duplicate(op.label, crop)
        if twin is not None:
            raise DuplicatePhoto(twin)
        folder = self.team_dirs[team] / op.name
        folder.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S")
        path = folder / f"{stem}.png"
        n = 1
        while path.exists():
            path = folder / f"{stem}_{n}.png"
            n += 1
        cv2.imwrite(str(path), crop)
        if background is not None:
            bg = background_path(path)
            bg.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(bg), background)
        emb = self._embed_crop(crop, op, background)
        self.files.append(path)
        self.file_op.append(self.index(op.label))  # type: ignore[arg-type]
        self._set_views(self._per_file + [emb])
        self._save_cache(team, self.params.cache_key(self.embedder.name), self._per_file)
        return path

    def find_duplicate(self, label: str, crop: np.ndarray) -> Path | None:
        """Gallery photo of `label` with the same pixels as `crop` (`same_photo`)."""
        j = self.index(label)
        for f, k in zip(self.files, self.file_op):
            if k != j:
                continue
            old = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if old is not None and same_photo(old, crop):
                return f
        return None

    def remove(self, path: Path) -> None:
        """Delete a gallery photo (undo of `add`)."""
        path = Path(path)
        i = self.files.index(path)
        team = self.operatives[self.file_op[i]].team
        path.unlink(missing_ok=True)
        background_path(path).unlink(missing_ok=True)
        del self.files[i], self.file_op[i]
        per_file = self._per_file[:i] + self._per_file[i + 1 :]
        self._set_views(per_file)
        self._save_cache(team, self.params.cache_key(self.embedder.name), per_file)

    # scores

    def scores(self, emb: np.ndarray, exclude_file: int | None = None, masked: bool = False) -> np.ndarray:
        """(n, n_operatives) score: mean of the `top_k` best cosine similarities
        to each operative's gallery views (the masked views with `masked`).
        -inf for an operative without views."""
        views, view_op, view_file = (self.masked_emb, self.masked_op, self.masked_file) if masked else (self.view_emb, self.view_op, self.view_file)
        n, m = len(emb), len(self.operatives)
        out = np.full((n, m), -np.inf)
        if n == 0 or len(views) == 0:
            return out
        sim = np.asarray(emb, np.float32) @ views.T
        for j in range(m):
            cols = view_op == j
            if exclude_file is not None:
                cols &= view_file != exclude_file
            if self.hidden:
                cols &= ~np.isin(view_file, list(self.hidden))
            if cols.any():
                out[:, j] = _top_photos(sim[:, cols], view_file[cols], self.params.top_k)
        return out

    def leave_one_out(self) -> tuple[int, int, list[tuple[str, str]]]:
        """Each gallery photo against the rest of the gallery (size limit on,
        no assignment). Returns (right, total, [(true, guessed) of the misses])."""
        right, misses = 0, []
        n = self._view_count()
        for i, emb in enumerate(self._per_file):
            masked = self.can_mask and self._has_mask(emb)
            sc = self.scores(emb[n : n + 1] if masked else emb[:1], exclude_file=i, masked=masked)[0]
            op = self.operatives[self.file_op[i]]
            allowed = self.allowed(op.base_mm)
            sc = np.where(allowed, sc, -np.inf)
            if not np.isfinite(sc).any():
                continue
            guess = self.operatives[int(np.argmax(sc))].label
            if guess == op.label:
                right += 1
            else:
                misses.append((op.label, guess))
        return right, right + len(misses), misses

    def base_sizes(self) -> tuple[float, ...] | None:
        """Base sizes of the operatives, sorted. None when there is no
        operative, or one without a known size (`roster_sizes`)."""
        sizes = [op.base_mm for op in self.operatives]
        if not sizes or any(s is None for s in sizes):
            return None
        return tuple(sorted(set(sizes)))  # type: ignore[arg-type]

    def allowed(self, diameter_mm: float | None) -> np.ndarray:
        """Operatives that can stand on a base of this diameter."""
        return np.array([op.base_mm is None or diameter_mm is None or op.base_mm == diameter_mm for op in self.operatives], bool)


def _top_photos(sim: np.ndarray, files: np.ndarray, k: int) -> np.ndarray:
    """Per row: best rotation of each photo (a crop matches one angle only),
    then the mean of the k best photos."""
    uniq = np.unique(files)
    best = np.stack([sim[:, files == f].max(axis=1) for f in uniq], axis=1)
    best = np.sort(best, axis=1)[:, ::-1]
    return best[:, : min(k, best.shape[1])].mean(axis=1)


def _cache_name(team_dir: Path, f: Path) -> str:
    return f.relative_to(team_dir).as_posix()


def _mtime(f: Path) -> float:
    return round(f.stat().st_mtime, 3)


def _load_cache(path: Path, key: str) -> dict[str, tuple[float, np.ndarray]]:
    if not path.exists():
        return {}
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) != key:
            return {}
        return {str(n): (float(t), e) for n, t, e in zip(z["names"], z["mtimes"], z["emb"])}
    except (OSError, KeyError, ValueError):
        return {}


# --- matching -----------------------------------------------------------------


@dataclass
class Ident:
    """Identity of one base. `label` is None when unknown. `candidates` are
    the allowed operatives with their scores, best first (the fix menu)."""

    label: str | None
    score: float
    margin: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    sure: bool = False  # named, and the margin is at least `min_margin` (or confirmed)
    confirmed: bool = False  # the user picked or accepted this label
    masked: bool = False  # the masked crop was compared with the masked gallery

    def to_dict(self) -> dict:
        def num(v: float) -> float | None:
            return round(float(v), 4) if np.isfinite(v) else None

        return {
            "label": self.label,
            "score": num(self.score),
            "margin": num(self.margin),
            "sure": self.sure,
            "confirmed": self.confirmed,
            "candidates": [[lab, round(float(s), 4)] for lab, s in self.candidates[:5]],
        }


def assign(scores: np.ndarray, allowed: np.ndarray, labels: list[str], min_score: float, min_margin: float, unique: bool = True) -> list[Ident]:
    """Labels for n bases from an (n, m) score matrix and an (n, m) allowed
    mask. With `unique` each operative goes to at most one base (Hungarian).
    `margin` is the assigned score minus the best other allowed score; it is
    negative when the assignment took the second choice, and below
    `min_margin` the label counts as unsure (shown in yellow)."""
    from scipy.optimize import linear_sum_assignment

    n, m = scores.shape
    s = np.where(allowed & np.isfinite(scores), scores, -np.inf)
    pick = np.full(n, -1)
    if n and m:
        if unique:
            big = 1e3
            cost = np.where(np.isfinite(s), -s, big)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if np.isfinite(s[r, c]):
                    pick[r] = c
        else:
            best = np.argmax(s, axis=1)
            pick = np.where(np.isfinite(s[np.arange(n), best]), best, -1)
    out = []
    for i in range(n):
        order = [j for j in np.argsort(-s[i]) if np.isfinite(s[i, j])]
        cands = [(labels[j], float(s[i, j])) for j in order]
        j = int(pick[i])
        if j < 0 or s[i, j] < min_score:
            best = cands[0][1] if cands else -np.inf
            out.append(Ident(None, best, -np.inf, cands))
            continue
        other = max((s[i, k] for k in order if k != j), default=-np.inf)
        margin = float(s[i, j] - other) if np.isfinite(other) else float("inf")
        out.append(Ident(labels[j], float(s[i, j]), margin, cands, sure=margin >= min_margin))
    return out


def identify(
    crops: list[np.ndarray],
    diameters_mm: list[float],
    gallery: Gallery,
    backgrounds: Sequence[np.ndarray] = (),
    centers_mm: Sequence[Sequence[float]] = (),
) -> list[Ident]:
    """Identity of each base from its stored crop (`store_crop`) and its size
    class. With `backgrounds` (empty-board crops) and centres and a gallery
    that `can_mask`, a base with a usable mask compares masked crops; all other
    bases compare full crops."""
    p = gallery.params
    if not crops:
        return []
    cut: list[np.ndarray | None] = [None] * len(crops)
    if gallery.can_mask and len(backgrounds) == len(crops) and len(centers_mm) == len(crops):
        centres = np.asarray(centers_mm, float)
        for i, (c, b, d) in enumerate(zip(crops, backgrounds, diameters_mm)):
            others = [(*(centres[j] - centres[i]), diameters_mm[j]) for j in range(len(crops)) if j != i]
            cut[i] = mask_crop(c, b, d, p, others)
    emb = gallery.embedder.embed([prepare(c if m is None else m, d, p) for c, m, d in zip(crops, cut, diameters_mm)])
    masked = np.array([m is not None for m in cut], bool)
    scores = np.full((len(crops), len(gallery.operatives)), -np.inf)
    scores[~masked] = gallery.scores(emb[~masked])
    scores[masked] = gallery.scores(emb[masked], masked=True)
    allowed = np.stack([gallery.allowed(d) for d in diameters_mm]) if gallery.operatives else np.zeros((len(crops), 0), bool)
    out = assign(scores, allowed, gallery.labels(), p.min_score, p.min_margin)
    for x, m in zip(out, masked):
        x.masked = bool(m)
    return out


# --- user fixes ---------------------------------------------------------------


@dataclass
class LabelSession:
    """Click logic of the ident command, no window calls. Holds the bases of
    one snapshot (center and diameter in board mm), their crops and identities.
    `pick(at_mm)` selects the base under the mouse, `choose(label)` sets its
    name and adds its crop to the gallery, `undo` removes the last added photo."""

    centers_mm: np.ndarray
    diameters_mm: list[float]
    crops: list[np.ndarray]
    idents: list[Ident]
    gallery: Gallery
    selected: int | None = None
    added: list[tuple[int, Path, Ident]] = field(default_factory=list)
    backgrounds: list[np.ndarray] = field(default_factory=list)  # empty: no reference

    def base_at(self, at_mm: Sequence[float]) -> int | None:
        """Base whose disc (plus 5 mm) holds the point, nearest centre first."""
        return base_at(self.centers_mm, self.diameters_mm, at_mm)

    def pick(self, at_mm: Sequence[float]) -> int | None:
        self.selected = self.base_at(at_mm)
        return self.selected

    def menu(self, i: int | None = None) -> list[str]:
        """Names for the fix menu of base i: allowed operatives, best score
        first, then the allowed operatives without gallery photos."""
        i = self.selected if i is None else i
        if i is None:
            return []
        names = [lab for lab, _ in self.idents[i].candidates]
        allowed = self.gallery.allowed(self.diameters_mm[i])
        names += [op.label for k, op in enumerate(self.gallery.operatives) if allowed[k] and op.label not in names]
        return names

    def choose(self, label: str, i: int | None = None) -> Path | None:
        """Set the name of base i (default: the selected base) and add its crop
        to the gallery. A label that another base holds moves to this base;
        the other base becomes unknown. Returns the new gallery photo, or None
        when the gallery has this photo already (the name is set, no file)."""
        i = self.selected if i is None else i
        if i is None:
            raise ValueError("no base selected")
        try:
            path = self.gallery.add(label, self.crops[i], self.diameters_mm[i], self.backgrounds[i] if self.backgrounds else None)
        except DuplicatePhoto:
            path = None
        self.added.append((i, path, self.idents[i]))
        for k, x in enumerate(self.idents):
            if k != i and x.label == label:
                self.idents[k] = Ident(None, x.score, -np.inf, x.candidates)
        old = self.idents[i]
        self.idents[i] = Ident(label, old.score, old.margin, old.candidates, sure=True, confirmed=True)
        self.selected = None
        return path

    def add(self, center_mm: Sequence[float], diameter_mm: float, crop: np.ndarray, ident: Ident, background: np.ndarray | None = None) -> int:
        """Add a base that the detector missed (the user clicked it). It
        becomes the last base; returns its index. `background` is needed
        when the other bases have one."""
        self.centers_mm = np.vstack([np.asarray(self.centers_mm, float).reshape(-1, 2), np.asarray(center_mm, float)[None]])
        self.diameters_mm.append(float(diameter_mm))
        self.crops.append(crop)
        self.idents.append(ident)
        if self.backgrounds or background is not None:
            self.backgrounds.append(background if background is not None else crop)
        return len(self.idents) - 1

    def accept_all(self) -> list[Path | None]:
        """Confirm every named, not yet confirmed base and add its crop to the
        gallery. One entry per base: the new photo, or None when the gallery
        has it already."""
        out = []
        for i, x in enumerate(self.idents):
            if x.label is not None and not x.confirmed:
                out.append(self.choose(x.label, i))
        return out

    def remove(self, i: int) -> list[Path]:
        """Remove base i from the session: a false base (terrain, board print).
        The later bases move one index down. The gallery photos that a choice
        of this session added for this base are removed too; returns them."""
        removed = [path for k, path, _ in self.added if k == i and path is not None]
        for path in removed:
            self.gallery.remove(path)
        self.centers_mm = np.delete(np.asarray(self.centers_mm, float).reshape(-1, 2), i, axis=0)
        for items in (self.diameters_mm, self.crops, self.idents, self.backgrounds):
            if i < len(items):
                del items[i]
        self.added = [(k - (k > i), path, old) for k, path, old in self.added if k != i]
        if self.selected is not None:
            self.selected = None if self.selected == i else self.selected - (self.selected > i)
        return removed

    def undo(self) -> tuple[bool, Path | None]:
        """Take back the last `choose`: (done, removed gallery photo). The
        photo is None when that choice added no file."""
        if not self.added:
            return False, None
        i, path, old = self.added.pop()
        if path is not None:
            self.gallery.remove(path)
        self.idents[i] = old
        return True, path


# --- done test ----------------------------------------------------------------


@dataclass
class IdentTruth:
    """Hand-checked names of the bases in a photo (`<name>.ident.json`)."""

    teams: list[str]
    bases: list[tuple[np.ndarray, str]]  # (center_mm, label)
    tol_mm: float = 10.0

    @classmethod
    def load(cls, path: str | Path) -> IdentTruth:
        raw = json.loads(Path(path).read_text())
        bases = [(np.array(b["center_mm"], float), str(b["label"])) for b in raw["bases"]]
        return cls([str(t) for t in raw["teams"]], bases, float(raw.get("tol_mm", 10.0)))

    def save(self, path: str | Path) -> None:
        """JSON text with one base per line, so a diff shows which base changed."""
        rows = [json.dumps({"center_mm": [round(float(v), 2) for v in c], "label": label}, ensure_ascii=False) for c, label in self.bases]
        body = "[\n" + ",\n".join(f"    {r}" for r in rows) + "\n  ]" if rows else "[]"
        Path(path).write_text(f'{{\n  "teams": {json.dumps(self.teams, ensure_ascii=False)},\n  "tol_mm": {self.tol_mm:g},\n  "bases": {body}\n}}\n')


def score_truth(truth: IdentTruth, centers_mm: np.ndarray, idents: list[Ident]) -> dict:
    """Compare the first-try labels with the truth. A truth base matches the
    nearest found base within `tol_mm`. Missed bases count as wrong. Done test
    of priority 4: at least 9 of 10 right."""
    right, wrong, missed = 0, [], []
    used: set[int] = set()
    for c, label in truth.bases:
        if len(centers_mm):
            d = np.linalg.norm(np.asarray(centers_mm) - c, axis=1)
            d[list(used)] = np.inf
            i = int(np.argmin(d))
        if not len(centers_mm) or d[i] > truth.tol_mm:
            missed.append(label)
            continue
        used.add(i)
        if idents[i].label == label:
            right += 1
        else:
            wrong.append((label, idents[i].label))
    total = len(truth.bases)
    acc = right / total if total else 1.0
    return {"right": right, "total": total, "accuracy": round(acc, 3), "wrong": wrong, "missed": missed, "done_ok": acc >= 0.9}
