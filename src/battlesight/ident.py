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
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Sequence
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
_BAD_NAME = re.compile(r"[/\\:#]|^\.|^$")


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
    reuse_move_mm: float = 2.0  # a base that moved less than this may keep its last scores
    reuse_crop_diff: float = 0.03  # ... when its crop changed less than this (mean grey difference, 0..1); 0 = never reuse

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
            reuse_move_mm=cfg.ident_reuse_move_mm,
            reuse_crop_diff=cfg.ident_reuse_crop_diff,
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


Progress = Callable[[int, int, str], None]  # (steps done, all steps, text of the next step)


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class ColorEmbedder:
    """Hue-saturation and lightness histograms. No ML: for tests and as a
    rough fallback (`ident.model = "color-hist"`)."""

    name = "color-hist"
    loaded = True  # nothing to load

    def load(self, input_px: int = 0, progress: Progress | None = None) -> None:
        """Nothing to load: for the same calls as `DinoEmbedder`."""

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
    first run downloads the weights), on Apple MPS when available. `load`
    loads it before the first use (the live app does it at start, on a
    thread of its own). A lock makes the load happen once: a second thread
    waits for the first."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    LOAD_STEPS = ("import torch", "import transformers", "load weights", "move to device", "warm up")

    def __init__(self, model: str = "facebook/dinov2-small", device: str | None = None, batch: int = 32):
        self.name = model
        self._device = device
        self._batch = batch
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self, input_px: int = 0, progress: Progress | None = None) -> None:
        """Load the model. `progress(done, total, text)` is called before
        each step of `LOAD_STEPS` and once at the end. With `input_px`, a
        first run on a blank crop of that size warms up the device (the
        first run on Apple MPS is slow)."""
        total = len(self.LOAD_STEPS)

        def step(k: int) -> None:
            if progress is not None:
                progress(k, total, self.LOAD_STEPS[k] if k < total else "done")

        with self._lock:
            if self._model is None:
                self._load(step)
            step(total - 1)
            if input_px > 0:
                self._run(self._model, [np.zeros((input_px, input_px, 3), np.uint8)])
            step(total)

    def _load(self, step: Callable[[int], None] = lambda _k: None):
        step(0)
        import torch

        step(1)
        from transformers import AutoModel

        if self._device is None:
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        step(2)
        try:  # cached weights first: no network check at the table
            model = AutoModel.from_pretrained(self.name, local_files_only=True)
        except OSError:
            model = AutoModel.from_pretrained(self.name)  # first run: download
        step(3)
        self._model = model.eval().to(self._device)
        return self._model

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), np.float32)
        model = self._model
        if model is None:
            with self._lock:  # another thread can be loading it
                model = self._model or self._load()
        return self._run(model, images)

    def _run(self, model, images: list[np.ndarray]) -> np.ndarray:
        import torch

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


_EMBEDDERS: dict[str, Embedder] = {}
_EMBEDDERS_LOCK = threading.Lock()


def make_embedder(model: str) -> Embedder:
    """One embedder per model name in a process: a new gallery (for example
    after a team change) uses the model that is loaded already."""
    with _EMBEDDERS_LOCK:
        if model not in _EMBEDDERS:
            _EMBEDDERS[model] = ColorEmbedder() if model == ColorEmbedder.name else DinoEmbedder(model)
        return _EMBEDDERS[model]


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
        raise ValueError(f"{what} {name!r}: must not be empty, start with '.', or hold / \\ : #")
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
    """(team, operative name) of a type label or a copy label (the copy
    number stays on the name: use `type_of` first for the plain name)."""
    team, _, op = label.partition("/")
    return team, op


COPY_SEP = "#"  # separates the copy number in a copy label: <team>/<operative>#<k>


def copy_label(type_label: str, k: int) -> str:
    """Label of copy k (1, 2, ...) of an operative that a team can field more than once."""
    return f"{type_label}{COPY_SEP}{k:d}"


def type_of(label: str) -> str:
    """Type label of a copy label; a type label returns itself."""
    return label.partition(COPY_SEP)[0]


def copy_number(label: str) -> int | None:
    """Copy number of a copy label, None for a type label."""
    _, sep, k = label.partition(COPY_SEP)
    return int(k) if sep and k.isdigit() else None


def display_name(label: str) -> str:
    """Name for the screen: "Boy #2" for a copy label, "Boy" for a type label."""
    name = split_label(type_of(label))[1]
    k = copy_number(label)
    return name if k is None else f"{name} #{k}"


@dataclass
class Operative:
    """An operative type of a team file. `max` = copies that a team can field
    (`max` in team.toml, default 1). The live app makes one Operative for each
    copy in play with `copy` set; a gallery operative has no `copy`."""

    team: str
    name: str
    base_mm: float | None = None
    wounds: int | None = None  # maximum wounds (W), None when team.toml has no value
    expendable: bool = False  # `expendable = true` in team.toml
    max: int = 1  # copies that a team can field
    copy: int | None = None  # copy number of one copy in play (only with max > 1)
    counts_as: str | None = None  # name of the original operative (`counts_as` in team.toml): a variant model with its own photos; base size, wounds, `expendable` and `max` are the original's
    targets: tuple[str, ...] = ()  # all names of `counts_as`, in file order; `counts_as` is the one in use (`Gallery.out_of_play`)

    def __post_init__(self) -> None:
        if not self.targets and self.counts_as is not None:
            self.targets = (self.counts_as,)

    @property
    def original_label(self) -> str:
        """Type label of the operative that this one counts as (its own type label when it is no variant)."""
        return self.type_label if self.counts_as is None else label_of(self.team, self.counts_as)

    @property
    def type_label(self) -> str:
        return label_of(self.team, self.name)

    @property
    def label(self) -> str:
        """Copy label for a copy, else the type label."""
        return self.type_label if self.copy is None else copy_label(self.type_label, self.copy)

    @property
    def display(self) -> str:
        return display_name(self.label)


def check_wounds(value, where: str) -> int:
    """A maximum wounds value: a whole number of 1 or more. Else a ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: wounds must be a whole number of 1 or more, not {value!r}")
    return value


APPROVED_OPS_FILE = Path(__file__).resolve().parents[2] / "data" / "approved-ops-2025.toml"


def load_archetypes(path: Path = APPROVED_OPS_FILE) -> tuple[str, ...]:
    """Archetype names: the keys of `[tac_ops]` in the approved ops file."""
    with path.open("rb") as fh:
        return tuple(tomllib.load(fh)["tac_ops"])


ARCHETYPES = load_archetypes()


@dataclass
class TeamInfo:
    """Team-level values of `team.toml`: killteam archetypes (a team has one
    or more), an optional faction counter text and an optional count of
    operatives (a number, or a range)."""

    archetypes: tuple[str, ...] = ()
    faction_counter: str | None = None
    operative_count: tuple[int, int] | None = None  # (min, max) count of operatives


def check_max(value, where: str) -> int:
    """A maximum count of copies: a whole number of 1 or more. Else a ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: max must be a whole number of 1 or more, not {value!r}")
    return value


def check_archetypes(values, where: str) -> tuple[str, ...]:
    """Archetypes: names from ARCHETYPES, no repeats. Else a ValueError."""
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{where}: archetype must be a list of {list(ARCHETYPES)}, not {values!r}")
    for v in values:
        if v not in ARCHETYPES:
            raise ValueError(f"{where}: archetype {v!r} is not one of {list(ARCHETYPES)}")
    if len(set(values)) != len(values):
        raise ValueError(f"{where}: archetype has a repeated value: {list(values)}")
    return tuple(values)


def parse_count_range(value, where: str) -> tuple[int, int]:
    """A whole number of 1 or more (5) or a range ("4-6" or "4..6"): (min, max).
    Else a ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{where}: count must be a whole number or a range, not {value!r}")
    if isinstance(value, int):
        lo = hi = value
    elif isinstance(value, str) and (m := re.fullmatch(r"\s*(\d+)\s*(?:-|\.\.)?\s*(\d+)?\s*", value)) and not (
        m.group(2) is None and re.search(r"-|\.\.", value)
    ):
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    else:
        raise ValueError(f"{where}: count must be a whole number or a range like 4-6, not {value!r}")
    if lo < 1 or hi < lo:
        raise ValueError(f"{where}: count must be 1 or more and the range must go up, not {value!r}")
    return lo, hi


def read_team_info(team_dir: Path) -> TeamInfo:
    """`archetype` and `faction-counter` of `team.toml` (empty when missing).
    A bad value is a ValueError."""
    path = team_dir / TEAM_FILE
    if not path.exists():
        return TeamInfo()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    counter = raw.get("faction-counter")
    if counter is not None and not isinstance(counter, str):
        raise ValueError(f"{path}: faction-counter must be a string, not {counter!r}")
    count = raw.get("operative-count")
    return TeamInfo(
        check_archetypes(raw.get("archetype", []), str(path)),
        counter or None,
        None if count is None else parse_count_range(count, f"{path} operative-count"),
    )


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
        if COPY_SEP in str(name):
            raise ValueError(f"{path} {name}: an operative name must not hold {COPY_SEP!r}")
        v = v if isinstance(v, dict) else {}
        base, wounds = v.get("base_mm"), v.get("wounds")
        out[str(name)] = (None if base is None else float(base), None if wounds is None else check_wounds(wounds, f"{path} {name}"))
    return out


def read_counts_as(team_dir: Path) -> dict[str, tuple[str, ...]]:
    """{variant name: original names} of the operatives with `counts_as` (a
    name, or a list of names) in `team.toml`. Each original must be an
    operative of the file that is no variant itself, and the originals of one
    variant must not have different base sizes. A bad value is a ValueError."""
    path = team_dir / TEAM_FILE
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    ops = raw.get("operatives", {})
    out = {}
    for name, v in ops.items():
        if not isinstance(v, dict) or "counts_as" not in v:
            continue
        value = v["counts_as"]
        targets = value if isinstance(value, list) else [value]
        if not targets or not all(isinstance(t, str) and t for t in targets):
            raise ValueError(f"{path} {name}: counts_as must be an operative name or a list of names, not {value!r}")
        if len(set(targets)) != len(targets):
            raise ValueError(f"{path} {name}: counts_as has a repeated name: {targets}")
        for target in targets:
            if target == name:
                raise ValueError(f"{path} {name}: counts_as names the operative itself")
            if target not in ops:
                raise ValueError(f"{path} {name}: counts_as {target!r} is not an operative of the file")
            if isinstance(ops[target], dict) and "counts_as" in ops[target]:
                raise ValueError(f"{path} {name}: counts_as {target!r}, which counts as another operative itself")
        sizes = {float(ops[t]["base_mm"]) for t in targets if isinstance(ops[t], dict) and "base_mm" in ops[t]}
        if len(sizes) > 1:
            raise ValueError(f"{path} {name}: the operatives of counts_as have different base sizes: {sorted(sizes)}")
        out[str(name)] = tuple(targets)
    return out


def read_expendable(team_dir: Path) -> set[str]:
    """Names of the operatives with `expendable = true` in `team.toml`. A value
    that is not true or false is a ValueError."""
    path = team_dir / TEAM_FILE
    if not path.exists():
        return set()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    out = set()
    for name, v in raw.get("operatives", {}).items():
        flag = v.get("expendable", False) if isinstance(v, dict) else False
        if not isinstance(flag, bool):
            raise ValueError(f"{path} {name}: expendable must be true or false, not {flag!r}")
        if flag:
            out.add(str(name))
    return out


def read_copies(team_dir: Path) -> dict[str, int]:
    """{operative name: max} of the operatives with `max` in `team.toml` (the
    others can be fielded once). A bad value is a ValueError."""
    path = team_dir / TEAM_FILE
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    out = {}
    for name, v in raw.get("operatives", {}).items():
        if isinstance(v, dict) and "max" in v:
            out[str(name)] = check_max(v["max"], f"{path} {name}")
    return out


def load_team_file(team_dir: Path) -> dict[str, float | None]:
    """{operative name: base diameter in mm or None} from `team.toml`."""
    return {name: base for name, (base, _wounds) in read_team_file(team_dir).items()}


def team_operatives(team: str, team_dir: Path, variants: bool = False) -> list[Operative]:
    """Operatives of a team: the `team.toml` entries in file order, then the
    operative folders that the file does not list, sorted. Needs no photos
    and no model. An operative with `counts_as` is a variant: it has the
    values of its original, and only `variants` lists it (the gallery needs
    it for its photos; a team list never does)."""
    listed = read_team_file(team_dir)
    expendable = read_expendable(team_dir)
    copies = read_copies(team_dir)
    counts_as = read_counts_as(team_dir)
    folders: set[str] = set()
    if team_dir.is_dir():
        folders = {q.name for q in team_dir.iterdir() if q.is_dir() and not q.name.startswith(".")}
    names = [*listed, *sorted(folders - set(listed))]
    ops = [Operative(team, name, *listed.get(name, (None, None)), name in expendable, copies.get(name, 1)) for name in names if variants or name not in counts_as]
    by_name = {op.name: op for op in ops}
    for op in ops:
        if op.name in counts_as:  # a variant takes the values of its (first) original
            op.targets = counts_as[op.name]
            _take_values(op, by_name[op.targets[0]])
    return ops


def _take_values(variant: Operative, original: Operative) -> None:
    """Make `variant` count as `original`: it gets the values of the original."""
    variant.counts_as = original.name
    variant.base_mm, variant.wounds = original.base_mm, original.wounds
    variant.expendable, variant.max = original.expendable, original.max


def roster_sizes(root: str | Path, teams: Iterable[str]) -> tuple[float, ...] | None:
    """Base sizes of the operatives of the teams, sorted, from the team files
    (no photos, no model). None when there is no operative, or when an
    operative has no known size: then all sizes are possible."""
    sizes = [op.base_mm for t in teams for op in team_operatives(t, Path(root) / t)]
    if not sizes or any(s is None for s in sizes):
        return None
    return tuple(sorted(set(sizes)))  # type: ignore[arg-type]


def save_team_file(
    team_dir: Path,
    team: str,
    operatives: dict[str, float | None],
    wounds: dict[str, int | None] | None = None,
    info: TeamInfo | None = None,
    expendable: Iterable[str] | None = None,
    copies: dict[str, int] | None = None,
    counts_as: dict[str, str | Sequence[str]] | None = None,
) -> None:
    """Write `team.toml`: team values, then base size, maximum wounds,
    `expendable` and `max` (copies, only above 1) of each operative, in the
    order of `operatives` (the order of the team lists in the live app).
    A variant (`counts_as`: variant name -> original names) gets `counts_as`
    only (a name for one original, else a list). Without `info`, `expendable`, `copies` or `counts_as` the values of
    the existing file stay."""
    wounds = wounds or {}
    if info is None:
        info = read_team_info(team_dir)
    expendable = read_expendable(team_dir) if expendable is None else set(expendable)
    copies = read_copies(team_dir) if copies is None else copies
    counts_as = read_counts_as(team_dir) if counts_as is None else counts_as
    lines = [
        f"# Battle Sight gallery of team {team!r}. One table per operative.",
        f"# archetype: killteam archetypes, one or more of {', '.join(ARCHETYPES)}.",
        "# faction-counter: optional text (a string).",
        "# operative-count: a whole number (5) or a range as text (\"4-6\").",
        "# base_mm: base diameter (one of [bases] diameters_mm). Only bases of this size can get this name.",
        "# wounds: maximum wounds (W) of the operative, for the health bar of the live app.",
        "# expendable: true for an expendable operative (optional, default false).",
        "# max: copies of the operative that the team can field (optional, default 1).",
        "# counts_as: name of an original operative, or a list of names (optional). The operative is another model of the original (own photos);",
        "#   it has the base size, wounds, expendable and max of the original and shows as the original.",
        "#   With a list, the original is the first name of the list that is in the team list.",
        "# Photos: one folder per operative next to this file, written by `battlesight enroll` and by fixes.",
        "",
    ]
    if info.archetypes:
        lines.append(f"archetype = {json.dumps(list(info.archetypes))}")
    if info.faction_counter:
        lines.append(f"faction-counter = {json.dumps(info.faction_counter)}")
    if info.operative_count is not None:
        lo, hi = info.operative_count
        lines.append(f"operative-count = {lo}" if lo == hi else f'operative-count = "{lo}-{hi}"')
    if info.archetypes or info.faction_counter or info.operative_count is not None:
        lines.append("")
    for name in operatives:
        lines.append(f"[operatives.{json.dumps(name)}]")
        if name in counts_as:
            value = counts_as[name]
            targets = [value] if isinstance(value, str) else list(value)
            lines.append(f"counts_as = {json.dumps(targets[0] if len(targets) == 1 else targets)}")
            lines.append("")
            continue
        if operatives[name] is not None:
            lines.append(f"base_mm = {operatives[name]:g}")
        if wounds.get(name) is not None:
            lines.append(f"wounds = {wounds[name]:d}")
        if name in expendable:
            lines.append("expendable = true")
        if copies.get(name, 1) > 1:
            lines.append(f"max = {copies[name]:d}")
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
        self.team_info: dict[str, TeamInfo] = {}
        self.files: list[Path] = []
        self.file_op: list[int] = []
        self.view_emb = np.zeros((0, 0), np.float32)
        self.view_op = np.zeros(0, int)
        self.view_file = np.zeros(0, int)
        self.masked_emb = np.zeros((0, 0), np.float32)
        self.masked_op = np.zeros(0, int)
        self.masked_file = np.zeros(0, int)
        self.hidden: set[int] = set()  # file indexes that `scores` must not use (the benchmark hides copies of a test photo)
        self._out_of_play: set[str] = set()  # see `out_of_play`
        self.record = True  # False: `LabelSession.choose` adds no photo (live app on `--image-source`: one frame, the same photo each time)
        self.copies: dict[str, int] = {}  # type label -> copies in play (live app); a type not here: its `max`
        self.version = 0  # +1 each time the views change: the score rows of an `IdentMemory` are old then
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

    @property
    def out_of_play(self) -> set[str]:
        """Labels that no base can get (removed from a team list of the live
        app). Assign a new set to change it: a variant with a list in
        `counts_as` then counts as the first operative of the list that is in
        play."""
        return self._out_of_play

    @out_of_play.setter
    def out_of_play(self, labels: set[str]) -> None:
        self._out_of_play = set(labels)
        self._resolve_variants()

    def _resolve_variants(self) -> None:
        by_label = {op.label: op for op in self.operatives}
        for op in self.operatives:
            if op.targets:
                labels = [label_of(op.team, t) for t in op.targets]
                _take_values(op, by_label[next((x for x in labels if x not in self._out_of_play), labels[0])])

    def columns(self) -> list[int]:
        """Operative index of each column of the name assignment: one column
        for each copy in play (`copies`, else the operative's `max`), so a
        type goes to at most that many bases. A variant (`counts_as`) has no
        column: its photos count for its original (`members`)."""
        return [k for k, op in enumerate(self.operatives) if op.counts_as is None for _ in range(max(1, self.copies.get(op.label, op.max)))]

    def members(self, k: int) -> list[int]:
        """Operative indexes of an original and its variants."""
        label = self.operatives[k].label
        return [j for j, op in enumerate(self.operatives) if j == k or op.original_label == label]

    def canonical(self, label: str) -> str:
        """The label of the original for a type label or a copy label of a
        variant (the copy number stays); other labels return themselves."""
        i = self.index(type_of(label))
        if i is None or self.operatives[i].counts_as is None:
            return label
        original = self.operatives[i].original_label
        k = copy_number(label)
        return original if k is None else copy_label(original, k)

    def counts(self) -> dict[str, int]:
        c = {op.label: 0 for op in self.operatives}
        for i in self.file_op:
            c[self.operatives[i].label] += 1
        return c

    def __len__(self) -> int:
        return len(self.files)

    def ensure_operative(
        self, team: str, name: str, base_mm: float | None, wounds: int | None = None, expendable: bool | None = None, counts_as: str | None = None
    ) -> Operative:
        """The operative, created (and written to team.toml) when new. Raises
        ValueError when it exists with another base size. `wounds` (maximum
        wounds) and `expendable` set or replace the value in team.toml (None
        keeps it). `counts_as` (the name of an original operative of the team)
        makes the operative a variant of it: it takes the values of the
        original, so `base_mm` must match, and `wounds` and `expendable` are
        refused."""
        if team not in self.team_dirs:
            raise ValueError(f"team {team!r} is not loaded, loaded: {sorted(self.team_dirs)}")
        name = check_name(name, "operative")
        if counts_as is not None:
            counts_as = check_name(counts_as, "operative")
            orig = self.index(label_of(team, counts_as))
            if counts_as == name:
                raise ValueError(f"{name}: --counts-as names the operative itself")
            if orig is None:
                raise ValueError(f"--counts-as: {label_of(team, counts_as)} is not in the gallery: enroll it first")
            original = self.operatives[orig]
            if original.counts_as is not None:
                raise ValueError(f"--counts-as: {counts_as} counts as {original.counts_as} itself")
            if wounds is not None or expendable is not None:
                raise ValueError(f"{name}: a variant uses the wounds and the expendable mark of {counts_as}, do not set them")
            if base_mm is not None and original.base_mm not in (None, base_mm):
                raise ValueError(f"{name}: {counts_as} has a {original.base_mm:g} mm base, not {base_mm:g} mm")
            existing = self.index(label_of(team, name))
            if existing is not None:
                if counts_as in self.operatives[existing].targets:
                    return self.operatives[existing]
                if self.operatives[existing].counts_as is not None or self.counts()[label_of(team, name)] or any(name in o.targets for o in self.operatives):
                    raise ValueError(f"{label_of(team, name)} exists already and does not count as {counts_as}")
                self.operatives.pop(existing)  # an operative without photos that nothing counts as: now a variant
                self._fix_file_ops(existing)
            op = Operative(team, name, original.base_mm, original.wounds, original.expendable, original.max, counts_as=counts_as)
            self.operatives.append(op)
            self._save_team(team)
            return op
        existing_op = self.index(label_of(team, name))
        if existing_op is not None and self.operatives[existing_op].counts_as is not None and (wounds is not None or expendable is not None):
            raise ValueError(f"{name} counts as {self.operatives[existing_op].counts_as}: set wounds and expendable on the original")
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
            if expendable is not None and op.expendable != expendable:
                op.expendable, changed = expendable, True
            if changed:
                self._save_team(team)
            return op
        op = Operative(team, name, base_mm, wounds, bool(expendable))
        self.operatives.append(op)
        self._save_team(team)
        return op

    def _fix_file_ops(self, removed: int) -> None:
        """Operative indexes of the photos after operative `removed` left the list."""
        self.file_op = [k - (k > removed) for k in self.file_op]

    def _save_team(self, team: str) -> None:
        ops = [op for op in self.operatives if op.team == team]
        orig = [op for op in ops if op.counts_as is None]
        save_team_file(
            self.team_dirs[team],
            team,
            {op.name: op.base_mm for op in ops},
            {op.name: op.wounds for op in orig},
            self.team_info[team],
            [op.name for op in orig if op.expendable],
            {op.name: op.max for op in orig},
            {op.name: op.targets for op in ops if op.targets},
        )

    def set_team_info(
        self, team: str, archetypes: Iterable[str] | None = None, faction_counter: str | None = None, operative_count: int | str | None = None
    ) -> TeamInfo:
        """Set the team values in team.toml; None keeps a value, an empty
        `faction_counter` or `operative_count` ("") clears it. Raises
        ValueError for a bad value."""
        if team not in self.team_dirs:
            raise ValueError(f"team {team!r} is not loaded, loaded: {sorted(self.team_dirs)}")
        old = self.team_info[team]
        new = TeamInfo(
            old.archetypes if archetypes is None else check_archetypes(list(archetypes), f"team {team!r}"),
            old.faction_counter if faction_counter is None else (faction_counter.strip() or None),
            old.operative_count
            if operative_count is None
            else (None if operative_count == "" else parse_count_range(operative_count, f"team {team!r} operative-count")),
        )
        if new != old:
            self.team_info[team] = new
            self._save_team(team)
        return new

    # loading and the embedding cache

    def reload(self) -> None:
        ops: list[Operative] = []
        files: list[Path] = []
        file_op: list[int] = []
        self.team_info = {team: read_team_info(d) for team, d in self.team_dirs.items()}
        for team, d in self.team_dirs.items():
            for op in team_operatives(team, d, variants=True):
                ops.append(op)
                folder = d / op.name
                if folder.is_dir():
                    for f in sorted(folder.iterdir()):
                        if f.suffix.lower() in CROP_SUFFIXES:
                            files.append(f)
                            file_op.append(len(ops) - 1)
        self.operatives, self.files, self.file_op = ops, files, file_op
        self._resolve_variants()
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
        self.version += 1

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
            if self.canonical(guess) == self.canonical(op.label):
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
        """Operatives that can stand on a base of this diameter and are in play
        (a variant is in play when its original is)."""
        return np.array([op.label not in self.out_of_play and op.original_label not in self.out_of_play and (op.base_mm is None or diameter_mm is None or op.base_mm == diameter_mm) for op in self.operatives], bool)


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
    photo_label: str | None = None  # type label of the variant (`counts_as`) whose photos match best, when not the original's: a fix photo goes there

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
    `min_margin` the label counts as unsure (shown in yellow). `labels` can
    repeat a label (one column for each copy of an operative): the margin
    compares with other labels only, and the candidates name each label once."""
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
        order = [j for j in np.argsort(-s[i], kind="stable") if np.isfinite(s[i, j])]
        cands, seen = [], set()
        for j in order:  # columns of copies of one operative: one entry
            if labels[j] not in seen:
                seen.add(labels[j])
                cands.append((labels[j], float(s[i, j])))
        j = int(pick[i])
        if j < 0 or s[i, j] < min_score:
            best = cands[0][1] if cands else -np.inf
            out.append(Ident(None, best, -np.inf, cands))
            continue
        other = max((s[i, k] for k in order if labels[k] != labels[j]), default=-np.inf)
        margin = float(s[i, j] - other) if np.isfinite(other) else float("inf")
        out.append(Ident(labels[j], float(s[i, j]), margin, cands, sure=margin >= min_margin))
    return out


REUSE_THUMB_PX = 48


def _thumb(crop: np.ndarray, diameter_mm: float, p: IdentParams) -> np.ndarray:
    """Small grey picture of what the model sees (`prepare` at angle 0), for the unchanged check."""
    g = cv2.cvtColor(prepare(crop, diameter_mm, p), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (REUSE_THUMB_PX, REUSE_THUMB_PX), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


@dataclass
class _Seen:
    center: np.ndarray
    diameter: float
    thumb: np.ndarray
    scores: np.ndarray  # (n_operatives,) raw scores of the base
    masked: bool


@dataclass
class IdentMemory:
    """The last identification of a run, so that `identify` embeds only the
    bases that changed. The score row of a base is reused when a base of the
    same diameter stood within `reuse_move_mm` of it and its crop looks the
    same (`reuse_crop_diff`). The memory is in RAM only: a new run starts
    empty. It is dropped (`clear`) when the gallery or the empty-board
    reference changes. `report` has one entry per base of the last call:
    `reused`, `reason`, `moved_mm` (to the nearest base of the same size),
    `diff` (crop difference, None when not compared); `embed_ms` is the
    time of the masks and the embedding."""

    key: tuple | None = None
    seen: list[_Seen] = field(default_factory=list)
    report: list[dict] = field(default_factory=list)
    embed_ms: float = 0.0

    def clear(self) -> None:
        self.key = None
        self.seen = []


def _gallery_key(gallery: Gallery, masking: bool) -> tuple:
    return (id(gallery), gallery.version, frozenset(gallery.hidden), masking)


def identify(
    crops: list[np.ndarray],
    diameters_mm: list[float],
    gallery: Gallery,
    backgrounds: Sequence[np.ndarray] = (),
    centers_mm: Sequence[Sequence[float]] = (),
    memory: IdentMemory | None = None,
) -> list[Ident]:
    """Identity of each base from its stored crop (`store_crop`) and its size
    class. With `backgrounds` (empty-board crops) and centres and a gallery
    that `can_mask`, a base with a usable mask compares masked crops; all other
    bases compare full crops. With `memory` (and centres), a base that did not
    move and looks the same as in the last call keeps its scores and is not
    embedded again."""
    p = gallery.params
    if not crops:
        if memory is not None:
            memory.report, memory.seen, memory.embed_ms = [], [], 0.0
        return []
    n = len(crops)
    masking = gallery.can_mask and len(backgrounds) == n and len(centers_mm) == n
    centres = np.asarray(centers_mm, float).reshape(-1, 2) if len(centers_mm) == n else None
    scores = np.full((n, len(gallery.operatives)), -np.inf)
    masked = np.zeros(n, bool)
    reused = np.zeros(n, bool)
    report: list[dict] = [{"reused": False, "reason": "no earlier result", "moved_mm": None, "diff": None} for _ in range(n)]
    thumbs: list[np.ndarray | None] = [None] * n
    kept: dict[int, _Seen] = {}  # new base -> the old entry that it keeps
    if memory is not None and centres is not None and p.reuse_crop_diff > 0:
        key = _gallery_key(gallery, masking)
        if memory.key != key:
            memory.clear()
            memory.key = key
        free = list(range(len(memory.seen)))
        for i in range(n):
            thumbs[i] = _thumb(crops[i], diameters_mm[i], p)
            near = sorted(((float(np.hypot(*(memory.seen[j].center - centres[i]))), j) for j in free if memory.seen[j].diameter == diameters_mm[i]))
            if not near:
                report[i].update(reason="new base" if memory.seen else "no earlier result")
                continue
            dist, j = near[0]
            report[i]["moved_mm"] = round(dist, 2)
            if dist > p.reuse_move_mm:
                report[i]["reason"] = "moved"
                continue
            diff = float(np.abs(thumbs[i] - memory.seen[j].thumb).mean())
            report[i]["diff"] = round(diff, 4)
            if diff > p.reuse_crop_diff:
                report[i]["reason"] = "crop changed"
                continue
            free.remove(j)
            kept[i] = memory.seen[j]
            reused[i] = True
            report[i].update(reused=True, reason="not moved")
            scores[i], masked[i] = memory.seen[j].scores, memory.seen[j].masked
    todo = [i for i in range(n) if not reused[i]]
    t0 = time.perf_counter()
    if todo:
        cut: dict[int, np.ndarray | None] = {i: None for i in todo}
        if masking:
            allc = np.asarray(centers_mm, float)
            for i in todo:
                others = [(*(allc[j] - allc[i]), diameters_mm[j]) for j in range(n) if j != i]
                cut[i] = mask_crop(crops[i], backgrounds[i], diameters_mm[i], p, others)
        emb = gallery.embedder.embed([prepare(crops[i] if cut[i] is None else cut[i], diameters_mm[i], p) for i in todo])
        m = np.array([cut[i] is not None for i in todo], bool)
        part = np.full((len(todo), len(gallery.operatives)), -np.inf)
        part[~m] = gallery.scores(emb[~m])
        part[m] = gallery.scores(emb[m], masked=True)
        for r, i in enumerate(todo):
            scores[i], masked[i] = part[r], m[r]
    if memory is not None:
        memory.embed_ms = (time.perf_counter() - t0) * 1000.0
        memory.report = report
        if centres is not None and p.reuse_crop_diff > 0:
            memory.seen = [
                kept[i] if i in kept else _Seen(centres[i].copy(), diameters_mm[i], thumbs[i], scores[i].copy(), bool(masked[i]))
                for i in range(n)
            ]  # a kept base keeps its old centre and picture: slow drift adds up and ends the reuse
    allowed = np.stack([gallery.allowed(d) for d in diameters_mm]) if gallery.operatives else np.zeros((n, 0), bool)
    cols = gallery.columns()
    labels = gallery.labels()
    # An original and its variants compete as one operative: the best score of the models.
    usable = np.where(allowed & np.isfinite(scores), scores, -np.inf)
    group = usable.copy()
    for k in set(cols):
        group[:, k] = usable[:, gallery.members(k)].max(axis=1)
    out = assign(group[:, cols], np.isfinite(group[:, cols]), [labels[k] for k in cols], p.min_score, p.min_margin)
    for i, (x, mk) in enumerate(zip(out, masked)):
        x.masked = bool(mk)
        # The fix menu names each model on its own, with its own score.
        order = [j for j in np.argsort(-usable[i], kind="stable") if np.isfinite(usable[i, j])]
        x.candidates = [(labels[j], float(usable[i, j])) for j in order]
        if x.label is not None:
            members = gallery.members(gallery.index(x.label))  # type: ignore[arg-type]
            best = max(members, key=lambda j: usable[i, j])
            if labels[best] != x.label:
                x.photo_label = labels[best]
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

    def choose(self, label: str, i: int | None = None, photo: str | None = None) -> Path | None:
        """Set the name of base i (default: the selected base) and add its crop
        to the gallery (the photo of a copy label goes to its type). A label
        that another base holds moves to this base; the other base becomes
        unknown, or, for a copy label when this base had another copy of the
        same type, gets that copy (a swap). Returns the new gallery photo, or
        None when the gallery has this photo already (the name is set, no file).
        The label of a variant (`counts_as`) names its original; the photo goes
        to the variant, or to `photo` (a type label) when given."""
        i = self.selected if i is None else i
        if i is None:
            raise ValueError("no base selected")
        photo = photo or type_of(label)
        label = self.gallery.canonical(label)
        path = None
        if self.gallery.record:
            try:
                path = self.gallery.add(photo, self.crops[i], self.diameters_mm[i], self.backgrounds[i] if self.backgrounds else None)
            except DuplicatePhoto:
                pass
        self.added.append((i, path, self.idents[i]))
        mine = self.idents[i].label
        swap = mine if mine is not None and mine != label and copy_number(label) is not None and type_of(mine) == type_of(label) else None
        for k, x in enumerate(self.idents):
            if k != i and x.label == label:
                self.idents[k] = Ident(swap, x.score, x.margin if swap else -np.inf, x.candidates, sure=swap is not None, confirmed=swap is not None)
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
                out.append(self.choose(x.label, i, x.photo_label))
        return out

    def remove(self, i: int, keep_photos: bool = False) -> list[Path]:
        """Remove base i from the session: a false base (terrain, board print).
        The later bases move one index down. The gallery photos that a choice
        of this session added for this base are removed too (not with
        `keep_photos`); returns the removed ones."""
        removed = [] if keep_photos else [path for k, path, _ in self.added if k == i and path is not None]
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
        if idents[i].label is not None and type_of(idents[i].label) == label:  # a copy counts as its type
            right += 1
        else:
            wrong.append((label, idents[i].label))
    total = len(truth.bases)
    acc = right / total if total else 1.0
    return {"right": right, "total": total, "accuracy": round(acc, 3), "wrong": wrong, "missed": missed, "done_ok": acc >= 0.9}
