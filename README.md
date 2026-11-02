# Battle Sight

Augmented reality overlay for Warhammer 40k: Kill Team. A camera above the
board, a window with the video and measurement overlays. See `specs/plan.md`.

## Setup (macOS)

1. Install `uv` if missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. In this folder: `uv sync` (creates `.venv` with Python 3.12, OpenCV, NumPy, Pillow).
3. Run the tests: `uv run pytest`

## Priority 1: board corners

### Print the tags

```
uv run tools/make_tags.py
```

Writes `data/tags/corner_tags.pdf` (A4). Print at 100%, no "fit to page".
Measure the 100 mm ruler on the print. Cut on the grey lines.

Place each card so that:

- the corner with the black triangle sits on the board corner,
- the two card edges run flush with the two board edges,
- "TOP" points toward the top edge of the board (away from you, if the origin
  corner is at your top-left on screen).

Card id 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left, as
seen on screen. Tape the cards down. A moved card looks like a moved board.

### Camera

Built-in camera for a first test:

```
uv run battlesight corners
```

Phone camera through Camo:

1. Install Camo on the Mac (reincubate.com/camo) and on the Xiaomi 13T. Connect by USB.
2. In Camo on the Mac pick the main lens, 4K, lock focus and exposure.
3. Find the camera index: `uv run tools/list_cameras.py`
4. Run: `uv run battlesight corners --source 1 --width 3840 --height 2160`

macOS asks for camera permission for the terminal app on the first run.

### What you see

- Green outline and id on each tag. Cyan dot = the tag's own top-left corner.
- White cross = computed board corner. Thin green quad joins the four corners.
- Banner: yellow "locking reference" for the first 10 still frames, then
  green "4 corners OK", yellow "BOARD MOVED" over 2 mm, red "TAG LOST".

Keys: `q` quit, `r` re-lock the reference after you moved the board on purpose,
`s` save a frame to `data/snapshots/`.

Tolerances live in `data/board.toml`.

## Layout

```
src/battlesight/   config.py camera.py calib.py overlay.py app.py
tools/             make_tags.py list_cameras.py
tests/             synthetic-board tests, no camera needed
data/              board.toml, tags/, snapshots/, calib/
```
