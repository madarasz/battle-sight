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

Writes `data/tags/corner_tags.pdf` (A4, all four cards on one page). Print at
100%, no "fit to page". Measure the 100 mm ruler on the print. Cut on the grey lines.

Each card is a 90 mm square. The tag sits in the outer quadrant. A black L
in the inner quadrant marks the board corner. The 5 mm white margin around
the tag is the quiet zone the detector needs. Do not trim it.

1. Put each card on the table at its board corner.
2. Rest the board corner on the card. Both board edges touch the inner side
   of the black L. Both lines stay fully visible, with no white gap between
   line and board. A covered line means the board is too far out, a white
   gap means it is too far in.

The card lies under the board, so nothing covers the play area. The tag sits
diagonally outside the board corner, and the tag's inner margin corner is
the board corner. The label beside the tag tells which corner: id 0 =
top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left, as seen on screen.

The app measures from the cards, not the board. The board's weight holds
the cards, so a bumped board usually drags its cards with it (warning). If
the board slides over a still card, there is no warning: lift the corner
and put it back against the L.

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
