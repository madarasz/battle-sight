# Battle Sight

Battle Sight helps you measure in Warhammer 40k: Kill Team. A camera above
the table looks down on the board. A window on your computer shows the live
video with measurement lines on top of it: the map grid, the operative
bases, and later the range rings and the distances between operatives.

## Work in Progress

1. ✅ Board corners: AprilTag cards, drift warning
2. ✅ Undistort and rectify: lens calibration, top-down board image, map grid
3. ✅ Bases: find round bases, size and position
4. 🚧 Operative ID: name each base from a photo gallery
5. 🚧 Rings and distances: range rings, rim-to-rim distance
6. 🚧 Tokens: find tokens, list operatives in range
7. 🚧 Doors: click hatchways once per map, draw 1 in and 2 in bands

✅ done · ⏳ next · 🚧 not started

## Setup

You need a Mac, a camera above the board (the built-in camera for a test, or
a phone through Camo), a printer, and scissors.

[docs/setup.md](docs/setup.md) gives all steps:

1. Install the app.
2. Print and cut the four corner cards.
3. Put the cards and the board on the table.
4. Connect the camera.
5. Set your board and map.
6. Calibrate the lens (optional).
7. Take a photo of the empty board (for base detection).

## How to use

Open the Terminal app and go to the Battle Sight folder. Start one of the
modes below. The app uses the camera that you saved in setup step 4
(`tools/list_cameras.py`). To use another camera for one run, add
`--source`, for example `--source 0` for the built-in camera.

### Board view (for a game)

```bash
uv run battlesight rectify
```

The window shows the board from straight above, upright, with the map grid
on it. The app removes the camera tilt and the lens distortion, so the
grid lines sit on the printed lines of the board.

| Key | Action |
|---|---|
| `v` | Change between the top-down view and the camera view. |
| `g` | Show or hide the grid. |
| `r` | Lock the board position again, after you moved the board on purpose. |
| `s` | Save a photo to `data/snapshots/`. |
| `q` | Quit. |

### Find the bases

```bash
uv run battlesight bases
```

The app finds the round bases of the operatives on the board. It shows each
base with a green circle and its size in mm (25, 28, 32, 40 or 50).

Before the operatives go on the board, set up the terrain and push `e`. The
app keeps this photo of the empty board. Then put the operatives on the
board and push `space`. The app compares the board with the empty-board
photo and looks for bases only where something changed. Without the
empty-board photo the app also finds some tokens and terrain parts that look
like bases.

| Key | Action |
|---|---|
| `space` | Find the bases now. The result stays on the screen until the next `space`. |
| `e` | Keep the current image as the empty-board photo. Do this on the empty board only. |
| `c` | Forget the empty-board photo. |
| `v` | Change between the top-down view and the camera view. |
| `s` | Save a photo to `data/snapshots/`. |
| `q` | Quit. |

The Terminal window also lists each base: size, position on the board in mm,
and how much of its edge the camera saw.

### Corner check (to set up the table)

```bash
uv run battlesight corners
```

The window shows the camera image and the four corner cards. Use this mode
to put the camera and the board in position before a game.

| Key | Action |
|---|---|
| `r` | Lock the board position again, after you moved the board on purpose. |
| `s` | Save a photo to `data/snapshots/`. |
| `q` | Quit. |

### Saved photos

```bash
uv run battlesight image my-photo.jpg --show
```

The app does the same analysis on a saved photo and shows the result.
Use it to check a photo that you saved with `s`. To find the bases with an
empty-board photo, add `--reference empty.jpg`.

## What you see

The banner at the top tells the board status:

| Banner | Meaning | What to do |
|---|---|---|
| Yellow "locking reference" | The app learns the board position. It takes about 10 frames. | Do not touch the board. |
| Green "4 corners OK" | All four cards are visible. The board did not move. | Play. |
| Yellow "BOARD MOVED" | A card moved more than 2 mm. | Put the board back, or press `r` if you moved it on purpose. |
| Red "TAG LOST" | The camera cannot see a card. | Remove what covers the card (a hand, a model, a dice tray). |

On the camera image:

- Green outline and number on each card: the app found the card.
- White cross: the board corner that the app calculated.
- Thin green lines: the board outline between the four corners.
- Yellow lines: the map grid.
- Green circle with a number: a base that the app found, and its size in mm.
  The circle is the bottom edge of the base, where it touches the board.

In the top-down view, the board is upright, the cards are in the corners of
the image, and the grid sits on the printed lines.

## Problems

- **"BOARD MOVED" does not go away.** The board moved after the app locked
  its position. Push the board back, or press `r` to lock the new position.
- **The board slid but there is no warning.** The app measures from the
  cards, not from the board. If the board slides over a card that stays
  still, the app cannot see it. Lift the corner and put it back against the
  black L of the card.
- **Error that width and height are the wrong way round.** Change
  `width_mm` and `height_mm` in `data/board.toml`. See
  [docs/setup.md](docs/setup.md), step 5.
- **The camera does not start.** Make sure that the Terminal app has camera
  permission (System Settings > Privacy & Security > Camera). Find the
  correct camera number with `uv run tools/list_cameras.py`. If Camo does
  not show, or you use a Wi-Fi stream, see
  [docs/setup.md](docs/setup.md), Troubleshooting.
- **A base is not found.** The app must see at least half of the black
  edge of the base. A model that leans over the edge, a wall, or a shadow can
  hide too much of it. Move the camera nearer to the top of the board, or
  turn the model.
- **A token or a terrain part shows as a base.** Take the empty-board photo
  (`e` in `battlesight bases`) after you set up the terrain and the tokens,
  before the operatives go on the board.
- **The size of a base is wrong.** Make sure that `diameters_mm` in
  `data/board.toml` lists only the base sizes of your teams.
- **The grid does not sit on the printed lines.** Check the map settings
  ([docs/setup.md](docs/setup.md), step 5). Then calibrate the lens (step 6).
