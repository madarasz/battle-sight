# Battle Sight

Battle Sight helps you measure in Warhammer 40k: Kill Team. A camera above
the table looks down on the board. A window on your computer shows the live
video with measurement lines on top of it: the map grid, the operative
bases, and later the range rings and the distances between operatives.

## Work in Progress

1. ✅ Board corners: AprilTag cards, drift warning
2. ✅ Undistort and rectify: lens calibration, top-down board image, map grid
3. ✅ Bases: find round bases, size and position
4. ⏳ Operative ID: name each base from a photo gallery (done test waits for real gallery photos)
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
8. Teach the app your operatives (for operative names).
9. Change the window layout settings (optional).

## How to use

Open the Terminal app and go to the Battle Sight folder. Start the app:

```bash
uv run battlesight
```

The app uses the camera that you saved in setup step 4
(`tools/list_cameras.py`). To use another camera for one run, add
`--source`, for example `--source 0` for the built-in camera.

The app uses all teams that you taught it (setup step 8). To use only the
two teams of this game, add their names:

```bash
uv run battlesight --team kommandos veterans
```

### The window

The video is at the left. The panel at the right has a control for the
functions that you can use now. The keys and the menus always work. Hold the
mouse pointer on a control to see its key.

The window has two layouts. Push `l`, or use the menu View > Layout, to change
the layout. The app keeps your choice for the next start. The video, the
result on the screen and the names do not change.

| Layout | Panel |
|---|---|
| `inspector` | The panel changes with your work: live view, result, or the base that you clicked. |
| `panel` | The panel is always the same and has all the controls. |

### Live view

The window shows the board from straight above, upright. The app removes the
camera tilt and the lens distortion, so the grid lines sit on the printed
lines of the board. The name of this function is "Adjust perspective".

Push `v` to turn "Adjust perspective" off. The window then shows the camera
image with the four corner cards and the map grid. Use it to put the camera
and the board in position before a game. The window also shows the camera
image when the camera cannot see all four cards.

When you move the board or the camera, the app finds the new position by
itself. Keep the board still for a moment after the move.

### Find the bases and name the operatives

1. Before the operatives go on the board, set up the terrain and the tokens,
   and push `e`. The app keeps this photo of the empty board. Then it looks
   for bases only where something changed. Without the empty-board photo,
   the app also finds some tokens and terrain parts that look like bases.
2. Put the operatives on the board.
3. Push `space`. The app finds the bases and writes the name of the
   operative under each base:
   - Green name: the app is sure.
   - Yellow name: the app is not sure. Check it.
   - Red `?`: the app does not know this operative.
4. If a name is wrong, click the base, then click the correct name in the
   list. In the `inspector` layout, you can also click the row of the
   operative in the panel. The panel then shows a photo of the base and the
   list of names. In the `panel` layout, the list opens at the base. The app
   learns from each correction, so it makes fewer errors in the next games.
5. Push `Esc` to go back to the live view.

Push `b` to find the bases without names (each base with a green circle and
its size in mm). The result stays on the screen until the next `space` or
`b`, or until you push `Esc`. The Terminal window also lists each base.

### Keys

Push `h` to see this list in the window. The menus have the same actions.

| Key or mouse | Action |
|---|---|
| `space` | Find the bases and name the operatives. |
| `b` | Find the bases only, no names. |
| `v` | Turn "Adjust perspective" on or off. On: the board from straight above. Off: the camera image. A result always opens with it on. |
| `g` | Show or hide the grid. |
| `e` | Keep the current image as the empty-board photo. Do this on the empty board only. |
| `c` | Forget the empty-board photo. |
| Click a base | Show the list of names for this base size. |
| Click a name | Give this name to the base. The app keeps the photo. |
| `a` | All names are correct: the app keeps the photos of all bases. |
| `u` | Undo the last correction. |
| `s` | Save a photo to `data/snapshots/`. |
| `l` | Change to the next window layout. |
| `h` | Show the key list. |
| `Esc` | Close the list of names. When no list is open, go back to the live view. |
| `q` | Quit. |

### Saved photos

```bash
uv run battlesight image-pipeline my-photo.jpg --show
```

The app does the same analysis on a saved photo and shows the result.
Use it to check a photo that you saved with `s`. To find the bases with an
empty-board photo, add `--reference empty.jpg`. The app also names the
operatives of all teams that you taught it (setup step 8). To use only some
teams, add `--team kommandos veterans`.

## What you see

The video has no status text. The status line at the bottom of the window
tells the board status, in all layouts. While the app finds the bases, it shows "working ...", and the keys do not work.

| Status line | Meaning | What to do |
|---|---|---|
| Yellow "locking reference" | The app learns the board position. It takes about 10 frames. | Do not touch the board. |
| Green "4 corners OK" | All four cards are visible. The board did not move. | Play. |
| Yellow "BOARD MOVED" | A card moved more than 2 mm. | Keep the board still. When all four cards moved together, the app saves the new position after about 10 frames. |
| Green "board position updated" | The board or the camera moved, and the app saved the new position. | Play. |
| Red "CARD MOVED" | One card moved, but the other cards did not. The card is not on its board corner now. | Put the card back against the board corner. |
| Red "TAG LOST" | The camera cannot see a card. | Remove what covers the card (a hand, a model, a dice tray). |
| Red "camera down" | The app gets no image from the camera. It tries again by itself. | Check the camera and its cable or network. |
| Red "CORNERS NOT SEEN" (second line, `enroll`) | The camera cannot see one or more corner cards now. The line names the cards. `space` cannot find the board. | Remove what covers the card. Move the camera if a card is outside the image. |
| Message at the right of the status line | The result of your last key or click, for 6 seconds. Red: it did not work, for example "no base found". Yellow: do the next step, for example "click the base". Green: done, for example "photo 3 added". | Read the message. For a red message, correct the cause and try again. |

In the `inspector` layout, the panel shows:

- At the top: the board status, and the view controls.
- In the live view: the buttons to find bases, and if the app has an
  empty-board photo.
- With a result: how many operatives have a name, and one row for each base.
  Green is sure, yellow is not sure, red `?` is unknown.
- After a click on a base: the size of the base, a photo of it, and the names
  with their scores. Push `Esc` to close this.
- Debug (click it to open): tags found, fit error, frames per second, and the
  time for tag detection. The `panel` layout always shows these values.

On the camera image:

- Green outline and number on each card: the app found the card.
- White cross: the board corner that the app calculated.
- Thin green lines: the board outline between the four corners.
- Yellow lines: the map grid.
- Green circle with a number: a base that the app found, and its size in mm.
  The circle is the bottom edge of the base, where it touches the board.
- Name under a base (after `space`): the operative and a score from 0 to 1.
  Green is sure, yellow is not sure, red `?` is unknown. A white circle
  shows the base that you clicked.

With "Adjust perspective" on, the board is upright, the cards are in the corners of
the image, and the grid sits on the printed lines.

## Problems

- **"BOARD MOVED" does not go away.** Something moves the cards, for example
  a hand or a shaking table. Keep the board still for a moment. If "CARD
  MOVED" shows, put the named card back on its board corner.
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
  (`e`) after you set up the terrain and the tokens,
  before the operatives go on the board.
- **The size of a base is wrong.** Make sure that `diameters_mm` in
  `data/board.toml` lists only the base sizes of your teams.
- **An operative gets the wrong name.** Click the base and select the
  correct name. If it happens frequently, add more photos of the two
  operatives (setup step 8), and check with `battlesight gallery`.
- **The name list is empty.** The gallery has no operative with this base
  size. Teach the app this operative (setup step 8), or check `base_mm` in
  `data/teams/<team>/team.toml`.
- **The first `space` or `enroll` is slow.** The app downloads the
  recognition model one time. Later, the first `space` takes a few seconds.
- **The grid does not sit on the printed lines.** Check the map settings
  ([docs/setup.md](docs/setup.md), step 5). Then calibrate the lens (step 6).
