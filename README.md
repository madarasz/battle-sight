# Battle Sight

Battle Sight helps you measure in Warhammer 40k: Kill Team. A camera above
the table looks down on the board. A window on your computer shows the live
video with measurement lines on top of it: the map grid, the operative
bases, the range rings and the distances between operatives.

## Work in Progress

1. ✅ Board corners: AprilTag cards, drift warning
2. ✅ Undistort and rectify: lens calibration, top-down board image, map grid
3. ✅ Bases: find round bases, size and position
4. ✅ Operative ID: name each base from a photo gallery
5. ✅ Rings and distances: range rings, rim-to-rim distance
6. ✅ Terrain layouts: walls, hatchways, objectives and drop zones of each layout card
7. ✅ Team layout: operative list inspector
8. ✅ Game state: operative health, open/closed status of hatchways and breach points
9. ⏳ Tokens: find tokens, list operatives in range

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
two teams of this game, choose them in the team lists of the window, or add
their names when you start the app:

```bash
uv run battlesight --team kommandos veterans
```

### The window

The keys and the menus always work. Hold the mouse pointer on a control to
see its key.

The window has three layouts. Push `l`, or use the menu View > Layout, to
change the layout. The app keeps your choice for the next start. The video,
the result on the screen and the names do not change.

| Layout | Panel |
|---|---|
| `rosters` (default) | A control bar above the video, and one team list at each side of the video. |
| `inspector` | The panel at the right changes with your work: live view, result, or the base that you clicked. |
| `panel` | The panel at the right is always the same and has all the controls. |

In the `rosters` layout, the control bar has these sections:

- **View**: the terrain type, the layout card, and the view check boxes.
- **Commands**: "Identify operatives", "Accept all identification" (only when
  some names are not confirmed), "Add operative" (not available yet), and "Go
  back to Live" (only when the result is on the screen).
- **Inspector**: the operative or the hatchway that you clicked. Click `✕` to
  clear it. Here you change the wounds of an operative and open or close a
  hatchway.
- **Reference**: take or clear the empty-board photo, and "Save screenshot".
- **Debug**: shows the debug values at the right end of the status line.

At each side of the video, choose the team of that player in the list. You
cannot choose the same team on the two sides. The names of its operatives
show below it, with marks at the right end of the line:

- Wounds, for example `12/15`: the operative has 12 of its 15 wounds. This
  shows only when the team file has the wounds of the operative.
- Eye: the name of this operative is confirmed.
- Percentage, for example `78%`: the app found this operative, but nobody
  confirmed the name. The percentage tells how sure the app is.
- `?`: the app did not find this operative.

Click an operative that the app found to show it in the inspector. When you
change a team, push `space` again to identify the operatives of the new team.

The top-right corner of the video shows the mode:

- A red circle and "Live": the live video.
- A blue pause sign and "Snapshot": a result.
- A green square and "Minimap": the minimap (see "Terrain map").

### Terrain map

At the top of the panel, choose the map of this game:

1. **Terrain type**: the terrain set, for example `Tomb World`.
2. **Layout**: the layout card, for example `Layout 6`. Choose `None` for no
   map.

The app draws the walls, pillars, hatchways, breach points, objective markers,
teleporter pads and light terrain (`C1` to `C5`) of the layout on the board, and a dashed circle 1 in around
each objective marker. Use the **Layout** check box (View menu) to show or
hide them. Use the **Drop zones** check box to show or hide the deployment
zones. Use the **Terrain keys** check box to show the label of each wall
(for example `A1`, `B3`), above a horizontal wall and at the right of a
vertical wall, a `T` on each teleporter pad, and the label of each light
terrain piece (for example `C1`). The keys are on top of the
other terrain. The drop zones and the terrain keys are off at the start. The app keeps the terrain type
and layout for the next start.

Use the **Minimap** check box to show a drawing of the board in place of the
camera image. The minimap shows the layout, the grid, and a full circle for
each operative that the app found, with its name and its health bar. Clicks,
rings and distances work as on the camera image. The minimap is available
only when you chose a layout, and it is off at the start. Push `space` to
find the operatives again after they move.

Put the board so that it agrees with the layout card: corner card 0 is at the
top-left corner of the card picture. If the drawing does not sit on the real
walls, turn the board or the cards.

To add a layout, give a photo of the layout card to the AI agent (Claude Code)
and ask it to record the map. It writes a file in `data/maps/tomb-world/`.

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
4. If a name is wrong, click the base. In the `rosters` layout, click
   "Change ID" in the inspector, then click the correct name. In
   the `inspector` layout, the panel shows the list of names. In the `panel`
   layout, right-click the base: the list opens at the base. The app learns
   from each correction, so it makes fewer errors in the next games.
   If the app found an operative where there is none (terrain or board
   print), click it, then click "Remove ID". The base goes away until the
   next `space`. If you named this base before, the app also removes its
   photo from the gallery.
5. Push `Esc` ("Go back to Live") to go back to the live video. The circles
   and the names stay on the live video until the next `space`.

Push `b` to find the bases without names (each base with a green circle and
its size in mm). The Terminal window also lists each base.

Click a hatchway or a breach point of the terrain map to show it in the
inspector.

### Measure rings and distances

You can measure after `space` or after `b`, on the result and on the live
video.

1. Click a base. A white circle shows the base that you selected.
2. Push a key from `1` to `9`, or move the "Distance" slider in the
   inspector. The app draws a ring that many inches from the edge of the
   base. Push `0`, or move the slider to `Off`, to remove the ring.
3. Click a second base. The app draws a line between the two base edges, and
   shows the distance in inches in the status line, for example `5.93"`.
4. Click another base to move the line to it. Click the second base again to
   remove the line.
5. Click the selected base, or an empty part of the board, to stop the
   measurement.

The measurement is from base edge to base edge. It is on the board surface.

#### Measure around walls

On Tomb World you cannot measure through walls, pillars, closed hatchways and
closed breach points. With a layout, the **Measure around walls** check box
(View section, or key `w`) is on, and the app obeys this rule:

- The distance to a second base is along the shortest path around the walls.
  The line bends at the wall corners. An open hatchway or breach point lets
  the line through. When all paths are closed, the app shows `no path`.
- The ring becomes a magenta area: all points that you can get to with a
  measurement of that many inches or less. The area stops at walls and goes
  around their ends.

A pillar or a wall at the side of the board stops the measurement as far as
the edge of the board: there is no way between it and the edge.

Open or close a hatchway (key `o`) and the measurement changes with it. Other
operatives do not stop the measurement. Clear the check box to measure in a
straight line again. Without a layout, the measurement is always straight.

#### Measure movement

The **Measure movement** check box (below the "Distance" slider, or key `m`)
changes the ring into the area that the selected operative can move to:

1. Click a base, then set the "Distance" slider (or push `1` to `9`) to the
   move distance in inches.
2. Set the **Measure movement** check box. The magenta area shows all places
   that the base can cover with that move.
3. Move the mouse pointer into the area. The app draws the best move to that
   point in grey: the base at each bend and at the end, the inches of each
   straight part, and the inches in all below the end. The status line shows
   the same, for example `move 5" (3 + 2)`.

The rules of the move:

- A move is made of straight parts. Each part counts as its length, rounded
  up to a full inch (2.75 in counts as 3 in).
- The base cannot go through walls and pillars, and it stays on the board.
  The full base must go past a corner, not only its centre.
- Through an open hatchway or breach point, only the centre of the base must
  go through. The base size does not matter there. This costs 1 in more. The
  base cannot stop while it is over the wall.
- A base of the other team stops the move. Bases of the same team and bases
  without a name do not: the operative can move through them. Thus name the
  operatives first (`space`).
- The operative cannot stop on another base. The area has a hole at each
  other operative.
- The operative cannot stop on an open hatchway or breach point. The area
  does not include them.

The distance to a second base stays as it was. Without a layout, only other
bases and the board edge stop the move.

### Wounds and hatchways

The app keeps two values of the game for you:

- **Wounds of each operative.** Each operative starts with all its wounds.
  Click an operative that has a name. In the inspector, click `-` or `+`, or
  type a number in the "Wounds" box and push Return. You can also push the
  `-` and `+` keys. The number stays between 0 and the maximum wounds of the
  operative. A health bar above the operative shows its wounds. The team
  list shows them as `12/15`. The wounds stay when you push `space` again.
- **Open or closed hatchways and breach points.** All hatchways and breach
  points start closed. Click a hatchway or a breach point. In the inspector,
  click "Open" or "Close", or push `o`. When you choose a layout again, all
  hatchways and breach points are closed again.

The app must know the maximum wounds of an operative. Put them in the team
file, or give them when you teach the app the operative (`docs/setup.md`,
step 8). When the maximum is not known, the inspector shows a warning and no
wounds controls.

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
| Click a base | Select this base. When a base is selected: measure the distance to the clicked base. |
| Click a hatchway or a breach point | Select it. |
| `1` to `9` | Draw a ring around the selected base, this many inches from its edge. |
| `0` | Remove the ring. |
| `w` | Turn "Measure around walls" on or off (only with a layout). |
| `m` | Turn "Measure movement" on or off. |
| `-` and `+` | Remove or add one wound of the selected operative. |
| `o` | Open or close the selected hatchway or breach point. |
| Right-click a base | Select this base and show the list of names for this base size. |
| Click a name | Give this name to the base. The app keeps the photo. |
| `a` | All names are correct: the app keeps the photos of all bases. |
| `u` | Undo the last correction. |
| `s` | Save a screenshot (the camera image) to `data/snapshots/`. |
| `l` | Change to the next window layout. |
| `h` | Show the key list. |
| `Esc` | Go back to the live video. The names stay. In the live video: stop the selection. |
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
tells the board status, in all layouts. It also tells the ring size and the distance between two selected bases. While the app finds the bases, it shows "working ...", and the keys do not work.

| Status line | Meaning | What to do |
|---|---|---|
| Yellow "locking reference" | The app learns the board position. It takes about 10 frames. | Do not touch the board. |
| Green "4 corners OK" | All four cards are visible. The board did not move. | Play. |
| Yellow "BOARD MOVED" | A card moved more than 2 mm. | Keep the board still. When all four cards moved together, the app saves the new position after about 10 frames. |
| Green "board position updated" | The board or the camera moved, and the app saved the new position. | Play. |
| Red "CARD MOVED" | One card moved, but the other cards did not. The card is not on its board corner now. | Put the card back against the board corner. |
| Red "TAG LOST" | The camera cannot see a card. When one or two cards are covered and the other cards did not move, the display stays as it is. When more cards are covered, the display goes back to the camera image. | Remove what covers the card (a hand, a model, a dice tray). |
| Red "camera down" | The app gets no image from the camera. It tries again by itself. | Check the camera and its cable or network. |
| Red "CORNERS NOT SEEN" (second line, `enroll`) | The camera cannot see one or more corner cards now. The line names the cards. `space` cannot find the board. | Remove what covers the card. Move the camera if a card is outside the image. |
| Message at the right of the status line | The result of your last key or click, for 6 seconds. Red: it did not work, for example "no base found". Yellow: do the next step, for example "click the base". Green: done, for example "photo 3 added". | Read the message. For a red message, correct the cause and try again. |

In the `rosters` layout, the inspector shows:

- For an operative: the team, the name and the base size, for example
  `Raveners - Wrecker (40mm)`, a photo of the base, the "Wounds" controls
  (`-`, the number, the maximum, `+`), the "Distance" slider (ring size, 0 to
  9 in), the distance to a second base, the "Change ID" button and the
  "Remove ID" button. The list of names shows only after you click
  "Change ID". "Remove ID" removes a base that is not an operative.
- For a hatchway or a breach point: its name, "Closed" or "Open", and a
  button that opens or closes it.

In the `inspector` layout, the panel shows:

- At the top: the terrain type and layout lists, and the view controls.
- In the live view: the "Identify operatives" button, and if the app has an
  empty-board photo.
- With a result: how many operatives have a name, and one row for each base.
  Green is sure, yellow is not sure, red `?` is unknown.
- After a click on a base: the size of the base, a photo of it, the
  "Wounds" controls, the ring buttons, the distance to a second base, the
  names with their scores, and the "Remove ID" button. Click the base again, or an empty part of the
  board, to close this.
- After a click on a hatchway or a breach point: its name, "Closed" or
  "Open", and a button that opens or closes it.
- Debug (click it to open): tags found, fit error, frames per second, and the
  time for tag detection. The `panel` layout always shows these values.

On the camera image:

- Green outline and number on each card: the app found the card.
- White cross: the board corner that the app calculated.
- Thin green lines: the board outline between the four corners.
- Yellow lines: the map grid.
- Terrain (when you chose a layout): white walls, grey pillars, yellow
  hatchways, orange breach points (filled when closed; when open, a gap in the
  wall with a short yellow or orange line at each end), white rings with a cross for objective
  markers and a white dashed circle 1 in around them, blue rings for
  teleporter pads, dark see-through areas for light terrain (the shape of the
  card symbol, not of the real piece; it has no effect on measurements). With **Drop zones** on: a light grey and an orange area
  for the deployment zones.
- Green circle with a number: a base that the app found, and its size in mm.
  The circle is the bottom edge of the base, where it touches the board.
- Name under a base (after `space`): the operative, on a dark box. When you
  chose the teams, the circle and the name have the colour of the team: red
  for the team at the left, blue for the team at the right, and grey with
  `?` for an unknown operative. With no team chosen: green is sure, yellow
  is not sure, red `?` is unknown. The lists show how sure the app is as a
  percentage, for example `78%`.
- Health bar above an operative (after `space`): its wounds. A longer bar
  is an operative with more maximum wounds. Green: the operative is unharmed
  (all its wounds are left). Orange: the operative is wounded (half of its
  wounds or more are left). Red: the operative is
  injured (fewer than half of its wounds are left). An empty bar with a red
  edge: no wounds are left.
- White outline: the hatchway or breach point that you selected.
- White circle: the base that you selected.
- Magenta circle around the base, with space between them: the range ring.
- Magenta circle on a base and a magenta line with a number: the second base
  and the distance in inches from edge to edge.
- Magenta area with a heavy edge, and a magenta line with bends: the range
  and the distance with "Measure around walls" on.
- Grey circles with lines between them, and numbers: the move to the mouse
  pointer with "Measure movement" on. Each circle is the base at a bend or at
  the end of the move.

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
  turn the model. With the empty-board photo (`e`), the app finds a base
  with less of its edge in view.
- **A base of a new operative is not found.** The app looks only for the
  base sizes of your gallery teams. Teach the app this operative (setup
  step 8) with its base size.
- **A token or a terrain part shows as a base.** Take the empty-board photo
  (`e`) after you set up the terrain and the tokens,
  before the operatives go on the board.
- **The inspector shows "has no maximum wounds".** The team file does not
  have the wounds of this operative. Start the enroll mode with the wounds,
  for example `uv run battlesight enroll --team kommandos --operative "Boss Nob" --wounds 12`,
  and push `q`. Or add `wounds = 12` to the operative in
  `data/teams/<team>/team.toml`. Then start the app again.
- **The size of a base is wrong.** Check `base_mm` of each operative in
  `data/teams/<team>/team.toml`. Without a gallery, make sure that
  `diameters_mm` in `data/board.toml` lists only the base sizes of your teams.
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
