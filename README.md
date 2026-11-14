# Battle Sight

Battle Sight helps you measure in Warhammer 40k: Kill Team. A camera above
the table looks down on the board. A window on your computer shows the live
video with measurements on top of it: the map grid, the operative bases, the
range rings and the distances between operatives.

## Work in Progress

1. ✅ Board corners: AprilTag cards, drift warning
2. ✅ Undistort and rectify: lens calibration, top-down board image, map grid
3. ✅ Bases: find round bases, size and position
4. ✅ Operative ID: name each base from a photo gallery
5. ✅ Rings and distances: range rings, rim-to-rim distance
6. ✅ Terrain layouts: walls, hatchways, objectives and drop zones of each layout card
7. ✅ Team layout: operative list inspector
8. ✅ Game state: operative health, open/closed status of hatchways and breach points
9. ✅ Measurement around walls, movement measurement
10. ✅ Minimap, Control marks (contesting objective, melee engagement)
11. ✅ Scoreboard, turn number, CP counter, faction ability counter, show CritOp
12. ✅ Edit roster: Add / Remove operative from the list
13. ⏳ Support multiple identical models: copies in the team list, told apart by position
14. 🚧 Operative states: injured, dead, borrowed, poisioned, etc.
optional 🚧 Tokens: find tokens, list operatives in range
optional 🚧 Turn counter (where to put it?)

✅ done · ⏳ next · 🚧 not started

## Setup

You need a Mac, a camera above the board (the built-in camera for a test, or
a phone through Camo, a Wi-Fi stream, or scrcpy over USB), a printer, and
scissors.

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

## Start the app

Open the Terminal app and go to the Battle Sight folder:

```bash
uv run battlesight
```

To use another camera for one run, add `--source`, for example `--source 0`
for the built-in camera. To use only two teams, add their names:

```bash
uv run battlesight --team kommandos veterans
```

## The window

Every control has a key. Hold the mouse pointer on a control to see its
key. Push `h` to see all key bindings.

The window has three layouts. Use the menu View > Layout to change the
layout. The app keeps your choice.

| Layout | Panel |
|---|---|
| `rosters` (default) | A control bar above the video, and one team list at each side. |
| `inspector` | The panel at the right changes with your work. |
| `panel` | The panel at the right is always the same and has all the controls. |

The top-right corner of the video shows the mode:

- Red circle, "Live": the live video.
- Blue pause sign, "Snapshot": a result. Click "Go back to Live" to leave it.
- Green square, "Minimap": the minimap.

### Team lists

In the `rosters` layout, choose the team of each player in the lists at the
sides of the video. The operatives of the team show below the list. The
wounds show at the right end of each line, for example `12/15`. A mark in
front of each name shows the identification:

- Eye: the name is confirmed.
- Percentage, for example `78%`: the app found this operative, but nobody
  confirmed the name.
- `?`: the app did not find this operative.

An identified operative shows in the colour of its team. Click an operative
to select it: its line gets a grey background and the inspector shows it. For
an operative that the app did not find, the inspector shows only its name
and wounds. After you change a team, click "Identify operatives" again.

The team list starts with all operatives of the team. To remove an operative
that does not play, select it and click "Remove operative". The app then does
not give its name to a base. To add it again, click "Add operative" and
click its name. Click "Identify operatives" again after a change.

Some operatives can be in a team more than once, for example `Boy 1` and
`Boy 2`. To add one more copy, click "Add operative" and click its name. The
app tells the copies apart by where they were in the last snapshot. When it
is not sure, the names show as unsure. To swap two copies, right click one
base and click the correct copy.

## Use the app

### 1. Choose the map

Choose the **Terrain type** and the **Layout** (the layout card). The app
draws the walls, hatchways, breach points, objectives and other terrain on
the board. Use the View check boxes to show or hide these parts:

- **Layout**: the terrain.
- **Drop zones**: the deployment zones.
- **Terrain keys**: the label of each wall and light terrain piece.

Put the board so that it agrees with the layout card: corner card 0 is at
the top-left corner of the card picture. If the drawing does not sit on the
real walls, turn the board or the cards.

To add a layout, give a photo of the layout card to the AI agent (Claude
Code) and ask it to record the map.

### 2. Find the operatives

1. Before the operatives go on the board, set up the terrain and the tokens,
   and click "Take empty-board reference" (Reference section). The app keeps this photo
   of the empty board and finds the bases more easily.
2. Put the operatives on the board.
3. Click "Identify operatives". The app finds the bases and writes the name of the
   operative under each base. Green: sure. Yellow: not sure, check it. Red
   `?`: unknown.
4. If a name is wrong, click the base and choose the correct name. In the
   `rosters` layout, click "Change ID" first ("Cancel" closes the list). The
   app learns from each correction.
5. If the app found an operative where there is none, click it and click
   "Remove ID".
6. If an operative has no base on the video, click "Locate operative", then
   click the centre of its base. The inspector shows the board under the
   pointer. The app finds the base and shows a list of names. Choose the
   correct name. Click "Cancel" to stop. You can also do this before
   "Identify operatives", to add each operative yourself.
7. Click "Go back to Live". The circles and names stay until the next
   "Identify operatives".

To find the bases without names, use the menu item "Find bases only".

### 3. Measure

You can measure after "Identify operatives" or "Find bases only".

1. Click a base. A white circle shows the selected base.
2. Move the "Distance" slider in the inspector. The app draws a ring that
   many inches from the edge of the base. Set the slider to `Off` to remove
   it.
3. Right click a second base. The app draws a line between the two base edges and
   shows the distance in inches in the status line.
4. Right click the second base again to remove the line. Click another
   base to select it. Click the selected base, or an empty part of the
   board, to stop.

With a layout, the **Measure around walls** check box is on. The
distance follows the shortest path around walls and closed hatchways, and
the ring becomes an area. The status line shows `no path` when all paths are
closed. Clear the check box to measure in a straight line.

The **Measure movement** check box shows the area that the
selected operative can move to with the distance that you set. Move the
mouse pointer in the area to see the best move to that point, with the
inches of each part. A yellow dashed circle shows 1 in around each enemy
operative. A white crossed swords sign on an enemy shows that the move goes
into its circle. Set the **Avoid melee** check box to keep the area and the
move out of these circles.

### 4. Track the game

- **Wounds.** Click an operative that has a name. Click `-` or `+` in the
  inspector, or type a number in the "Wounds" box and push Return. A health bar above the operative shows its
  wounds, and the team list shows them as `12/15`.
- **Score.** Choose the Tac Op
  and the Primary. Click the skulls to give victory points for Crit Op and
  Tac Op in each turning point, and for Kill Op. Click the last lit skull to
  remove a point. Click `-` or `+` at the top to change the command points
  (CP). The Primary points show at the end of the Primary line. The total
  shows at the bottom, in the team colour. The counters start again at each app start.
- **Hatchways and breach points.** All start closed. Click one, then click
  "Open" or "Close" in the inspector. The measurements follow.
- **Control marks** (View check box, on at the start). Yellow crossed swords
  show two operatives of different teams in melee. Yellow arrow heads show
  an operative in control range of an objective. Operatives without a name
  get no marks.

### 5. Other views

- **Minimap.** Shows a drawing of the board in place of the camera image,
  with a disc for each operative. Clicks and measurements work as on the
  camera image. It needs a layout.
- **Adjust perspective** check box. On: the board from straight above. Off:
  the camera image with the four corner cards. Use "off" to put the camera
  and the board in position before a game.
- **Save screenshot** button. Saves the camera image to `data/snapshots/`.

When you move the board or the camera, the app finds the new position by
itself. Keep the board still for a moment after the move.

### Saved photos

```bash
uv run battlesight image-pipeline my-photo.jpg --show
```

The app does the same analysis on a saved photo and shows the result. To use
an empty-board photo, add `--reference empty.jpg`. To use only some teams,
add `--team kommandos veterans`.

## What you see

The status line at the bottom of the window shows the board status, the ring
size and the distance. While the app finds the bases, it shows "working ...",
and the keys do not work. The status line also shows the result of your last
key or click for 6 seconds (red: it did not work, yellow: do the next step,
green: done).

| Status line | Meaning | What to do |
|---|---|---|
| Yellow "locking reference" | The app learns the board position. | Do not touch the board. |
| Green "4 corners OK" | All four cards are visible. | Play. |
| Yellow "BOARD MOVED" | A card moved. | Keep the board still. |
| Green "board position updated" | The app saved the new board position. | Play. |
| Red "CARD MOVED" | One card is not on its board corner now. | Put the card back against the board corner. |
| Red "TAG LOST" | The camera cannot see a card. | Remove what covers the card (a hand, a model, a dice tray). |
| Red "camera down" | The app gets no image from the camera. It tries again by itself. | Check the camera and its cable or network. |
| Red "CORNERS NOT SEEN" | "Identify operatives" cannot find the board. The line names the cards. | Remove what covers the card, or move the camera. |

On the video:

- Green circle with a number: a base that the app found, and its size in mm.
- Name under a base: the operative. With teams chosen, the circle and name
  are red for the left team, blue for the right team, and grey with `?` for
  an unknown operative.
- Health bar above an operative: green = unharmed, orange = wounded, red =
  injured, empty with a red edge = no wounds left.
- White circle: the selected base. White outline: the selected hatchway or
  breach point.
- Magenta circle or area: the range ring.
- Magenta line with a number: the distance between two bases, in inches.
- Grey circles with lines: the move to the mouse pointer.
- Terrain: white walls, grey pillars, yellow hatchways, orange breach
  points, white rings for objectives, blue rings for teleporter pads, dark
  areas for light terrain.

In the `rosters` layout, the inspector shows the selected operative (team,
name, base size, photo, wounds, distance slider, "Change ID" and "Remove ID")
or the selected hatchway or breach point (name, status, "Open"/"Close").

## Problems

- **"BOARD MOVED" does not go away.** Something moves the cards, for example
  a hand or a shaking table. Keep the board still for a moment. If "CARD
  MOVED" shows, put the named card back on its board corner.
- **The board slid but there is no warning.** The app measures from the
  cards, not from the board. Lift the corner and put it back against the
  black L of the card.
- **Error that width and height are the wrong way round.** Change
  `width_mm` and `height_mm` in `data/board.toml`. See
  [docs/setup.md](docs/setup.md), step 5.
- **The camera does not start.** Make sure that the Terminal app has camera
  permission (System Settings > Privacy & Security > Camera). Find the
  correct camera number with `uv run tools/list_cameras.py`. If Camo does
  not show, or you use a Wi-Fi stream or scrcpy, see
  [docs/setup.md](docs/setup.md), Troubleshooting.
- **A base is not found.** The app must see at least half of the black edge
  of the base. Move the camera nearer to the top of the board, or turn the
  model. Click "Take empty-board reference" to help the app.
- **A base of a new operative is not found.** Teach the app this operative
  (setup step 8) with its base size.
- **A token or a terrain part shows as a base.** Click "Take empty-board reference"
  after you set up the terrain and the tokens, before the operatives
  go on the board.
- **An operative gets the wrong name.** Click the base and choose the
  correct name. If it happens frequently, add more photos of the operatives
  (setup step 8).
- **The name list is empty.** The gallery has no operative with this base
  size. Teach the app this operative (setup step 8).
- **The inspector shows "has no maximum wounds".** Add the wounds of the
  operative: `uv run battlesight enroll --team kommandos --operative "Boss Nob" --wounds 12`,
  close the window, then start the app again.
- **The size of a base is wrong.** Check `base_mm` of each operative in
  `data/teams/<team>/team.toml`.
- **The first "Identify operatives" is slow.** The app downloads the recognition model one
  time.
- **The grid does not sit on the printed lines.** Check the map settings
  ([docs/setup.md](docs/setup.md), step 5). Then calibrate the lens (step 6).
