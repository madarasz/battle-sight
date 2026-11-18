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
13. ⏳ Support multiple identical models: copies in the team list, told apart by position (UNTESTED)
14. ⏳ Shoot at measurements (UNTESTED)
15. ✅  Tomb World light terrain: real outlines, found on the board, cover for Shoot at
16. ✅ KillOp calculation, dead operatives
17. 🚧 Operative states: injured, dead, borrowed, poisioned, etc.
optional 🚧 Tokens: find tokens, list operatives in range

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
9. Record your light terrain pieces (optional).
10. Change the window layout settings (optional).

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

Hold the mouse pointer on a control to see its key. Push `h` to see all key
bindings. Use the menu View > Layout to change the window layout.

## Use the app

### 1. Before the game

1. Choose the terrain type and the layout card. Put the board so that corner
   card 0 is at the top-left corner of the card picture.
2. Set up the terrain and the tokens, then click "Take empty-board
   reference". This helps the app find the bases.
3. If you recorded light terrain pieces, click "Identify all terrain".
4. Choose the team of each player.
5. Remove the operatives that do not play ("Remove operative"). To add a
   copy of an operative, click "Add operative".

To add a new layout card, give a photo of the card to the AI agent (Claude
Code) and ask it to record the map.

### 2. Find the operatives

1. Put the operatives on the board.
2. Click "Identify all operatives". Check the names with a percentage or a
   `?`.
3. If a name is wrong, click the base and choose the correct name. The app
   learns from each correction.
4. If the app found an operative where there is none, click "Remove ID".
5. If a base is not found, click "Locate operative", then click the centre
   of the base.
6. To swap two copies of the same operative, right click one base and choose
   the correct copy.

Click "Identify all operatives" again after the operatives move or after a
change to the teams. The names stay on the live video until then.

### 3. Measure

1. Click a base to select it. Set the distance to see a range ring.
2. Right click a second base to see the distance between the two bases.

With a layout, measurements go around walls and closed hatchways. Clear
"Measure around walls" to measure in a straight line.

"Measure movement" shows where the selected operative can move. Put the
mouse pointer in the area to see the best move. "Avoid control range" keeps
the move out of enemy control range.

"Shoot at" (with the target selected): put the mouse pointer on the shooter.
The terrain in the way shows red. Push Esc to stop.

### 4. Track the game

Click an operative to change its wounds. An operative with 0 wounds is dead.
Click a hatchway or breach point to open or close it. All start closed.

Turn, Crit Op, Tac Op, Primary, victory points and command points are in the
game bar and below each team list. They start again at each app start.
The app counts the Kill Op from the dead enemy operatives. The last Kill Op
skull is the point for more kills than the opponent.

### Saved photos

```bash
uv run battlesight image-pipeline my-photo.jpg --show
```

The app does the same analysis on a saved photo and shows the result. To use
an empty-board photo, add `--reference empty.jpg`. To use only some teams,
add `--team kommandos veterans`.

## What you see

- Name colour: green = sure, yellow = not sure, check it, red `?` = unknown.
- Health bar: green = unharmed, orange = wounded, red = injured.
- Skull on a base or in a team list: dead operative.
- Magenta: range ring and distance.
- Yellow lines around a hatchway: 1 in and 2 in from it.
- Yellow crossed swords: melee. Yellow arrow heads: objective control.

| Status line | Meaning | What to do |
|---|---|---|
| Yellow "locking reference" | The app learns the board position. | Do not touch the board. |
| Yellow "BOARD MOVED" | A card moved. | Keep the board still. |
| Red "CARD MOVED" | One card is not on its board corner now. | Put the card back against the board corner. |
| Red "TAG LOST" | The camera cannot see a card. | Remove what covers the card. |
| Red "camera down" | The app gets no image from the camera. | Check the camera and its cable or network. |
| Red "CORNERS NOT SEEN" | "Identify all operatives" cannot find the board. | Remove what covers the named card, or move the camera. |

## Problems

- **The board slid but there is no warning.** The app measures from the
  cards, not from the board. Put the board corner back against the black L
  of the card.
- **Error that width and height are the wrong way round.** Change
  `width_mm` and `height_mm` in `data/board.toml`. See
  [docs/setup.md](docs/setup.md), step 5.
- **The camera does not start.** Make sure that the Terminal app has camera
  permission (System Settings > Privacy & Security > Camera). Find the
  correct camera number with `uv run tools/list_cameras.py`. For Camo, a
  Wi-Fi stream or scrcpy, see [docs/setup.md](docs/setup.md),
  Troubleshooting.
- **The terrain drawing does not sit on the real walls.** Turn the board or
  the cards.
- **A base is not found.** The app must see at least half of the black edge
  of the base. Move the camera nearer to the top of the board, or turn the
  model. Take an empty-board reference.
- **A base of a new operative is not found, or the name list is empty.**
  Teach the app this operative (setup step 8) with its base size.
- **A token or a terrain part shows as a base.** Take the empty-board
  reference after you set up the terrain and the tokens.
- **An operative gets the wrong name frequently.** Add more photos of the
  operatives (setup step 8).
- **The inspector shows "has no maximum wounds".** Add the wounds of the
  operative: `uv run battlesight enroll --team kommandos --operative "Boss Nob" --wounds 12`,
  then start the app again.
- **The size of a base is wrong.** Check `base_mm` of each operative in
  `data/teams/<team>/team.toml`.
- **The first "Identify all operatives" is slow.** The app downloads the
  recognition model one time.
- **The grid does not sit on the printed lines.** Check the map settings
  ([docs/setup.md](docs/setup.md), step 5). Then calibrate the lens (step 6).
