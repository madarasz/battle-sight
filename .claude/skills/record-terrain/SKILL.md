---
name: record-terrain
description: Record the real footprint outline of one light terrain piece (Tomb World C1..C5) from a photo of its underside into a Battle Sight piece file in data/terrain/<terrain>/. Use when the user gives a photo of the bottom of a terrain piece and asks to record, save or create its outline, footprint or geometry.
---

# Record a light terrain piece from a photo of its underside

You look at a photo of the underside of one piece and write one piece file:
`data/terrain/tomb-world/<key>.toml`. You trace the outline by eye. The
helper `outline.py` in this folder shows the photo with pixel coordinates
and does the mathematics (scale, mirror, piece frame). Do not write the
piece file by hand. Format reference: `docs/cli.md`, "Light terrain pieces".

## Input from the user

- The photo: the piece upside down on a light table, camera straight above,
  the long side of the piece along a side of the photo (within 20 degrees).
- `key`: the label on the layout cards (`C1`..`C5`).
- `length` and `width` in mm: the size of the footprint along its long
  side, and the size at a right angle to it (as a caliper or two rulers
  measure it: between two parallel lines that touch the piece).

If key, length or width is missing, ask for it. Do not estimate a size from the photo.

## What the footprint is

The footprint is the outer edge of the rim: the surface that touches the
board when the piece is right side up. In the photo this rim is the top
surface, nearest to the camera.

- Trace the outer edge of the rim, not the inner edge of the hollow.
- When the camera is not exactly above the piece, the outer wall shows
  below the rim on one side (a band between the rim and the table, usually
  in paint colour). That band is not footprint: stay on the rim edge.
- A shadow on the table is not footprint.
- Paint on the rim is footprint. Painted pieces often have paint along the outer edge of the rim. Do not cut it
  off as shadow or outer wall. A shadow is a smooth area on the table, on
  one side of the piece only (away from the light). A paint patch has
  texture, touches the dark rim, and is also on the side of the light.
- The rim is often a band 1 to 3 mm wide in a lighter colour than the
  hollow. Trace its outer edge (to the table), not its inner edge (to the
  dark hollow).
- When you are not sure if a band is rim or outer wall, compare the box of
  each choice with length : width. The choice that agrees better is
  usually correct.
- Follow each corner and notch that is 1 mm or larger. 30 to 80 points is usual.
- Flash, mould lines and paint drops smaller than 1 mm: ignore.

## Steps

All commands from the repository root. Put work files in the scratchpad or a temporary folder, not in the repository.

1. Look at the full photo with the grid:
   `uv run .claude/skills/record-terrain/outline.py view PHOTO --out view.png`
   Read the image. Note the box around the piece in photo pixels.
2. Get a first outline:
   `uv run .claude/skills/record-terrain/outline.py suggest PHOTO --out points.json`
   It is the area that is not table colour: only a start. It includes
   shadow and outer wall, and on a rim in a light colour it goes to the
   inner edge of the rim. Expect to move most points. If it fails, write
   `points.json` by eye: a JSON list of `[x, y]`
   in photo pixels, in order along the outline.
   When `suggest` is far off, a short script in the scratchpad can give a
   better start: sample the colours of table, shadow, rim and paint with
   OpenCV (HSV), make a mask (for example dark pixels, less the orange
   shadow hue with high saturation), keep it inside a box around the piece,
   and take the largest outer contour (`cv2.approxPolyDP`, 2 to 3 px).
   Check the result in the views like any other outline: a mask also loses
   bright paint and highlights, and can leak into table stains.
3. Check it in parts. For each quarter of the piece (with some overlap):
   `... outline.py view PHOTO --points points.json --out q1.png --box X0 Y0 X1 Y1`
   Read the image. The grid labels are photo pixels and each point has its
   index in the file. Compare the yellow line with the rim edge. Add
   `--no-grid` (marks at the top and left edge only), without `--points`
   too, when the grid or the line hides a thin rim edge.
4. Correct `points.json`: move, add or delete points. Read coordinates from
   the grid of the zoomed view, not from the full photo. When you replace a
   run of points, write the new points in the direction of the file (the
   point numbers in the view show it), else the lines cross.
5. Repeat 3 and 4 until the line is on the rim edge in each view. Do a
   minimum of two full rounds. Stop when no part is more than about 1 % of
   the piece length away from the edge.
6. Simplify:
   `... outline.py simplify points.json --length L --out points.json`
   It removes steps smaller than 0.2 mm (`--tolerance-mm`). Do this when
   the outline has more than about 80 points (a jagged rim, an outline from
   a mask). Then do one more round of step 3: the point numbers changed.
7. Save:
   `... outline.py save points.json --key KEY --length L --width W --source PHOTO_NAME --check check.png`
   `save` finds the frame angle (within 20 degrees of the photo axes)
   where the outline agrees best with length : width, and prints the
   scale difference that stays, the frame angle, and how sure the scale is.
   - Exit 1 (scales differ by more than 8 % at each angle): the outline or
     a size is wrong, the camera was tilted, or the piece is turned more
     than 20 degrees in the photo. Check the two ends of the long side and
     of the short side first. Do not change the sizes to make it pass and
     do not write the piece file by hand: keep `points.json`, report its
     path and the numbers, and ask the user to measure again.
   - A warning (3 % to 8 %), or a scale sure within more than 2 %: say it
     in the report.
8. Read `check.png`. It shows the outline as seen from above, so it is the
   mirror image of the photo. Check that it is the mirror image, that the
   shape has no crossed lines, and the sizes.
9. Check that the app accepts the file:
   `uv run python -c "from battlesight import terrain as t; print(t.load_pieces('data/terrain/tomb-world'))"`
   The problem list must be empty.

## Report

Give: the file path, the number of points, the three numbers that `save` printed, the parts
of the outline where the rim edge was not clear (shadow, blur, outer wall),
and the path of the last view image with the outline, so that the user can check it.
Make this image after the last change to `points.json`: the user gives
corrections by the point numbers in it.

## Notes

- The live app finds the piece on the board with this outline only
  ("Identify terrains"). A wrong outline gives a wrong or no result.
- The mirror step is in `save`. Do not mirror the points yourself.
- The direction of the piece x axis along the long side is free (the two
  choices differ by 180 degrees; the app finds or the user sets the angle).
