---
name: record-map
description: Record a Kill Team Tomb World terrain layout from a photo of its layout card (or of the set-up board) into a Battle Sight map file in data/maps/tomb-world/. Use when the user gives a layout card image and asks to record, save, add or create a map or layout.
---

# Record a Tomb World map from a photo

You read a photo of a Tomb World layout card and write one map file:
`data/maps/tomb-world/layout-<card number>.toml`. No script detects the
terrain. You read the photo, write the file, draw it with the `map` command,
and compare. Worked example: `data/maps/tomb-world/layout-6.toml` (card 6).
Format reference: `docs/cli.md`, section "Terrain maps".

## Grid

- The card shows the board in the same orientation as the board under the
  camera (corner tag 0 at the card top-left).
- 7 x 6 grid sections. A grid point is `[col, row]`: col 0..7 from left to
  right, row 0..6 from top to bottom. Row 0 and col 0 are the outer grid
  lines (12 mm in from the board edge), not the board edge.
- Points are on the grid or a half grid (steps of 0.5). When an item is off
  the half grid, use `at_mm = [x, y]` in board mm: x = 12 + 97 * col,
  y = 12 + 97 * row.
- The photo has perspective and a fold. Count grid lines; do not measure
  pixels across the whole card.

## What to record

| On the card | In the file |
|---|---|
| Black line between squares = wall, with its label (`A1`..`A4` = 2 grid long, `B1`..`B4` = 1 grid long) | `[[wall]]` with `key` (the label), `from`, `to` (pillar centres) |
| Black square at a wall end or junction (a lighter one too) | point in `pillars` |
| Middle of every `Ax` wall, except `A2` | point in `pillars` (the card does not draw it) |
| Square with an ✕ at a wall end | point in `wall_ends` (half pillar past the wall end; the app adds it to the wall length) |
| Hexagon marker next to a wall | `opening = true` (the app makes it a breach point on `A1` and `B2`, a hatchway on the other walls) |
| No hexagon marker next to the wall | `opening = false` |
| Circle with a skull (white or red, same meaning) | `[[objective]]` |
| Black circle with `T` | `[[teleporter]]` |
| Black rectangle or octagon with `C1`..`C5` (light terrain) | `[[light]]` with `key` (the label), `at` or `at_mm` (centre), `long = "x"` or `"y"` (board axis of the long side; C1 and C2 only, default `"x"`) |
| Dark grey area, dark orange area | `[[zone]]`, `name = "grey"` / `"orange"`, `from` top-left, `to` bottom-right grid point. A side on an outer grid line goes on to the board edge. |

Ignore: green squares along a wall, faded
grey squares (unused pillar positions, mostly at the board edge), dashed
lines, the light grey and light orange halves.

## Light terrain

The file holds the footprint of the card symbol, not the real outline of the
piece. Sizes are in `[terrain] light_mm` of `data/board.toml` (long x short,
mm): C1 81 x 63 octagon, C2 62 x 46, C3 48 x 48, C4 45 x 45, C5 40 x 40. A
grid section is 97 mm, so C2 is about 2/3 x 1/2 of a section.

- Measure the centre, not a corner. Most C2 and C4 centres are on a grid
  point; C1, C3 and C5 are often off the half grid.
- To measure, warp the card photo to board mm with the four outer grid
  corners (homography to x = 12 + 97 * col, y = 12 + 97 * row), then find the
  dark shape around the label with a local threshold and take its bounding
  box. Check each outline by eye: shadows and a touching objective merge with
  the shape (use a darker threshold then).
- Snap the position with "Placement of objects".

## Openings

An opening sits at the centre between two pillars, never on a pillar.

- `B` wall, or `A2` (no middle pillar): leave out `opening_at`. The opening
  is at the wall middle.
- `A` wall with a middle pillar: give `opening_at` = the centre of the half
  next to the hexagon marker. Example: wall `[3, 1]` to `[5, 1]`, marker next
  to the right half: `opening_at = [4.5, 1]`.

The marker can be on either side of the wall. Only its position along the
wall matters.

## Placement of objects

The card designer puts light terrain, objectives and teleporters in one of
these ways. They are in order: the first is the most common. Measure the
position on the photo, then snap it to the first rule that fits within
about 5 mm, separately for x and for y. Write the rule in a comment.

1. Its centre is on a grid line (or a half grid line). Use `at`, for
   example `at = [4, 2]`.
2. Its side touches a grid line: a printed (whole) grid line, not a half
   grid line. Centre = line + half the size (light
   terrain: half its side along that axis; objective: 20 mm, teleporter:
   30 mm). Example: C5 (40 mm) with its left side on col 4 (x = 400):
   x = 420.
3. Its side touches another object (light terrain, objective, teleporter,
   wall end or pillar). Place the other object first, then this one next to
   it. Example: an objective that touches the left side of that C5:
   x = 400 - 20 = 380. A wall end reaches 6.5 mm past its grid point, a
   pillar 6.5 mm from its grid point.

The board edge counts as a grid line for rule 2 (for example C1 81 mm along
x at the right edge: x = 703 - 40.5). Rules 2 and 3 give a point off the
half grid: use `at_mm` and give the sum in the comment. Only when no rule
fits, keep the measured point and say so in the comment.

Objects never overlap. When a rule gives an overlap with a neighbour (for
example an objective on a half grid point 2 mm into a C2), use rule 3
instead: the two touch. After you write the file, check that no light
terrain, objective or teleporter overlaps another or a wall.

## Steps

1. Find the grid lines and number them (col 0..7, row 0..6).
2. List every wall: label (`key`), `from`, `to`, marker or not, which half the marker
   is next to.
3. List the pillars: all wall ends and junctions, plus the middle of each `A`
   wall except `A2`. Put a wall end (✕) in `wall_ends`, not in `pillars`.
4. Light terrain `C1`..`C5`: label, centre, long axis (see "Light terrain").
5. Objectives and teleporters. Snap light terrain, objectives and
   teleporters with "Placement of objects" (rule 1, then 2, then 3).
6. Deployment zones.
7. Write the file. Start from `layout-6.toml`: `schema = 1`, `name =
   "Layout <n>"`, `terrain = "Tomb World"`, `source`. Give each wall its
   label as `key`. The same label can be on more than one wall.
8. Check and draw it:

   ```bash
   uv run battlesight map data/maps/tomb-world/layout-<n>.toml --out out/maps/
   ```

   Exit 1 prints the problem (for example a wall end with no pillar). Fix it
   and run again. The loader checks the key against the wall length and
   the middle pillar (an A wall except A2 needs one). Then open
   `out/maps/layout-<n>.png` and compare it with the photo, wall by wall:
   wall keys (above or right of each wall), walls, pillars, openings (yellow hatchway, orange
   breach), objectives (white ring), pads (cyan ring), light terrain (dark
   fill with its key), zones.
9. If a photo of the real board with this layout exists (for example
   `tests/images/<name>.empty.png`), draw the map on it to check the
   positions, the same way as for card 6: rectify the photo with
   `Pipeline(cfg, until="rectify")`, then `overlay.draw_terrain` with
   `rectify.mm_to_rect`.
10. Tell the user the counts (the `map` command prints them) and each item
   that you were not sure of (an off-grid objective, a faint marker).
