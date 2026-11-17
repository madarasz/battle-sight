# Measurement around walls and movement: specification

Status: in code 2026-11-11 (`walls.py`, `moves.py`, live actions, all three layouts). Not checked with a tape measure on the real board. Extends priority 5 (rings and distances) of `specs/plan.md` with the terrain of priority 6.

Read first: `CLAUDE.md` (`walls.py`, `moves.py`, `measure.py`, `terrain.py`), `docs/cli.md` "Rings and distances", `README.md` "Measure rings and distances".

## 1. Goal

The plain ring and the straight rim-to-rim line are wrong on Tomb World, where terrain stops a measurement and a move. The app must give three results that obey the game rules:

1. The distance between two operatives, around the walls.
2. The area within N inches of an operative, around the walls.
3. The area that an operative can move to with N inches, and the best move to the mouse pointer.

All results are 2D in the board plane, in board mm, from the base rim. No heights.

## 2. Game rules that the app obeys

### 2.1 Measurement (distance, range)

- R1. A measurement cannot go through a wall, a pillar, a wall end, a closed hatchway or a closed breach point.
- R2. An open hatchway or breach point lets the measurement through its gap (hatchway 22 mm, breach point 35 or 28 mm).
- R3. The measurement is the shortest path of straight segments. The line has no width: it can touch a corner and run along a wall face.
- R4. Terrain at the side of the board (on an outer grid line) blocks as far as the physical board edge. The printed margin between a pillar and the edge is no way around it.
- R5. Other operatives do not block a measurement.
- R6. The measurement is from base rim to base rim, or from the rim of the selected base to a point.

### 2.2 Movement

- M1. A move is made of straight increments. Each increment counts as its length rounded up to a full inch (2.75 in counts as 3 in). This is the same as steps of 1 in or less that each count as 1 in.
- M2. The full base must stay clear of walls, pillars, wall ends and closed openings. Thus the base centre keeps one base radius away from them, also around corners.
- M3. The base stays on the board.
- M4. Through an open hatchway or breach point only the centre of the base must go. The base size does not matter there. This costs 1 in more. The base cannot stop while it is over the wall parts beside the gap, and it cannot stop on the open hatchway or breach point itself.
- M5. An operative of the other team blocks the move: the bases cannot overlap at any time.
- M6. The operative can move through operatives of its own team, but it cannot stop on one.
- M7. Light terrain does not change a measurement or a move (drawn only).
- M8. Melee zone: 1 in around each enemy base, rim to rim. With movement on, the zone shows as a yellow dashed circle. A move whose base comes into a zone at any point gets a white crossed swords sign on that enemy.
- M9. "Avoid control range" (check box, off at the start, enabled only with movement on): the base keeps out of all melee zones, for the area and for each move. An enemy that is already within 1 in at the start blocks only with its base.

Not in scope: teleporter pads, vantage and height, climb and jump, dash and charge limits, movement through closed openings, tokens.

## 3. User interface

| Control | Action | Key | Enabled | Start value |
|---|---|---|---|---|
| "Measure around walls" check box (View section) | `measure.walls` | `w` | with a layout | on at each layout load |
| "Measure movement" check box (below the ring control) | `measure.move` | `m` | always | off |
| "Avoid control range" check box (next to it) | `measure.avoid_melee` | none | with movement on | off |
| Mouse pointer on the board | `board.hover` (`x`, `y` image pixels, or null) | none | sent only with movement on, a ring and a selected base | none |

What the ring control (keys `1` to `9`, slider) shows for the selected base:

| Movement | Around walls + layout | Ring shows |
|---|---|---|
| off | off, or no layout | plain circle: base radius + N in |
| off | on | range area around walls (2.1) |
| on | any | movement area (2.2). Terrain always applies when a layout is loaded |

- Areas: magenta fill (alpha 0.25) and a heavy magenta edge. A hole in the area is not filled.
- Distance to a second base (left click): a magenta line with bends, inches at half its length. With "Measure around walls" off it is the straight line. Movement does not change it.
- Move to the pointer: grey circles (the base at each bend and at the end), the two lines that the rim follows, the inches of each increment at its middle (when there is more than one), the inches in all in a box below the end.
- Status bar: `around walls 6.10"`, `around walls no path`, `move 5" (3 + 2)`.
- The movement area is the base footprint: all points that some part of the base can cover. The pointer in the outer band of the area (where the centre cannot go) snaps to the nearest position that the centre can get to. The pointer out of the area gives no move.
- A base without a name has no team. When the selected base has no name, no base blocks (M5 needs `space`). All other bases are holes in the movement area (M6), with or without a name.

State fields (`LiveSession.state`): `around_walls`, `distance` = `{mm, in, around}` (`mm` / `in` null = no path), `movement`, `move` = `{in, increments}` or null.

## 4. Terrain model

`walls.obstacles(shapes, size_mm, outer)` gives axis-aligned rectangles `(x0, y0, x1, y1)` from `terrain.Shape` items (walls are horizontal or vertical in the map files):

- wall: pillar centre to pillar centre, `terrain.wall_mm` thick. An open opening (`Shape.open`) on it cuts a gap of its width.
- pillar, wall end, closed hatchway (22 x 10 mm), closed breach point.
- R4: a rectangle that gets to an outer grid line (`BoardConfig.grid_lines_mm`) goes on to 1 mm past the board edge.

`walls.open_gaps(shapes)` gives the rectangles of the open openings (for M4). Zones, discs, keys and light terrain are not obstacles.

## 5. Measurement algorithm (`walls.Navigator`)

Exact vector geometry, no raster for lengths.

1. `clear(p, q)`: a segment is clear when it does not go through the open interior of a rectangle (slab clipping, rectangles less 1e-6 mm). Rectangles overlap at pillars, so there is no leak between two of them.
2. Nodes: the rectangle corners that are not in another rectangle and not off the board.
3. Visibility graph: edge between two nodes when clear, weight = length. Rows are made on demand and kept.
4. A base is a disc. Its link to a node uses the rim point on the line to the base centre (exact) and 64 rim points. A rim point counts only when the centre sees it (a detected rim can be a little in a wall).
5. `path(a, b)`: a clear centre line gives `measure.rim_gap` (same number as without walls). Else Dijkstra from the rim of A to the rim of B. No path gives None.
6. `reach(base, range)`: Dijkstra with the range as limit. Sources: base centre (range + radius), 16 rim points (range), each reached corner (range less its distance). Each source fills its disc on a mask (`RANGE_PX_PER_MM` = 4) without the shadow of each rectangle. The shadow of a rectangle side goes out to an arc beyond the disc (a straight far side leaks next to a long wall). The union, with the base disc, is cut at the board edge. Contours go back to board mm.

A grid Dijkstra or fast marching field was rejected for lengths: 2 to 8 % error.

## 6. Movement algorithm (`moves.move_field`, `MoveField`)

Raster of base centre positions, `MOVE_PX_PER_MM` = 1.

Masks:

- `stand` (the base can stop): board inset by the base radius, less each rectangle grown by the base radius (round corners), less the enemy discs and the friend discs (radius = both radii), less each open gap rectangle grown by the base radius. The clearance is `SLACK_MM` (0.5) less than the radius, for a base that touches a wall.
- `free` (the centre can move): `stand` with the friend discs, and a zone around each open gap (base radius + 3 mm around the gap line, less the rectangles grown by 1 mm). In the zone the base can be over the wall parts (M4).
- `barrier`: a line 2 pixels thick in each open gap. To go over it costs 1 in more.

Field (`steps`, inches to each position):

1. One straight line from the start gets the exact length rounded up (rays over the full raster).
2. Level N + 1 = each position that a position of level N sees within 1 in. Only the edge of level N is a source. Far from terrain this is a plain dilation; near terrain, 240 rays per source on the raster.
3. A step over a barrier gives level N + 2.
4. A start that is a little in terrain gets out over `ESCAPE_MM` (3).

Results:

- `outline()`: reached `stand` positions, grown by the base radius, contours at 4 times zoom for a smooth edge.
- `route(point)`: goes back from the point level by level to the start (a strict line check first, then a loose one for raster edges). Then it picks the positions of that chain that give the fewest inches, then the fewest increments. An increment up to `ROUND_SLACK_IN` (0.06 in, 1.5 mm) over a full inch counts as that inch.

`LiveSession._enemies`: named bases of another team than the selected base. All other bases are friends.

## 7. Accuracy and speed

| Item | Value |
|---|---|
| Path length, range edge (vector) | within 0.5 mm of hand-computed values in the tests; mask 0.25 mm |
| Movement area edge | about 1 mm (raster), up to about 1 mm short for each bend |
| Route total against the area | never more inches than the field; can be 1 in less near a level edge |
| Layout 6, 9 in range area | about 0.1 s |
| Layout 6, path | 0.02 to 0.03 s |
| Layout 6, 9 in movement field | about 0.15 s |
| Route to the pointer | about 2 ms |

`LiveSession` keeps the navigator (per layout and set of open openings), the last path, range area, movement field and route. The live view draws each frame, so nothing is computed again until a base, the layout, an opening, the ring or the pointer changes. The shell sends one pointer position at a time and keeps only the newest.

## 8. Tests

- `tests/test_walls.py`: R1 to R6 on synthetic rectangles with hand-computed lengths.
- `tests/test_moves.py`: M1 to M6.
- `tests/test_live.py`: actions, state fields, images, openings, enemies by team, no stop on another base.
- `tests/test_ui.py`: check boxes in each layout, status text, pointer positions one at a time.

Done test (open): on the real board, 5 distances around walls and 5 moves agree with a tape measure within 3 mm, and each shown move total agrees with a count by hand.

## 9. Open points

- Tape measure check on the real board (section 8).
- Equal-cost routes: the shown one is not always the one with the fewest bends.
- A bend of a route can be in the zone of an open gap (base over the wall). The rules permit to go through, the display of a bend there can look wrong.
- Teleporter pads are not in the movement area.
- Volkus (no walls on a grid): only the board edge and bases apply. Its terrain needs its own obstacle shapes.

## 10. Targeting lines (`shoot.py`, "Shoot at")

Status: in code 2026-11-15. Rules from the "Intervening" page of the core rules.

- T1. The shooter draws straight lines, 1 mm in diameter, from one point of its base to each facing part of the target base. The app uses the mouse pointer as that point. The lines fill the triangle from the pointer to the two tangent points on the target rim (`shoot.tangent_points`), less the target base.
- T2. Heavy terrain: walls, pillars, wall ends, closed hatchways and closed breach points (`walls.obstacles` without the board-edge extension). An open opening cuts a gap in its wall. Light terrain is not heavy.
- T3. A heavy piece that the lines touch is intervening (red fill and edge on the video).
- T4. OBSCURED: a part of an intervening piece in the lines is outside the 1 in zones (rim + 1 in) around both the target and the shooter.
- T5. IN COVER: a part of an intervening piece in the lines is in the 1 in zone around the target. T4 and T5 can both show.
- T7. Light terrain (C1 to C5) gives cover as T5 and never obscures. Its footprint is the recorded outline of the piece (`data/terrain/<folder>/<key>.toml`) at the pose found on the board ("Identify terrains"); without a piece file or a pose, the card symbol. A light piece that gives cover shows red.
- T8. No cover while the target is within 2 in of the shooter: rim to rim with a shooter base (T6), else from the pointer to the target rim. This is also for heavy terrain. State `too_near`.
- T6. The shooter zone is 1 in around the shooter base: the base under the pointer, other than the target and not of the target's team (a base without a name can be the shooter). That base also gets a yellow dashed 1 in circle; the target always gets one. With the pointer on no such base, the shooter zone is 1 in around the pointer.

UI: "Shoot at" (`measure.shoot`, no key, enabled with a selected base) in the inspector header after "Remove ID". While it is on the button reads "Cancel"; `Esc` (`snapshot.back`) and a new selection also stop it. The inspector crop is the board under the pointer (`LiveSession._aim_crop`, as for "Locate operative"), and OBSCURED / IN COVER take the place of the ring, movement and distance. The shell sends `board.hover` while `shooting`.

Algorithm (`shoot.aim`): each heavy rectangle, grown by 0.5 mm for the line width, is sampled every 0.5 mm; a sample is in the lines when it is in the triangle and outside the target base. The sample zone tests give T4 and T5. Light terrain: the part of the bounding box of each outline in the target zone is sampled every 0.5 mm; a sample in the lines, in the zone, and in the outline or within 0.5 mm of its edge (`_touches_polygon`, even-odd rule) gives cover. `LiveSession._aim` keeps the result until the pointer, a base, the layout or an opening changes. State: `shooting`, `shot` = `{"obscured", "in_cover", "cover_by", "too_near", "intervening", "shooter"}` or null (no pointer on the board, or the pointer on the target).

Light terrain pose (`lightfit.py`, 2026-11-16, search changed 2026-11-17): "Identify terrains" searches the outline of each piece file in the top-down image, only at poses with the outline in the card rectangle plus `[light] margin_mm`. The score is the part of the outline on image edges of the same direction, against the median over all angles at that place. The Canny thresholds come from the gradients of the searched area (90th percentile, at least 30), so dark pieces and blurred photos also give edges. No colour and no image of the piece: the outline comes from an underside photo (skill `record-terrain`) and each owner paints differently. "Locate terrain" does the same search around a clicked point for each piece file, with or without a layout, only at poses with the click in the outline (else an objective ring next to the piece takes the search away), and takes the piece only with one clear result (`[light] min_margin` 0.05). Measured with `tools/bench_light.py` on 5 test photos (21 pieces, 23 points with no piece), limit 0.47: clicks 105 of 105 right (before: 16 of 21 at the centre, about 61 of 84 off the centre), no wrong key, 1 false find of 23; card search 21 of 21 (before 17), 3 false of 115 (scores 0.47 to 0.52, on bases). Done test open: all 5 pieces on the real board. Rejected: a colour mask of "not board" pixels (misses parts of the pieces, marks tokens and prints), an image template (0.40 against 0.20, but needs a photo of each owner's paint), the empty-board reference (the terrain is in it), fixed Canny thresholds (lost dark and blurred pieces), an edge tolerance only to the outside of the outline (matched everything), a colour contrast across the outline (bases and markers have it too). No parallax correction: the far side of a high piece can be off by some mm at the board edge. A wrong result: "Remove ID".

Open: the shooter cannot choose a second point; vantage and the Obscured exceptions (for example a shooter within 1 in of the terrain that it looks over) are only the zone rule T4.
