# Measure around walls, movement, targeting

Modules: `walls.py`, `moves.py`, `shoot.py` (pure geometry in board mm), their handlers in `live.py`. Tests: `tests/test_walls.py`, `test_moves.py`, `test_shoot.py`. Specification (rules R1 to R6, M1 to M7, algorithms, accuracy, open points): `specs/measure-and-move.md`. Update it when a rule or the model changes.

## walls.py

- Tomb World rule: no measurement through walls, pillars, closed hatchways and breach points.
- `obstacles(shapes)` gives axis-aligned rectangles (an open opening cuts a gap in its wall; terrain on an outer grid line goes to the board edge, so the printed margin is no way around an edge pillar).
- `Navigator.path` = shortest rim-to-rim path (visibility graph on the rectangle corners + Dijkstra, exact; a clear centre line gives `rim_gap`).
- `Navigator.reach` = outline of the area within a path length (discs of the range that is left, seen from the base and from each reached corner, shadows cast on a 4 px per mm mask, contours back to mm). Also `distance_to`, `contains`.
- The line has no width and other bases do not block it.
- Live action `measure.walls` ("Measure around walls", key `w`, enabled with a layout, on at each layout load): `LiveSession._path_around` / `_range_around` keep the last result (the live view draws each frame); `overlay.draw_path_mm` and `draw_range_mm` draw them; `state["distance"]["around"]`, `mm` / `in` None = no path.

## moves.py

- Check box "Measure movement", action `measure.move`, key `m`; pointer position through `board.hover`.
- Rule: straight steps of 1 in or less, each counts 1 in (= an increment rounded up); the full base stays clear of terrain, enemy bases (named, other team: `LiveSession._enemies`) and the board edge; all other bases (`friends`) can be moved through but not stopped on (hole in the area); through an open gap only the centre must go, 1 in more, no stop over the wall or on the open opening.
- `move_field` = raster (1 px per mm) level field (`stand` = where the base can stop, `free` = `stand` + zones around open gaps, `barrier` = gap lines).
- `MoveField.outline` (covered area), `MoveField.route` (increments to a point, snaps a point on the covered area to the nearest centre in reach). The route total can be less than the field level near a level edge (raster); never more.
- `overlay.draw_move_mm` draws the base at each bend.
- With movement on, each enemy gets a yellow dashed circle 1 in around it and, when the route comes into it (`LiveSession._move_melee`, `state["move_melee"]`), a white `draw_melee_sign`.
- `measure.avoid_melee` ("Avoid control range", enabled with movement) gives `move_field(avoid_mm=1 in)`, which keeps area and route out of these zones (not for an enemy already within 1 in at the start).
- `LiveSession._field` / `_route` keep the last result.

## shoot.py

- "Shoot at", action `measure.shoot`, no key, enabled with a selected base; `Esc` or a new selection stops it. `specs/measure-and-move.md` section 10.
- `tangent_points`, `facing_outline` (pointer + facing rim, for `overlay.draw_targeting_mm`).
- `aim` -> `Shot`: `pieces` = intervening heavy rectangles from `walls.obstacles`, `obscured` = a part in the lines outside the 1 in zones of both target and shooter (shooter base, else the pointer), `in_cover` = a part in the 1 in zone of the target.
- `aim(light=[outlines])`: light terrain gives cover the same way and never obscures (`Shot.light` = indexes of the covering outlines, `cover_by`); no cover while the target is within 2 in of the shooter (`NO_COVER_MM`, `Shot.too_near`; also for heavy terrain).
- Origin = the mouse pointer (`board.hover`, the shell sends it while `shooting`); shooter = the base under the pointer that is not the target or its team (`LiveSession._shooter_index`).
- `LiveSession._aim` caches the result; `state["shooting"]`, `state["shot"]`. The inspector crop is `_aim_crop` while shooting.
