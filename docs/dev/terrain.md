# Terrain layouts and light terrain (priority 6)

Modules: `terrain.py`, `lightfit.py`, their handlers in `live.py`. Tests: `tests/test_terrain.py`, `test_lightfit.py`. Skills: `.claude/skills/record-map/` (map files), `.claude/skills/record-terrain/` (piece files). CLI: `docs/cli.md` "Terrain maps".

Design rule: walls and openings are not detected. Each layout is a map file, written from the layout card photo with the skill `record-map`.

## Map files (terrain.py)

- `data/maps/<terrain>/<layout>.toml` (committed), written by an agent from a layout card photo with the skill `record-map` (reading rules there).
- `load_map` / `parse_map` check the file (`MapError`) and turn grid points `[col, row]` (0.5 steps) or `at_mm` into board mm.
- Walls: each wall has its card `key` (A = 2 grid, B = 1 grid; A walls except `[terrain] no_middle_pillar` have a middle pillar) and `opening = true|false` (breach on `[terrain] breach_walls`, else hatchway); an opening sits at the centre between two pillars (`opening_at` picks the half of an A wall with a middle pillar).
- Light terrain `[[light]]` (`C1`..`C5`, `at` / `at_mm`, `long = "x"|"y"`, sizes in `[terrain] light_mm`, card-symbol footprint) is drawn only: not in `walls.obstacles`, no effect on measurement or movement.
- `shapes` gives footprints with the `[terrain]` sizes of `data/board.toml`; `overlay.draw_terrain` draws them.
- Command `map` checks and draws files.

## Piece files (terrain.py)

- `data/terrain/<map folder>/<key>.toml` (`Piece`: real outline in the piece frame, origin = bounding box centre, x = long side; no height: the search and the cover use the footprint only). Images are git-ignored.
- Written by an agent from a photo of the underside of the piece with the skill `record-terrain`. The agent traces the rim in photo pixels; its helper `outline.py` shows the photo with a pixel grid and `to_piece` gives the frame angle (best fit of length : width within 20 degrees of the photo axes, not a minimum area rectangle), one scale from the measured length and width, the mirror to the view from above, and the piece frame.
- `pieces_for(map)` loads them; `shapes(map, cfg, pieces, poses)` puts the outline at `poses[k]` = (centre, angle; + = clockwise on the screen; `Shape.located`, drawn as a narrow line in the wall colour); without a pose the card symbol stays.
- A missing piece file is not an error: the card symbol stays.

## Live actions for terrain

- `list_maps` feeds `map.terrain_type`, `map.layout`. No keys, no options: `ui/parts.MapPicker` drop-down lists at the top of each layout panel, the choice is part of the saved game state (`--state`); a normal start has no layout.
- `view.terrain` ("Layout"), `view.zones` ("Drop zones", off at start), `view.keys` ("Terrain keys", off at start: wall keys and `T` on pads, drawn last), `view.light` ("Light terrain", on at start).
- `view.minimap` ("Minimap" radio button with `view.live` / `view.snapshot`; off at start, needs a layout): `LiveSession._render_minimap` draws the layout, grid, a full disc per base, names in the disc (`draw_idents(inside=True)`), health bars and the selection on a dark board in place of the camera image, with the top-down scale; `state["minimap"]`, `state["mode"]` does not change.

## lightfit.py

- Pose of the light terrain pieces in the top-down bases image, no window, from the outline only (no colour, no image of the piece: each owner paints differently). No parallax correction.
- `edge_maps`: Canny on the Lab channels at 2 px per mm, thresholds from the 90th gradient percentile of the searched area, at least 30; 12 direction classes, grown by 0.75 mm.
- `outline_points`, `origin_box` (poses with the outline in the card rectangle plus `[light] margin_mm`).
- `locate`: coarse 4 degrees and 1 mm, fine 0.5 degrees and 0.5 mm; score = (part of the outline on edges - median over the angles) / (1 - median); under `[light] min_score` (0.47) the nominal pose, `confirmed = False`; `hold_mm` = a point the outline must hold.
- `locate_at`: "Locate terrain" search of all piece files around a click, `ClickResult`; `progress(done, total)` after each piece file.
- Ground truth `tests/images/<name>.light.json` (pieces: key, centre, angle; `not_pieces` points). `tools/bench_light.py` measures both searches; do not change the limit or the edge values without that measurement (`docs/specs/measure-and-move.md` section 10).

## LiveSession light terrain

- State: `pieces`, `light_poses`, `light_shown`. The poses are not saved.
- `_locate_light` (action `light.locate`, "Identify all terrain"): each piece with a file and no pose, in the snapshot image while frozen, else the newest frame; never run by a snapshot or a layout load.
- `_start_locating_light` / `_locate_light_at` (`light.locate_at`, "Locate terrain", works with no layout): the next click is on a piece. While it waits, `LiveFrame.crop` is the board under the pointer (`_aim_crop` of `_locate_image`, the top-down image at the start, also searched by the click) and the inspector shows the "Locate operative" page (`ui/parts.aim_texts`). `lightfit.locate_at` searches each piece file of the terrain type (`_read_pieces`) at poses that hold the click. One clear result (the best found piece `[light] min_margin` above the next) sets the pose of the nearest layout item of that key, else a `loose_light` piece. No clear result: nothing changes, a notice tells why, the scores go to standard output. `_locate_light_mm` is the search at a board point, returns the light shape index.
- `_locate_selected_light` (`light.locate_here`, "Identify terrain here" in the inspector of a selected card symbol): the search at its centre, then the found piece is selected.
- `_light_crop`: `LiveFrame.crop` of a selected identified piece, cached per selection.
- `_cover_card_symbols`: a card symbol with an identified piece on it gets `Shape.covered`: not drawn, not clicked, no cover; `_light_outlines(cover=True)` for `shoot.aim`.
- `_remove_light` (`light.remove`, "Remove ID").
- `_build_shapes` keeps the opening order, raises `_light_version` for the `_aim` cache.
- A click on a piece sets `selected_light`. `ui/parts.LightControl` (piece image; "Remove ID" for an identified piece, else "Identify terrain here"; no disabled button) in the inspector of `rosters` and `inspector`.
