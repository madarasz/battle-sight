# Live session, selection, measurement, game state (priority 5)

Modules: `live.py`, `measure.py`, `score.py`. Tests: `tests/test_live.py`, `test_measure.py`, `test_score.py`. Front end: `src/battlesight/ui/CLAUDE.md`. Around walls, movement, shooting: `docs/dev/walls-moves-shoot.md`. Terrain: `docs/dev/terrain.md`.

## LiveSession contract

- `live.py` is the game core of the live app: no window, no camera, no GUI toolkit. All calls come from one thread.
- `LiveSession.step(frame)` (None = camera down) returns a `LiveFrame` (image with board overlays only, JSON-safe `state`, `scale`).
- Zoom: `set_display_width(px)` sets the width of the whole image at 100%. `set_display_view(ZoomView)` (None = 100%) sets the visible region (fractions of the whole image) and its width on the screen. A zoomed render (`BoardWindow.out_transform` -> `OutTransform`) warps or crops only that region at the screen scale, then draws the overlays on it: `_to_px` maps through `_xf` while drawing, `_src_px` stays in source pixels. `overlay.style_scale` sets the line and text size: on the screen they grow as `zoom ** ZOOM_STYLE_POWER` (0.5: 2x at 4x zoom), slower than the image. `LiveFrame.region` = (whole size, region x y w h) in window pixels; None = the whole image. Click and hover positions and `position_px` are window pixels of the whole image at 100% at all zooms. These are front-end calls, not `ACTIONS`: the view is not game state and is not saved.
- `LiveSession.dispatch(action_id, **args)` does one action of `ACTIONS`. `Action.enabled` and `checked` only tell a front end how to show a control; `dispatch` does not read them. `shell = True` actions (help, quit) belong to the front end.
- A new live feature is an entry in `live.ACTIONS` with a handler in `LiveSession`.
- `Action.radio`, `live.shown_view`: radio group `view.live` / `view.snapshot` / `view.minimap`.
- Model preload: `cmd_live` calls `LiveSession.start_preload()`. It loads the identification model (`ident.make_embedder(...).load`) on a daemon thread of its own, the only thread besides the session thread; the two share only `_preload` (one dict, replaced in one assignment). State keys: `model_ready` (false while it loads) and `progress` (`{text, done, total}`, text `PRELOAD_TEXT`, else `null`). `snapshot.name` and `base.add` are disabled while it loads, and their handlers refuse with a notice (a key works also when the button is off). A failed load is a notice (`model not loaded: ...`) on the next `step`; the first identification then loads the model as before. A loaded model (also `color-hist`) starts no thread. Tests replace `start_preload` with a no-op (`tests/conftest.py`, `REAL_PRELOAD` for the preload tests).
- Long actions (`_task(text)`, a context manager): `busy` while they run, state `progress` = `{text, done, total}` with one text per action (`TASK_TEXT`), `on_busy` gets a frame at the start, `on_progress` gets the progress at each step (the worker thread runs the action, so no frame goes out until the end). Steps: `snapshot.name` / `snapshot.bases` = the gallery load (first time), each pipeline stage (`Pipeline.analyze(progress=...)`), the name session; `base.add` click = the base search, its name; `light.locate` = one per layout piece; `light.locate_at` and `light.locate_here` = one per piece file (`lightfit.locate_at(progress=...)`). A long action has priority over the preload in `progress`.
- Do not name an action `ring.*` unless it is a ring size (`ui/parts.ring_controls` takes all of them).

## Snapshot and views

- `LiveSession.frozen` = the snapshot is on the screen. `Esc` (`snapshot.back`, "Go back to Live", key only; also from the minimap) or radio `view.live` only clears it; radio `view.snapshot` sets it again (with no snapshot yet: `_still_snapshot`, the current frame with no bases), so bases, names and selection stay on the live video (mapped with the live `BoardView`) until the next snapshot.
- `view.minimap`: see `docs/dev/terrain.md`.

## measure.py

Pure geometry in board mm, no window: `ring_radius_mm`, `rim_gap` (distance and end points on the rims, 0 for touching bases), `base_at` (hit test, `LabelSession.base_at` uses it), `offset_outline`, `near_polygon`, `melee_pairs`, `objective_contacts`.

## Selection and clicks

- `LiveSession` holds `selected`, `target`, `ring_in`, also without names (after `b`).
- Left click (`board.click`): select a base (replaces the selection).
- Right click (`board.select`): with a selected base, on another base: set or clear the target (`state["right_click"] = "measure"`; no effect while movement is on, and `measure.move` clears the target); else replace the selection, the shell then opens the name popup in layouts with `name_popup` (`right_click = "name"`).
- A click without a base under it selects a hatchway or breach point (`selected_opening`; yellow line 1 in and dashed 2 in around it, `measure.offset_outline`; bases with rim within 2 in get a yellow rim circle, `state["near_opening"]`, `measure.near_polygon`). A click on a light piece sets `selected_light`.
- `base.select(index)` selects a base; `operative.select(label)` (team list click) selects its base, or without one the operative alone (`selected_operative`, wounds only; `selected_label` for both).
- A new selection clears the ring and the target.
- Without movement, the other bases with the rim within the ring distance (around the walls when on) get a yellow rim circle (`LiveSession._in_ring`, `state["in_ring"]`).
- `base.add` is "Locate operative" in the UI: the next left click searches a missed base near it (`pipeline.find_base_near`), `Esc` stops it; while it waits, `LiveFrame.crop` is the board under the pointer.

## Teams, rosters, copies

- `team.set(slot, name)` changes the teams at run time (gallery loads again at the next `space`).
- The team list shows the operatives by maximum wounds, highest first (`LiveSession._listed`: no value last, copies of one type together in copy order, equal wounds in file order). `in_play` keeps the order of adding; only the display sorts.
- `roster.remove(slot)` / `roster.add(slot, label)` change the team list of a side (`LiveSession.in_play`, reset at team change; None = `default_copies` of each operative, see `_listed`); removed operatives go to `Gallery.out_of_play` (not `allowed`, never named) and out of the known base sizes.
- Copies (`max` in `team.toml`, `docs/specs/operative-copies.md`): copy labels `<team>/<operative>#<k>` in `in_play`, wounds and selection; `Gallery.copies` gives `assign` one column per copy; `LiveSession._pair_copies` gives copies to bases by `LiveSession.positions` (`[copies]` config); never compare a copy label with a gallery label without `idt.type_of`.
- Move trails: `_close_snapshot` keeps the named bases of the closing session in `last_positions` (label -> centre, diameter). After a name snapshot, `_find_moves` puts each label that moved at least `ident.reuse_move_mm` in `moved_from`; `_draw_moves` draws `overlay.draw_move_trail_mm` (dashed old base, arrow rim to rim, `SELECTED_COLOR`) for each base whose ident is not `confirmed`, so `names.accept` and `base.choose` end it. Also on the minimap. Both not saved (`state.NOT_SAVED`): a new run starts with no old places.
- Models (`counts_as` variants, `docs/specs/operative-copies.md` section 12): `LiveSession.models` (live label -> model type label) and `model_locked`, both saved. `_update_models` runs after `_pair_copies`; `_read_model` (auto, `[models] switch_margin`); `base.choose` with a type label locks the model, with a copy label the models move with the bases; `base.model_auto` unlocks. State: `candidates` (models, `auto` row), `copy_choices` ("Change ID"), `bases[].model`, `model_name`, `model_locked`. Never show a model in the overlay or the team list.

## Wounds and openings

- `LiveSession.wounds` (`{label: current}`, maximum from `team.toml`, kept across snapshots); actions `wounds.minus` / `wounds.plus` / `wounds.set`, state `selected_wounds`, `wounds` / `max_wounds` in `bases` and `rosters`; `overlay.draw_health_bars`, bar length in proportion to the maximum.
- `LiveSession.opened` (open openings, all closed at each layout load); `opening.toggle` key `o`; drawn as a gap in the wall with two jambs through `terrain.Shape.open`.
- `ui/parts.WoundsControl` and `OpeningControl` show them in the inspector of `rosters` and `inspector`.

## Control marks

- `view.marks` ("Control marks", off at start, enabled only with a named base).
- With names, `measure.melee_pairs` (two teams, rims within 1 in) gives an `overlay.draw_melee_sign` per pair.
- With a layout, `measure.objective_contacts` (rim within `terrain_objective_range_mm` of the marker edge) gives `overlay.draw_objective_arrows` (heads to the centre, or 4 heads in X to the rim when the centre is less than 2 heads away).
- Both yellow (`overlay.MARK_COLOR`). `LiveSession._control` -> `state["control"]`.

## Saved game state (`state.py`)

- `state.capture(session)` is the JSON state, `state.write(session, folder)` writes `state.json`, `frame.png`, `empty_board.png`; `state.apply(session, folder)` restores (`LiveSession.load_state`). File list and rules: `docs/cli.md`, "Saved game state".
- Auto-save: `LiveSession.dispatch` calls `autosave()` after each action except `live.NO_AUTOSAVE` (`board.hover`, `state.save`). A new action saves the state by default.
- `LiveSession.restore_snapshot` detects the bases on the saved frame (`until="bases"`), then sets the saved circles, crops and names.
- **A new `LiveSession` attribute needs a decision.** Put it in `state.SAVED` (with its key in `state.json`, and code in `capture` and `apply`), or in `state.NOT_SAVED` with the reason. `tests/test_state.py` lists `vars(session)` and fails for an attribute in neither. A new `Score` field is saved by itself (`Score.to_state`); a derived one goes in `score.DERIVED`.
- The front end layout name is `LiveSession.ui_layout` (set by the action `view.layout`).

## score.py and game state

- `Score` of one side (pure state): skull cells Crit / Tac per TP 2..4 with 2 skulls, Kill one cell of 6, category max 6, Primary bonus = ceil(category / 2), `hidden` = 0, CP from 3.
- `TAC_OPS` from `data/approved-ops-2025.toml`, `tac_ops_for(archetypes)`.
- `LiveSession.scores` (one per side, saved in `state.json`), actions `score.skull` / `score.tac_op` / `score.primary` / `score.cp` (all with `slot`), `state["scores"]`.
- Both sides: `score.TURNS` (1..4), `CRIT_OPS` (`[crit_ops] names`); `LiveSession.turn` / `crit_op` (saved), actions `game.turn` (`delta`, kept in 1..4) / `game.crit_op` (`name`), `state["turn"]`, `turns`, `crit_op`, `crit_ops`.
- Kill Op is automatic: an operative with 0 wounds is dead (`LiveSession._is_dead`, `dead` in `bases` / `rosters`); `score.kill_grade` (table `data/kill-op.toml`, row = not expendable operatives in the enemy team list) and `kill_scores` (+1 for the higher grade); `LiveSession._update_kill_scores` at each `state()`; `score.skull` with `kill` is an error.
- Dead operatives: no melee mark, `overlay.draw_dead_marks` at the base centre, skull mark in the team list.
