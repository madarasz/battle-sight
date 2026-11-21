# src/battlesight: package guide

Module map. Read the topic file in `docs/dev/` before you change a module. `ui/CLAUDE.md` has the Qt front end.

## Module map

| Module | Job | Details |
|---|---|---|
| `app.py` | argparse CLI: no command = live app (`cmd_live`), subcommands `enroll`, `gallery`, `calibrate`, `image-pipeline`, `map` | below, `docs/cli.md` |
| `live.py` | game core of the live app: `LiveSession`, `ACTIONS`, `BoardTracker` | `docs/dev/live-session.md` |
| `pipeline.py` | single-frame stages `corners`, `rectify`, `bases`, `ident` | below |
| `config.py` | `BoardConfig` from `data/board.toml` | `docs/dev/board-and-camera.md` |
| `camera.py`, `scrcpy.py` | video sources, `CameraSettings` (`data/camera.toml`) | `docs/dev/board-and-camera.md` |
| `calib.py` | AprilTag corners, `DriftMonitor`, `Lens`, ChArUco | `docs/dev/board-and-camera.md` |
| `rectify.py` | `BoardView` homography, top-down warp, grid check | `docs/dev/board-and-camera.md` |
| `circles.py` | base detection (priority 3) | `docs/dev/bases-and-ident.md` |
| `ident.py` | operative identity, `Gallery`, `assign` (priority 4) | `docs/dev/bases-and-ident.md` |
| `bench.py`, `ident_bench.py` | bases and name benchmarks | `docs/dev/bases-and-ident.md` |
| `measure.py` | rings, rim gaps, hit tests (priority 5) | `docs/dev/live-session.md` |
| `walls.py`, `moves.py`, `shoot.py` | measure around walls, movement, targeting | `docs/dev/walls-moves-shoot.md` |
| `terrain.py`, `lightfit.py` | map files, light terrain pieces and their poses (priority 6) | `docs/dev/terrain.md` |
| `score.py` | score, turn, Crit Op, Kill Op | `docs/dev/live-session.md` |
| `state.py` | saved game state: folder with `state.json`, frame, reference; auto-save, restore; the list of saved and not saved `LiveSession` attributes | `docs/dev/live-session.md`, `docs/cli.md` |
| `overlay.py` | draw-in-place helpers on BGR frames | below |
| `colors.py` | all UI colours from `data/color-settings.toml` | below |

## Rules for all modules

- The core must not depend on Qt. Only `ui/` imports PySide6. A later FastAPI and browser front end must reuse the core.
- All lengths are in mm. Board coordinates: origin at the top-left board corner, x right, y down. Corner index 0..3 = TL, TR, BR, BL, and `tags.ids[k]` is the tag at corner k.
- Board size, tag sizes, tolerances, base, token, and door sizes are config values, not constants.
- A new live feature is an entry in `live.ACTIONS` with a handler in `LiveSession`, not a new subcommand or option.
- New `LiveSession` state is saved: add it to `state.SAVED` (and `capture`, `apply`) or give a reason in `state.NOT_SAVED`, in the same change. A test fails otherwise.
- A new priority is a new stage in `pipeline.py`.
- Never a colour literal in the code (see `colors.py`).
- Never call `cv2.imshow` or `cv2.waitKey` in the live app.

## app.py

Entry point `battlesight = battlesight.app:main` (hatchling, `src` layout). `app.py` holds the argparse parser: no command = the live app (`cmd_live` builds a `live.LiveSession` and calls `ui.shell.run_window`; its options are on the main parser), plus the subcommands `enroll`, `gallery`, `calibrate`, `image-pipeline`, `map`. `_apply_camera_settings` fills the camera options left out on the command line from `data/camera.toml`: option > file > built-in default. `docs/cli.md` "All commands" lists every command.

## pipeline.py

Single-frame analysis, no camera or window. Stages are `corners`, `rectify`, `bases`, `ident`. Stages register in run order with `@stage(name, draw=..., views=...)` (`views` returns extra named images, `--out` writes them as `<name>.<view>.jpg`) and return a `StageResult` (ok, message, JSON-safe `data`). A failed stage skips the later ones. `Pipeline.analyze(frame, until=...)` stops early (the live `b` key). `Pipeline.analyze` gives a `FrameResult` (`to_dict()` is the JSON report, with `schema` version), `Pipeline.annotate` draws all stage overlays. Add each new priority as a stage here, so the live app, the `image-pipeline` subcommand and tests pick it up. The `image-pipeline` subcommand exit codes: 0 all stages ok, 1 a stage failed, 2 bad input. `Pipeline(cfg, until, camera)` takes the `CameraSettings`.

Scratch keys between stages: rectify puts `view` and `rectified`; bases puts `bases` (list of `Circle`) and `bases_image`; ident puts `base_crops` and `idents`.

## overlay.py

Draw-in-place helpers on BGR frames. Line and font sizes scale from a 1080p baseline, so overlays stay readable at 4K.

## colors.py

All UI colours, read at import from `data/color-settings.toml` (committed). One constant per function, never per colour: key `team_1` = `TEAM_1_COLOR`. Values are BGR tuples for OpenCV, Qt strings (`#rrggbb`, `#aarrggbb`) for the `[window]` table; `qt_hex` converts BGR. A missing, unknown or bad key raises `ColorError` at import. A new colour is a new key in the file and a constant in `colors.py` (`tests/test_colors.py` checks keys and constants match).
