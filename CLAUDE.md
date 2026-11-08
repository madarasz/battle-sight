# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Battle Sight is an augmented reality overlay for Warhammer 40k: Kill Team. A fixed camera looks down on the board. The app finds the board with AprilTags, rectifies the image to board millimetres, and draws measurements (inch grid, rings, rim-to-rim distances, token ranges, door bands) on the video.

Read these before design work:

- `specs/about-project`: the owner's goals, in their own words.
- `specs/plan.md`: decisions, the 7 priorities with "done when" tests, risks, stack, and target layout. The plan is the source of truth for scope. Priorities 1 (board corners) and 2 (rectify) are done. Priority 3 (bases) is in code; its done test waits for a test photo with an empty-board reference. Priority 4 (operative ID) is in code; its done test waits for a real gallery and a labelled game photo.

Work goes in priority order (1 corners, 2 undistort and rectify, 3 bases, 4 operative ID, 5 rings and distances, 6 tokens, 7 doors). Each priority has a measurable done test in the plan. Use it as the acceptance check.

## Documentation rules

- `README.md` is for non-technical users who install and run the app. It tells what the app does, how to start each mode, the keys, what the overlay shows, and fixes for common problems. Keep code internals, tests, config internals, module names, and development steps out of it. Write in ASD-STE100 Simplified Technical English.
- `README.md` starts with a "Work in Progress" list: one line per priority with a status emoji (✅ done, ⏳ in progress or next, 🚧 not started). Update it when a priority starts or passes its done test. Keep each line short.
- `README.md` "Setup" is only a short list that links to `docs/setup.md`. Put all setup and calibration steps (install, print tags, place cards, camera, `data/board.toml` values a user sets, lens calibration) in `docs/setup.md`, for the same non-technical reader.
- Technical and development information (architecture, algorithms, tolerances, tests, layout) goes in this file, `docs/cli.md`, or `docs/testing.md`.
- When a new priority adds a user-facing mode, key, or overlay, update the README "How to use" and "What you see" sections. When it adds a setup step, update `docs/setup.md`.

## Commands

Use `uv` for all work. Python is pinned to 3.12 (ML wheels lag on newer versions).

```bash
uv sync                                   # create .venv, install deps
uv run pytest                             # unit tests only (no camera needed)
uv run pytest -m photos                   # photo regression tests on tests/images
uv run pytest -m ""                       # all tests
uv run pytest tests/test_calib.py::test_detects_four_tags_flat   # one test
uv run battlesight                        # live app: corners, top-down view (v), space = bases + names, b = bases only, h = keys
uv run battlesight --team kommandos veterans   # live app with 1 or 2 gallery teams (default: all teams)
uv run battlesight --source http://PHONE-IP:8080/video  # one run with another source
uv run battlesight enroll --team kommandos --operative "Boss Nob"   # gallery crops, live or --images
uv run battlesight gallery --team kommandos   # list gallery, leave-one-out check
uv run battlesight image-pipeline shot.jpg --team kommandos --json -   # all stages on saved photos
uv run battlesight calibrate              # lens calibration -> [lens] in data/camera.toml
uv run battlesight image-pipeline tests/images              # pipeline on saved photos, exit 0/1/2
uv run battlesight image-pipeline shot.jpg --out out/ --json -     # overlays + per-image JSON, report on stdout
uv run battlesight image-pipeline shot.jpg --show                  # view the overlays
uv run battlesight image-pipeline game.jpg --reference empty.jpg --out out/  # bases in reference mode
uv run tools/list_cameras.py              # list camera indices
uv run tools/list_cameras.py --use 1 --width 3840 --height 2160  # check index 1, save to data/camera.toml
uv run tools/list_cameras.py --url http://PHONE-IP:8080/video    # check a Wi-Fi stream, save it
uv run tools/make_tags.py                 # print sheet -> data/print/corner_tags.pdf
uv run tools/make_charuco.py              # lens calibration sheet -> data/print/charuco.pdf
uv run tools/bench_bases.py               # bases benchmark on tests/images: table, out/bench/
uv run tools/bench_bases.py --sweep min_coverage=0.4,0.5 --no-images   # parameter sweep, best f1 first
uv run tools/bench_bases.py --baseline tests/images/bases_baseline.json   # exit 1 when worse
uv run tools/annotate_bases.py tests/images/<name>.jpg --prefill   # click base ground truth
```

`docs/testing.md` describes the two test sets, how to add a test photo, and how to update expected files. Update it when you add a test file or a stage.

`docs/cli.md` is the full CLI reference: options, exit codes, JSON report schema, and how to add a stage or a command. Update it when you add or change a subcommand, an option, a stage, or a JSON field.

No linter or formatter is configured.

## Architecture

- `src/battlesight/` is the package (hatchling, `src` layout). Entry point `battlesight = battlesight.app:main`. `app.py` holds the argparse parser: no command = the live app (`cmd_live`, `_Live` state; its options are on the main parser), plus the subcommands `enroll`, `gallery`, `calibrate`, `image-pipeline`. A new live feature is a key in `cmd_live` (and the `HELP_KEYS` list), not a new subcommand or option. `docs/cli.md` "All commands" lists every command.
- `pipeline.py`: single-frame analysis, no camera or window. Stages are `corners`, `rectify`, `bases`, `ident`. Stages register in run order with `@stage(name, draw=..., views=...)` (`views` returns extra named images, `--out` writes them as `<name>.<view>.jpg`) and return a `StageResult` (ok, message, JSON-safe `data`). A failed stage skips the later ones. `Pipeline.analyze(frame, until=...)` stops early (the live `b` key). `Pipeline.analyze` gives a `FrameResult` (`to_dict()` is the JSON report, with `schema` version), `Pipeline.annotate` draws all stage overlays. Add each new priority as a stage here, so the live app, the `image-pipeline` subcommand and tests pick it up. The `image-pipeline` subcommand exit codes: 0 all stages ok, 1 a stage failed, 2 bad input.
- `circles.py` (priority 3): `find_circles` finds round bases of the known diameters (`[bases]` in `data/board.toml`, base bottom) in a top-down image at `bases.px_per_mm` (4). Edge votes along the gradient for proposals, 120 rays with a dark-rim test, robust free fit, nearest size class, fixed-radius refit, `coverage` = visible rim part. `foreground_mask` diffs against an empty-board reference. `Pipeline(reference=frame)` / `set_reference` warps the reference from its own tags into `cache["bases_reference"]`; the `bases` stage then runs in `reference` mode, else `single_frame` (finds some dark-ringed tokens and barricades). The stage puts `bases` (list of `Circle`) in `scratch`. Bottom edge lies on the play surface: no parallax correction. `docs/cli.md` "Base detection" has the steps and config keys.
- `ident.py` (priority 4): crop around each base from the bases image (`store_crop` keeps 40 mm board each side, `prepare` cuts `ident.crop_pad_mm`, turns, masks a circle, resizes to `input_px`), `Embedder` (`DinoEmbedder` = DINOv2 via `transformers`, MPS, lazy load; `ColorEmbedder` = `model = "color-hist"`, no ML, used by all unit tests), `Gallery` (`data/teams/<team>/team.toml` with `base_mm`, `<operative>/*.png`, `embeddings.npz` cache, `rotations` angles per crop, score = mean of `top_k` best photos, `leave_one_out`), `assign` (size-class limit, Hungarian one-to-one, `min_score` unknown, `min_margin` unsure), `LabelSession` (click fix logic, no window calls), `IdentTruth` / `score_truth` (done test). Labels are `<team>/<operative>`. Stage `ident` after `bases`: always ok, names nothing without a gallery (`Pipeline(gallery=...)`, `image --team`). The bases stage puts `bases_image` in `scratch`, the ident stage `base_crops` and `idents`. `overlay.Menu` is the click menu (hit test unit tested). `docs/cli.md` "Operative identity" has the steps and config keys.
- `bench.py`: bases benchmark, pure functions. Ground truth `<name>.bases.json` schema 2 (`Truth`, `Decoy` = `not_bases`, schema 1 still loads), `match` (one-to-one global greedy within `tol_mm`, false detections get a `kind`, `inside_base` or `other`), `PhotoScore` (`done_ok` = priority 3 done test), `metrics` / `total` (recall, precision, f1, size accuracy, centre error of clear clicked or detector rims), `report` (`bench.json`), `compare` (baseline rules), text table. Also the photo file rules (`find_photos`, `reference_path`: `<name>.empty.*` is a reference). `pipeline.detect_bases(img, cfg, reference)` is the body of the bases stage, so `tools/bench_bases.py` runs corners and rectify once and sweeps `[bases]` values cheaply. `tools/annotate_bases.py` holds the click logic in `Annotator` (no window calls, unit tested).
- `tests/test_photos.py` (marker `photos`, off by default through `addopts` in `pyproject.toml`): runs the pipeline on each photo in `tests/images` and compares with the reviewed `<name>.expected.json` next to it (the `--out` report), corners within `CORNER_TOL_MM` (1 mm, converted with the tag's pixel size). Never update an expected file only to make a test pass. Bases: `<name>.bases.json` is ground truth (`clear` / `hidden` rims, `tol_mm`, `source`, `not_bases`), `<name>.empty.jpg` an optional empty-board reference; `test_bases_done_criteria` runs only with both. `test_bases_benchmark_no_worse` compares the benchmark totals with `tests/images/bases_baseline.json`; `pytest -m photos` prints the metrics table. The baseline moves only on purpose (`tools/bench_bases.py --write-baseline`, new numbers in the commit message), never only to make a test pass.
- `config.py`: `BoardConfig` (frozen dataclass) loads `data/board.toml`. All lengths are in mm. Board coordinates: origin at the top-left board corner, x right, y down. Corner index 0..3 = TL, TR, BR, BL, and `tags.ids[k]` is the tag at corner k.
- `calib.py`: `TagDetector` (OpenCV `cv2.aruco` with `DICT_APRILTAG_36h11`, no contrib build) returns `{tag_id: TagObs}`. Each card lies under a board corner, tag outside the board, diagonal to the corner. `board_corners_px` takes tag corner k+2 (the one nearest the board) and pushes it toward the board by the white card margin to get board corner k. `DriftMonitor` auto-locks a reference after `settle_frames` still frames, then reports OK / MOVED / CARD_MOVED / LOST. After a move it waits for `settle_frames` still frames again: when all four cards moved as one piece (`card_offset_mm`, leave-one-tag-out homography, within `drift.tolerance_mm`) it saves the new position (`relocked`), else CARD_MOVED names the card. No manual re-lock key. It converts pixel shift to mm with each tag's apparent size.
- `calib.py` also holds `Lens` (K, distortion, `undistort_px` / `distort_px`), `lens_for_frame(camera, size)` (the `[lens]` calibration of `CameraSettings` when its aspect ratio fits the frame, else a nominal pinhole from `CameraSettings.focal_35mm`), and the ChArUco target and `calibrate_lens`.
- `rectify.py`: `fit_view` fits a `BoardView` (homography board-surface mm -> undistorted px) to the 16 tag corners, with the board thickness correction from the planar camera pose. `warp_board` makes the top-down image. `measure_edges` and `measure_grid_lines` check it (board edges, printed grid lines). The rectify stage puts `view` and `rectified` in `scratch` for later stages.
- Map grid: `[grid]` in `data/board.toml` and `BoardConfig.grid_lines_mm()`. Tomb World uses its 7 x 6 printed sections inside a 12 mm margin, Volkus a 1 in pitch. Never hard-code an inch grid. `docs/cli.md` "Map grid" documents it.
- `camera.py`: `Camera` wraps `cv2.VideoCapture` and reconnects after repeated read failures. `read()` returns `None` while the camera is down. Callers must handle that. A device index uses AVFoundation on macOS (Camo, built-in camera). A URL (`is_stream`) uses FFmpeg with open and read timeouts, and `LatestFrameReader` reads it on a thread and keeps only the newest frame. `CameraSettings` loads and saves `data/camera.toml` (git-ignored, machine-local): `[camera]` from `tools/list_cameras.py`, `[lens]` from `calibrate`. Each writer keeps the other table. `app.py` `_apply_camera_settings` fills the camera options left out on the command line from it: option > file > built-in default. `docs/setup.md` has the camera troubleshooting (Camo extension blocked by MDM, stream network checks).
- `overlay.py`: draw-in-place helpers on BGR frames. Line and font sizes scale from a 1080p baseline, so overlays stay readable at 4K.
- `tests/conftest.py`: `render_board` makes a synthetic top-down board with the board resting on real square AprilTag cards, tags outside the corners, with an optional homography, pixel shift, printed grid lines, and `bases` (`Base`: black rim, lighter top, model over part of the rim), `walls`, `discs` (tokens, prints). `tomb_world_grid()` is a test config with the Tomb World size and sections. `tests/test_rectify.py` adds a virtual pinhole camera for exact thickness and distortion tests. It returns the image and an `mm2px` function for ground truth. New vision tests should use this harness, not camera captures. `make_config(**over)` builds a test config without the TOML file.

- Corner cards (`tools/make_tags.py`): 90 mm square, tag in the outer quadrant, black L in the inner quadrant at the board corner. The 5 mm white margin is the tag quiet zone. The app measures from the cards, not the board: a board that slides over a still card gives no warning.
- Rectify: the top-down image scale is `rectify.px_per_mm` (default 2 px per mm). `board.width_mm` is the edge from tag 0 to tag 1 as the cards lie (703 x 606 mm in the test photos). When width and height are swapped, the stage fails and says so. Board thickness: the cards lie on the table and the play surface is `board.thickness_mm` higher (Tomb World 1.0 mm, caliper). Without the correction, 1 mm gives up to about 0.5 mm error at the board edge, 3 mm about 1.5 mm. With `grid.printed = true` the stage finds the printed lines and reports their offsets, a check that does not use the tags.
- Lens: without a `[lens]` table in `data/camera.toml` the app uses a nominal lens (`focal_35mm`, default 24 mm, no distortion). Phones correct most distortion themselves. All camera values live in `data/camera.toml`, none in `data/board.toml` (a `[camera]` table there is an error). `Pipeline(cfg, until, camera)` takes the `CameraSettings`; tests and photo tests use the default (nominal lens), and `tests/conftest.py` points the CLI at a temporary camera file.

Layout:

```
src/battlesight/   config.py camera.py calib.py rectify.py circles.py ident.py bench.py overlay.py pipeline.py app.py
tools/             make_tags.py make_charuco.py list_cameras.py bench_bases.py annotate_bases.py
tests/             synthetic-board tests, test-photo tests (images/ = test photos), no camera needed
data/              board.toml, print/ (printable sheets), snapshots/ (live saves), teams/<team>/ (gallery, git-ignored)
docs/              setup.md (user setup), cli.md, testing.md
```

`specs/bases-benchmark.md` is the plan for the bases benchmark tool, ground truth click tool, and metric baseline (parts 1 to 4 done; section 11 tuning and re-annotation are next).

Target modules for later priorities (`doors.py`, `measure.py`, `ui/`) and data folders are listed in `specs/plan.md` section 7.

## Design constraints from the plan

- All measurement is 2D in the board plane. No vantages or heights in the MVP. Measure to the base rim, never to the model body (parallax moves the model about 19 mm at a board corner).
- Base detection (geometry: background subtraction plus known-radius circle fit) stays separate from operative identity (DINOv2 embedding plus few-shot gallery). A new team must only need gallery photos, never a retrained detector.
- Walls are not tracked. Doors are clicked once per map, snapped to the grid, and saved as a named map file.
- The core must not depend on Qt. A later FastAPI and browser front end must reuse it. An OpenCV `imshow` window is enough for priorities 1 and 2. PySide6 comes later.
- Board size, tag sizes, tolerances, base, token, and door sizes are config values, not constants. A second board (Volkus, 559 x 760 mm) is planned.

## CLI design principles

- Use as few command-line options as possible. When a key (later a UI button) can turn a feature on or off, or change a mode or view, do not add an option for it. Options are only for values that the user cannot change while the app runs (camera source, config file, teams).
