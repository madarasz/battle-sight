# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Battle Sight is an augmented reality overlay for Warhammer 40k: Kill Team. A fixed camera looks down on the board. The app finds the board with AprilTags, rectifies the image to board millimetres, and draws measurements (inch grid, rings, rim-to-rim distances, token ranges, door bands) on the video.

Read these before design work:

- `specs/about-project`: the owner's goals, in their own words.
- `specs/plan.md`: decisions, the 7 priorities with "done when" tests, risks, stack, and target layout. The plan is the source of truth for scope. Its "No code yet" status line is out of date: priority 1 (board corners) is built.

Work goes in priority order (1 corners, 2 undistort and rectify, 3 bases, 4 operative ID, 5 rings and distances, 6 tokens, 7 doors). Each priority has a measurable done test in the plan. Use it as the acceptance check.

## Commands

Use `uv` for all work. Python is pinned to 3.12 (ML wheels lag on newer versions).

```bash
uv sync                                   # create .venv, install deps
uv run pytest                             # unit tests only (no camera needed)
uv run pytest -m photos                   # photo regression tests on data/snapshots/tests
uv run pytest -m ""                       # all tests
uv run pytest tests/test_calib.py::test_detects_four_tags_flat   # one test
uv run battlesight corners                # live corner tracking, built-in camera
uv run battlesight corners --source 1 --width 3840 --height 2160  # phone through Camo, 4K
uv run battlesight image data/snapshots/tests              # pipeline on saved photos, exit 0/1/2
uv run battlesight image shot.jpg --out out/ --json -     # overlays + per-image JSON, report on stdout
uv run battlesight image shot.jpg --until corners --show  # stop after a stage, view overlay
uv run tools/list_cameras.py              # find camera index
uv run tools/make_tags.py                 # print sheet -> data/tags/corner_tags.pdf
```

`docs/testing.md` describes the two test sets, how to add a test photo, and how to update expected files. Update it when you add a test file or a stage.

`docs/cli.md` is the full CLI reference: options, exit codes, JSON report schema, and how to add a stage or a command. Update it when you add or change a subcommand, an option, a stage, or a JSON field.

No linter or formatter is configured.

## Architecture

- `src/battlesight/` is the package (hatchling, `src` layout). Entry point `battlesight = battlesight.app:main`. `app.py` holds argparse subcommands. Add new modes as new subcommands there.
- `pipeline.py`: single-frame analysis, no camera or window. Stages register in run order with `@stage(name, draw=...)` and return a `StageResult` (ok, message, JSON-safe `data`). A failed stage skips the later ones. `Pipeline.analyze` gives a `FrameResult` (`to_dict()` is the JSON report, with `schema` version), `Pipeline.annotate` draws all stage overlays. Add each new priority as a stage here, so the `image` subcommand and tests pick it up. The `image` subcommand exit codes: 0 all stages ok, 1 a stage failed, 2 bad input.
- `tests/test_photos.py` (marker `photos`, off by default through `addopts` in `pyproject.toml`): runs the pipeline on each photo in `data/snapshots/tests` and compares with the reviewed `<name>.expected.json` next to it (the `--out` report), corners within `CORNER_TOL_MM` (1 mm, converted with the tag's pixel size). Never update an expected file only to make a test pass.
- `config.py`: `BoardConfig` (frozen dataclass) loads `data/board.toml`. All lengths are in mm. Board coordinates: origin at the top-left board corner, x right, y down. Corner index 0..3 = TL, TR, BR, BL, and `tags.ids[k]` is the tag at corner k.
- `calib.py`: `TagDetector` (OpenCV `cv2.aruco` with `DICT_APRILTAG_36h11`, no contrib build) returns `{tag_id: TagObs}`. Each card lies under a board corner, tag outside the board, diagonal to the corner. `board_corners_px` takes tag corner k+2 (the one nearest the board) and pushes it toward the board by the white card margin to get board corner k. `DriftMonitor` auto-locks a reference after `settle_frames` still frames, then reports OK / MOVED / LOST. It converts pixel shift to mm with each tag's apparent size.
- `camera.py`: `Camera` wraps `cv2.VideoCapture` (AVFoundation on macOS) and reconnects after repeated read failures. `read()` returns `None` while the camera is down. Callers must handle that.
- `overlay.py`: draw-in-place helpers on BGR frames. Line and font sizes scale from a 1080p baseline, so overlays stay readable at 4K.
- `tests/conftest.py`: `render_board` makes a synthetic top-down board with the board resting on real square AprilTag cards, tags outside the corners, with an optional homography and pixel shift. It returns the image and an `mm2px` function for ground truth. New vision tests should use this harness, not camera captures. `make_config(**over)` builds a test config without the TOML file.

Target modules for later priorities (`rectify.py`, `circles.py`, `ident.py`, `doors.py`, `measure.py`, `ui/`) and data folders are listed in `specs/plan.md` section 7.

## Design constraints from the plan

- All measurement is 2D in the board plane. No vantages or heights in the MVP. Measure to the base rim, never to the model body (parallax moves the model about 19 mm at a board corner).
- Base detection (geometry: background subtraction plus known-radius circle fit) stays separate from operative identity (DINOv2 embedding plus few-shot gallery). A new team must only need gallery photos, never a retrained detector.
- Walls are not tracked. Doors are clicked once per map, snapped to the grid, and saved as a named map file.
- The core must not depend on Qt. A later FastAPI and browser front end must reuse it. An OpenCV `imshow` window is enough for priorities 1 and 2. PySide6 comes later.
- Board size, tag sizes, tolerances, base, token, and door sizes are config values, not constants. A second board (Volkus, 559 x 760 mm) is planned.
