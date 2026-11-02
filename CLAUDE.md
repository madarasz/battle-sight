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
uv run pytest                             # all tests (no camera needed)
uv run pytest tests/test_calib.py::test_detects_four_tags_flat   # one test
uv run battlesight corners                # live corner tracking, built-in camera
uv run battlesight corners --source 1 --width 3840 --height 2160  # phone through Camo, 4K
uv run tools/list_cameras.py              # find camera index
uv run tools/make_tags.py                 # print sheet -> data/tags/corner_tags.pdf
```

No linter or formatter is configured.

## Architecture

- `src/battlesight/` is the package (hatchling, `src` layout). Entry point `battlesight = battlesight.app:main`. `app.py` holds argparse subcommands. Add new modes as new subcommands there.
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
