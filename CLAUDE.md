# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It is the index. Details are in the files of "Where to find what".

## Project

Battle Sight is an augmented reality overlay for Warhammer 40k: Kill Team. A fixed camera looks down on the board. The app finds the board with AprilTags, rectifies the image to board millimetres, and draws measurements (inch grid, rings, rim-to-rim distances, token ranges, door bands) on the video.

Read these before design work:

- `docs/specs/about-project`: the owner's goals, in their own words.
- `docs/specs/plan.md`: decisions, the 7 priorities with "done when" tests, risks, stack, and target layout. The plan is the source of truth for scope.

Work goes in priority order. Each priority has a measurable done test in the plan. Use it as the acceptance check.

| # | Priority | Status | Details |
|---|---|---|---|
| 1 | Board corners | done | `docs/dev/board-and-camera.md` |
| 2 | Undistort and rectify | done | `docs/dev/board-and-camera.md` |
| 3 | Bases | in code; done test passes on the 2 test photos with an empty-board reference, not on the 2 without | `docs/dev/bases-and-ident.md` |
| 4 | Operative ID | in code; done test waits for a real gallery and a labelled game photo | `docs/dev/bases-and-ident.md` |
| 5 | Rings and distances | in code; done test waits for a tape measure check on the real board | `docs/dev/live-session.md` |
| 6 | Terrain layouts (map files, skill `record-map`, overlay) | done | `docs/dev/terrain.md` |
| 7 | Tokens | next | `docs/specs/plan.md` section 7 (data folders) |

## Where to find what

Claude Code loads a `CLAUDE.md` of a subfolder when it reads a file in that folder.

| Topic | File |
|---|---|
| Module map, package rules, `app.py`, `pipeline.py`, `overlay.py`, `colors.py` | `src/battlesight/CLAUDE.md` |
| Qt front end, layouts, widget rules | `src/battlesight/ui/CLAUDE.md` |
| Config, camera, scrcpy, tags, drift, lens, rectify, map grid | `docs/dev/board-and-camera.md` |
| Base detection, identity, gallery, benchmarks | `docs/dev/bases-and-ident.md` |
| `LiveSession`, selection, wounds, openings, rosters, copies, control marks, score | `docs/dev/live-session.md` |
| Measure around walls, movement, shooting | `docs/dev/walls-moves-shoot.md` |
| Map files, piece files, light terrain search, minimap | `docs/dev/terrain.md` |
| Test harness, photo tests, baselines | `tests/CLAUDE.md`, `docs/testing.md` |
| Tool scripts (cameras, print sheets, benchmarks, annotation) | `tools/CLAUDE.md` |
| CLI options, exit codes, JSON report, config keys | `docs/cli.md` |
| User setup and calibration | `docs/setup.md` |
| Recording maps and terrain pieces | `.claude/skills/record-map/`, `.claude/skills/record-terrain/` |
| Rules for `README.md` and `docs/setup.md` | `.claude/skills/write-user-docs/` |
| Specs: UI plan, benchmarks, measure and move rules, operative copies | `docs/specs/ui.md`, `docs/specs/bases-benchmark.md`, `docs/specs/ident-benchmark.md`, `docs/specs/measure-and-move.md`, `docs/specs/operative-copies.md` |

Folders:

```
src/battlesight/   config.py camera.py scrcpy.py calib.py rectify.py circles.py ident.py measure.py score.py walls.py moves.py shoot.py terrain.py lightfit.py bench.py ident_bench.py colors.py overlay.py pipeline.py live.py app.py ui/ (shell.py parts.py settings.py layouts/)
tools/             make_tags.py make_charuco.py list_cameras.py bench_bases.py bench_ident.py bench_light.py annotate_bases.py
tests/             synthetic-board tests, test-photo tests (images/ = test photos), no camera needed
data/              board.toml, color-settings.toml (UI colours), maps/<terrain>/ (terrain map files), terrain/<terrain>/ (light terrain piece files; images git-ignored), print/ (printable sheets), snapshots/ (live saves), teams/<team>/ (gallery, git-ignored), ui.toml (layout choice, git-ignored), camera.toml (git-ignored)
docs/              setup.md (user setup), cli.md, testing.md, dev/ (developer topics), specs/ (plan, goals, feature specs)
```

## Keep the docs current

When you change code, update the file of its topic in "Where to find what" in the same change:

- A module, action, state key or config key: its `docs/dev/` file or folder `CLAUDE.md`.
- A subcommand, option, stage, or JSON field: `docs/cli.md`.
- A test file or a stage: `docs/testing.md` and `tests/CLAUDE.md`.
- A rule or model of measurement or movement: `docs/specs/measure-and-move.md`.
- A priority status: the table above and the list in `docs/dev-log.md`.
- A user-facing mode, overlay or setup step: `README.md` or `docs/setup.md` (skill `write-user-docs`).

## User documents

`README.md` and `docs/setup.md` are for non-technical users, in ASD-STE100 Simplified Technical English. Load the skill `write-user-docs` before you change them. Developer information goes in the `CLAUDE.md` files, `docs/dev/`, `docs/cli.md`, or `docs/testing.md`.

## Commands

Use `uv` for all work. Python is pinned to 3.12 (ML wheels lag on newer versions). No linter or formatter is configured. Tool scripts: `tools/CLAUDE.md`.

```bash
uv sync                                   # create .venv, install deps
uv run pytest                             # unit tests only (no camera needed), on all cores (xdist)
uv run pytest -m core -n 0                # one category: core, vision, live, ui (see "Which tests to run")
uv run pytest -m photos                   # photo regression tests on tests/images
uv run pytest -m ""                       # all tests
uv run pytest -n 0 tests/test_calib.py::test_detects_four_tags_flat   # one test, one process (also for -x, --pdb, -s)
uv run battlesight                        # live app: corners, top-down view (v), space = bases + names, b = bases only, l = next UI layout, h = keys
uv run battlesight --team kommandos veterans   # live app with 1 or 2 gallery teams (default: all teams)
uv run battlesight --source http://PHONE-IP:8080/video  # one run with another source
uv run battlesight --image-source tests/images/tomb-world-1.jpg  # live app on one image, for debugging
uv run battlesight enroll --team kommandos --operative "Boss Nob"   # gallery crops, live or --images
uv run battlesight gallery --team kommandos   # list gallery, leave-one-out check
uv run battlesight image-pipeline shot.jpg --team kommandos --json -   # all stages on saved photos
uv run battlesight calibrate              # lens calibration -> [lens] in data/camera.toml
uv run battlesight map data/maps/tomb-world/layout-6.toml --out out/maps/   # check a terrain map file, draw it
uv run battlesight image-pipeline tests/images              # pipeline on saved photos, exit 0/1/2
uv run battlesight image-pipeline shot.jpg --out out/ --json -     # overlays + per-image JSON, report on stdout
uv run battlesight image-pipeline shot.jpg --show                  # view the overlays
uv run battlesight image-pipeline game.jpg --reference empty.jpg --out out/  # bases in reference mode
uv run .claude/skills/record-terrain/outline.py view c2.jpg --out view.png   # helper of the skill record-terrain (view, suggest, simplify, save)
```

### Which tests to run

While you work, run the tests of the changed code. Use `-n 0` for one file or for `core`: the workers take longer to start than these tests. Categories and test speed: `tests/CLAUDE.md`.

| Changed code | Run |
|---|---|
| `measure`, `walls`, `moves`, `shoot`, `score`, `terrain`, `config`, `calib`, `camera`, `scrcpy` | its test file, then `-m "core or live"` (`live.py` uses them) |
| `ui/` | `-m ui` |
| `live.py`, `overlay.py` | `-m "live or ui"` |
| `lightfit` | `tests/test_lightfit.py`, then `-m live`, then `tools/bench_light.py` when a search result can change |
| `.claude/skills/record-terrain/outline.py` | `tests/test_terrain.py` |
| `circles`, `rectify`, `pipeline`, `ident`, `bench`, `ident_bench`, `tools/bench_*`, `tools/annotate_bases.py` | `-m vision`, then `-m photos` when a detection or naming result can change (also for `[bases]` or `[ident]` values in `data/board.toml`) |
| `app.py` | `-m "vision or ui"` (CLI tests are in both) |
| `data/*.toml`, `tests/conftest.py`, `pyproject.toml` | all unit tests |
| Docs, specs, skills only | no tests |

Before a commit, and at the end of a task that changed code, run all unit tests (`uv run pytest`). Run `-m photos` when the vision code changed.

Never update an expected file or move a baseline only to make a test pass.

## Design constraints from the plan

- All measurement is 2D in the board plane. No vantages or heights in the MVP. Measure to the base rim, never to the model body (parallax moves the model about 19 mm at a board corner).
- Base detection (geometry: background subtraction plus known-radius circle fit) stays separate from operative identity (DINOv2 embedding plus few-shot gallery). A new team must only need gallery photos, never a retrained detector.
- Walls and openings are not detected. Each layout is a map file, written from the layout card photo with the skill `record-map`.
- The core must not depend on Qt. A later FastAPI and browser front end must reuse it.
- Board size, tag sizes, tolerances, base, token, and door sizes are config values, not constants. A second board (Volkus, 559 x 760 mm) is planned.

## CLI design principles

- Use as few command-line options as possible. When a key (later a UI button) can turn a feature on or off, or change a mode or view, do not add an option for it. Options are only for values that the user cannot change while the app runs (camera source, config file, teams).
- A new live feature is an entry in `live.ACTIONS` with a handler in `LiveSession`, not a new subcommand or option.
