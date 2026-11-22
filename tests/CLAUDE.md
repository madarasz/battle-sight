# tests: test guide

`docs/testing.md` describes the two test sets, how to add a test photo, and how to update expected files. Update it when you add a test file or a stage. The root `CLAUDE.md` "Which tests to run" says which tests to run for a change.

## Categories and speed

- `pyproject.toml` runs the tests on all cores (`-n auto`, pytest-xdist); the unit tests take about 30 s.
- `conftest.py` points `live.SAVES_DIR` to a temporary folder; `make_session` gives each session its own `saves_dir`. No test writes `data/saves`.
- Each test file has one category marker (`core`, `vision`, `live`, `ui`), set by file name in `CATEGORIES` of `tests/conftest.py`. A new test file must go there, else collection stops.
- Serial times: `core` about 1 s, `vision` about 25 s, `live` about 20 s, `ui` about 30 s. Use `-n 0` for one file or for `core`.
- The `named` fixture (`tests/test_live.py`, also used by `tests/test_ui.py`) enrolls the test gallery once per process and copies it into each test.
- `tests/conftest.py` keeps each `circles.find_circles` result for the process (most tests run the same synthetic board); the function must stay deterministic. `BATTLESIGHT_TEST_NO_MEMO=1` turns the memo off.

## Synthetic board harness (`conftest.py`)

- `render_board` makes a synthetic top-down board with the board resting on real square AprilTag cards, tags outside the corners, with an optional homography, pixel shift, printed grid lines, and `bases` (`Base`: black rim, lighter top, model over part of the rim), `walls`, `discs` (tokens, prints). It returns the image and an `mm2px` function for ground truth.
- `tomb_world_grid()` is a test config with the Tomb World size and sections.
- `make_config(**over)` builds a test config without the TOML file.
- `tests/test_rectify.py` adds a virtual pinhole camera for exact thickness and distortion tests.
- New vision tests use this harness, not camera captures.
- Tests use the default camera (nominal lens); `conftest.py` points the CLI at a temporary camera file.
- No test may depend on a value of `data/color-settings.toml` (the user changes it): never a colour literal or a fixed colour threshold. Compare with the values that the app reads from the file (`colors` constants; `near_color` in `tests/test_live.py` for anti-aliased lines). `tests/test_colors.py` reads the file itself.
- `conftest.py` replaces `LiveSession.start_preload` with a no-op (no DINOv2 load on a thread); `REAL_PRELOAD` is the real one.
- All unit tests use `ColorEmbedder` (`model = "color-hist"`), no ML.

## Photo tests (`test_photos.py`, marker `photos`)

- Off by default through `addopts` in `pyproject.toml`. Run `uv run pytest -m photos`.
- Runs the pipeline on each photo in `tests/images` and compares with the reviewed `<name>.expected.json` next to it (the `--out` report), corners within `CORNER_TOL_MM` (1 mm, converted with the tag's pixel size).
- The photo tests stay on one worker (`xdist_group`), so each photo runs the pipeline once; their metrics tables go to the main process in the reports.
- Files per photo in `tests/images`:
  - `<name>.expected.json`: reviewed pipeline report.
  - `<name>.bases.json`: bases ground truth (`clear` / `hidden` rims, `tol_mm`, `source`, `not_bases`, `roster_mm`).
  - `<name>.empty.jpg`: optional empty-board reference.
  - `<name>.ident.json`: operative names ground truth.
  - `<name>.light.json`: light terrain ground truth (pieces: key, centre, angle; `not_pieces` points).
- Bases and names have no all-found test: not all bases can be found. `test_bases_benchmark_no_worse` and `test_ident_benchmark_no_worse` fail only when the result is worse than `tests/images/bases_baseline.json` or `ident_baseline.json` (totals and per photo). `pytest -m photos` prints the metrics tables with the `done` column of each done test.

## Never

- Never update an expected file only to make a test pass.
- Never move a baseline only to make a test pass. A baseline moves only on purpose (`tools/bench_bases.py --write-baseline`, new numbers in the commit message).
