# Tests

Battle Sight has two test sets. Both use `pytest`. Neither needs a camera.

| Set | Marker | Command | Input | Checks |
|---|---|---|---|---|
| Unit tests | none | `uv run pytest` | Synthetic boards made in code | Each function and the CLI give the correct result. |
| Photo tests | `photos` | `uv run pytest -m photos` | Real photos in `data/snapshots/tests` | The pipeline result on each photo agrees with a reviewed expected result. |

To run all tests:

```
uv run pytest -m ""
```

`pyproject.toml` sets `addopts = "-m 'not photos'"`. Thus `uv run pytest`
runs only the unit tests. A `-m` option on the command line replaces this
default.

## Unit tests

| File | Tests |
|---|---|
| `tests/test_bases.py` | Circle fit on synthetic boards: each base size (centre within 0.5 mm, right class), neighbour sizes 25, 28, 32, a model over 40 % of the rim, a rim hidden too much (no result), touching bases, a base against a wall, light tokens and other sizes rejected, printed grid gives no bases, region limit. Reference mode: a static dark print is removed, noise is not foreground. Pipeline stage in single-frame and reference mode, no bases, bad reference, `image --reference`, the `[bases]` config table. |
| `tests/test_bench.py` | Bases benchmark: one-to-one global greedy match (with the case where a per-truth match differs), tolerance, unknown size, decoy and `inside_base` categories. Metrics for a hand-made case, a negative photo, no detections, `done_ok`. Each baseline rule, a new photo, a better run. Ground truth schema 1 and 2, bad `rim` and `kind`. `tools/bench_bases.py` end to end on two synthetic photos (one with a reference): `bench.json`, `--mode`, `--set`, `--sweep`, `--write-baseline` and `--baseline`, exit codes, debug images. Click tool logic: fit of a known circle, class keys, hidden rim, estimate, save and load, undo, delete. |
| `tests/test_camera.py` | Camera source parsing: device index, stream URL, file path. The stream reader keeps only the newest frame, waits for a new frame, and stops at the end of the stream. Camera settings file: defaults without a file, save and load for a device and a stream, command options override the file, `list_cameras.py` keeps the lens when it saves a new source. No camera or network needed. |
| `tests/test_config.py` | Load of `data/board.toml`. Rejection of bad tag ids and bad grid sections. Grid lines for Tomb World sections and for a Volkus 1 in pitch. |
| `tests/test_calib.py` | Tag corner geometry in mm. Tag detection (flat, perspective and blur, low pixel density, wrong tag ids). Board corner position. Drift monitor: reference lock, moved board, one moved tag, lost tag. |
| `tests/test_pipeline.py` | Pipeline stages: a good board, a missing tag, skip of later stages after a failure, an error in a stage, an unknown stage name. The `image` command: exit codes 0, 1, 2, the JSON report, the `--out` files, `--until corners`. |
| `tests/test_rectify.py` | Board plane fit with an exact virtual camera: homography, camera position, board thickness correction, lens distortion. Top-down image of a rendered board: board edges, corner angles, printed grid lines, a board that slid over still cards, lens distortion removal. Stage report, failure on a bad fit, hint for swapped board width and height. Lens in the camera file: nominal lens and its focal length, calibrated lens at another scale, other aspect ratio. ChArUco lens calibration on virtual camera views, and the `calibrate` command (writes `[lens]`, keeps `[camera]`). |

The unit tests use `render_board` in `tests/conftest.py`. It makes a top-down
image of the board on the four tag cards, with an optional perspective,
pixel shift, and printed grid lines (`print_grid=True`). With `bases`, `walls`
and `discs` it also draws round bases (black rim, lighter top, optional
model over part of the rim), dark wall pieces, and flat tokens or prints.
It also gives a function from board mm to image pixels. The tests use this function as the
true position. `tomb_world_grid()` gives a test config with the Tomb World
board size and its 7 x 6 printed sections. Use this harness for new vision
unit tests. Do not use camera captures in unit tests.

`tests/test_rectify.py` also uses a virtual pinhole camera
(`cv2.projectPoints`). It projects the tag corners on the table and the
board points on the play surface. This gives exact true positions for the
thickness and distortion tests.

## Photo tests

The photo tests are in `tests/test_photos.py`. They use the board config in
`data/board.toml`.

### Files

Each test photo has an expected file next to it, and optional files for the
bases stage:

```
data/snapshots/tests/
  tomb-world-1.jpg
  tomb-world-1.expected.json   # reviewed report of `image --out`
  tomb-world-1.bases.json      # bases ground truth, optional
  tomb-world-1.empty.jpg       # empty-board reference, optional
  tomb-world-2.jpg
  ...
  bases_baseline.json          # bases benchmark baseline (tools/bench_bases.py --write-baseline)
```

There are three kinds of photos for the bases stage:

| Kind | Files | Use |
|---|---|---|
| Game, single | `<name>.jpg`, `<name>.bases.json` | Single-frame mode. |
| Game pair | `<name>.jpg`, `<name>.empty.jpg`, `<name>.bases.json` | Reference mode, and single-frame mode on the same photo (`bench_bases.py --mode single`). The done test of priority 3. |
| Negative | `<name>.jpg`, `<name>.bases.json` with `"bases": []` | False bases only. |

`<name>.empty.jpg` (or `.png`) is the same board with the terrain and the
tokens, but no operatives, from the same camera and light. When it exists,
the test runs the bases stage in reference mode. The tests do not use it as
a photo of its own. The two Tomb World game photos have no empty-board
photo.

A negative photo (terrain and tokens, no operatives) can also be a test
photo of its own. Give it a `<name>.bases.json` with an empty `bases`
list, so each base found in it is a false base. Do not name it
`<name>.empty.jpg`: that name makes it a reference, not a test photo.

`<name>.bases.json` is the ground truth (schema 2). Files without
`schema` are version 1: a clear base reads as `source: "detector"`, a
hidden base as `source: "estimate"`, with no `not_bases`.

```json
{
  "schema": 2,
  "comment": "free text",
  "bases": [
    {"center_mm": [329.8, 186.5], "diameter_mm": 40, "rim": "clear", "source": "clicked", "note": "tyranid, top centre", "points_mm": [[349.8, 186.5], "..."]},
    {"center_mm": [527.0, 268.0], "diameter_mm": 40, "rim": "hidden", "tol_mm": 8, "source": "estimate", "note": "tyranid against wall"},
    {"center_mm": [164.6, 240.3], "diameter_mm": null, "rim": "hidden", "tol_mm": 10, "source": "estimate", "note": "model on a token"}
  ],
  "not_bases": [
    {"center_mm": [551.0, 350.0], "diameter_mm": 60, "kind": "teleporter", "note": "dark teleporter"}
  ]
}
```

| Field | Meaning |
|---|---|
| `bases[].center_mm` | Centre of the base bottom in board mm. |
| `bases[].diameter_mm` | Base bottom diameter. `null` when not known: any size class matches, no size check. |
| `bases[].rim` | `clear`: at least half of the bottom rim is visible. `hidden`: less (a model, a wall or a token hides it). |
| `bases[].tol_mm` | Match radius. Default 2 mm (the done test). Use a larger value for a position known less well. |
| `bases[].source` | `clicked` (rim points clicked in `tools/annotate_bases.py`), `detector` (a checked detector fit), `estimate` (approximate). |
| `bases[].points_mm` | Clicked rim points, so a person can check the fit. Optional. |
| `not_bases[]` | Known round things that are not bases. Only labels: a false base within half of `diameter_mm` of the centre gets `kind` as its category. |
| `not_bases[].kind` | One of `teleporter`, `token`, `print`, `barricade`, `wall`, `model`, `other`. |

The clear-rim ground truth of `tomb-world-1` and `-2` comes from detector
fits (schema 1), checked by eye. Their centre errors in the benchmark are
thus near 0 and do not show the true error. Click them again with
`tools/annotate_bases.py` ("Click the ground truth" below).

The expected file is the image report that `battlesight image --out` writes
(see [cli.md](cli.md), "JSON report"). A person checked the overlay image
before the report became the expected file.

### What the tests check

For each photo:

| Test | Passes when |
|---|---|
| `test_photo_passes_all_stages` | All pipeline stages pass. |
| `test_photo_size_matches` | The image size after the EXIF rotation is the same as in the expected file. |
| `test_corners_match_expected` | The found tag ids are the same as in the expected file. Each of the four board corners and each of the 16 tag corners is within the tolerance of its expected position. |
| `test_rectify_match_expected` | The lens source is the same. 81 board points through the expected and the found homography are within `RECTIFY_TOL_MM` (1 mm). Each board edge offset and each printed grid line offset is within 1 mm of the expected value. |
| `test_bases_match_expected` | Same `mode`. Each expected base is found within `BASES_TOL_MM` (1 mm) with the same size class. No new bases. |
| `test_bases_clear_rims_found` | Each `clear` base of `<name>.bases.json` is found within its `tol_mm` (2 mm) with the right size, in any mode. Skipped without `<name>.bases.json`. |
| `test_bases_benchmark_no_worse` | One test for all photos. The bases benchmark totals are not worse than `bases_baseline.json` (rules in "Bases benchmark" below). Skipped without the baseline file. |
| `test_bases_done_criteria` | The done test of priority 3 (`PhotoScore.done_ok`): every true base found (clear and hidden), right size, no false bases, each clear base within 2 mm. Needs `<name>.empty.jpg` and `<name>.bases.json`, else skipped. |
| `test_rectify_done_criteria` | The done test of priority 2 in `specs/plan.md`, with no expected file: all four board edges found and tilted less than `EDGE_ANGLE_TOL_DEG` (0.3 deg), board corners at 90 deg within `CORNER_ANGLE_TOL_DEG` (0.5 deg), each printed grid line found and within `GRID_TOL_MM` (3.5 mm) of its nominal place, neighbouring grid lines spaced right within `SPACING_TOL_MM` (1.5 mm). |

`GRID_TOL_MM` is large because it includes the card placement error: in the
two test photos the board right edge is 2.5 to 3.3 mm inside the L on the
cards, and the grid lines near it move with the board (largest line offset
3.0 mm). The spacing test does
not depend on the card placement. A 1.5 mm error over a 97 mm Tomb World
section is 0.4 mm per inch. The plan allows 1 mm per inch. The test photos
give at most 1.4 mm.

A change of `board.thickness_mm` moves the fitted board plane (up to about
1 mm at the board edge for each 1 mm of thickness at these camera heights).
`test_rectify_match_expected` then fails. That is a correct change: make the
expected files again ("Update an expected file" below).

The corner tolerance is `CORNER_TOL_MM = 1.0` mm in `tests/test_photos.py`. The test
changes the pixel distance to mm with the size of the tag at that corner
(tag edge in pixels divided by `tags.size_mm`). Thus the same tolerance
applies to photos with different resolutions and camera heights.

In single-frame mode the two Tomb World photos give these results (the
ground truth was checked on 4 px per mm crops):

| Photo | Clear bases found | Hidden bases found | False bases |
|---|---|---|---|
| `tomb-world-1` | 5 of 5 | 1 of 3 (the model on the pink token, size unknown) | 2: a round barricade, a 50 mm spinner token |
| `tomb-world-2` | 4 of 4 | 1 of 4 (a tyranid, as 32 mm) | 5: two circles on the large model, a hole in a wall grating, a wall part, the spinner token |

The hidden tyranids next to walls show less than half of their rim. Their
`tol_mm` is large because their true centre is not known to 2 mm. The done
test needs photos with an empty-board reference.

The tests ignore the `ms` run times and the `message` texts.

A failure message gives the corner and the distance:

```
AssertionError: board corner 2 off by 1.68 mm (tolerance 1.0 mm)
```

### Add a photo

1. Put the photo in `data/snapshots/tests`. To get a frame from the real
   camera, push `s` in `battlesight corners`, then move the file from
   `data/snapshots/`.
2. Run the pipeline on it:

   ```
   uv run battlesight image data/snapshots/tests/<name>.jpg --out out/
   ```

3. Open `out/<name>.overlay.jpg`. Make sure that each white cross is on its
   board corner and each green outline is on its tag. Open
   `out/<name>.rectified.jpg`. Make sure that the board is upright, its
   edges are on the cyan outline, and the yellow grid lines are on the
   printed section lines.
   Open `out/<name>.bases.jpg`. Make sure that each green circle is on the
   bottom edge of a base.
4. Copy the report to the expected file:

   ```
   cp out/<name>.json data/snapshots/tests/<name>.expected.json
   ```

5. Optional, for the bases tests: click `<name>.bases.json` with
   `tools/annotate_bases.py` ("Click the ground truth" below). For the done
   test, take `<name>.empty.jpg` too ("Add a game pair" below), and run
   step 2 with `--reference data/snapshots/tests/<name>.empty.jpg`.
6. Run `uv run pytest -m photos`. A new photo with ground truth gives no
   baseline failure ("new photo, not in baseline"). Write the baseline again
   when you want to keep its numbers (see "Bases benchmark").
7. Commit the photo and its files together.

A photo without an expected file makes `test_photo_size_matches` and
`test_corners_match_expected` fail. The failure message gives the commands
above.

### Update an expected file

Update an expected file only when the change is correct. Example: a better
corner detector that puts the corners nearer to the true board corners.

1. Run step 2 and step 3 of "Add a photo" on the photo.
2. Compare the new overlay with the old one.
3. Copy the new report over the expected file. Tell in the commit message
   why the expected result changed.

Do not update an expected file only to make a failed test pass.

### New stages

`test_photo_passes_all_stages` runs all stages, so it includes a new stage
at once. After you add a stage (see [cli.md](cli.md), "Add a stage"):

1. Make the expected files again for all photos, with the procedure in
   "Update an expected file". The new files hold the data of the new stage.
2. Add a `test_<stage>_match_expected` test to `tests/test_photos.py`. Use a
   tolerance in mm, as in `test_corners_match_expected`.
3. Add the test to the table in "What the tests check".

## Bases benchmark

`tools/bench_bases.py` ([cli.md](cli.md)) and the photo tests use the same
code, `src/battlesight/bench.py`. Plan: `specs/bases-benchmark.md`.

### Matching

1. All (true base, detection) pairs with distance `<= tol_mm`.
2. Sort by distance, then by the order in the file.
3. Take the pairs in that order. Skip a pair when its true base or its
   detection is already taken (one to one, global greedy).
4. True bases without a pair are missed. Detections without a pair are
   false.
5. Category of a false detection: the `kind` of the nearest `not_bases`
   entry within half its diameter; else `inside_base` when the centre lies
   inside a true base (half its diameter, or 20 mm for an unknown size);
   else `other`.

### Metrics

For each photo and for the total (counts summed over the photos, then the
ratios). Also by rim (`clear`, `hidden`) and by mode.

| Metric | Meaning |
|---|---|
| `truth`, `found` | Number of true bases, number of detections. |
| `matched` | Number of matches. |
| `size_ok`, `wrong_size` | Matches with the right and the wrong size class (true size known). |
| `missed` | `truth - matched`. |
| `false`, `false_by_kind` | False detections, and their number by category. |
| `recall` | `matched / truth` (1.0 when `truth` is 0). |
| `precision` | `matched / found` (1.0 when `found` is 0). |
| `f1` | `2 p r / (p + r)` (0 when both are 0). |
| `size_accuracy` | `size_ok / (size_ok + wrong_size)` (1.0 when no size is known). |
| `center_err_mm` | Median, 90th percentile and maximum of the centre error. Only clear rims with `source` `clicked` or `detector`. |
| `done_ok` | Nothing missed, no wrong size, no false base, each clear match within 2 mm: the done test of priority 3. |
| `ms` | Base detection time. |

The by-rim metrics have no `found`, `false` or `precision`: a false
detection has no rim.

### Table

`uv run pytest -m photos` prints the table after the tests (a
`pytest_terminal_summary` hook in `tests/conftest.py`). It prints nothing
when the photo tests did not run.

```
photo                 mode    truth found match size_ok miss false  recall  prec    f1  err_med err_max  done    ms
tomb-world-1          single      8     8     6       5    2     2    0.75  0.75  0.75     0.04    0.07    no  1953
...
TOTAL                            16    19    11       9    5     8    0.69  0.58  0.63     0.05    0.09   0/3
target                                                     0     0    1.00  1.00  1.00        -    2.00   3/3
false by kind: inside_base 2, other 6
clear rims: recall 1.00 (9/9), hidden rims: recall 0.29 (2/7)
```

The `target` row is the done test. It is not a pass rule now.

### Baseline

`data/snapshots/tests/bases_baseline.json` is the output of
`--write-baseline`: the config, the git commit, the per-photo and the total
metrics, no match lists. It is committed.

A run is worse than the baseline when one of these is true (totals):

| Rule | Limit |
|---|---|
| `matched` (all rims) | lower |
| `matched` (clear rims) | lower |
| `false` | higher |
| `wrong_size` | higher |
| `center_err_mm.max` | more than 0.5 mm higher |
| a photo with `done_ok` true in the baseline | now false |
| a photo in the baseline | missing in the run |

A new photo (not in the baseline) gives no failure. When the run is better
on a rule and worse on none, the tool prints "better than baseline: run
with --write-baseline to keep it".

Move the baseline only on purpose: run
`uv run tools/bench_bases.py --write-baseline data/snapshots/tests/bases_baseline.json`,
and commit it with the new numbers in the commit message. Do not write a
baseline only to make a test pass.

`"holdout": ["<name>", ...]` in the baseline file keeps photos out of the
tuning: a sweep ranks the combinations without them and shows their `f1`
in its own column. `--write-baseline` keeps the list.

### Click the ground truth

```
uv run tools/annotate_bases.py data/snapshots/tests/<name>.jpg --prefill
```

Click 5 or more points on the bottom rim of each base (where the black
bevel touches the board), push `Enter`. Push `h` when less than half of the
rim is visible. Give each base a note (`t`). Mark round terrain, tokens and
prints that the detector finds as `not_bases`: mouse on the cyan proposal,
`x`, type the kind. With `--prefill`, `a`
accepts a detection, but check it at 4x zoom first. Push `s`. The keys are
in [cli.md](cli.md).

### Add a game pair

A game pair is the done test of priority 3.

1. Set the terrain and the tokens. Take the empty photo.
2. Put the operatives on the board. Take the game photo. Do not move tokens
   or terrain between the two photos.
3. A hand-held phone is allowed: the rectify stage maps both photos to the
   same board mm. Tall terrain does not map the same. A wall top 75 mm above
   the board moves about 75 x d / H mm when the camera moves d mm at height
   H (d = 100 mm, H = 700 mm: about 11 mm). The foreground mask then marks
   wall edges as changed. Hold the phone at about the same place, straight
   above the board centre. The benchmark shows `changed_fraction`.
4. Lock exposure and white balance if the camera app allows it. Do not change
   the lamps between the two photos.
5. Save them as `<name>.jpg` and `<name>.empty.jpg`. Add the expected file
   ("Add a photo") and click `<name>.bases.json`.

## In CI or a script

```
uv run pytest
uv run pytest -m photos
```

Both commands give exit code 0 when all tests pass. The photo tests need the
photos in the checkout. When `data/snapshots/tests` has no photos, pytest
skips the photo tests.
