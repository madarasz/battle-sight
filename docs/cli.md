# Battle Sight CLI

The `battlesight` command has one subcommand for each mode. Run all commands
with `uv run` from the repository root.

```
uv run battlesight --help
uv run battlesight <command> --help
uv run battlesight --version
```

| Command | Input | Use |
|---|---|---|
| `corners` | live camera | Track the four corner tags. Warn when a tag is lost or the board moved. |
| `rectify` | live camera | Show the board top-down, with the map grid. |
| `bases` | live camera | Find the operative bases on a key push (snapshot). Take the empty-board reference. |
| `calibrate` | live camera or saved photos | Calibrate the lens with the ChArUco sheet. Write the `[lens]` table of `data/camera.toml`. |
| `image` | saved photos | Run the analysis pipeline on photos. Write a JSON report and overlay images. Use it for tests and automation. |

## `corners`: live camera

```
uv run battlesight corners [--source 0] [--width 1920] [--height 1080] [--fps N]
                           [--camera-file data/camera.toml]
                           [--config data/board.toml] [--display-width 1600]
```

The camera options come from three places. The first that has a value wins:

1. the command line option,
2. the camera file (`data/camera.toml`, saved by `tools/list_cameras.py`),
3. the built-in default in the table below.

The command prints the camera file path when it uses one.

| Option | Default | Meaning |
|---|---|---|
| `--source` | `0` | Camera index (Camo, built-in camera) or network stream URL (`http://...`, `rtsp://...`). A stream is read on a background thread that keeps only the newest frame. The phone app sets its size, so `--width` and `--height` have no effect. See [setup.md](setup.md), step 4. |
| `--width`, `--height` | `1920`, `1080` | Requested frame size. The camera can give a different size. The command prints the real size. |
| `--fps` | camera default | Requested frame rate. |
| `--camera-file` | `data/camera.toml` | Saved camera settings. A missing file is not an error: the built-in defaults apply. |
| `--config` | `data/board.toml` | Board and tag config. |
| `--display-width` | `1600` | Window width on screen. The analysis always uses the full frame. |

Keys in the window:

| Key | Action |
|---|---|
| `q`, `Esc` | Quit. |
| `r` | Reset the reference. Use it after you move the board on purpose. |
| `s` | Save the current frame to `data/snapshots/snap_<date>_<time>.png`. |

The README section "What you see" tells what the overlay shows.

## `rectify`: live top-down view

```
uv run battlesight rectify [--source 0] [--width 1920] [--height 1080] [--fps N]
                           [--camera-file data/camera.toml]
                           [--config data/board.toml] [--display-width 1600]
```

The options are the same as for `corners`. The command finds the four tags,
fits the board plane, and shows the board from straight above at
`rectify.px_per_mm`, with the lens distortion removed and the map grid on
top (see "Map grid" below).

When the drift reference is locked and the status is OK, the view uses the
locked reference. Thus the image does not jitter. When the board moved, the
view follows the tags in each frame, and the banner shows "BOARD MOVED".
Push `r` to lock the new position.

| Key | Action |
|---|---|
| `q`, `Esc` | Quit. |
| `v` | Change between the top-down view and the camera view with the grid. |
| `g` | Grid on or off. |
| `r` | Reset the drift reference. |
| `s` | Save the current camera frame to `data/snapshots/`. |

The banner on the right tells which lens is in use: `calibrated` (from the
`[lens]` table of `data/camera.toml`) or `nominal`.

## `bases`: find the bases

```
uv run battlesight bases [--reference data/snapshots/empty_board.png]
                         [--source 0] [--width 1920] [--height 1080] [--fps N]
                         [--camera-file data/camera.toml]
                         [--config data/board.toml] [--display-width 1600]
```

Snapshot mode (plan section 1, "Measure mode"): the command shows the live
camera, and runs the full pipeline (`corners`, `rectify`, `bases`) on one
frame when you push `space`. The result stays on the screen until the next
`space`. The command prints each base to stdout: diameter class, centre in
board mm, rim coverage.

| Option | Default | Meaning |
|---|---|---|
| `--reference` | `data/snapshots/empty_board.png` | Empty-board reference. Loaded at start when the file exists. The `e` key writes it. Git ignores the file (`data/snapshots/*.png`). |

The other options are the same as for `corners`.

| Key | Action |
|---|---|
| `space` | Detect on the current frame. |
| `e` | Use the current frame as the empty-board reference and write it to `--reference`. The frame must pass `corners` and `rectify`, else the command keeps the old reference and prints why. |
| `c` | Clear the reference: single-frame mode. |
| `v` | Change between the top-down view (`bases` view) and the camera view with all overlays. |
| `s` | Save the current camera frame to `data/snapshots/`. |
| `q`, `Esc` | Quit. |

## `calibrate`: lens calibration

```
uv run battlesight calibrate [--source 0] [--width 1920] [--height 1080]
                             [--camera-file data/camera.toml]
                             [--images PATH [PATH ...]]
                             [--min-views 10]
```

Do this one time for each camera mode (resolution, lens, zoom). Without a
calibration, the app uses a nominal lens: focal length from `focal_35mm` in
`data/camera.toml` (default 24 mm), no distortion. That is good for phones that correct the
distortion in the phone, but a calibration is better.

1. Print the target: `uv run tools/make_charuco.py`. This writes
   `data/calib/charuco.pdf` (A4 landscape, ChArUco 10 x 7, 25 mm squares).
   Print at 100 % or fit to page. The print size does not change the lens
   values (focal length, centre, distortion): they are in pixels and do not
   depend on the square size. The scale must be the same in both directions.
2. Glue the print to something flat and stiff.
3. Run the command with the same camera as for play. With no camera
   options, it uses the saved camera file.
4. Hold the target in the camera view. Green dots show the found corners.
   Push `space` to keep the view. Keep 10 to 20 views: the target at the
   centre, at each edge and corner of the frame, some tilted 20 to 40 deg.
5. Push `c`. The command calibrates and writes `[lens]` to the camera file.
   The other values in the file stay.

With `--images`, the command reads the views from image files or folders.
All images must have the same size.

| Option | Default | Meaning |
|---|---|---|
| `--images` | none | Image files or folders. Use them instead of the camera. |
| `--camera-file` | `data/camera.toml` | Camera file. The camera options come from it, and the command writes `[lens]` to it. |
| `--min-views` | `10` | Fewest views with the target. The command always needs at least 3. |

The command prints the reprojection error (`rms`). Under 0.5 px is good.

Exit codes: `0` calibration written, `1` too few views, `2` bad input or no
camera.

The app uses `[lens]` only for frames with the same aspect ratio as the
calibration images. It scales it to other sizes with that aspect ratio (4K
calibration, 1080p frames). For other frames it uses the nominal lens.

## `image`: saved photos

```
uv run battlesight image PATH [PATH ...] [--until STAGE] [--config FILE] [--camera-file FILE]
                         [--reference FILE] [--out DIR] [--json FILE|-] [--show]
                         [--display-width 1600] [-q]
```

`PATH` is an image file or a folder. For a folder, the command reads all
images directly in it (not in subfolders), in name order. Accepted types:
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`. The command applies
the EXIF orientation of phone photos.

| Option | Default | Meaning |
|---|---|---|
| `--until STAGE` | all stages | Run the stages up to and including `STAGE`, then stop. |
| `--config` | `data/board.toml` | Board and tag config. |
| `--camera-file` | `data/camera.toml` | Camera file. Only the lens (`focal_35mm`, `[lens]`) is used. The photo tests do not read it: they always use the nominal 24 mm lens. |
| `--reference FILE` | none | Empty-board photo (terrain set up, no operatives) from the same camera and light. The `bases` stage then runs in reference mode for all images. An unreadable file, or one where `corners` or `rectify` fails, gives exit code 2. |
| `--out DIR` | none | Write `<name>.overlay.jpg`, one `<name>.<view>.jpg` per stage view, and `<name>.json` for each image to `DIR`. The command makes `DIR` if it does not exist. |
| `--json FILE` | none | Write one report for all images to `FILE`. Use `-` to write it to stdout. |
| `--show` | off | Show each overlay in a window. Push a key for the next image. Push `q` or `Esc` to stop. |
| `--display-width` | `1600` | Window width for `--show`. |
| `-q`, `--quiet` | off | Do not print a status line for each image. |

### Examples

Check all test photos:

```
uv run battlesight image data/snapshots/tests
```

Write overlays and JSON files to a folder:

```
uv run battlesight image data/snapshots/tests --out out/
```

`--out` writes `<name>.overlay.jpg` (camera frame with all overlays),
`<name>.rectified.jpg` (top-down board with the map grid, from the
`rectify` stage), `<name>.bases.jpg` (top-down board with the found bases),
and `<name>.json`.

Find the bases with an empty-board reference:

```
uv run battlesight image game.jpg --reference empty.jpg --out out/
```

Send the report to another tool:

```
uv run battlesight image shot.jpg --json - | jq '.images[0].stages.corners.data.board_corners_px'
```

Look at one photo after the corners stage only:

```
uv run battlesight image shot.jpg --until corners --show
```

### Output

Each image gives one status line on stdout. With `--json -`, the status lines
go to stderr, so stdout holds only the JSON.

```
ok   data/snapshots/tests/tomb-world-1.jpg  corners ok: 4/4 tags
FAIL data/snapshots/tests/blank.png  corners FAIL: 0/4 tags, missing [0, 1, 2, 3]
```

Errors (file not found, file is not an image) go to stderr as `error: ...`.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All stages passed on all images. |
| `1` | One or more stages failed on one or more images. All images were read. |
| `2` | An input is missing or is not an image, or no image was found. Bad options also give `2` (argparse). |

The command does all images before it stops, also after a failure. One bad
image does not stop the others.

### Stages

The pipeline runs the stages in a fixed order. Each stage uses the results of
the stages before it. When a stage fails, the command skips the stages after
it on that image.

| Stage | Priority | Passes when |
|---|---|---|
| `corners` | 1 | All four corner tags are found. |
| `rectify` | 2 | The board plane fits the 16 tag corners: the largest error is not more than `rectify.max_reproj_mm`. |
| `bases` | 3 | Always, when it runs (no bases is a valid result). Fails only when the reference has another image size. |

Each new priority adds a stage to this list. `--until` accepts only the
stages that exist. `uv run battlesight image --help` shows them.

### JSON report

The `--json` report:

```json
{
  "schema": 1,
  "config": "data/board.toml",
  "stages": ["corners", "rectify", "bases"],
  "reference": null,
  "ok": true,
  "errors": [],
  "images": [ /* one image report for each image */ ]
}
```

One image report (also the content of `<name>.json` from `--out`):

```json
{
  "schema": 1,
  "source": "data/snapshots/tests/tomb-world-1.jpg",
  "size": [3072, 4096],
  "ok": true,
  "stages": {
    "corners": {
      "ok": true,
      "skipped": false,
      "message": "4/4 tags",
      "ms": 210.4,
      "data": {
        "tags": {
          "0": {"corner": "top-left", "corners_px": [[x, y], ...], "side_px": 158.0}
        },
        "missing": [],
        "board_corners_px": [[x, y], [x, y], [x, y], [x, y]]
      }
    }
  }
}
```

| Field | Meaning |
|---|---|
| `reference` | Path of the `--reference` file, or `null`. Only in the combined report. |
| `schema` | Report format version. It increases when a field changes or is removed. A new stage or a new field in `data` does not change it. |
| `size` | Image `[width, height]` in pixels, after the EXIF rotation. |
| `ok` | `true` when all stages in the report passed. |
| `stages.<name>.skipped` | `true` when an earlier stage failed. |
| `stages.<name>.ms` | Run time of the stage. Do not compare it in tests. |
| `stages.<name>.data` | Results of the stage. Each stage has its own fields. |

Fields of the `corners` stage:

| Field | Meaning |
|---|---|
| `tags` | Found tags by id: board corner name, the four tag corners in pixels (tag order TL, TR, BR, BL), mean edge length in pixels. |
| `missing` | Ids of the tags that were not found. |
| `board_corners_px` | Board corners in pixels, order TL, TR, BR, BL. `null` for a corner without a tag. |

Fields of the `rectify` stage:

| Field | Meaning |
|---|---|
| `lens` | Lens in use: `source` (`calibrated` or `nominal`), image `size`, `fx`, `fy`, `cx`, `cy` in pixels, `dist` (OpenCV k1 k2 p1 p2 k3), `rms_px` of the calibration. |
| `H_mm_to_px` | 3 x 3 homography from board mm on the play surface to undistorted image pixels. Apply the lens distortion after it to get frame pixels. |
| `reproj_rms_mm`, `reproj_max_mm` | Fit error of the 16 tag corners, in mm. A large value means a tag card is off its corner, a tag is printed at the wrong size, or the board size in the config is wrong. |
| `camera_mm` | Camera centre: x and y in board mm, height above the table in mm. |
| `thickness_mm`, `surface_scale` | Board thickness from the config, and the scale it gives about the camera nadir (1.0 for thickness 0). |
| `board_corners_px` | Play surface corners in frame pixels, order TL, TR, BR, BL. |
| `rectified` | Top-down image: `px_per_mm`, `margin_mm` (table around the board), `size` in pixels. |
| `edges` | The board edges found in the top-down image, by name (`top`, `right`, `bottom`, `left`). `offset_mm`: distance from the nominal edge at its midpoint, positive into the board. `angle_deg`: tilt. `n`: samples used, `rms_mm`: fit error. `null` when not found. |
| `corner_angles_deg` | Board corner angles TL, TR, BR, BL from the found edges. `null` when an edge is missing. |
| `grid_lines` | Only with `grid.printed = true`. Offsets in mm of the printed grid lines from their nominal places: `x` for the vertical lines (left to right), `y` for the horizontal lines (top to bottom). Positive = right or down. `null` for a line that is hidden too much. |

The edge and grid offsets are checks. They do not make the stage fail. A
large edge offset means the board is not on the L lines of the cards (the
board slid, or the board size in the config is wrong).

Fields of the `bases` stage:

| Field | Meaning |
|---|---|
| `mode` | `reference` (an empty-board reference is set) or `single_frame`. |
| `px_per_mm` | Scale of the top-down image for the circle fit (`bases.px_per_mm`). |
| `diameters_mm` | The size classes from the config. |
| `changed_fraction` | Reference mode: part of the board that differs from the reference. A value near 1 means other light or a moved camera: take the reference again. `null` in single-frame mode. |
| `bases` | Found bases, best rim coverage first. Each: `center_mm` (board mm), `center_px` (frame pixels), `diameter_mm` (size class), `fit_diameter_mm` (free circle fit before the class was chosen), `coverage` (0..1, part of the rim that supports the fit), `rms_mm` (fit residual), `contrast` (median Lab L step from the rim to the board). |

## Base detection

`src/battlesight/circles.py`, plan priority 3. Geometry only, no training.
The stage warps the frame to a top-down image at `bases.px_per_mm` (default
4 px per mm, the camera gives about 3.5 to 4) with a 5 mm border. Then:

1. With a reference: Lab distance to the reference image (warped the same
   way from its own tags), blur, threshold `bases.diff_threshold`. Centres
   must lie near changed pixels, and `bases.min_foreground` of the base disc
   must be changed.
2. Proposals: each strong edge pixel with a dark band inside (at 0.5 and
   1.0 mm) and lighter board outside (at 1.25 mm) votes for the centre one
   radius inward along its gradient, for each diameter. The peaks of the
   vote maps are the proposals (`max_proposals` per diameter).
3. Verification: 120 rays from the proposal. On each ray the rim edge is the
   strongest dark-to-light step within 3.5 mm of the radius that passes the
   rim test: `outer - rim > bases.min_contrast` and
   `rim < bases.rim_ratio * outer`. A robust free circle fit gives the centre
   and the fitted diameter. Three rounds, each with rays from the new centre.
4. Class: the nearest diameter in `bases.diameters_mm`, when it is within
   `bases.diameter_tol_mm`. A last fit with the class radius gives the
   centre. `coverage` is the part of the 120 rays with an edge within 1 mm
   of that circle. It must be at least `bases.min_coverage`.
5. Overlapping circles: the best coverage wins. Bases can touch, so two
   circles overlap when their centres are closer than 0.8 times the sum of
   the radii.

The diameter is the bottom of the base, where it touches the board. The
bottom edge lies on the play surface, so the rectify homography maps it
without parallax. The top of the base is smaller and higher; the rim test
uses only the dark bevel next to the bottom edge. On the far side of a base
at a board corner (about 35 deg view angle) the top edge can hide the
bottom edge by about 0.5 mm.

Single-frame mode (no reference) also finds round things with a dark rim:
tokens with a dark ring, round barricades, holes in a wall grating. Bases
with less than `min_coverage` of the rim visible (a model or a wall over
it) are not reported. Reference mode removes the static ones. Plan section
9 has the results on the test photos. `tools/bench_bases.py` (below)
measures them.

`[bases]` in `data/board.toml`:

| Key | Default | Meaning |
|---|---|---|
| `diameters_mm` | `[25, 28, 32, 40, 50]` | Size classes, base bottom. |
| `px_per_mm` | `4.0` | Scale of the top-down image for the fit. |
| `diameter_tol_mm` | `3.0` | Largest distance of the fitted diameter from a class. |
| `min_coverage` | `0.5` | Smallest visible part of the rim. |
| `min_contrast` | `12.0` | Smallest Lab L step (0..255) from the rim band to the board. |
| `rim_ratio` | `0.75` | Rim band L at most this part of the board L outside. |
| `diff_threshold` | `25.0` | Reference mode: Lab distance that counts as changed. |
| `min_foreground` | `0.25` | Reference mode: changed part of the base disc. |

## Map grid

The grid that `rectify` and the overlays draw depends on the map. Set it in
the `[grid]` section of `data/board.toml`. All values are in board mm.

| Map | Grid | Config |
|---|---|---|
| Tomb World | 7 x 6 printed sections inside a 12 mm margin. Each section is 97 mm. An inch grid does not agree with the print. | `margin_mm = 12.0`, `sections = [7, 6]`, `printed = true` |
| Volkus | 1 in squares from the board edge. | `margin_mm = 0.0`, `pitch_mm = 25.4`, no `sections` |

| Key | Default | Meaning |
|---|---|---|
| `margin_mm` | `0.0` | Distance from each board edge to the first grid line. |
| `sections` | none | `[nx, ny]`: split the space inside the margin into nx by ny equal sections. The lines include both margin lines. |
| `pitch_mm` | `25.4` | Line spacing when `sections` is not set. The lines start at the margin. |
| `printed` | `false` | The grid lines are printed on the board. The `rectify` stage then finds them in the top-down image and reports `grid_lines`: a check of the rectification that does not use the tags. |

The Tomb World values come from the test photos: the thick section lines are
12 mm from each edge of a 703 x 606 mm board, (703 - 24) / 7 = (606 - 24) /
6 = 97 mm apart. For a new map, find the lines in `out/<name>.rectified.jpg`
and set the margin and sections or pitch.

## Automated tests

[testing.md](testing.md) tells about the unit tests and the photo tests. The
photo tests use the `<name>.json` files from `--out` as expected results.
The bases benchmark (`tools/bench_bases.py`, below) gives metrics and a
baseline for the bases stage.

In CI or a shell script, use the exit code of the `image` command:

```
uv run battlesight image data/snapshots/tests -q --json report.json || echo "pipeline failed"
```

## Add a stage

Add the stage in `src/battlesight/pipeline.py`, after the existing stages:

```python
def _draw_tokens(frame, result, cfg):
    ...  # draw in place on the BGR camera frame

def _views_tokens(frame, result, cfg):
    return {"tokens": ...}  # optional extra images, written as <name>.tokens.jpg

@stage("tokens", draw=_draw_tokens, views=_views_tokens)
def _tokens(ctx: Context) -> StageResult:
    view = ctx.scratch["view"]              # BoardView from the rectify stage
    top_down = ctx.scratch["rectified"]     # top-down image, no overlays
    bases = ctx.scratch["bases"]            # list of circles.Circle from the bases stage
    ...
    return StageResult(ok=True, message="...", data={...})
```

Rules:

- `data` must be JSON-safe: numbers, strings, lists, dicts, `None`. Put NumPy
  arrays and other objects in `ctx.scratch` for later stages and for drawing.
- Put expensive objects (detectors, models) in `ctx.cache`. The cache stays
  for all images of one command.
- The stage must not open a window or read a camera.
- `views` is optional. It returns extra named images. `--out` writes each as
  `<name>.<view>.jpg`, `--show` shows each in its own window.

The `image` command, `--until`, and `test_photo_passes_all_stages` use the
new stage with no other change. Add the stage to the "Stages" table above.
Then update the photo test expected files (see [testing.md](testing.md),
"New stages").

## `tools/bench_bases.py`: bases benchmark

Plan: `specs/bases-benchmark.md`. The tool measures the `bases` stage on
photos with ground truth (`<name>.bases.json`). It gives numbers, not only
pass or fail. Use it to tune `[bases]` and to stop changes that make the
result worse.

```
uv run tools/bench_bases.py [PATH ...] [--config data/board.toml]
                            [--mode auto|single|reference]
                            [--set KEY=VALUE ...] [--sweep KEY=V1,V2,... ...]
                            [--out out/bench] [--json FILE|-]
                            [--baseline FILE] [--write-baseline FILE]
                            [--no-images] [-q]
```

| Option | Default | Meaning |
|---|---|---|
| `PATH` | `data/snapshots/tests` | Photos or folders. Only photos with a `<name>.bases.json` count. `<name>.empty.*` is a reference, not a photo. |
| `--config` | `data/board.toml` | Board config. |
| `--mode` | `auto` | `auto`: reference mode when `<name>.empty.*` exists. `single`: never use the reference. `reference`: only photos with a reference. |
| `--set` | none | Override a `[bases]` key for this run: `--set min_coverage=0.4 --set diameters_mm=32,40`. Unknown key or bad value: exit 2. |
| `--sweep` | none | Run all combinations. Values split by `,`. For `diameters_mm` split the alternatives by `;`: `--sweep "diameters_mm=32,40;25,28,32,40,50"`. Prints one total row per combination, best `f1` first, then fewer `false`. The table, `bench.json` and the images are for the best one. A `px_per_mm` sweep makes a new bases image for each value (slow). |
| `--out` | `out/bench` | Folder for `<name>.bench.jpg` and `bench.json`. |
| `--json` | none | Also write `bench.json` to a file, or `-` for stdout (the table then goes to stderr). |
| `--baseline` | none | Compare with a baseline file. Exit 1 when a rule is worse (see [testing.md](testing.md), "Bases benchmark"). |
| `--write-baseline` | none | Write the metrics as a new baseline. Keeps the `holdout` list of an existing file. |
| `--no-images` | off | No debug images (faster sweeps). |
| `-q` | off | No progress lines on stderr. |

The tool uses the nominal lens, as the photo tests do. It does not read
`data/camera.toml`. It runs `corners` and `rectify` once per photo and per
reference, and then only the base detection (`pipeline.detect_bases`) for
each combination.

Exit codes: 0 done, 1 worse than `--baseline`, 2 bad input (no photo with
ground truth, unreadable file, a photo or reference that fails `corners` or
`rectify`, bad `--set` or `--sweep`, bad ground truth file).

Text output: one row per photo, `TOTAL`, `target` (the priority 3 done
test), false bases by kind, recall by rim. Then one line for each miss
(`MISS`), wrong size (`SIZE`) and false base (`FALSE`), with the board mm,
so you can find it in the debug image.

`<name>.bench.jpg`: the top-down image at `rectify.px_per_mm` with green
circles (match, right size), orange (match, wrong size, `found/true`, line
to the true centre), red (false base, kind and coverage), yellow dashed
(missed base), grey dotted (`not_bases` entry). The banner has the photo
result.

`bench.json`:

```json
{
  "schema": 1,
  "config": {"diameters_mm": [25.0, 28.0, 32.0, 40.0, 50.0], "min_coverage": 0.5, "...": "all [bases] values used"},
  "overrides": {"min_coverage": 0.4},
  "git": "f56156b+dirty",
  "photos": [
    {"name": "tomb-world-1", "mode": "single_frame", "changed_fraction": null,
     "metrics": {"truth": 8, "found": 8, "matched": 6, "...": "..."},
     "by_rim": {"clear": {"...": "..."}, "hidden": {"...": "..."}},
     "matches": [{"truth": {"...": "..."}, "found": {"...": "..."}, "error_mm": 0.04, "size_ok": true}],
     "missed": [{"...": "truth entry"}],
     "false": [{"found": {"...": "..."}, "kind": "token"}]}
  ],
  "total": {"metrics": {"...": "..."}, "by_rim": {"...": "..."}, "by_mode": {"...": "..."}, "false_by_kind": {"other": 6}},
  "sweep": [{"overrides": {"...": "..."}, "total": {"...": "..."}, "holdout": null}]
}
```

`git` is `git rev-parse --short HEAD`, with `+dirty` when the work tree has
changes, or `null`. `sweep` is there only with `--sweep`. `holdout` (a list
of photo names) is there when the baseline has one. The metric fields are
in [testing.md](testing.md), "Bases benchmark".

Examples:

```
uv run tools/bench_bases.py                                            # all test photos
uv run tools/bench_bases.py --set diameters_mm=32,40                   # only the sizes of the teams on the photos
uv run tools/bench_bases.py --sweep min_coverage=0.4,0.45,0.5 --no-images
uv run tools/bench_bases.py --baseline data/snapshots/tests/bases_baseline.json
```

## `tools/annotate_bases.py`: click the ground truth

```
uv run tools/annotate_bases.py data/snapshots/tests/<name>.jpg [--config data/board.toml] [--prefill]
                               [--display-width 1600] [--display-height 950]
```

The tool runs `corners` and `rectify` (nominal lens), makes the bases image
(`bases.px_per_mm`), and shows it in a window. It loads
`<name>.bases.json` when it exists and writes it (schema 2) on `s`. The
ground truth then does not come from the detector.

| Key | Action |
|---|---|
| left click | Add a point on the bottom rim of a base. 5 or more around the visible part is best. |
| `Enter` | 3 or more points: fit a free circle and snap the diameter to the nearest class. `source: "clicked"`, `tol_mm` 2, the points go in `points_mm`. The rim is `clear` when the points span at least 180 deg. Fewer than 3 points: an estimate at the click, `source: "estimate"`, `tol_mm` 8, rim `hidden`. |
| `1` to `9` | With points: fit with size class 1 to 9 of `bases.diameters_mm`. Without points: set the size of the base under the mouse. |
| `0` | Size unknown (`null`). |
| `Backspace` | Drop the points. |
| `h` | Rim `clear` <-> `hidden` (base under the mouse, else the last one). |
| `t` | Note for the entry under the mouse. Type it in the terminal. |
| `x` | `not_bases` entry. With 3 or more points: of their fit. Else: of the base or the proposal under the mouse (the base moves, the proposal goes away). Type its kind in the terminal. |
| `a` | With `--prefill`: accept the detector proposal (cyan dashed) under the mouse, `source: "detector"`. The rim is `clear` when its coverage is 0.5 or more. |
| `p` | With `--prefill`: show all proposals. By default a proposal on a saved entry (within its `tol_mm`, at least 3 mm, or inside a `not_bases` circle) is hidden. |
| `d` | Delete the entry or proposal under the mouse. |
| `u` | Undo. |
| `+` / `-`, `z` / `Z`, mouse wheel | Zoom in / out at the mouse: 1x, 2x, 4x, 8x of the screen fit. |
| arrows, right-drag | Pan. |
| `s` | Save. |
| `q`, `Esc` | Quit. Asks in the terminal to save when there are changes. |

The banner shows the board mm under the mouse, the counts, the points, and
the free fit diameter, and the shown / all proposals. Green circles are clear bases, yellow hidden bases,
dashed circles estimates, cyan dashed circles detector proposals (not
saved), grey dotted circles `not_bases`, magenta the fit of the current
points. A label shows the size and the first letter of the source (`c`
clicked, `d` detector, `e` estimate).

The edit logic is the `Annotator` class (no window calls). `tests/test_bench.py` drives it.

## `tools/list_cameras.py`: find and save the camera

```
uv run tools/list_cameras.py [--max 6] [--width 3840] [--height 2160]
uv run tools/list_cameras.py --use INDEX [--width 3840] [--height 2160] [--fps N]
uv run tools/list_cameras.py --url URL
```

| Option | Default | Meaning |
|---|---|---|
| (none) | | List the device indices `0` to `--max - 1`: default size, size after the request, frame rate. Saves nothing. |
| `--use` | none | Check this device index at `--width` x `--height` and save it. When the camera gives another size, the script saves the size it gives. |
| `--url` | none | Check this stream (30 frames, measured frame rate) and save the URL. Width and height are not saved for a stream. |
| `--width`, `--height` | `3840`, `2160` | Frame size to request from a device. |
| `--fps` | none | Frame rate to save with `--use`. |
| `--no-save` | off | Only check. Do not write the camera file. |
| `--camera-file` | `data/camera.toml` | Camera settings file to write. |

Exit code 0 when the camera works, 1 when it does not open or gives no frame.
The script writes only `[camera]`. `focal_35mm` and `[lens]` stay. When the
source changes and the file has a lens calibration, the script tells you to
calibrate again if the calibration is for another camera.

## Camera file

`data/camera.toml` holds everything about the camera. It is local to the
machine (it can hold the phone IP address), so git ignores it. Without the
file, the app uses the built-in defaults: camera 0, 1920 x 1080, nominal
24 mm lens.

```toml
[camera]                     # tools/list_cameras.py writes it
source = 1                   # device index, or a URL string such as "http://192.168.1.20:8080/video"
width = 3840                 # devices only
height = 2160                # devices only
fps = 30.0                   # optional
focal_35mm = 24.0            # nominal lens: 35 mm equivalent focal length

[lens]                       # battlesight calibrate writes it
image_size = [3840, 2160]    # frame size of the calibration views
camera_matrix = [[3012.4, 0.0, 1921.7], [0.0, 3010.9, 1078.2], [0.0, 0.0, 1.0]]
dist_coeffs = [0.0412, -0.1033, 0.0002, -0.0004, 0.0581]   # OpenCV k1 k2 p1 p2 k3
rms_px = 0.31                # reprojection error
views = 15
date = "2026-11-03 14:20:11"
```

`CameraSettings` in `src/battlesight/camera.py` reads and writes it. The
live commands and `image` read the lens from it (`--camera-file`). The board
config (`data/board.toml`) must not have a `[camera]` table: the app stops
with an error that tells you to move it.

## Add a command

Add a `cmd_<name>(args)` function and a subparser in `main()` in
`src/battlesight/app.py`. Return the exit code from the function. Add a
section for the command to this document.
