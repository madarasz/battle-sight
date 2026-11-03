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
                         [--out DIR] [--json FILE|-] [--show]
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
| `--out DIR` | none | Write `<name>.overlay.jpg` and `<name>.json` for each image to `DIR`. The command makes `DIR` if it does not exist. |
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
`rectify` stage), and `<name>.json`.

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

Each new priority adds a stage to this list. `--until` accepts only the
stages that exist. `uv run battlesight image --help` shows them.

### JSON report

The `--json` report:

```json
{
  "schema": 1,
  "config": "data/board.toml",
  "stages": ["corners", "rectify"],
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

In CI or a shell script, use the exit code of the `image` command:

```
uv run battlesight image data/snapshots/tests -q --json report.json || echo "pipeline failed"
```

## Add a stage

Add the stage in `src/battlesight/pipeline.py`, after the existing stages:

```python
def _draw_bases(frame, result, cfg):
    ...  # draw in place on the BGR camera frame

def _views_bases(frame, result, cfg):
    return {"bases": ...}  # optional extra images, written as <name>.bases.jpg

@stage("bases", draw=_draw_bases, views=_views_bases)
def _bases(ctx: Context) -> StageResult:
    view = ctx.scratch["view"]              # BoardView from the rectify stage
    top_down = ctx.scratch["rectified"]     # top-down image, no overlays
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
