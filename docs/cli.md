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
| `image` | saved photos | Run the analysis pipeline on photos. Write a JSON report and overlay images. Use it for tests and automation. |

## `corners`: live camera

```
uv run battlesight corners [--source 0] [--width 1920] [--height 1080] [--fps N]
                           [--config data/board.toml] [--display-width 1600]
```

| Option | Default | Meaning |
|---|---|---|
| `--source` | `0` | Camera index or stream URL. Find the index with `uv run tools/list_cameras.py`. |
| `--width`, `--height` | `1920`, `1080` | Requested frame size. The camera can give a different size. The command prints the real size. |
| `--fps` | camera default | Requested frame rate. |
| `--config` | `data/board.toml` | Board and tag config. |
| `--display-width` | `1600` | Window width on screen. The analysis always uses the full frame. |

Keys in the window:

| Key | Action |
|---|---|
| `q`, `Esc` | Quit. |
| `r` | Reset the reference. Use it after you move the board on purpose. |
| `s` | Save the current frame to `data/snapshots/snap_<date>_<time>.png`. |

The README section "What you see" tells what the overlay shows.

## `image`: saved photos

```
uv run battlesight image PATH [PATH ...] [--until STAGE] [--config FILE]
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

Send the report to another tool:

```
uv run battlesight image shot.jpg --json - | jq '.images[0].stages.corners.data.board_corners_px'
```

Look at one photo after one stage only:

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

Each new priority adds a stage to this list. `--until` accepts only the
stages that exist. `uv run battlesight image --help` shows them.

### JSON report

The `--json` report:

```json
{
  "schema": 1,
  "config": "data/board.toml",
  "stages": ["corners"],
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
def _draw_rectify(frame, result, cfg):
    ...  # draw in place on the BGR frame

@stage("rectify", draw=_draw_rectify)
def _rectify(ctx: Context) -> StageResult:
    corners = ctx.scratch["board_corners_px"]  # from the corners stage
    ...
    return StageResult(ok=True, message="...", data={...})
```

Rules:

- `data` must be JSON-safe: numbers, strings, lists, dicts, `None`. Put NumPy
  arrays and other objects in `ctx.scratch` for later stages and for drawing.
- Put expensive objects (detectors, models) in `ctx.cache`. The cache stays
  for all images of one command.
- The stage must not open a window or read a camera.

The `image` command, `--until`, and `test_photo_passes_all_stages` use the
new stage with no other change. Add the stage to the "Stages" table above.
Then update the photo test expected files (see [testing.md](testing.md),
"New stages").

## Add a command

Add a `cmd_<name>(args)` function and a subparser in `main()` in
`src/battlesight/app.py`. Return the exit code from the function. Add a
section for the command to this document.
