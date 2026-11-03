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
| `tests/test_config.py` | Load of `data/board.toml`. Rejection of bad tag ids. |
| `tests/test_calib.py` | Tag corner geometry in mm. Tag detection (flat, perspective and blur, low pixel density, wrong tag ids). Board corner position. Drift monitor: reference lock, moved board, one moved tag, lost tag. |
| `tests/test_pipeline.py` | Pipeline stages: a good board, a missing tag, skip of later stages after a failure, an error in a stage, an unknown stage name. The `image` command: exit codes 0, 1, 2, the JSON report, and the `--out` files. |

The unit tests use `render_board` in `tests/conftest.py`. It makes a top-down
image of the board on the four tag cards, with an optional perspective and
pixel shift. It also gives a function from board mm to image pixels. The
tests use this function as the true position. Use this harness for new
vision unit tests. Do not use camera captures in unit tests.

## Photo tests

The photo tests are in `tests/test_photos.py`. They use the board config in
`data/board.toml`.

### Files

Each test photo has an expected file next to it:

```
data/snapshots/tests/
  tomb-world-1.jpg
  tomb-world-1.expected.json
  tomb-world-2.jpg
  tomb-world-2.expected.json
```

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

The tolerance is `CORNER_TOL_MM = 1.0` mm in `tests/test_photos.py`. The test
changes the pixel distance to mm with the size of the tag at that corner
(tag edge in pixels divided by `tags.size_mm`). Thus the same tolerance
applies to photos with different resolutions and camera heights.

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
   board corner and each green outline is on its tag.
4. Copy the report to the expected file:

   ```
   cp out/<name>.json data/snapshots/tests/<name>.expected.json
   ```

5. Run `uv run pytest -m photos`.
6. Commit the photo and the expected file together.

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

## In CI or a script

```
uv run pytest
uv run pytest -m photos
```

Both commands give exit code 0 when all tests pass. The photo tests need the
photos in the checkout. When `data/snapshots/tests` has no photos, pytest
skips the photo tests.
