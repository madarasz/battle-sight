# Bases benchmark: implementation plan

Status: parts 1 to 4 (sections 5 to 10, 12) done 2026-11-03, first baseline written. Section 11 (tuning): first two changes done 2026-11-10 (roster sizes, fixed-radius fit with known sizes; see 11.1). New photos and a holdout (11 step 2) are still open. Date: 2026-11-03. Priority 3 of `specs/plan.md`.

Read first: `CLAUDE.md`, `specs/plan.md` (priority 3 row and section 9),
`docs/cli.md` "Base detection", `docs/testing.md` "Photo tests".

## 1. Goal

Measure how well the `bases` stage works on real photos, with numbers, not
only pass or fail. Then use the numbers to tune the algorithm and its
parameters, and stop changes that make the result worse.

The work has four parts:

1. A shared matching and metrics module (`src/battlesight/bench.py`).
2. A benchmark tool (`tools/bench_bases.py`): per-photo and total metrics, debug
   images, a saved baseline, a parameter sweep.
3. A click tool for ground truth (`tools/annotate_bases.py`), so the ground
   truth does not come from the detector.
4. Photo tests that compare the metrics with the baseline, and print the
   metrics table.

Out of scope: changes to the detection algorithm. Do them after this work,
one change at a time, each checked with the benchmark (section 11).

## 2. Current state

- `src/battlesight/circles.py`: `find_circles(img, params, region, foreground)`,
  `foreground_mask(img, reference, threshold)`, `CircleParams.from_config(cfg, origin_mm)`.
- `src/battlesight/pipeline.py`: stage `bases`. It warps the frame with
  `bases_image(frame, view, cfg)` (4 px per mm, 5 mm border,
  `BASES_MARGIN_MM`), makes the board `region`, makes the foreground mask when
  `cache["bases_reference"]` is set, calls `find_circles`, and writes
  `data["bases"]` (list of `Circle.to_dict()` plus `center_px`) and
  `data["mode"]` (`reference` or `single_frame`).
- `data/board.toml` `[bases]`: `diameters_mm`, `px_per_mm`, `diameter_tol_mm`,
  `min_coverage`, `min_contrast`, `rim_ratio`, `diff_threshold`,
  `min_foreground`. `BoardConfig.bases_*` fields.
- Test photos in `tests/images`:

  | Photo | Kind | Ground truth |
  |---|---|---|
  | `tomb-world-1.jpg` | game, no reference | `tomb-world-1.bases.json`: 5 clear, 3 hidden |
  | `tomb-world-2.jpg` | game, no reference | `tomb-world-2.bases.json`: 4 clear, 4 hidden |

  The negative photo `tomb-world-empty.jpg` (terrain and tokens, no
  operatives) was removed on 2026-11-07. It was not a reference for the two
  game photos (other token places, other light: 46 % and 49 % of the board
  changes), and its printed grid failed the rectify spacing test. A new
  negative photo and an empty-board pair will replace it.

- The clear-rim ground truth of `tomb-world-1` and `-2` comes from detector
  fits, checked by eye on 4 px per mm crops. Hidden-rim positions are
  estimates (`tol_mm` 8 to 10). Replace both with clicked ground truth
  (section 8).
- `tests/test_photos.py` has `test_bases_match_expected`,
  `test_bases_clear_rims_found`, `test_bases_false_positives_no_worse`,
  `test_bases_done_criteria` (skipped without `<name>.empty.jpg`), and the
  helpers `load_truth`, `_match_truth`, `reference_path`.
- Known errors on the current photos (from the owner's review):
  - 50 mm false bases on round prints and tokens (spinner tokens). No team on
    the photos has 50 mm bases.
  - Two teleporters (terrain, round, 60 mm) with smaller round patterns
    inside give false bases (on the removed `tomb-world-empty` a 40 mm
    circle at (551, 350) mm).
  - Terrain parts (round barricades, wall gratings) give false bases.
  - Two bases are found as a smaller circle inside the real base: a pattern
    on the model or the base top.
  - Bases with most of the rim hidden (model over the rim, a wall) are missed.

## 3. Photo set and file names

Three kinds of photos. All go in `tests/images`.

| Kind | Files | Use |
|---|---|---|
| Game, single | `<name>.jpg`, `<name>.bases.json` | Single-frame mode. |
| Game pair | `<name>.jpg`, `<name>.empty.jpg`, `<name>.bases.json` | Reference mode, and single-frame mode on the same photo (`--mode single`). The done test of priority 3. |
| Negative | `<name>.jpg`, `<name>.bases.json` with `"bases": []` | False bases only. |

Every `<name>.jpg` also has `<name>.expected.json` (the existing regression
file). A `<name>.empty.jpg` is not a photo of its own: `PHOTOS` in
`tests/test_photos.py` already skips stems that end in `.empty`. The tool
must use the same rule. Put the rule in `bench.py` and import it in the test.

Rules for a new game pair (put them in `docs/setup.md` step 7 and in
`docs/testing.md`):

- Same session: set the terrain and tokens, take the empty photo, then put
  the operatives on the board, and take the game photo. Do not move tokens
  or terrain between the two.
- A hand-held phone is allowed. The rectify stage maps both photos to the
  same board mm. Tall terrain does not map the same: a wall top 75 mm above
  the board moves about 75 x d / H mm in the top-down image when the camera
  moves d mm at height H (d = 100 mm, H = 700 mm: about 11 mm). The
  foreground mask then marks wall edges as changed. Hold the phone at about
  the same place, straight above the board centre. The benchmark reports
  `changed_fraction` so the effect is visible.
- Lock exposure and white balance if the camera app allows it. Do not change
  the lamps between the two photos.

## 4. Ground truth file, version 2

Keep the current fields. Add `schema`, `source`, and `not_bases`. Readers
must accept version 1 files (no `schema`): treat them as version 2 with
`not_bases: []`, `source: "detector"` for a clear base and `source:
"estimate"` for a hidden base.

```json
{
  "schema": 2,
  "comment": "free text",
  "bases": [
    {"center_mm": [329.8, 186.5], "diameter_mm": 40, "rim": "clear", "source": "clicked", "note": "tyranid, top centre"},
    {"center_mm": [527.0, 268.0], "diameter_mm": 40, "rim": "hidden", "tol_mm": 8, "source": "estimate", "note": "tyranid against wall"},
    {"center_mm": [164.6, 240.3], "diameter_mm": null, "rim": "hidden", "tol_mm": 10, "source": "estimate", "note": "model on a token"}
  ],
  "not_bases": [
    {"center_mm": [551.0, 350.0], "diameter_mm": 60, "kind": "teleporter", "note": "dark teleporter"},
    {"center_mm": [542.9, 160.9], "diameter_mm": 45, "kind": "token", "note": "spinner token"}
  ]
}
```

| Field | Meaning |
|---|---|
| `bases[].center_mm` | Centre of the base bottom, board mm. |
| `bases[].diameter_mm` | Base bottom diameter, or `null` (size not known: any class matches, no size check). |
| `bases[].rim` | `clear`: at least half of the bottom rim is visible. `hidden`: less. |
| `bases[].tol_mm` | Match radius. Default `BASE_CENTER_TOL_MM` = 2.0 (done test). |
| `bases[].source` | `clicked` (rim points clicked, section 8), `detector` (a checked detector fit), `estimate` (approximate). |
| `not_bases[]` | Known round things that are not bases. Only for labels: a false base within `0.5 * diameter_mm` of the centre gets `kind` as its category. |
| `not_bases[].kind` | One of `teleporter`, `token`, `print`, `barricade`, `wall`, `model`, `other`. |

## 5. Matching and metrics (`src/battlesight/bench.py`)

Pure functions, no window, no file reads except where named. The tests and
the tool use the same code.

### 5.1 Data classes

```python
@dataclass(frozen=True)
class Truth:          # one entry of bases[]
    center_mm: tuple[float, float]
    diameter_mm: float | None
    rim: str          # "clear" | "hidden"
    tol_mm: float
    source: str
    note: str

@dataclass(frozen=True)
class Decoy:          # one entry of not_bases[]
    center_mm: tuple[float, float]
    diameter_mm: float
    kind: str
    note: str

@dataclass
class Match:          # one paired truth and detection
    truth: Truth
    found: dict       # one entry of the stage data["bases"]
    error_mm: float   # centre distance
    size_ok: bool | None   # None when truth.diameter_mm is None

@dataclass
class PhotoScore:
    name: str
    mode: str                      # "single_frame" | "reference"
    changed_fraction: float | None
    matches: list[Match]
    missed: list[Truth]
    false: list[tuple[dict, str]]  # (detection, category)
    ms: float                      # bases stage run time
```

### 5.2 Matching rule

1. Make all (truth, detection) pairs with distance `<= truth.tol_mm`.
2. Sort the pairs by distance, then by truth order.
3. Take pairs in that order. Skip a pair when its truth or its detection is
   already taken (one-to-one, global greedy).
4. Truths without a pair are `missed`. Detections without a pair are `false`.
5. Category of a false detection: the `kind` of the nearest decoy within
   `0.5 * decoy.diameter_mm`; else `inside_base` when its centre lies inside
   a true base circle (distance < `0.5 * truth.diameter_mm`, or 20 mm for an
   unknown size); else `other`.

The current `_match_truth` in `tests/test_photos.py` is a per-truth greedy
match. Replace it with this function. Add a unit test for a case where the
two rules differ (two truths, one detection between them).

### 5.3 Metrics

For one photo and for the total (sum of counts over photos, then the
ratios). Also split by `rim` (`clear`, `hidden`, `all`) and by mode.

| Metric | Definition |
|---|---|
| `truth`, `found` | Count of true bases, count of detections. |
| `matched` | Count of matches. |
| `size_ok` | Matches with `size_ok is True`. |
| `wrong_size` | Matches with `size_ok is False`. |
| `missed` | `truth - matched`. |
| `false` | Count of false detections. Also `false_by_kind` (dict). |
| `recall` | `matched / truth` (1.0 when `truth` = 0). |
| `precision` | `matched / found` (1.0 when `found` = 0). |
| `f1` | `2 p r / (p + r)` (0 when both 0). |
| `size_accuracy` | `size_ok / (size_ok + wrong_size)` (1.0 when no known sizes). |
| `center_err_mm` | Median, 90th percentile and maximum of `error_mm`, clicked or detector truths with `rim = clear` only. Estimates do not count. |
| `done_ok` | `missed = 0`, `wrong_size = 0`, `false = 0`, and every clear match within 2 mm. The priority 3 done test for that photo. |
| `ms` | Bases stage time. |

The plan's done test is: recall 1.0, size accuracy 1.0, no false bases,
centre error under 2 mm, on a Tomb World board with walls. Show these values
as the "target" row in the table.

## 6. Benchmark tool (`tools/bench_bases.py`)

### 6.1 Command

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
| `PATH` | `tests/images` | Photos or folders. Only photos with a `<name>.bases.json`. Skip `*.empty.*`. |
| `--config` | `data/board.toml` | Board config. |
| `--mode` | `auto` | `auto`: reference mode when `<name>.empty.*` exists. `single`: never use the reference. `reference`: only photos with a reference. |
| `--set` | none | Override a `[bases]` key for this run, for example `--set min_coverage=0.4 --set diameters_mm=32,40`. Unknown key: exit 2. |
| `--sweep` | none | Run all combinations of the given values. Print one total row per combination, best `f1` first, then fewer `false`. Images only for the best one. |
| `--out` | `out/bench` | Folder for `<name>.bench.jpg` and `bench.json`. |
| `--json` | none | Write `bench.json` to a file or `-` (stdout; table then goes to stderr). |
| `--baseline` | none | Compare with a baseline file. Print the change of each total metric. Exit 1 when a metric is worse (section 7). |
| `--write-baseline` | none | Write the totals and per-photo metrics as a new baseline. |
| `--no-images` | off | Do not write debug images (faster sweeps). |

The camera lens is the nominal lens (`CameraSettings()`), as in the photo
tests. Do not read `data/camera.toml`.

Exit codes: 0 done, 1 worse than `--baseline`, 2 bad input (no photo with
ground truth, unreadable file, reference that fails `corners` or `rectify`,
bad `--set`).

### 6.2 Speed

Corners and rectify take about 1 s per photo. A sweep must not run them
again for each combination.

1. Move the body of the `bases` stage into a function in `pipeline.py`:

   ```python
   def detect_bases(img: np.ndarray, cfg: BoardConfig, reference: np.ndarray | None) -> tuple[list[circles.Circle], np.ndarray | None]:
       """Bases in the bases image `img`. Returns (circles, foreground mask or None)."""
   ```

   The stage calls it. Its output must not change: `test_bases_match_expected`
   proves it.
2. The tool runs `Pipeline(cfg, until="rectify")` once per photo, and once
   per reference, keeps `bases_image(...)` of both, then calls
   `detect_bases` for each combination with `dataclasses.replace(cfg, bases_...=...)`.
   `px_per_mm` in `--sweep` forces a new `bases_image`: allow it, but warn
   that it is slow.

### 6.3 Text output

One row per photo, then the total, then the target:

```
photo                 mode    truth found match size_ok miss false  recall prec  f1    err_med err_max  done  ms
tomb-world-1          single      8     8     6       5    2     2   0.75  0.75  0.75   0.31    0.62   no    610
tomb-world-2          single      8    10     5       4    3     5   0.62  0.50  0.56   0.28    0.55   no    640
tomb-world-empty      single      0     1     0       0    0     1   1.00  0.00  0.00      -       -   no    590
TOTAL                            16    19    11       9    5     8   0.69  0.58  0.63   0.30    0.62  0/3
target                                                      0     0   1.00  1.00  1.00      -    2.00  3/3
false by kind: teleporter 1, token 2, barricade 1, inside_base 2, other 2
clear rims: recall 1.00 (9/9), hidden rims: recall 0.29 (2/7)
```

(The numbers above are an example layout, not measured values.)

Then one line for each miss, wrong size, and false base, so a person can find
it in the debug image:

```
tomb-world-2  MISS   40 mm hidden at (338.0, 182.0)  tyranid against wall corner
tomb-world-2  SIZE   found 32 mm, true 40 at (431.0, 183.0), 7.9 mm off  tyranid next to barricade
tomb-world-1  FALSE  50 mm at (542.9, 160.9) cov 0.55  token (spinner token)
```

### 6.4 `bench.json`

```json
{
  "schema": 1,
  "config": {"diameters_mm": [25, 28, 32, 40, 50], "min_coverage": 0.5, "...": "all [bases] values used"},
  "overrides": {"min_coverage": 0.4},
  "git": "8648c42",
  "photos": [
    {"name": "tomb-world-1", "mode": "single_frame", "changed_fraction": null,
     "metrics": {"truth": 8, "found": 8, "matched": 6, "...": "section 5.3"},
     "by_rim": {"clear": {"...": "..."}, "hidden": {"...": "..."}},
     "matches": [{"truth": {"...": "..."}, "found": {"...": "..."}, "error_mm": 0.12, "size_ok": true}],
     "missed": [{"...": "truth"}],
     "false": [{"found": {"...": "..."}, "kind": "token"}]}
  ],
  "total": {"metrics": {"...": "..."}, "by_rim": {"...": "..."}, "false_by_kind": {"token": 2}},
  "sweep": [{"overrides": {"...": "..."}, "total": {"...": "..."}}]
}
```

`git` is the output of `git rev-parse --short HEAD`, or `null`. Add `+dirty`
when `git status --porcelain` is not empty.

### 6.5 Debug image `<name>.bench.jpg`

The `bases` view image (top-down, `rectify.px_per_mm`) with:

| Mark | Colour | Meaning |
|---|---|---|
| Solid circle, size label | green | Match, right size. |
| Solid circle, `found/true` label | orange | Match, wrong size. Also a thin line to the true centre. |
| Solid circle, coverage label | red | False detection. Label also has the category. |
| Dashed circle, note | yellow | Missed true base (true size, or 30 mm for unknown). |
| Thin dotted circle | grey | `not_bases` entry. |

A legend in a corner and the total line of the photo in the banner. Use the
`overlay.py` helpers and its 1080p scale rule. Put a dashed-circle helper in
`overlay.py`.

## 7. Baseline and thresholds

### 7.1 Baseline file

`tests/images/bases_baseline.json`: the output of
`--write-baseline`. Same content as `bench.json` without `matches`,
`missed`, `false` lists and without `sweep`. It records the config and the
git commit. Commit it.

### 7.2 Pass rules (`--baseline` and the photo test)

A run is worse than the baseline when one of these is true (totals, then
each photo, mode `auto`):

| Rule | Limit |
|---|---|
| `matched` (all rims) | lower than the baseline |
| `matched` (clear rims) | lower than the baseline |
| `false` | higher than the baseline |
| `wrong_size` | higher than the baseline |
| `center_err_mm.max` | more than 0.5 mm higher than the baseline |
| `matched` of a photo | lower than the baseline |
| `false` of a photo | higher than the baseline |
| a photo with `done_ok` true in the baseline | now false |
| a photo in the baseline | missing in the run |

A photo that is new (not in the baseline) gives no failure. The tool prints
"new photo, not in baseline".

When the run is better on a rule and worse on none, the tool prints
"better than baseline: run with --write-baseline to keep it". The baseline
only moves on purpose, in a commit with the new numbers in the message.
Never write a baseline only to make the test pass (the same rule as for
expected files).

### 7.3 Targets

The targets (section 5.3) are not a pass rule now. The photo test prints
them. When a photo reaches `done_ok`, the baseline rule keeps it there.

## 8. Ground truth click tool (`tools/annotate_bases.py`)

```
uv run tools/annotate_bases.py tests/images/<name>.jpg [--config data/board.toml] [--prefill]
```

1. Run corners and rectify (nominal lens). Make the bases image at
   `bases.px_per_mm` (4 px per mm).
2. Show it in an OpenCV window. Zoom with `+` and `-` (1x, 2x, 4x of the
   screen fit), pan with the arrow keys or a right-drag. Show the board mm
   of the mouse in the banner.
3. Left-click points on the bottom rim of a base (at least 3, better 5 or
   more around the visible part). `Enter` fits a free circle
   (`circles._fit_circle`), and snaps the diameter to the nearest class in
   `bases.diameters_mm`. Show the fitted circle and the free diameter.
   Keys `1` to `9`: pick the class by index instead. `0`: size unknown
   (`null`).
4. `h` toggles the rim between `clear` and `hidden`. `t` asks for a note
   on the terminal (`input()`), `x` marks the circle as a `not_bases` entry
   and asks for its kind.
5. With fewer than 3 points, `Enter` makes an estimate: centre at the click,
   `source: "estimate"`, `tol_mm` 8.
6. `--prefill` loads the current detections as proposals (dashed). `a`
   accepts the proposal under the mouse as `source: "detector"`, `d`
   deletes the entry under the mouse.
7. `u` undo, `s` save `<name>.bases.json` (schema 2, keeps `comment`), `q`
   quit (asks to save when there are changes).
8. Clicked bases get `source: "clicked"` and `tol_mm` 2.0. Store the clicked
   points too (`"points_mm"`), so a person can check the fit later.

Put the logic (state, fit, save, load) in plain functions or a class without
OpenCV window calls, so a unit test can drive it. The window code only maps
keys and clicks to those calls.

Then re-annotate `tomb-world-1` and `tomb-world-2` with clicks (owner task,
or the agent with `--prefill` and a careful check), and add the
`not_bases` entries for the teleporters, spinner tokens, and barricades.

## 9. Photo tests

In `tests/test_photos.py`:

1. Replace `_match_truth` and `load_truth` with `bench.match` and
   `bench.load_truth`.
2. `test_bases_match_expected` and `test_bases_clear_rims_found` are removed
   (2026-11-10). The benchmark rules in 7.2 guard the results.
3. Not all bases can be found, so no test requires it.
4. Replace `test_bases_false_positives_no_worse` with
   `test_bases_benchmark_no_worse`: compute the scores for all photos with
   ground truth (module fixture, reuse the `results` fixture: no second
   pipeline run), compare with `bases_baseline.json` by the rules in 7.2.
   Skip when the baseline file does not exist, with a message that gives
   the `--write-baseline` command.
5. `test_bases_done_criteria` is removed (2026-11-10). The `done` column of
   the metrics table shows `bench.PhotoScore.done_ok`.
6. Print the metrics table after the photo tests: a `pytest_terminal_summary`
   hook in `tests/conftest.py` prints the table that the benchmark fixture
   stored on `config.stash`. It prints nothing when the photo tests did not
   run. Thus `uv run pytest -m photos` shows the numbers on every run.

## 10. Unit tests (`tests/test_bench.py`, no marker)

- Matching: exact pair, pair outside `tol_mm`, two detections near one truth
  (one match, one false), global greedy case (section 5.2), unknown size,
  decoy category, `inside_base` category.
- Metrics: counts and ratios for a hand-made case, empty truth (negative
  photo), no detections, `done_ok` true and false.
- Baseline compare: each rule in 7.2 gives "worse", a new photo gives no
  failure, a better run gives the "better" message.
- Ground truth load: schema 1 file reads as schema 2, bad `rim` value
  rejected, bad `kind` rejected.
- Tool end to end: write two synthetic photos with `render_board` (bases,
  walls, a dark-ring print as a decoy) and their `.bases.json` in
  `tmp_path`, one with a `.empty.png` reference. Run `tools/bench_bases.py`
  through `runpy` or a `main(argv)` function. Check `bench.json`, the exit
  code, `--set`, `--sweep` (row count, best first), `--write-baseline` then
  `--baseline` (exit 0), a worse `--set` against that baseline (exit 1),
  and that `<name>.bench.jpg` exists.
- Annotate logic: points on a known circle give its centre within 0.1 mm and
  the right class, save and load round trip, estimate entry with fewer than 3
  points.
- `detect_bases` refactor: the stage result on a synthetic board is the same
  before and after (existing tests in `tests/test_bases.py` cover it).

## 11. Tuning process (after parts 1 to 4)

1. Write the baseline for the current code on the current photos.
2. Owner adds photos: at least 3 game pairs (empty and game, same session),
   2 negative photos, other light, other camera positions. Annotate each
   with `annotate_bases.py`. Keep 1 pair out of tuning ("holdout": put its
   name in `bases_baseline.json` under `"holdout"`; the sweep table shows
   holdout totals in their own column).
3. Write the baseline again with the new photos (current code).
4. One change at a time. For each: run the benchmark, look at the debug
   images, keep the change only when the totals are better and the holdout
   is not worse. Write the baseline with the change, commit both.
5. Candidate changes, in order of expected gain (from the owner's review):

   | Error | Change |
   |---|---|
   | 50 mm bases that no team has | Roster in the config: `bases.diameters_mm` per game, or a `[roster]` with the sizes on the board. First test: `--set diameters_mm=32,40`. |
   | Teleporter and token patterns | Ring-in-ring rule: drop a circle when a larger dark ring that is not a base size has the same centre (within 3 mm). Reference mode removes them too. |
   | Smaller circle inside the real base | Concentric rule: of two circles with centres within 0.3 x the smaller radius, keep the larger when its coverage is at least 0.8 of the smaller one. Also check the band outside the rim against the board colour (or the reference). |
   | Terrain as bases | Reference mode. Without it: a model check (colour or texture variance inside the circle above the board value). |
   | Hidden bases | Reference mode: lower `min_coverage` (sweep 0.3 to 0.5), centre hint from the changed blob. |
   | Hand-held pairs | `changed_fraction` high near walls: ignore changed pixels in a band around strong edges that the reference also has, or align the two bases images with a small ECC warp before the diff. |

   Done 2026-11-10 (section 11.1): the roster row, and the hidden bases row
   in a stronger form.

6. Later, with priority 4 (operative ID): a known operative gives a known
   base size and a last known position. Then fit one size with a lower
   coverage limit. Order: rough position from the changed blob, then the ID
   from a crop, then the exact fit. Add this as a new mode in the benchmark
   (`--sizes-from truth`: use the true size of each base as the only class
   near it) to measure the gain before priority 4 exists.

### 11.1 Changes of 2026-11-10

Measured on the 4 photos (34 true bases). No holdout exists yet, so these
values are fitted on the test photos: check them again with new photos.

| Change | Matched | False | Notes |
|---|---|---|---|
| Baseline before | 25 | 10 | f1 0.72, done 0 of 4 |
| Roster: only the base sizes of the teams (32, 40 mm) | 25 | 2 | The app takes them from the gallery `team.toml`, the benchmark from `roster_mm` in `<name>.bases.json`. |
| Fixed-radius fit with a roster, `reference_min_coverage` 0.4 with a reference | 30 | 1 | f1 0.92, done 2 of 4, max centre error 0.93 mm |

Evidence for the fixed radius: at the true centre of the 9 missed bases, the
fixed-radius fit has a coverage of 0.41 to 0.52 and is under 1.1 mm off for 7
of them. The free fit fails, or gives 42 mm at 3 mm off (it goes to the model
or its shadow).

Tried and not kept:

| Change | Result |
|---|---|
| Fixed radius with all 5 config sizes (no roster) | 27 matched, 10 false: no gain in false bases, and with coverage 0.4 in reference mode 2 false per photo. Thus the fixed radius needs a roster. |
| Coverage 0.4 without a reference | 31 matched, 32 false (62 on single-frame runs of all photos). The changed area of a reference is what stops the false circles. |
| Free search, then fixed search for the rest | 1 match less than the fixed search alone. |
| Ray edge nearest to the known radius, not the strongest | More false bases (22 at coverage 0.5), fewer matches. |

Left: 4 bases on the 2 photos without a reference (3 hidden rims, 1 model on
a teleporter), 1 false base (a round barricade on `tomb-world-1`).

## 12. Documentation

- `docs/testing.md`: the photo kinds and file names (section 3), ground
  truth schema 2, the click tool, the metrics and their meaning, the
  baseline rules, the printed table, how to add a game pair.
- `docs/cli.md`: a section for `tools/bench_bases.py` and
  `tools/annotate_bases.py` (options, exit codes, `bench.json`).
- `docs/setup.md` step 7: the rules for a game pair (section 3).
- `CLAUDE.md`: the two tool commands, `bench.py` in the architecture list,
  the baseline rule.
- `specs/plan.md` section 9, priority 3: the baseline numbers.
- `README.md`: no change (developer tools).

## 13. Work order

| Step | Work | Done when |
|---|---|---|
| 1 | `bench.py`: data classes, `load_truth` (schema 1 and 2), `match`, `score`, `compare`. Unit tests. | `uv run pytest tests/test_bench.py` passes. |
| 2 | `detect_bases` refactor in `pipeline.py`. | All unit and photo tests pass with no change to the expected files. |
| 3 | `tools/bench_bases.py`: run, table, `bench.json`, images, `--mode`, `--set`. | Tool runs on `tests/images`, end-to-end unit test passes. |
| 4 | `--sweep`, `--baseline`, `--write-baseline`. | Unit tests for compare and sweep pass. |
| 5 | Photo tests (section 9) and the terminal summary. Write the first baseline. | `uv run pytest -m photos` passes and prints the table. |
| 6 | `tools/annotate_bases.py` and its logic tests. | Logic tests pass. The owner can click a base and save. |
| 7 | Documentation (section 12). | Docs match the code. |

Steps 1 to 5 give the numbers. Step 6 makes the ground truth independent of
the detector. Re-annotation and new photos (section 11 step 2) are owner
work, not part of these steps.

## 14. Known issue, not part of this work

Closed 2026-11-07: the owner removed `tomb-world-empty.jpg` from the test
set and will take a new photo. Record of the issue:
`test_rectify_done_criteria[tomb-world-empty.jpg]` failed: neighbouring
printed grid lines y2 and y3 are 1.90 mm too close (limit `SPACING_TOL_MM`
1.5 mm). The y offsets are +1.5, +1.5, +2.1, +0.2, +0.1, -1.0, -0.7 mm: the
step is at y3 = 303 mm, the middle of the board. The likely cause is a gap
between the two board halves in this setup (the owner changed the board).
The fit of the tags (max 1.7 mm) and the edges (tilt under 0.1 deg, corners
90 +- 0.15 deg) are good. Decide with the owner: fix the board and take
the photo again, or record this photo as a known exception.
