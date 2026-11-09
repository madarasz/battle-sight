# Identification benchmark: implementation plan

Status: parts 1 to 4 (sections 5 to 10, 12) done 2026-11-09, first baseline written. Section 11 (tuning) not started. Date: 2026-11-10. Priority 4 of `specs/plan.md`.

Read first: `CLAUDE.md`, `specs/plan.md` (priority 4 row and section 9),
`specs/bases-benchmark.md`, `docs/cli.md` "Operative identity",
`docs/testing.md` "Operative names" and "Identification benchmark".

## 1. Goal

Measure how well the `ident` stage names the operatives on real photos, with
numbers, not only pass or fail. Then use the numbers to tune the stage, and
stop changes that make the names worse.

The work has four parts:

1. A shared scoring module (`src/battlesight/ident_bench.py`).
2. A benchmark tool (`tools/bench_ident.py`): per-photo and total metrics, a
   saved baseline.
3. Name assignment in the click tool (`tools/annotate_bases.py --team`), so a
   person can make `<name>.ident.json`.
4. A photo test that compares the metrics with the baseline, and prints the
   metrics table.

Out of scope: changes to the identification algorithm. Do them after this
work, one change at a time, each checked with the benchmark (section 11).

## 2. Current state

- `src/battlesight/ident.py`: `Gallery`, `identify`, `assign`, `IdentTruth`,
  `score_truth` (the done test).
- `src/battlesight/pipeline.py`: stage `ident`. It reads the bases of the
  `bases` stage, and puts `idents` in `scratch`.
- `data/board.toml` `[ident]`: `model`, `crop_pad_mm`, `input_px`,
  `rotations`, `top_k`, `min_score`, `min_margin`, `mask`, `mask_threshold`,
  `mask_min_base`, `mask_max_fraction`.
- Gallery: `data/teams/Deathwatch` and `data/teams/Raveners`, 5 operatives
  each, 21 to 30 photos per operative, 276 photos in all. Git ignores it.
- Test photos in `tests/images`:

  | Photo | Mode | True names |
  |---|---|---|
  | `tomb-world-1.jpg` | single frame | 7 |
  | `tomb-world-2.jpg` | single frame | 8 |
  | `tomb-world-4k-1.png` | reference | 10 |
  | `tomb-world-4k-2.png` | reference | 9 |

- First baseline, `[ident]` values of `data/board.toml` (`input_px` 280,
  `mask` true):

  | Photo | Right | Wrong | Unknown | Wrong size | Missed | Accuracy | Naming accuracy |
  |---|---|---|---|---|---|---|---|
  | `tomb-world-1` | 4 of 7 | 2 | 0 | 0 | 1 | 0.57 | 0.67 |
  | `tomb-world-2` | 3 of 8 | 1 | 0 | 2 | 2 | 0.38 | 0.75 |
  | `tomb-world-4k-1` | 9 of 10 | 0 | 0 | 1 | 0 | 0.90 | 1.00 |
  | `tomb-world-4k-2` | 6 of 9 | 0 | 0 | 0 | 3 | 0.67 | 1.00 |
  | Total | 22 of 34 | 3 | 0 | 3 | 6 | 0.65 | 0.88 |

- Known errors on the current photos:
  - 9 of the 12 lost names come from detection: 6 missed bases and 3 wrong
    sizes. A wrong size also stops the true name, because `assign` compares
    only operatives of the found size.
  - The 3 wrong names are all Raveners on the two single-frame photos:
    Felltalon as Prime, Tremorscythe as Felltalon, Prime as Felltalon.
  - No wrong name has `sure` true. No base is unknown.
  - The two single-frame photos have no empty-board reference, so they have
    no mask. Their right names score 0.76 and 0.77. The 4k photos score 0.86
    and 0.88.

## 3. Photo set and file names

All files go in `tests/images`. The benchmark uses the photo rules of
`bench.py` (`find_photos`, `reference_path`).

| File | Use |
|---|---|
| `<name>.jpg` or `.png` | Game photo. |
| `<name>.ident.json` | True names (section 4). Without it, the benchmark does not use the photo. |
| `<name>.empty.jpg` or `.png` | Optional empty-board reference. The `bases` stage then runs in reference mode, and the `ident` stage can use the mask. |
| `<name>.bases.json` | Bases ground truth. The identification benchmark does not read it. |

Rules for a useful test photo:

- Make the gallery from other photos than the test photo. A confirmed name
  in the live app adds a crop to the gallery, so do not confirm names on a
  frame that becomes a test photo.
- Name every base on the photo. An unnamed base is not in the test.
- Check each name by eye. The `--prefill` best guess is the output of the
  app. An unchecked guess makes the test compare the app with itself.

## 4. Ground truth file

`<name>.ident.json`:

```json
{
  "teams": ["Raveners", "Deathwatch"],
  "tol_mm": 10,
  "bases": [
    {"center_mm": [329.8, 186.5], "label": "Raveners/Felltalon"}
  ]
}
```

| Field | Meaning |
|---|---|
| `teams` | Gallery teams of the game. The gallery has only these teams. |
| `tol_mm` | Match radius from a true centre to a found base. Default 10. |
| `bases[].center_mm` | Centre of the base bottom, board mm. |
| `bases[].label` | True operative, `<team>/<operative>`. |

## 5. Scoring and metrics (`src/battlesight/ident_bench.py`)

The tests and the tool use the same code.

### 5.1 Data class

```python
@dataclass
class PhotoScore:
    name: str
    mode: str                                   # "single_frame" | "reference"
    truth: int                                  # number of true names
    right: list[float]                          # score of each right name
    wrong: list[tuple[str, str, bool]]          # (true, got, sure)
    unknown: list[str]                          # true labels of bases with no name
    wrong_size: list[tuple[str, float, float]]  # (true, found mm, true mm)
    missed: list[str]                           # no found base within tol_mm
    ms: float                                   # time of all stages
```

### 5.2 Matching rule

1. Take the true names in file order.
2. Each true name takes the nearest free found base within `tol_mm`. This is
   the rule of `ident.score_truth`, so `right` and `accuracy` agree with the
   done test.
3. A true name with no such base is `missed`.

### 5.3 Result of a true name

Examine these conditions in this order. The first that is true gives the
result.

| Result | Condition | Cause |
|---|---|---|
| `right` | The found base has the true name. | |
| `wrong_size` | The operative has a `base_mm`, and the found size is not that size. | Detection |
| `unknown` | The found base has no name. | Naming |
| `wrong` | The found base has another name. | Naming |

The size of an operative comes from the gallery (`team.toml`), not from
`<name>.bases.json`.

### 5.4 Metrics

For one photo and for the total (sum of counts over photos, then the ratios).

| Metric | Definition |
|---|---|
| `truth` | Count of true names. |
| `right`, `wrong`, `unknown`, `wrong_size`, `missed` | Count of each result. |
| `sure_wrong` | Wrong names with `sure` true. The user sees no warning on these. |
| `accuracy` | `right / truth` (1.0 when `truth` = 0). The done test. |
| `naming_accuracy` | `right / (right + wrong + unknown)` (1.0 when the sum = 0). Only bases with a correct detection and size. |
| `score_right_mean` | Mean score of the right names, or `null`. For information. No rule uses it. |
| `done_ok` | `accuracy >= 0.9` (`DONE_ACCURACY`). For the total: all photos pass. |
| `done`, `photos` | Photos that pass, photos in the set. |
| `ms` | Time of all stages. |

The plan's done test is: 9 of 10 bases named right on the first try. The
table shows this target.

### 5.5 Gallery state

`gallery_state(root, teams)` gives `teams`, `files` and `fingerprint`. The
fingerprint is the first 16 hex digits of a SHA-256 of the name and the
content of each file in the team folders: gallery photos, `bg/` crops,
`team.toml`. The embedding cache (`.npz`) and hidden files do not count.

## 6. Benchmark tool (`tools/bench_ident.py`)

### 6.1 Command

```
uv run tools/bench_ident.py [PATH ...] [--config data/board.toml]
                            [--gallery-root data/teams] [--json FILE|-]
                            [--baseline FILE] [--write-baseline FILE] [-q]
```

| Option | Default | Meaning |
|---|---|---|
| `PATH` | `tests/images` | Photos or folders. Only photos with a `<name>.ident.json`. Skip `*.empty.*`. |
| `--config` | `data/board.toml` | Board config. |
| `--gallery-root` | `data/teams` | Folder with one folder per team. |
| `--json` | none | Write the report to a file or `-` (stdout; table then goes to stderr). |
| `--baseline` | none | Compare with a baseline file. Exit 1 when a rule is worse (section 7). |
| `--write-baseline` | none | Write the totals and per-photo metrics as a new baseline. |
| `-q` | off | No progress lines. |

The camera lens is the nominal lens, as in the photo tests. Do not read
`data/camera.toml`. Each photo uses its empty-board reference when it has one.

Exit codes: 0 done, 1 worse than `--baseline`, 2 bad input (no photo with
names, unreadable file, a team with no gallery or no gallery photos, a failed
stage, a baseline of another gallery).

### 6.2 Text output

One row per photo, then the total, then the target. Then one line for each
lost name, so a person can find it on the photo:

```
photo                 mode     truth right wrong  sure  unk size miss    acc  naming  score  done     ms
tomb-world-1          single       7     4     2     0    0    0    1   0.57    0.67   0.76    no   4428
...
TOTAL                             34    22     3     0    0    3    6   0.65    0.88   0.83   1/4
target: accuracy 0.90 on each photo. sure = wrong names with no unsure mark. naming = right / (right + wrong + unk).

tomb-world-1: WRONG Raveners/Felltalon named Raveners/Prime
tomb-world-2: SIZE Deathwatch/Gravis found 25 mm, true 40 mm
tomb-world-2: MISS Raveners/Felltalon
```

### 6.3 Report

```json
{
  "schema": 1,
  "config": {"model": "facebook/dinov2-small", "top_k": 3, "...": "all [ident] values used"},
  "gallery": {"teams": ["Deathwatch", "Raveners"], "files": 554, "fingerprint": "9cbf949945da25ac"},
  "git": "9035073+dirty",
  "photos": [
    {"name": "tomb-world-1", "mode": "single_frame",
     "metrics": {"truth": 7, "right": 4, "...": "section 5.4"},
     "wrong": [{"true": "Raveners/Felltalon", "got": "Raveners/Prime", "sure": false}],
     "unknown": [],
     "wrong_size": [{"true": "Deathwatch/Sergeant", "found_mm": 28.0, "true_mm": 32.0}],
     "missed": ["Raveners/Prime"]}
  ],
  "total": {"metrics": {"...": "..."}}
}
```

`git` is `git rev-parse --short HEAD`, or `null`. Add `+dirty` when
`git status --porcelain` is not empty.

## 7. Baseline and thresholds

### 7.1 Baseline file

`tests/images/ident_baseline.json`: the output of `--write-baseline`. Same
content as the report without the `wrong`, `unknown`, `wrong_size` and
`missed` lists. It records the config, the gallery state and the git commit.
Commit it.

### 7.2 Pass rules (`--baseline` and the photo test)

A run is worse than the baseline when one of these is true. The totals use
only the photos in both the run and the baseline, so a new photo cannot
change the result.

| Rule | Limit |
|---|---|
| total `right` | lower than the baseline |
| total `naming_accuracy` | lower than the baseline |
| total `sure_wrong` | higher than the baseline |
| `right` of a photo in the baseline | lower than the baseline |
| a photo with `done_ok` true in the baseline | now false |
| a photo in the baseline | missing in the run |

The per-photo `right` rule stops a gain on one photo from hiding a loss on
another. The tool prints "new photo, not in baseline" for a new photo. It
gives no failure.

When the run is better on a rule and worse on none, the tool prints "better
than baseline: run with --write-baseline to keep it". The baseline only moves
on purpose, in a commit with the new numbers in the message. Never write a
baseline only to make the test pass.

A better detection can add a base that the naming gets wrong. Then
`naming_accuracy` falls although the app is better. Examine that result, then
write the baseline again if the change is correct.

### 7.3 Gallery rule

The scores change when the gallery changes, with no code change. A baseline
holds only for its own gallery.

| Case | `--baseline` | Photo test |
|---|---|---|
| Same fingerprint | Compare. | Compare. |
| Other fingerprint | Exit 2, the message says why. | Skip, the message says why. |
| No gallery folder for a team | Exit 2. | Skip. |

The photo test gives a skip because the gallery is local to one machine. A
skip thus does not tell that the names are good. After a gallery change,
examine the table and write the baseline again.

### 7.4 Targets

The target is not a pass rule of the baseline. `test_ident_done_criteria` is
the separate done test. When a photo reaches `done_ok`, the baseline rule
keeps it there.

## 8. Name assignment (`tools/annotate_bases.py`)

```
uv run tools/annotate_bases.py tests/images/<name>.jpg [--prefill] [--team NAME ...]
```

1. `--team` turns on the `n` key. Without it the tool works as before.
2. `n` on a confirmed base opens the list of the operatives of that base size
   in the terminal, keys 1 to 9. Push the number in the window. `0` clears
   the name, `n` shows the next page, `Esc` cancels. No operative has a fixed
   key.
3. `--prefill` with `--team` runs the identification of the app. Each base
   with no name starts with the best guess. `a` gives an accepted proposal
   its guess. In the list, the guess is key 1.
4. A name from `<name>.ident.json` stays. A guess that another base has is
   not set.
5. `s` writes `<name>.bases.json`, `<name>.ident.json` (only the named bases,
   at the centre of the base), and `<name>.expected.json` when that file is
   missing. It never replaces an expected file.
6. A name with no base gives a warning and stays in the file. A name of a
   team outside `--team` is an error (exit 2).

The names are not in `<name>.bases.json`. `Truth.label` is edit state only.
The logic is in the `Annotator` class, without window calls.

## 9. Photo tests

In `tests/test_photos.py`:

1. Keep `test_ident_done_criteria` (`ident.score_truth`).
2. Add the module fixture `ident_report`: run `ident_bench.run` on all photos
   with names, and store the table on `config.stash`. Skip on `NoGallery`.
3. Add `test_ident_benchmark_no_worse`: compare with `ident_baseline.json` by
   the rules in 7.2. Skip when the baseline file does not exist, with a
   message that gives the `--write-baseline` command. Skip for another
   gallery (7.3).
4. The `pytest_terminal_summary` hook in `tests/conftest.py` prints the table
   after the bases table.

## 10. Unit tests (`tests/test_ident_bench.py`, no marker)

All use the colour-histogram embedder. No model download.

- Scoring: each result, the order of the conditions in 5.3, an operative with
  no known size, no found base, the same result as `score_truth`.
- Metrics: counts and ratios for a hand-made case, no truth.
- Report: the baseline form has no name lists.
- Baseline compare: each rule in 7.2, a better run, a gain on one photo that
  must not hide a loss on another, a new photo that cannot fail.
- Gallery state: content counts, cache and hidden files do not.
- Tool end to end on a synthetic gallery and photo: report, table,
  `--write-baseline` then `--baseline` (exit 0), a worse run (exit 1), another
  gallery (exit 2), bad input (exit 2).
- Annotate logic in `tests/test_bench.py`: the assign list, pages, clear,
  cancel, best guess, `<name>.ident.json` round trip, expected file only when
  missing.

## 11. Tuning process (after parts 1 to 4)

1. Check each name in the four `<name>.ident.json` files by eye. Write the
   baseline again if a name changes.
2. Find out if the gallery holds crops of the test photos. Remove them, or
   take new test photos. Write the baseline again.
3. Owner adds photos: at least 3 game pairs (empty and game, same session),
   other light, other camera positions, other operative positions. Name each
   with `annotate_bases.py`.
4. Add holdout photos: `"holdout"` in the baseline, as in
   `specs/bases-benchmark.md` section 11.
5. One change at a time. For each: run the benchmark, keep the change only
   when the totals are better and the holdout is not worse. Write the
   baseline with the change, commit both.
6. Candidate changes, in order of expected gain:

   | Error | Change |
   |---|---|
   | Missed bases, wrong sizes (9 of 12 lost names) | Bases tuning first (`specs/bases-benchmark.md` section 11). Measure the gain here too. |
   | Wrong size stops the true name | Add `--sizes-from truth` (section 13) to measure the gain. Then let the gallery give a known size a second, relaxed search. |
   | Felltalon, Prime and Tremorscythe confused | First examine the masks and the gallery photos. Sweep `crop_pad_mm`, `top_k`, `input_px`. Then try `facebook/dinov2-base`. |
   | No mask on photos without a reference | Take a reference for each game. Compare `mask` true and false on the same photos. |
   | `min_score` and `min_margin` not tuned for masks | Sweep them. No base is unknown now, so `min_score` has no effect. |
   | Live fixes feed the gallery | Measure the effect with a fixed copy of the gallery, before and after a game of fixes. |

## 12. Documentation

- `docs/testing.md`: the file `<name>.ident.json`, the causes of a loss, the
  metrics, the baseline rules, the gallery rule, the printed table.
- `docs/cli.md`: a section for `tools/bench_ident.py`, the `--team` part of
  `tools/annotate_bases.py`.
- `CLAUDE.md`: the tool commands, `ident_bench.py` in the architecture list,
  the baseline rule, a link to this file.
- `specs/plan.md` section 9, priority 4: the baseline numbers (open).
- `README.md`: no change (developer tools).

## 13. Not built

Add these only when the tuning needs them.

| Item | Use |
|---|---|
| `--set`, `--sweep` for `[ident]` keys | Parameter search. The tool must then keep the embeddings of the gallery photos when only `min_score`, `min_margin` or `top_k` changes. Other keys embed all crops again. |
| `--sizes-from truth` | Give each true base its true size, to measure the naming without detection errors. |
| Truth bases as input (`--bases-from truth`) | Name the true base positions, to measure the naming alone on all 34 bases. |
| Debug images | Crop of each lost name beside its best gallery photo. |
| `holdout` list | Keep photos out of the tuning, as in the bases benchmark. |
| Leave-one-out in the report | Record `Gallery.leave_one_out` with the gallery state. |

## 14. Work order

| Step | Work | Done when |
|---|---|---|
| 1 | `ident_bench.py`: scoring, metrics, report, compare, gallery state. Unit tests. | `uv run pytest tests/test_ident_bench.py` passes. Done. |
| 2 | `tools/bench_ident.py`: run, table, report, `--baseline`, `--write-baseline`. | The tool runs on `tests/images`, the end-to-end unit tests pass. Done. |
| 3 | `tools/annotate_bases.py --team`, `--prefill` best guess. | Logic tests pass. The owner can name a base and save. Done. |
| 4 | Photo test and terminal summary. Write the first baseline. | `test_ident_benchmark_no_worse` passes and `uv run pytest -m photos` prints the table. Done. |
| 5 | Documentation (section 12). | Docs match the code. Done, except `specs/plan.md`. |
| 6 | Check of names and gallery (section 11, steps 1 and 2). | Owner work. Open. |
| 7 | Tuning (section 11, steps 3 to 6). | The done test passes on the game pairs. Open. |

## 15. Known issues

- The priority 4 done test fails on 3 of the 4 photos. This is a measured
  result. Do not lower the limit or edit a `<name>.ident.json` to make it pass.
- `bench.compare` for the bases uses the totals of all photos, so a new photo
  can fail `test_bases_benchmark_no_worse`. `specs/bases-benchmark.md` section
  7.2 says that it gives no failure. `ident_bench.compare` does not have this
  problem.
- The baseline is local to one machine, because git ignores the gallery. On
  another machine `test_ident_benchmark_no_worse` gives a skip.
- `score_right_mean` can change by a small value between runs on MPS. No rule
  uses it.
