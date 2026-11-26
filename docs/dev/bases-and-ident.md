# Bases and operative identity (priorities 3 and 4)

Modules: `circles.py`, `ident.py`, `bench.py`, `ident_bench.py`, the `bases` and `ident` stages of `pipeline.py`. Tests: `tests/test_bases.py`, `test_ident.py`, `test_bench.py`, `test_ident_bench.py`, `test_photos.py`. Plans: `docs/specs/bases-benchmark.md`, `docs/specs/ident-benchmark.md`. Config keys and steps: `docs/cli.md` "Base detection" and "Operative identity".

Design rule: base detection (geometry: background subtraction plus known-radius circle fit) stays separate from operative identity (DINOv2 embedding plus few-shot gallery). A new team must only need gallery photos, never a retrained detector.

## circles.py and the bases stage

- `find_circles` finds round bases of the known diameters (`[bases]` in `data/board.toml`, base bottom) in a top-down image at `bases.px_per_mm` (4).
- Steps: edge votes along the gradient for proposals, 120 rays with a dark-rim test, robust free fit, nearest size class, fixed-radius refit, `coverage` = visible rim part.
- Bottom edge lies on the play surface: no parallax correction.
- Reference mode: `foreground_mask` diffs against an empty-board reference. `Pipeline(reference=frame)` / `set_reference` warps the reference from its own tags into `cache["bases_reference"]`; the `bases` stage then runs in `reference` mode, else `single_frame` (finds some dark-ringed tokens and barricades).
- The stage puts `bases` (list of `Circle`) and `bases_image` in `scratch`.
- Known sizes (roster): `Pipeline(roster=...)` / `set_roster`, else `Gallery.base_sizes()` (`team.toml`), else all `diameters_mm`; `LiveSession` sets it from `ident.roster_sizes` at the start.
  - With known sizes `detect_bases` searches only these and keeps the radius in the fit (`_verify_fixed`; a free fit goes to the model when part of the rim is hidden), and with a reference too the limit is `reference_min_coverage` (0.4).
  - Without known sizes the fit is free: do not use the fixed radius with all config sizes, and do not lower the coverage without a reference (both measured worse, `docs/specs/bases-benchmark.md` 11.1).
- `pipeline.BasesSearch` limits one search (an area of the frame, known size, `relaxed` = second search with `[bases] relaxed_*`, results need a click). `pipeline.find_base_near` is the live "Locate operative" search.
- `pipeline.detect_bases(img, cfg, reference)` is the body of the bases stage.

## ident.py and the ident stage

- Crop around each base from the bases image: `store_crop` keeps 40 mm board each side, `prepare` cuts `ident.crop_pad_mm`, turns, masks a circle, resizes to `input_px`.
- `Embedder`: `DinoEmbedder` = DINOv2 via `transformers`, MPS, lazy load or `load(input_px, progress)` (5 steps of `LOAD_STEPS`, then a warm-up run on a blank crop; a lock makes the load happen once when two threads ask); `make_embedder` keeps one embedder per model name in a process, so a new gallery (team change) does not load the model again; `ColorEmbedder` = `model = "color-hist"`, no ML, used by all unit tests.
- `Gallery`:
  - `data/teams/<team>/team.toml` with `base_mm`, optional `wounds` (maximum wounds, `Operative.wounds`, game state only; `enroll --wounds` sets it at the start) optional `max` (copies, `docs/specs/operative-copies.md`) and optional `default_copies` (copies in a new team list, 0 to `max`, default 1; `read_default_copies`; `enroll --default-copies` sets it). `counts_as = "Original"` makes an operative a variant model (own photos, values of the original; `team_operatives(..., variants=True)` lists it, a team list never does; `Gallery.members`, `canonical`, `columns` has no variant column; `identify` takes the best score of an original and its variants). `counts_as` can also be a list of names with one base size (`Operative.targets`): `Operative.counts_as` is then the first name that is not in `Gallery.out_of_play`; the setter of `out_of_play` resolves it again.
  - `<operative>/*.png`, `<operative>/bg/` empty-board crop of each photo (optional).
  - `mask_crop` (`ident.mask` switch in `data/board.toml`: grey on all but the operative, raw and masked embeddings side by side, `can_mask`, per-base fallback to the full crop).
  - `embeddings.npz` cache, `rotations` angles per crop, score = mean of `top_k` best photos, `leave_one_out`.
  - `Gallery.add` raises `DuplicatePhoto` for a crop with the same pixels as a photo of that operative (`same_photo`). `Gallery.record` False (live app on `--image-source`, `LiveSession.record_photos`): `LabelSession.choose` adds no photo.
  - `Gallery.hidden` = file indexes that `scores` skips. `Gallery.out_of_play` = removed operatives (not `allowed`, never named). `Gallery.copies` gives `assign` one column per copy.
- `IdentMemory` (in `pipeline.cache["ident_memory"]`, RAM only, never saved: a new run starts empty): the last `identify` call. A base of the same diameter within `reuse_move_mm` of an earlier base, with a thumbnail (`prepare` at angle 0, grey, 48 px) that differs by at most `reuse_crop_diff`, keeps the earlier score row and is not masked or embedded. `assign` always runs on all rows (a moved operative can free a name). A kept base keeps its old centre and thumbnail, so slow drift adds up and ends the reuse. The memory clears when `Gallery.version` (+1 in `_set_views`), `Gallery.hidden`, the masking mode, the gallery object or the empty-board reference changes. `IdentMemory.report` (stage data `reuse`) and `embed_ms` show each decision; the live app prints them (`LiveSession._reuse_text`). `identify` without `memory` (`_add_at`, benchmarks) reads every base.
- `assign`: size-class limit, Hungarian one-to-one, `min_score` unknown, `min_margin` unsure.
- `LabelSession`: click fix logic, no window calls; `choose` returns None and `undo` (done, path) for a fix that added no file. `LabelSession.selected` is not used by the live app.
- `IdentTruth` / `score_truth`: done test.
- Labels are `<team>/<operative>`; copy labels `<team>/<operative>#<k>`. Never compare a copy label with a gallery label without `idt.type_of`.
- Stage `ident` after `bases`: always ok, names nothing without a gallery (`Pipeline(gallery=...)`, `image --team`). It puts `base_crops` and `idents` in `scratch`.

## bench.py (bases benchmark)

- Pure functions. Ground truth `<name>.bases.json` schema 2 (`Truth`, `Decoy` = `not_bases`, `roster_mm` = base sizes of the teams, given to the stage by the tool and the photo tests; schema 1 still loads).
- `match`: one-to-one global greedy within `tol_mm`, false detections get a `kind`, `inside_base` or `other`.
- `PhotoScore` (`done_ok` = priority 3 done test), `metrics` / `total` (recall, precision, f1, size accuracy, centre error of clear clicked or detector rims), `report` (`bench.json`), `compare` (baseline rules), text table.
- Photo file rules: `find_photos`, `reference_path` (`<name>.empty.*` is a reference).

## ident_bench.py (name benchmark)

- Runs on `<name>.ident.json` photos.
- `score` splits losses by cause (`right`, `wrong`, `unknown` = naming; `wrong_size`, `missed` = detection, operative size from the gallery).
- `metrics` (`accuracy` = done test, `naming_accuracy` = only correctly detected bases), `report`, `compare` (baseline rules, per-photo `right` too).
- `gallery_state`: content hash of the git-ignored gallery; hashes only names and base sizes of `team.toml`. `same_gallery` guards each compare.
- `copies_of`: gallery photos that show the same frame as a found base of the test photo, from a name fix on `--image-source`; `run` hides them for that photo, metric `hidden`.
- `run` is shared by `tools/bench_ident.py` and `test_ident_benchmark_no_worse`.

## Baselines

`tests/images/bases_baseline.json` and `ident_baseline.json` move only on purpose (`--write-baseline`, new numbers in the commit message), never only to make a test pass.
