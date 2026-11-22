# tools: helper scripts

Run each with `uv run tools/<name>.py`. After a change here, run `uv run pytest -m vision`, then `-m photos` when a detection or naming result can change.

## Commands

```bash
uv run tools/list_cameras.py              # list camera indices
uv run tools/list_cameras.py --use 1 --width 3840 --height 2160  # check index 1, save to data/camera.toml
uv run tools/list_cameras.py --url http://PHONE-IP:8080/video    # check a Wi-Fi stream, save it
uv run tools/list_cameras.py --scrcpy --width 3840 --height 2160   # Android phone camera over USB (scrcpy server), save it
uv run tools/make_tags.py                 # print sheet -> data/print/corner_tags.pdf
uv run tools/make_charuco.py              # lens calibration sheet -> data/print/charuco.pdf
uv run tools/bench_bases.py               # bases benchmark on tests/images: table, out/bench/
uv run tools/bench_bases.py --sweep min_coverage=0.4,0.5 --no-images   # parameter sweep, best f1 first
uv run tools/bench_bases.py --baseline tests/images/bases_baseline.json   # exit 1 when worse
uv run tools/bench_ident.py               # identification benchmark on tests/images: table
uv run tools/bench_ident.py --baseline tests/images/ident_baseline.json   # exit 1 when worse, 2 for another gallery
uv run tools/dedupe_gallery.py [--dry-run]   # delete gallery photos that show the same frame (oldest stays)
uv run tools/bench_light.py -v            # light terrain search benchmark on tests/images (<name>.light.json): table
uv run tools/annotate_bases.py tests/images/<name>.jpg --prefill   # click base ground truth, writes a missing <name>.expected.json
uv run tools/annotate_bases.py tests/images/<name>.jpg --team kommandos veterans   # also assign operatives (n key) -> <name>.ident.json
```

## Notes

- `make_tags.py`: corner cards are 90 mm square, tag in the outer quadrant, black L in the inner quadrant at the board corner. The 5 mm white margin is the tag quiet zone.
- `bench_bases.py` runs corners and rectify once and calls `pipeline.detect_bases` (the body of the bases stage), so it sweeps `[bases]` values cheaply. Logic is in `bench.py`.
- `bench_ident.py` and `test_ident_benchmark_no_worse` share `ident_bench.run`.
- `bench_light.py` measures both light terrain searches of `lightfit.py`. Do not change the `[light]` limit or the edge values without it.
- `annotate_bases.py` holds the click logic in `Annotator` (no window calls, unit tested).
- Baselines (`tests/images/*_baseline.json`) move only on purpose, never only to make a test pass. See `docs/dev/bases-and-ident.md`.
