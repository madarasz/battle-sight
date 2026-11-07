"""Bases benchmark: matching, metrics, baseline compare, ground truth files,
the benchmark tool end to end, and the click tool logic. No camera needed."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from battlesight import bench
from battlesight.bench import Decoy, GroundTruth, Truth

from .conftest import Base, make_config, render_board

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name: str):
    """Import tools/<name>.py once (its dataclasses need it in sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def det(x, y, d=40.0, cov=0.9) -> dict:
    return {"center_mm": [x, y], "diameter_mm": d, "coverage": cov}


def tr(x, y, d=40.0, rim="clear", tol=2.0, source="clicked", note="") -> Truth:
    return Truth((x, y), d, rim, tol, source, note)


# --- matching ----------------------------------------------------------------


def test_match_exact_pair():
    m, missed, false = bench.match([tr(100, 100)], [det(100.5, 100)])
    assert len(m) == 1 and not missed and not false
    assert m[0].error_mm == pytest.approx(0.5)
    assert m[0].size_ok is True


def test_match_outside_tol():
    m, missed, false = bench.match([tr(100, 100)], [det(103, 100)])
    assert not m and len(missed) == 1 and len(false) == 1


def test_two_detections_near_one_truth():
    m, missed, false = bench.match([tr(100, 100)], [det(101.5, 100), det(100.2, 100)])
    assert len(m) == 1 and m[0].found["center_mm"] == [100.2, 100]
    assert len(false) == 1 and false[0][1] == "inside_base"


def test_global_greedy_differs_from_per_truth_greedy():
    # One detection between two truths: global greedy gives it to the nearer
    # truth (B), a per-truth greedy match to the first one in the file (A).
    truths = [tr(0, 0, tol=8, note="A"), tr(10, 0, tol=8, note="B")]
    m, missed, false = bench.match(truths, [det(6, 0)])
    assert [x.truth.note for x in m] == ["B"]
    assert [t.note for t in missed] == ["A"]
    assert not false


def test_unknown_size_matches_any_class():
    m, _, _ = bench.match([tr(100, 100, d=None)], [det(100, 100, d=50)])
    assert m[0].size_ok is None


def test_wrong_size():
    m, _, _ = bench.match([tr(100, 100, d=40)], [det(100, 100, d=32)])
    assert m[0].size_ok is False


def test_decoy_category():
    decoys = [Decoy((300.0, 300.0), 60.0, "teleporter"), Decoy((330.0, 300.0), 45.0, "token")]
    _, _, false = bench.match([], [det(305, 300), det(328, 300), det(500, 500)], decoys)
    assert [k for _, k in false] == ["teleporter", "token", "other"]


def test_inside_base_category():
    truths = [tr(100, 100, d=40), tr(300, 300, d=None, rim="hidden", tol=1)]
    _, _, false = bench.match(truths, [det(110, 100, d=25), det(315, 300, d=25), det(150, 100, d=25)])
    assert [k for _, k in false] == ["inside_base", "inside_base", "other"]


# --- metrics -----------------------------------------------------------------


def make_score(truths, found, decoys=(), mode="single_frame", name="p") -> bench.PhotoScore:
    return bench.score(name, GroundTruth(list(truths), list(decoys)), found, mode, None, 10.0)


def test_metrics_hand_case():
    truths = [tr(100, 100), tr(200, 100, d=32), tr(300, 100, rim="hidden", tol=8, source="estimate"), tr(400, 100, rim="hidden")]
    found = [det(100.5, 100), det(200, 100.3, d=40), det(303, 100), det(600, 100)]
    s = make_score(truths, found)
    m = s.metrics()
    assert (m["truth"], m["found"], m["matched"], m["size_ok"], m["wrong_size"], m["missed"], m["false"]) == (4, 4, 3, 2, 1, 1, 1)
    assert m["recall"] == pytest.approx(0.75) and m["precision"] == pytest.approx(0.75) and m["f1"] == pytest.approx(0.75)
    assert m["size_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    # Only clear, clicked or detector truths count for the centre error.
    assert m["center_err_mm"]["max"] == pytest.approx(0.5)
    assert m["false_by_kind"] == {"other": 1}
    assert not s.done_ok
    clear, hidden = s.metrics("clear"), s.metrics("hidden")
    assert (clear["truth"], clear["matched"], hidden["truth"], hidden["matched"]) == (2, 2, 2, 1)
    assert "false" not in clear


def test_metrics_negative_photo():
    s = make_score([], [det(100, 100)])
    m = s.metrics()
    assert m["truth"] == 0 and m["recall"] == 1.0 and m["precision"] == 0.0 and m["f1"] == 0.0
    assert not s.done_ok
    assert make_score([], []).done_ok


def test_metrics_no_detections():
    m = make_score([tr(100, 100)], []).metrics()
    assert m["recall"] == 0.0 and m["precision"] == 1.0 and m["f1"] == 0.0 and m["center_err_mm"] is None


def test_done_ok():
    assert make_score([tr(100, 100), tr(200, 100, d=None, rim="hidden", tol=8)], [det(101, 100), det(205, 100, d=50)]).done_ok
    # A clear rim with a large tolerance must still be within 2 mm.
    assert not make_score([tr(100, 100, tol=5)], [det(103, 100)]).done_ok
    assert not make_score([tr(100, 100)], [det(100, 100, d=32)]).done_ok


def test_total_sums_counts_then_ratios():
    a = make_score([tr(100, 100)], [det(100, 100)], name="a")
    b = make_score([tr(100, 100), tr(200, 200)], [det(500, 500)], name="b", mode="reference")
    t = bench.total([a, b])
    assert t["metrics"]["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert t["metrics"]["precision"] == pytest.approx(0.5)
    assert set(t["by_mode"]) == {"single_frame", "reference"}
    assert t["metrics"]["done"] == 1 and t["metrics"]["photos"] == 2


def test_table_and_details():
    s = make_score([tr(100, 100, note="tyranid"), tr(200, 100, d=32)], [det(200, 100, d=40), det(400, 400)], [Decoy((400.0, 400.0), 50.0, "token", "spinner")])
    table = bench.format_table([s])
    assert "TOTAL" in table and "target" in table and "false by kind: token 1" in table
    details = bench.format_details([s])
    assert "MISS" in details and "tyranid" in details and "SIZE" in details and "token (spinner)" in details


# --- baseline compare --------------------------------------------------------


def baseline_pair():
    truths = [tr(100, 100), tr(200, 100), tr(300, 100, rim="hidden", tol=8)]
    s = make_score(truths, [det(100.2, 100), det(200, 100.4), det(500, 500)], name="a")
    done = make_score([], [], name="neg")
    base = bench.report([s, done], {}, detail=False)
    return base, truths


def test_compare_same_run():
    base, _ = baseline_pair()
    c = bench.compare(copy.deepcopy(base), base)
    assert c.ok and not c.better and c.lines()[-1] == "same as baseline"


@pytest.mark.parametrize(
    "found, extra_photo, rule",
    [
        ([det(100.2, 100), det(500, 500)], True, "matched clear rims"),  # one clear base lost
        ([det(100.2, 100), det(200, 100.4), det(500, 500), det(600, 600)], True, "false"),
        ([det(100.2, 100), det(200, 100.4, d=32), det(500, 500)], True, "wrong_size"),
        ([det(100.2, 100), det(201.2, 100), det(500, 500)], True, "center_err_mm.max"),
        ([det(100.2, 100), det(200, 100.4), det(500, 500)], False, "missing in the run"),
    ],
)
def test_compare_worse_rules(found, extra_photo, rule):
    base, truths = baseline_pair()
    photos = [make_score(truths, found, name="a")] + ([make_score([], [], name="neg")] if extra_photo else [])
    c = bench.compare(bench.report(photos, {}), base)
    assert not c.ok
    assert any(rule in w for w in c.worse), c.worse


def test_compare_matched_all_rims():
    base, truths = baseline_pair()
    # Hidden base matched in the baseline? No: make a baseline with it, then lose it.
    s = make_score(truths, [det(100.2, 100), det(200, 100.4), det(303, 100)], name="a")
    base = bench.report([s], {}, detail=False)
    run = bench.report([make_score(truths, [det(100.2, 100), det(200, 100.4)], name="a")], {})
    c = bench.compare(run, base)
    assert any(w.startswith("matched ") for w in c.worse)


def test_compare_done_ok_lost():
    base, truths = baseline_pair()
    run = bench.report([make_score(truths, [det(100.2, 100), det(200, 100.4), det(500, 500)], name="a"), make_score([], [det(50, 50)], name="neg")], {})
    c = bench.compare(run, base)
    assert any("neg: done_ok" in w for w in c.worse)


def test_compare_new_photo_and_better():
    base, truths = baseline_pair()
    photos = [make_score(truths, [det(100.2, 100), det(200, 100.4)], name="a"), make_score([], [], name="neg"), make_score([], [], name="new")]
    c = bench.compare(bench.report(photos, {}), base)
    assert c.ok
    assert c.new == ["new"]
    assert any("false 1 -> 0" in b for b in c.better)
    assert "better than baseline: run with --write-baseline to keep it" in c.lines()
    assert "new photo, not in baseline: new" in c.lines()


# --- ground truth files ------------------------------------------------------


def test_schema_1_reads_as_schema_2():
    old = [{"center_mm": [1, 2], "diameter_mm": 40, "rim": "clear", "note": "x"}, {"center_mm": [9, 9], "diameter_mm": 40, "rim": "hidden", "tol_mm": 8}]
    gt = bench.parse_truth({"comment": "old", "bases": old})
    assert gt.bases[0].source == "detector" and gt.bases[0].tol_mm == 2.0 and gt.not_bases == []
    assert gt.bases[1].source == "estimate"
    assert gt.to_dict()["schema"] == 2 and gt.comment == "old"


def test_schema_2_round_trip(tmp_path):
    gt = GroundTruth(
        [tr(1, 2, note="a"), Truth((3.0, 4.0), None, "hidden", 8.0, "estimate", "b"), Truth((5.0, 6.0), 32.0, "clear", 2.0, "clicked", "c", ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))],
        [Decoy((7.0, 8.0), 60.0, "teleporter", "dark")],
        "comment",
    )
    path = tmp_path / "x.bases.json"
    bench.save_truth(path, gt)
    assert bench.load_truth(path) == gt
    assert json.loads(path.read_text())["schema"] == 2


def test_bad_rim_rejected():
    with pytest.raises(ValueError, match="rim"):
        bench.parse_truth({"schema": 2, "bases": [{"center_mm": [1, 2], "diameter_mm": 40, "rim": "half"}]})


def test_bad_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        bench.parse_truth({"schema": 2, "bases": [], "not_bases": [{"center_mm": [1, 2], "diameter_mm": 40, "kind": "tree"}]})


def test_photo_files(tmp_path):
    for n in ("a.jpg", "a.empty.jpg", "b.png", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in bench.find_photos([tmp_path])] == ["a.jpg", "b.png"]
    assert bench.reference_path(tmp_path / "a.jpg").name == "a.empty.jpg"
    assert bench.reference_path(tmp_path / "b.png") is None


# --- benchmark tool end to end -----------------------------------------------

BOARD_TOML = "[board]\nwidth_mm = 400.0\nheight_mm = 300.0\n[tags]\nids = [0, 1, 2, 3]\nsize_mm = 40.0\nmargin_mm = 5.0\n"
PRINT = ((100.0, 220.0), 40.0, 15)  # dark round print: a base in single-frame mode


@pytest.fixture(scope="module")
def bench_photos(tmp_path_factory):
    """Two synthetic photos with ground truth: `a` with an empty-board
    reference, `b` without. Both have the dark print, a decoy."""
    from .test_bases import camera_view

    d = tmp_path_factory.mktemp("bench")
    (d / "board.toml").write_text(BOARD_TOML)
    cfg = make_config(width_mm=400.0, height_mm=300.0)
    a_bases = (Base((250.0, 120.0), 28), Base((300.0, 220.0), 40))
    b_bases = (Base((280.0, 120.0), 32),)
    shots = {
        "a.png": camera_view(cfg, discs=(PRINT,), bases=a_bases)[0],
        "a.empty.png": camera_view(cfg, discs=(PRINT,))[0],
        "b.png": camera_view(cfg, discs=(PRINT,), bases=b_bases)[0],
    }
    photos = d / "photos"
    photos.mkdir()
    for name, img in shots.items():
        cv2.imwrite(str(photos / name), img)
    decoy = [Decoy(PRINT[0], PRINT[1], "print", "dark print")]
    for stem, bases in (("a", a_bases), ("b", b_bases)):
        truth = [Truth(b.center_mm, b.diameter_mm, "clear", note=f"{b.diameter_mm:g} mm") for b in bases]
        bench.save_truth(photos / f"{stem}.bases.json", GroundTruth(truth, decoy))
    return d


def run_tool(d: Path, *extra: str) -> tuple[int, dict | None]:
    tool = load_tool("bench_bases")
    out = d / "out"
    code = tool.main([str(d / "photos"), "--config", str(d / "board.toml"), "--out", str(out), "-q", *extra])
    rep = json.loads((out / "bench.json").read_text()) if code != 2 else None
    return code, rep


def test_tool_report_and_images(bench_photos, capsys):
    code, rep = run_tool(bench_photos)
    assert code == 0
    photos = {p["name"]: p for p in rep["photos"]}
    assert photos["a"]["mode"] == "reference" and photos["b"]["mode"] == "single_frame"
    assert photos["a"]["metrics"]["done_ok"] and photos["a"]["changed_fraction"] > 0
    assert photos["b"]["metrics"]["matched"] == 1 and photos["b"]["false"][0]["kind"] == "print"
    assert rep["total"]["metrics"]["matched"] == 3 and rep["total"]["false_by_kind"] == {"print": 1}
    assert rep["config"]["min_coverage"] == 0.5 and rep["overrides"] == {}
    assert (bench_photos / "out" / "a.bench.jpg").is_file() and (bench_photos / "out" / "b.bench.jpg").is_file()
    text = capsys.readouterr().out
    assert "TOTAL" in text and "FALSE" in text and "print (dark print)" in text


def test_tool_single_mode_and_set(bench_photos):
    code, rep = run_tool(bench_photos, "--mode", "single", "--no-images")
    assert code == 0 and {p["mode"] for p in rep["photos"]} == {"single_frame"}
    assert rep["total"]["metrics"]["false"] == 2
    code, rep = run_tool(bench_photos, "--set", "diameters_mm=28,40", "--no-images")
    assert rep["overrides"] == {"diameters_mm": [28.0, 40.0]} and rep["config"]["diameters_mm"] == [28.0, 40.0]
    assert rep["total"]["metrics"]["matched"] == 2  # the 32 mm base has no class now


def test_tool_reference_mode_only_pairs(bench_photos):
    code, rep = run_tool(bench_photos, "--mode", "reference", "--no-images")
    assert code == 0 and [p["name"] for p in rep["photos"]] == ["a"]


def test_tool_sweep(bench_photos, capsys):
    code, rep = run_tool(bench_photos, "--sweep", "min_contrast=200,12", "--sweep", "diameters_mm=28,40;25,28,32,40,50", "--no-images")
    assert code == 0
    rows = rep["sweep"]
    assert len(rows) == 4
    keys = [(-r["total"]["metrics"]["f1"], r["total"]["metrics"]["false"]) for r in rows]
    assert keys == sorted(keys)
    assert rep["overrides"] == rows[0]["overrides"] == {"min_contrast": 12.0, "diameters_mm": [25.0, 28.0, 32.0, 40.0, 50.0]}
    assert "sweep, best first" in capsys.readouterr().out


def test_tool_baseline(bench_photos, capsys):
    base = bench_photos / "baseline.json"
    code, _ = run_tool(bench_photos, "--write-baseline", str(base), "--no-images")
    assert code == 0
    saved = json.loads(base.read_text())
    assert "matches" not in saved["photos"][0] and "sweep" not in saved
    assert run_tool(bench_photos, "--baseline", str(base), "--no-images")[0] == 0
    assert "same as baseline" in capsys.readouterr().out
    assert run_tool(bench_photos, "--baseline", str(base), "--set", "min_contrast=200", "--no-images")[0] == 1
    assert "WORSE" in capsys.readouterr().out


def test_tool_bad_input(bench_photos, tmp_path):
    assert run_tool(bench_photos, "--set", "colour=3")[0] == 2
    assert run_tool(bench_photos, "--set", "min_coverage=lots")[0] == 2
    tool = load_tool("bench_bases")
    assert tool.main([str(tmp_path), "--config", str(bench_photos / "board.toml"), "--out", str(tmp_path / "o"), "-q"]) == 2
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((300, 300, 3), 160, np.uint8))
    bench.save_truth(tmp_path / "blank.bases.json", GroundTruth())
    assert tool.main([str(tmp_path), "--config", str(bench_photos / "board.toml"), "--out", str(tmp_path / "o"), "-q"]) == 2


# --- click tool logic --------------------------------------------------------

CLASSES = (25.0, 28.0, 32.0, 40.0, 50.0)


def rim_points(center, d, deg_from=0.0, deg_to=360.0, n=7):
    a = np.radians(np.linspace(deg_from, deg_to, n, endpoint=deg_to - deg_from < 360))
    return [(center[0] + 0.5 * d * np.cos(t), center[1] + 0.5 * d * np.sin(t)) for t in a]


def test_annotate_fit_known_circle():
    tool = load_tool("annotate_bases")
    ann = tool.Annotator(GroundTruth(), CLASSES)
    for p in rim_points((210.3, 145.8), 39.2):
        ann.add_point(p)
    c, d, k = ann.preview()
    assert np.linalg.norm(c - (210.3, 145.8)) < 0.1 and k == 40.0
    t = ann.commit()
    assert np.linalg.norm(np.subtract(t.center_mm, (210.3, 145.8))) < 0.1
    assert t.diameter_mm == 40.0 and t.source == "clicked" and t.tol_mm == 2.0 and t.rim == "clear"
    assert len(t.points_mm) == 7 and ann.points == [] and ann.dirty


def test_annotate_class_choice_and_hidden_rim():
    tool = load_tool("annotate_bases")
    ann = tool.Annotator(GroundTruth(), CLASSES)
    for p in rim_points((100.0, 100.0), 32.0, 200, 300, n=5):  # 100 deg of rim: hidden
        ann.add_point(p)
    t = ann.commit(choice=1)
    assert t.diameter_mm == 28.0 and t.rim == "hidden"
    assert ann.toggle_rim((100.0, 100.0)).rim == "clear"
    assert ann.set_size((100.0, 100.0), None).diameter_mm is None


def test_annotate_estimate_with_few_points():
    tool = load_tool("annotate_bases")
    ann = tool.Annotator(GroundTruth(), CLASSES)
    ann.add_point((50.0, 60.0))
    ann.add_point((52.0, 60.0))
    t = ann.commit()
    assert t.source == "estimate" and t.tol_mm == 8.0 and t.center_mm == (51.0, 60.0) and t.diameter_mm is None


def test_annotate_save_load_round_trip(tmp_path):
    tool = load_tool("annotate_bases")
    ann = tool.Annotator(GroundTruth(comment="keep me"), CLASSES, proposals=[det(300, 200, d=50, cov=0.7)])
    for p in rim_points((100.0, 100.0), 40.0):
        ann.add_point(p)
    ann.commit()
    ann.set_note((100.0, 100.0), "tyranid")
    assert ann.accept((301.0, 200.0)).source == "detector" and not ann.proposals
    assert ann.to_decoy((300.0, 200.0), "token").diameter_mm == 50.0
    path = tmp_path / "x.bases.json"
    ann.save(path)
    assert not ann.dirty
    gt = bench.load_truth(path)
    assert gt.comment == "keep me" and [t.note for t in gt.bases] == ["tyranid"]
    assert gt.not_bases[0].kind == "token"
    assert gt.to_dict() == ann.gt.to_dict()
    with pytest.raises(ValueError):
        ann.to_decoy((100.0, 100.0), "tree")


def test_annotate_undo_and_delete():
    tool = load_tool("annotate_bases")
    ann = tool.Annotator(GroundTruth([tr(100, 100)]), CLASSES)
    assert ann.delete((101.0, 100.0)) and ann.gt.bases == []
    assert ann.undo() and len(ann.gt.bases) == 1 and not ann.dirty
    assert not ann.delete((400.0, 400.0))
    assert ann.commit() is None  # no points


def test_annotate_proposals_on_entries_hidden():
    tool = load_tool("annotate_bases")
    gt = GroundTruth([tr(100, 100), tr(300, 100, rim="hidden", tol=8, source="estimate")], [Decoy((500.0, 100.0), 60.0, "teleporter")])
    props = [det(101, 100), det(305, 100), det(510, 100, d=40), det(700, 100)]
    ann = tool.Annotator(gt, CLASSES, props)
    assert ann.visible_proposals() == [3]
    assert ann.accept((305.0, 100.0)) is None  # hidden: not picked
    ann.show_all_proposals = True
    assert ann.visible_proposals() == [0, 1, 2, 3]


def test_annotate_proposal_to_decoy():
    tool = load_tool("annotate_bases")
    ann = tool.Annotator(GroundTruth(), CLASSES, [det(200, 200, d=50, cov=0.55)])
    dec = ann.to_decoy((202.0, 200.0), "token")
    assert dec.center_mm == (200.0, 200.0) and dec.diameter_mm == 50.0 and dec.kind == "token"
    assert ann.proposals == [] and ann.gt.not_bases == [dec] and ann.gt.bases == []
    assert ann.undo() and len(ann.proposals) == 1 and ann.gt.not_bases == []
    assert ann.to_decoy((600.0, 600.0), "token") is None
