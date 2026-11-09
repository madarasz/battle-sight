"""Identification benchmark: scores, baseline rules, gallery state, the tool."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from battlesight import ident as idt
from battlesight import ident_bench as ib
from battlesight.config import DEFAULT_CONFIG, BoardConfig

from .test_ident import enroll_from, make_gallery, paint_board

SIZES = {"T/A": 40.0, "T/B": 40.0, "T/C": 32.0, "T/D": None}


def truth(*items, tol=10.0):
    return idt.IdentTruth(["T"], [(np.array(c, float), label) for c, label in items], tol)


def x(label, sure=True, score=0.8):
    return idt.Ident(label, score, 0.1, [], sure=sure)


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "bench_ident.py"
    spec = importlib.util.spec_from_file_location("bench_ident", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_ident"] = mod
    spec.loader.exec_module(mod)
    return mod


# --- score ---------------------------------------------------------------------


def test_score_splits_the_losses_by_cause():
    t = truth(((100, 100), "T/A"), ((200, 100), "T/B"), ((300, 100), "T/C"), ((400, 100), "T/A"), ((500, 100), "T/B"))
    found = [((100, 100), 40.0), ((200, 100), 40.0), ((300, 100), 28.0), ((400, 100), 40.0)]
    s = ib.score("p", t, found, [x("T/A"), x("T/A", sure=False), x(None), x(None)], SIZES)
    assert s.right == [0.8]
    assert s.wrong == [("T/B", "T/A", False)]
    assert s.wrong_size == [("T/C", 28.0, 32.0)]  # a detection loss, not a naming loss
    assert s.unknown == ["T/A"]
    assert s.missed == ["T/B"]


def test_score_size_unknown_is_not_a_size_loss():
    s = ib.score("p", truth(((100, 100), "T/D")), [((100, 100), 25.0)], [x(None)], SIZES)
    assert s.unknown == ["T/D"] and s.wrong_size == []


def test_score_agrees_with_the_done_test():
    t = truth(((100, 100), "T/A"), ((200, 100), "T/B"), ((900, 900), "T/C"))
    found = [((101, 100), 40.0), ((200, 101), 40.0)]
    ids = [x("T/A"), x("T/A")]
    s = ib.score("p", t, found, ids, SIZES)
    d = idt.score_truth(t, np.array([c for c, _ in found], float), ids)
    assert (len(s.right), round(s.accuracy, 3), s.done_ok) == (d["right"], d["accuracy"], d["done_ok"])


def test_score_no_found_base():
    s = ib.score("p", truth(((100, 100), "T/A")), [], [], SIZES)
    assert s.missed == ["T/A"] and s.accuracy == 0.0


def test_metrics_counts_and_ratios():
    a = ib.score("a", truth(((100, 100), "T/A"), ((200, 100), "T/B")), [((100, 100), 40.0), ((200, 100), 40.0)], [x("T/A"), x("T/A")], SIZES)
    b = ib.score("b", truth(((100, 100), "T/A"), ((200, 100), "T/C")), [((100, 100), 40.0)], [x("T/A", score=0.6)], SIZES)
    m = ib.metrics([a, b])
    assert (m["truth"], m["right"], m["wrong"], m["sure_wrong"], m["missed"]) == (4, 2, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["naming_accuracy"] == round(2 / 3, 4)
    assert m["score_right_mean"] == 0.7 and (m["done"], m["photos"], m["done_ok"]) == (0, 2, False)


def test_metrics_of_no_truth():
    m = ib.metrics([ib.PhotoScore("p", "", 0)])
    assert m["accuracy"] == 1.0 and m["naming_accuracy"] == 1.0 and m["score_right_mean"] is None and m["done_ok"]


def test_report_baseline_form_has_no_lists():
    s = ib.score("p", truth(((100, 100), "T/A")), [], [], SIZES)
    assert ib.report([s], {}, {"fingerprint": "x"})["photos"][0]["missed"] == ["T/A"]
    assert "missed" not in ib.report([s], {}, {"fingerprint": "x"}, detail=False)["photos"][0]


# --- baseline rules ------------------------------------------------------------


def base_report():
    t = truth(((100, 100), "T/A"), ((200, 100), "T/B"), ((300, 100), "T/C"))
    found = [((100, 100), 40.0), ((200, 100), 40.0), ((300, 100), 32.0)]
    s = ib.score("p", t, found, [x("T/A"), x("T/B"), x("T/A", sure=False)], SIZES)
    return ib.report([s], {}, {"fingerprint": "abc", "files": 3})


def changed(**over):
    r = copy.deepcopy(base_report())
    r["photos"][0]["metrics"].update(over)
    return r


def test_compare_equal_is_ok():
    c = ib.compare(base_report(), base_report())
    assert c.ok and not c.better and not c.new


@pytest.mark.parametrize(
    "over, text",
    [({"right": 1}, "right 2 -> 1"), ({"right": 1, "wrong": 2}, "naming_accuracy"), ({"sure_wrong": 1}, "sure_wrong 0 -> 1")],
)
def test_compare_total_rules_worse(over, text):
    c = ib.compare(changed(**over), base_report())
    assert not c.ok and any(text in w for w in c.worse)


def test_compare_better_is_not_a_failure():
    c = ib.compare(changed(right=3, wrong=0), base_report())
    assert c.ok and "right 2 -> 3" in c.better and any("naming_accuracy" in b for b in c.better)


def test_compare_new_photo_cannot_fail():
    run = base_report()
    bad = {"name": "new", "metrics": dict(run["photos"][0]["metrics"], right=0, wrong=3, sure_wrong=3)}
    run["photos"].append(bad)
    run["total"]["metrics"]["sure_wrong"] = 3
    c = ib.compare(run, base_report())
    assert c.ok and c.new == ["new"] and not c.better


def test_compare_gain_on_one_photo_does_not_hide_a_loss_on_another():
    base = base_report()
    base["photos"].append({"name": "q", "metrics": dict(base["photos"][0]["metrics"], right=1)})
    run = copy.deepcopy(base)
    run["photos"][0]["metrics"]["right"] = 1
    run["photos"][1]["metrics"]["right"] = 2
    c = ib.compare(run, base)
    assert not c.ok and "p: right 2 -> 1" in c.worse and "q: right 1 -> 2" in c.better


def test_compare_photo_done_and_missing_and_new():
    base = base_report()
    base["photos"][0]["metrics"]["done_ok"] = True
    run = copy.deepcopy(base)
    run["photos"][0]["metrics"]["done_ok"] = False
    assert "p: done_ok true -> false" in ib.compare(run, base).worse
    run["photos"][0]["name"] = "other"
    c = ib.compare(run, base)
    assert "p: in the baseline, missing in the run" in c.worse and c.new == ["other"]


# --- gallery state -------------------------------------------------------------


def test_gallery_state_changes_with_content_only(tmp_path):
    op = tmp_path / "T" / "A"
    op.mkdir(parents=True)
    (op / "1.png").write_bytes(b"one")
    (tmp_path / "T" / "team.toml").write_text("x")
    first = ib.gallery_state(tmp_path, ["T"])
    assert first["files"] == 2 and first["teams"] == ["T"]

    (tmp_path / "T" / "embeddings.npz").write_bytes(b"cache")
    (op / ".DS_Store").write_bytes(b"junk")
    assert ib.gallery_state(tmp_path, ["T"]) == first

    (op / "1.png").write_bytes(b"two")
    assert ib.gallery_state(tmp_path, ["T"])["fingerprint"] != first["fingerprint"]
    (op / "1.png").write_bytes(b"one")
    assert ib.gallery_state(tmp_path, ["T"])["fingerprint"] == first["fingerprint"]
    (op / "bg").mkdir()
    (op / "bg" / "1.png").write_bytes(b"bg")
    assert ib.gallery_state(tmp_path, ["T"])["files"] == 3


def test_same_gallery():
    a = {"gallery": {"fingerprint": "a"}}
    assert ib.same_gallery(a, a) and not ib.same_gallery(a, {"gallery": {"fingerprint": "b"}})


# --- tool ----------------------------------------------------------------------

GAME = {"Nob": ((180.0, 170.0), 30.0), "Grot": ((420.0, 210.0), 120.0), "Burna": ((300.0, 430.0), 200.0)}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Colour-histogram gallery and one synthetic game photo with its names."""
    d = tmp_path_factory.mktemp("ident_bench")
    text = DEFAULT_CONFIG.read_text().replace('model = "facebook/dinov2-small"', 'model = "color-hist"').replace("mask = true", "mask = false")
    (d / "board.toml").write_text(text)
    cfg = BoardConfig.load(d / "board.toml")
    gallery = make_gallery(d, cfg)
    for shift in (0.0, 60.0):
        placed = {n: ((c[0] + shift, c[1] + shift), turn + 40.0 * shift) for n, (c, turn) in GAME.items()}
        enroll_from(gallery, cfg, paint_board(cfg, placed), {n: c for n, (c, _) in placed.items()})
    photos = d / "photos"
    photos.mkdir()
    moved = {n: ((c[0] + 25.0, c[1] + 15.0), turn + 15.0) for n, (c, turn) in GAME.items()}
    cv2.imwrite(str(photos / "game.png"), paint_board(cfg, moved))
    named = [(np.array(c), idt.label_of("orks", n)) for n, (c, _) in moved.items()]
    idt.IdentTruth(["orks"], named).save(photos / "game.ident.json")
    return d


def run_tool(d: Path, *extra: str) -> int:
    return load_tool().main([str(d / "photos"), "--config", str(d / "board.toml"), "--gallery-root", str(d / "gallery"), "-q", *extra])


def test_tool_report_and_table(setup, capsys):
    out = setup / "report.json"
    assert run_tool(setup, "--json", str(out)) == 0
    rep = json.loads(out.read_text())
    m = rep["total"]["metrics"]
    assert (m["truth"], m["right"]) == (3, 3) and rep["photos"][0]["name"] == "game"
    assert rep["gallery"]["teams"] == ["orks"] and rep["config"]["model"] == "color-hist"
    assert "TOTAL" in capsys.readouterr().out


def test_tool_baseline_round_trip(setup, tmp_path, capsys):
    base = tmp_path / "ident_baseline.json"
    assert run_tool(setup, "--write-baseline", str(base)) == 0
    raw = json.loads(base.read_text())
    assert "wrong" not in raw["photos"][0] and raw["gallery"]["fingerprint"]
    assert run_tool(setup, "--baseline", str(base)) == 0

    raw["total"]["metrics"]["right"] += 1
    raw["photos"][0]["metrics"]["right"] += 1
    base.write_text(json.dumps(raw))
    assert run_tool(setup, "--baseline", str(base)) == 1
    assert "WORSE: right 4 -> 3" in capsys.readouterr().out


def test_tool_baseline_of_another_gallery_is_refused(setup, tmp_path, capsys):
    base = tmp_path / "ident_baseline.json"
    run_tool(setup, "--write-baseline", str(base))
    raw = json.loads(base.read_text())
    raw["gallery"]["fingerprint"] = "other"
    base.write_text(json.dumps(raw))
    assert run_tool(setup, "--baseline", str(base)) == 2
    assert "another gallery" in capsys.readouterr().err


def test_tool_bad_input(setup, tmp_path, capsys):
    empty = tmp_path / "none"
    empty.mkdir()
    tool = load_tool()
    assert tool.main([str(empty), "--config", str(setup / "board.toml"), "-q"]) == 2
    assert "no photo" in capsys.readouterr().err
    code = tool.main([str(setup / "photos"), "--config", str(setup / "board.toml"), "--gallery-root", str(empty), "-q"])
    assert code == 2 and "no gallery" in capsys.readouterr().err
    assert run_tool(setup, "--baseline", str(tmp_path / "missing.json")) == 2
