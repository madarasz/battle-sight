import json

import cv2
import numpy as np
import pytest

from battlesight import pipeline
from battlesight.app import main
from battlesight.pipeline import Pipeline, StageResult, analyze, select_stages

from .conftest import make_config, render_board

CONFIG_TOML = """
[board]
name = "test"
width_mm = 606.0
height_mm = 703.0

[tags]
ids = [0, 1, 2, 3]
size_mm = 40.0
margin_mm = 5.0
"""


def test_corners_stage_finds_board(cfg):
    img, mm2px = render_board(cfg)
    r = analyze(img, cfg)
    assert r.ok
    c = r.stages["corners"]
    assert c.data["missing"] == []
    assert sorted(c.data["tags"]) == ["0", "1", "2", "3"]
    for k, mm in enumerate(cfg.board_corners_mm()):
        assert np.linalg.norm(np.array(c.data["board_corners_px"][k]) - mm2px(mm)) < 1.5
    json.dumps(r.to_dict())  # report must be JSON-safe


def test_corners_stage_reports_missing_tag(cfg):
    img, mm2px = render_board(cfg)
    x, y = mm2px(cfg.tag_corners_mm()[1].mean(axis=0)).astype(int)
    img[y - 60 : y + 60, x - 60 : x + 60] = 160  # cover tag 1
    r = analyze(img, cfg)
    assert not r.ok
    assert r.stages["corners"].data["missing"] == [1]
    assert r.stages["corners"].data["board_corners_px"][1] is None


def test_later_stages_skip_after_failure(cfg, monkeypatch):
    calls = []

    def later(ctx):
        calls.append(1)
        return StageResult(ok=True)

    monkeypatch.setitem(pipeline.STAGES, "later", pipeline.Stage("later", later))
    blank = np.full((400, 400, 3), 160, np.uint8)
    r = Pipeline(cfg).analyze(blank)
    assert not r.stages["corners"].ok
    assert r.stages["later"].skipped and calls == []
    assert select_stages("corners") == ["corners"]
    assert select_stages("rectify") == ["corners", "rectify"]
    assert select_stages()[:3] == ["corners", "rectify", "bases"]


def test_stage_exception_becomes_failure(cfg, monkeypatch):
    def boom(ctx):
        raise RuntimeError("bad frame")

    monkeypatch.setitem(pipeline.STAGES, "corners", pipeline.Stage("corners", boom))
    r = analyze(np.zeros((10, 10, 3), np.uint8), cfg)
    assert not r.ok and "bad frame" in r.stages["corners"].message


def test_analyze_until_stops_early():
    # The live app runs only up to `bases` on the b key.
    cfg = make_config()
    img, _ = render_board(cfg)
    pipe = Pipeline(cfg)
    assert list(pipe.analyze(img, until="corners").stages) == ["corners"]
    assert list(pipe.analyze(img, until="bases").stages) == ["corners", "rectify", "bases"]
    assert list(pipe.analyze(img).stages) == pipe.names


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        select_stages("nope")


@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "board.toml"
    p.write_text(CONFIG_TOML)
    return p


def test_cli_image_ok(tmp_path, config_file, capsys):
    img, _ = render_board(make_config())
    shots = tmp_path / "shots"
    shots.mkdir()
    cv2.imwrite(str(shots / "a.png"), img)
    out = tmp_path / "out"
    code = main(["image-pipeline", str(shots), "--config", str(config_file), "--out", str(out), "--json", "-"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["stages"] == ["corners", "rectify", "bases", "ident"]
    assert report["images"][0]["stages"]["corners"]["ok"]
    assert report["images"][0]["stages"]["rectify"]["ok"]
    assert (out / "a.overlay.jpg").is_file()
    assert (out / "a.rectified.jpg").is_file()
    assert json.loads((out / "a.json").read_text())["ok"]


def test_cli_image_stage_failure_exit_1(tmp_path, config_file):
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((400, 400, 3), 160, np.uint8))
    report = tmp_path / "r.json"
    assert main(["image-pipeline", str(p), "--config", str(config_file), "--json", str(report), "-q"]) == 1
    stages = json.loads(report.read_text())["images"][0]["stages"]
    assert stages["corners"]["data"]["missing"] == [0, 1, 2, 3]
    assert stages["rectify"]["skipped"]


def test_cli_image_bad_input_exit_2(tmp_path, config_file):
    bad = tmp_path / "not_an_image.jpg"
    bad.write_text("x")
    assert main(["image-pipeline", str(tmp_path / "missing.jpg"), "--config", str(config_file)]) == 2
    assert main(["image-pipeline", str(bad), "--config", str(config_file)]) == 2
