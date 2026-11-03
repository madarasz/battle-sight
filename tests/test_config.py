from pathlib import Path

import numpy as np
import pytest

from battlesight.config import DEFAULT_CONFIG, BoardConfig


def test_default_config_loads():
    cfg = BoardConfig.load(DEFAULT_CONFIG)
    assert cfg.tag_ids == (0, 1, 2, 3)
    assert (cfg.width_mm, cfg.height_mm) == (703.0, 606.0)
    assert cfg.grid_sections == (7, 6) and cfg.grid_margin_mm == 12.0 and cfg.grid_printed
    assert cfg.corner_index(2) == 2
    assert cfg.corner_index(99) is None


def test_bad_ids_rejected(tmp_path: Path):
    p = tmp_path / "b.toml"
    p.write_text('[board]\nwidth_mm=1\nheight_mm=1\n[tags]\nids=[0,0,1,2]\nsize_mm=40\n')
    try:
        BoardConfig.load(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_grid_lines_sections():
    from .conftest import tomb_world_grid

    xs, ys = tomb_world_grid().grid_lines_mm()
    assert len(xs) == 8 and len(ys) == 7
    assert np.allclose(xs, 12 + 97 * np.arange(8))
    assert np.allclose(ys, 12 + 97 * np.arange(7))


def test_grid_lines_pitch():
    from .conftest import make_config

    xs, ys = make_config(width_mm=559.0, height_mm=760.0).grid_lines_mm()  # Volkus, 1 in from the edge
    assert np.allclose(np.diff(xs), 25.4) and np.allclose(np.diff(ys), 25.4)
    assert xs[0] == 0 and xs[-1] <= 559 and 559 - xs[-1] < 25.4
    assert len(ys) == 30


def test_grid_and_rectify_from_toml(tmp_path: Path):
    p = tmp_path / "b.toml"
    p.write_text(
        '[board]\nwidth_mm=559\nheight_mm=760\nthickness_mm=2.5\n[tags]\nids=[0,1,2,3]\nsize_mm=40\n'
        '[grid]\npitch_mm=25.4\n[rectify]\npx_per_mm=3\n'
    )
    cfg = BoardConfig.load(p)
    assert cfg.thickness_mm == 2.5 and cfg.grid_sections is None and cfg.grid_pitch_mm == 25.4
    assert cfg.rectify_px_per_mm == 3.0


def test_camera_table_in_board_config_rejected(tmp_path: Path):
    p = tmp_path / "b.toml"
    p.write_text('[board]\nwidth_mm=1\nheight_mm=1\n[tags]\nids=[0,1,2,3]\nsize_mm=40\n[camera]\nfocal_35mm=26\n')
    with pytest.raises(ValueError, match="camera.toml"):
        BoardConfig.load(p)


def test_bad_sections_rejected(tmp_path: Path):
    p = tmp_path / "b.toml"
    p.write_text('[board]\nwidth_mm=1\nheight_mm=1\n[tags]\nids=[0,1,2,3]\nsize_mm=40\n[grid]\nsections=[7]\n')
    with pytest.raises(ValueError):
        BoardConfig.load(p)
