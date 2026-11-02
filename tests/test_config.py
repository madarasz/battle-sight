from pathlib import Path

from battlesight.config import DEFAULT_CONFIG, BoardConfig


def test_default_config_loads():
    cfg = BoardConfig.load(DEFAULT_CONFIG)
    assert cfg.tag_ids == (0, 1, 2, 3)
    assert cfg.width_mm == 606.0
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
