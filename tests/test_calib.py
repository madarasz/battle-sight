import numpy as np

from battlesight.calib import DriftMonitor, DriftStatus, TagDetector, board_corners_px

from .conftest import mild_perspective, render_board


def test_tag_corners_mm_geometry(cfg):
    tc = cfg.tag_corners_mm()
    # Tag 0 sits outside the top-left corner, one margin away, off the board.
    assert np.allclose(tc[0][2], [-5, -5])
    assert np.allclose(tc[0][0], [-45, -45])
    # Tag 2 sits outside the bottom-right corner.
    assert np.allclose(tc[2][0], [606 + 5, 703 + 5])
    assert np.allclose(tc[2][2], [606 + 45, 703 + 45])
    # No tag covers the play area.
    for quad in tc.values():
        inside = (quad[:, 0] > 0) & (quad[:, 0] < 606) & (quad[:, 1] > 0) & (quad[:, 1] < 703)
        assert not inside.any()


def test_detects_four_tags_flat(cfg):
    img, mm2px = render_board(cfg)
    obs = TagDetector(cfg).detect(img)
    assert sorted(obs) == [0, 1, 2, 3]
    corners = board_corners_px(cfg, obs)
    for k, mm in enumerate(cfg.board_corners_mm()):
        assert np.linalg.norm(corners[k] - mm2px(mm)) < 1.5, k


def test_detects_four_tags_perspective_and_blur(cfg):
    import cv2

    flat, _ = render_board(cfg)
    H = mild_perspective(flat.shape[1], flat.shape[0])
    img, mm2px = render_board(cfg, H=H)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    obs = TagDetector(cfg).detect(img)
    assert sorted(obs) == [0, 1, 2, 3]
    corners = board_corners_px(cfg, obs)
    for k, mm in enumerate(cfg.board_corners_mm()):
        assert np.linalg.norm(corners[k] - mm2px(mm)) < 2.5, k


def test_detects_at_low_density(cfg):
    # 1.5 px/mm is the plan's floor. Tag = 60 px, 7.5 px per module.
    img, _ = render_board(cfg, px_per_mm=1.5)
    obs = TagDetector(cfg).detect(img)
    assert sorted(obs) == [0, 1, 2, 3]


def test_ignores_foreign_tag_ids(cfg):
    from .conftest import make_config

    other = make_config(tag_ids=(10, 11, 12, 13))
    img, _ = render_board(other)
    assert TagDetector(cfg).detect(img) == {}


def test_drift_auto_lock_then_ok(cfg):
    img, _ = render_board(cfg)
    det, mon = TagDetector(cfg), DriftMonitor(cfg)
    obs = det.detect(img)
    for i in range(cfg.settle_frames - 1):
        r = mon.update(obs)
        assert r.status is DriftStatus.NO_REFERENCE
        assert 0 < r.settle_progress < 1
    r = mon.update(obs)
    assert r.status is DriftStatus.OK
    assert mon.reference is not None
    assert max(r.shift_mm.values()) < 0.05


def test_drift_detects_moved_board(cfg):
    px_per_mm = 2.0
    img, _ = render_board(cfg, px_per_mm=px_per_mm)
    det, mon = TagDetector(cfg), DriftMonitor(cfg)
    mon.set_reference(det.detect(img))
    moved, _ = render_board(cfg, px_per_mm=px_per_mm, shift_px=(8, 0))  # 4 mm
    r = mon.update(det.detect(moved))
    assert r.status is DriftStatus.MOVED
    assert sorted(r.moved) == [0, 1, 2, 3]
    for mm in r.moved.values():
        assert abs(mm - 4.0) < 0.5
    # Under tolerance: 1 mm shift is still OK.
    small, _ = render_board(cfg, px_per_mm=px_per_mm, shift_px=(2, 0))
    r = mon.update(det.detect(small))
    assert r.status is DriftStatus.OK


def test_drift_detects_single_moved_tag(cfg):
    img, _ = render_board(cfg)
    det, mon = TagDetector(cfg), DriftMonitor(cfg)
    obs = det.detect(img)
    mon.set_reference(obs)
    bumped = {k: v for k, v in obs.items()}
    t = obs[2]
    bumped[2] = type(t)(t.tag_id, t.corners + np.array([0, 6.0]), t.center + np.array([0, 6.0]), t.side_px)  # 3 mm
    r = mon.update(bumped)
    assert r.status is DriftStatus.MOVED
    assert list(r.moved) == [2]


def test_drift_detects_lost_tag(cfg):
    img, _ = render_board(cfg)
    det, mon = TagDetector(cfg), DriftMonitor(cfg)
    obs = det.detect(img)
    mon.set_reference(obs)
    partial = {k: v for k, v in obs.items() if k != 1}
    for _ in range(cfg.lost_after_frames - 1):
        assert mon.update(partial).status is DriftStatus.OK  # brief dropout tolerated
    r = mon.update(partial)
    assert r.status is DriftStatus.LOST
    assert r.lost == [1]
    # Tag returns: status recovers at once.
    assert mon.update(obs).status is DriftStatus.OK


def test_auto_lock_restarts_when_board_moves_during_settle(cfg):
    img, _ = render_board(cfg)
    moved, _ = render_board(cfg, shift_px=(10, 0))
    det, mon = TagDetector(cfg), DriftMonitor(cfg)
    a, b = det.detect(img), det.detect(moved)
    for _ in range(cfg.settle_frames - 1):
        mon.update(a)
    r = mon.update(b)
    assert r.status is DriftStatus.NO_REFERENCE
    assert r.settle_progress == 1 / cfg.settle_frames
