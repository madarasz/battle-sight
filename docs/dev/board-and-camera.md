# Board, camera, corners, rectify (priorities 1 and 2)

Modules: `config.py`, `camera.py`, `scrcpy.py`, `calib.py`, `rectify.py`. Tests: `tests/test_config.py`, `test_camera.py`, `test_scrcpy.py`, `test_calib.py`, `test_rectify.py`. User setup: `docs/setup.md`.

## config.py

- `BoardConfig` (frozen dataclass) loads `data/board.toml`. All lengths are in mm.
- Board coordinates: origin at the top-left board corner, x right, y down. Corner index 0..3 = TL, TR, BR, BL, and `tags.ids[k]` is the tag at corner k.
- All camera values live in `data/camera.toml`, none in `data/board.toml` (a `[camera]` table there is an error).
- Map grid: `[grid]` in `data/board.toml` and `BoardConfig.grid_lines_mm()`. Tomb World uses its 7 x 6 printed sections inside a 12 mm margin, Volkus a 1 in pitch. Never hard-code an inch grid. `docs/cli.md` "Map grid" documents it.
- A second board (Volkus, 559 x 760 mm) is planned.

## Corner cards

- Made by `tools/make_tags.py`: 90 mm square, tag in the outer quadrant, black L in the inner quadrant at the board corner. The 5 mm white margin is the tag quiet zone.
- Each card lies under a board corner, tag outside the board, diagonal to the corner.
- The app measures from the cards, not the board: a board that slides over a still card gives no warning.

## calib.py

- `TagDetector` (OpenCV `cv2.aruco` with `DICT_APRILTAG_36h11`, no contrib build) returns `{tag_id: TagObs}`.
- `board_corners_px` takes tag corner k+2 (the one nearest the board) and pushes it toward the board by the white card margin to get board corner k.
- `DriftMonitor` auto-locks a reference after `settle_frames` still frames, then reports OK / MOVED / CARD_MOVED / LOST.
  - After a move it waits for `settle_frames` still frames again: when all four cards moved as one piece (`card_offset_mm`, leave-one-tag-out homography, within `drift.tolerance_mm`) it saves the new position (`relocked`), else CARD_MOVED names the card. No manual re-lock key.
  - A LOST report is `held` when at most `HOLD_MAX_LOST` (2) tags are out of view and the tags in view did not move: `live.BoardTracker` (used by `LiveSession.step` and the live `enroll` window) then keeps the reference for the board fit, so the adjusted perspective stays while a player covers a card.
  - It converts pixel shift to mm with each tag's apparent size.
- `Lens` (K, distortion, `undistort_px` / `distort_px`), `lens_for_frame(camera, size)` (the `[lens]` calibration of `CameraSettings` when its aspect ratio fits the frame, else a nominal pinhole from `CameraSettings.focal_35mm`), the ChArUco target and `calibrate_lens`.
- Without a `[lens]` table in `data/camera.toml` the app uses a nominal lens (`focal_35mm`, default 24 mm, no distortion). Phones correct most distortion themselves.

## rectify.py

- `fit_view` fits a `BoardView` (homography board-surface mm -> undistorted px) to the 16 tag corners, with the board thickness correction from the planar camera pose.
- `warp_board` makes the top-down image. Scale is `rectify.px_per_mm` (default 2 px per mm).
- `measure_edges` and `measure_grid_lines` check it (board edges, printed grid lines). With `grid.printed = true` the stage finds the printed lines and reports their offsets, a check that does not use the tags.
- The rectify stage puts `view` and `rectified` in `scratch` for later stages.
- `board.width_mm` is the edge from tag 0 to tag 1 as the cards lie (703 x 606 mm in the test photos). When width and height are swapped, the stage fails and says so.
- Board thickness: the cards lie on the table and the play surface is `board.thickness_mm` higher (Tomb World 1.0 mm, caliper). Without the correction, 1 mm gives up to about 0.5 mm error at the board edge, 3 mm about 1.5 mm.

## camera.py and scrcpy.py

- `Camera` wraps `cv2.VideoCapture` and reconnects after repeated read failures. `read()` returns `None` while the camera is down. Callers must handle that.
- `ImageSource` has the same calls for one image file (`--image-source`).
- A device index uses AVFoundation on macOS (Camo, built-in camera).
- A URL (`is_stream`) uses FFmpeg with open and read timeouts, and `LatestFrameReader` reads it on a thread and keeps only the newest frame.
- Source `scrcpy` / `scrcpy:SERIAL` (`scrcpy.is_scrcpy`): `scrcpy.ScrcpyServer` pushes and starts `scrcpy-server` (camera, `raw_stream=true`) through adb with a forwarded TCP port, `Camera` reads the raw H.264 like a stream (open retried until a frame arrives); width, height, fps are the camera size and rate asked from the phone; `Camera.error` holds the reason of a failed open. `docs/cli.md` "scrcpy source".
- `CameraSettings` loads and saves `data/camera.toml` (git-ignored, machine-local): `[camera]` from `tools/list_cameras.py`, `[lens]` from `calibrate`. Each writer keeps the other table.
- `docs/setup.md` has the camera troubleshooting (Camo extension blocked by MDM, stream network checks).
