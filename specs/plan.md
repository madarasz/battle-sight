# Battle Sight: research and plan

Status: planning only. No code yet. Date: 2026-11-02.

## 1. Decisions made

| Topic | Decision |
|---|---|
| Stack | Python core. OpenCV for vision. PyTorch (Apple MPS) for ML. |
| Machine | This Mac, M4 Pro, 48 GB. |
| Camera | Xiaomi 13T phone, main lens (24 mm equiv.), 4K, fed to the Mac through Camo. Arm or tripod above board center. App must accept any camera source with enough pixels per mm. |
| Display | Monitor or laptop next to table. One window: video with overlay, side panel. |
| Board | MVP: Kill Team "Tomb World" board, 606 x 703 mm. Later: Volkus open board, 559 x 760 mm. |
| Board frame | AprilTags sit on the real board corners. Board coordinate origin is one corner. Grid offset is configurable. |
| Geometry | All measurement is 2D in the board plane. No vantages, no heights. |
| Walls | Not tracked, not measured. They block placement physically. No map overlay of wall layouts. |
| Doors | Hatchways and breach points are placed by clicking on the rectified image once per map, snapped to the tile grid, saved as a named map file. App draws 1 in and 2 in bands from their edges. |
| Bases | Round bases 25, 28, 32, 40, 50 mm. Later: 35 x 60 mm oval. |
| Tokens | Round tokens 20 and 40 mm. |
| Measure mode | Snapshot first. Key press triggers detect and measure on a clean frame. Live tracking later. |
| Interaction | Click an operative or token on screen. Hotkeys for ring radius, grid, snapshot. |
| Operative ID | All models visually distinct. Vision ID with few-shot gallery. No markers on models. |

## 2. Priorities and phases

Order from the owner. Each phase has a done test.

| # | Priority | Approach | Done when |
|---|---|---|---|
| 1 | Find board corners | AprilTag 36h11 at each corner. Detect every frame. Warn when a tag is lost or moved. | Four corners found in every clean frame. Drift of a bumped board detected within a second. |
| 2 | Undistort and rectify | One-time lens calibration. Per-frame homography from the 4 tags. Warp to a top-down image at a fixed scale (mm per pixel). Optional 1 in grid overlay. | Board edges straight, corners at 90 deg. Grid lines land on the printed board grid across the whole board. Known 1 in distance measures 25.4 mm plus or minus 1 mm anywhere. |
| 3 | Find bases and positions | Detection is separate from identity. Background subtraction against an empty-board reference, then edge points, then circle fit with the known diameter set. Later: a YOLO segmentation model trained on base rims. | All bases found on a Tomb World board with walls. Center error under 2 mm. Correct diameter class. |
| 4 | Identify operatives | Crop around each found base. DINOv2 embedding. Nearest neighbor in an enrolled gallery (10 to 20 photos per operative from this camera). User confirms or fixes with one click. Fixes feed the gallery. | 9 of 10 bases labeled right on first try. |
| 5 | Rings and distances | Select operative. Hotkey picks N in 1 to 9. Draw circle with radius = base radius + N in. Select a second operative: draw and print rim-to-rim distance. | Ring and distance match a tape measure within 2 mm. |
| 6 | Tokens | Detect 20 and 40 mm round tokens with the same circle fit, different diameter set. Rings 1 to 9 in. Mark every operative whose base rim lies within the selected distance. | Tokens found. In-range list matches tape measure. |
| 7 | Hatchways and breach points | Setup mode: user clicks two endpoints per door on the rectified image. Click snaps to the printed tile grid. Thickness from config. Saved as a named map file, reused next game. Game mode: click toggles open or closed. Draw 1 in or 2 in band from the rectangle edge (rounded-rectangle offset). Vision detection rejected: door types look alike from above, models hide walls, state needs user input anyway. Optional later: template match to auto-suggest positions on map load. | Map saved and reloaded. Bands drawn. Operative in or out of band matches tape measure. |

Answer to the question in priority 3: yes, bases come before identity. Detection is pure geometry and needs no training. Identity runs on the crop around each found base. Keeping them separate means a new team only needs gallery photos, never a retrained detector.

## 3. Feasibility verdict

Feasible. Risk per priority:

| # | Risk | Note |
|---|---|---|
| 1, 2 | Low | Standard OpenCV. |
| 3 | Medium | Known diameters make circle fit robust under partial occlusion by the model body. Dark board and wall shadows lower contrast. |
| 4 | Medium | Works when models look different from above. Enrollment photos must come from the real camera. |
| 5 | Low | Pure math after 3. |
| 6 | Low to medium | Metallic or transparent tokens reflect light. Printed ring on the token helps. |
| 7 | Low to medium | Rectangles of known size are easy when visible. Click fallback covers the rest. |

## 4. Camera geometry

Board size is a config value. Tomb World: 606 x 703 mm. Volkus: 559 x 760 mm.

Camera: Xiaomi 13T. Main lens 50 MP at 24 mm equivalent, f/1.9, OIS. 4K at 30 fps. Field of view in 16:9 video: about 74 deg x 46 deg.

Sample photos from this phone show the board filling about 95% of the frame width. At 4K that is about 4 px per mm. Target is 1.5 px per mm, so density is not a problem. 16:9 video crops the short side by 25% compared with the 4:3 photos, so the mount sits a little higher than for the photos.

Parallax at 1.0 m, at a board corner (about 465 mm from camera nadir):

- Base top, 3 mm high: shifts about 1.4 mm outward. Correctable, since camera pose is known.
- Model body, 40 mm high: shifts about 19 mm outward. Body covers the outer side of its base. Inner side of the base rim stays visible.

Conclusion: fit a circle of known radius to the visible part of the rim. Measure to the rim, never to the model.

Ways to get the phone feed into OpenCV on macOS:

1. **Camo** (Reincubate). USB or Wi-Fi. Shows up as a normal camera, so `cv2.VideoCapture` opens it. Pro tier for 4K, lens choice, manual focus and exposure. Pick this for the prototype.
2. **IP Webcam** (Android app). MJPEG or RTSP URL into OpenCV. Free. Higher latency, fine for snapshot mode.
3. **scrcpy** camera mode over USB. Free, 4K H.264, but no virtual camera on macOS. Fallback only.

Camera rules: rigid mount, manual exposure and focus lock, diffuse light from two sides, phone on charger with screen off. OIS jitter is corrected by per-frame tag detection.

## 5. Operative ID options (kept for reference)

1. Few-shot embeddings (DINOv2 or CLIP) and nearest neighbor. No training. MVP choice.
2. Trained classifier on crops from the real camera. Better accuracy. Needs an enrollment tool that auto-captures frames while the user rotates the model.
3. Track first. Identify once at deployment, then follow by position. Later phase. Also fixes identical models.
4. Small color mark on the base rim. Not needed, all models distinct.
5. Vision language model as fallback re-identifier. Slow and costly. Not planned.

## 6. Software stack

- Python 3.12 (not 3.14: ML wheels lag behind). Manage with `uv`.
- `opencv-python` 4.10+. AprilTag and ArUco live in the `objdetect` module. No contrib build needed.
- `numpy`, `scipy` (circle fit).
- `torch` with MPS, `torchvision`. DINOv2 via `torch.hub` or `transformers`.
- `ultralytics` (YOLO) later, for base segmentation and live tracking.
- UI: `PySide6` (Qt). Video widget with overlay painter, side panel, mouse clicks. An OpenCV `imshow` window is enough for priorities 1 and 2.
- Config and data: JSON or TOML. Calibration, gallery embeddings, token and door sizes.

Later option: FastAPI plus browser front end over WebSocket for a tablet at the table. Core must not depend on Qt.

## 7. Proposed repository layout

```
battle-sight/
  specs/              # this plan, about-project, images
  src/battlesight/
    camera.py         # capture, reconnect
    calib.py          # intrinsics, tag detection, homography, drift check
    rectify.py        # warp to board mm space, grid
    circles.py        # background subtraction, known-radius circle fit (bases and tokens)
    ident.py          # embeddings, gallery, kNN
    doors.py          # hatchway and breach point rectangles, open/closed state
    measure.py        # rim-to-rim distances, rings, bands, in-range tests
    overlay.py        # drawing in board coordinates
    ui/               # PySide6 window
  data/
    calib/            # camera intrinsics, board homography
    gallery/          # enrollment images and embeddings
    maps/             # clicked door positions per Tomb World layout
    sizes.toml        # base, token, door dimensions
  tools/
    calibrate_camera.py
    make_tags.py      # print sheet for AprilTags
    enroll_operative.py
```

## 8. Risks

- **Dark board and wall shadows** reduce contrast at base rims. Mitigation: diffuse light, background subtraction against the reference frame, known-radius fit.
- **Metallic or transparent tokens** reflect light. Mitigation: detect by printed ring, or user places token by click.
- **Camera bumped mid-game.** Mitigation: tag drift check every frame, auto re-homography, warning in the UI.
- **Few-shot ID confuses similar paint schemes.** Mitigation: confirmation step, corrections go into gallery, upgrade to trained classifier.
- **Phone stream drops or heats up.** Mitigation: USB, charger, screen off, app reconnects on its own.

## 9. Open implementation details

Resolved inside the phase that needs them.

- Priority 1: tag size and print sheet. Install Camo, confirm 4K reaches OpenCV.
- Priority 3: whether the empty-board reference is enough or a trained base segmenter is needed. Decide after first tests on the real board.
- Priority 7: door thickness in mm for config. Grid snap pitch (1 in, or half inch if walls sit between lines).
- Later: roster format, wounds and statuses, live tracking, oval bases, Volkus board.
