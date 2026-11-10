# Battle Sight: research and plan

Status: priorities 1 (board corners), 2 (undistort and rectify), 3 (bases), 4 (operative ID), 5 (rings and distances) and 6 (terrain layouts) are done. Priority 7 (tokens) is next. UI foundation: game core (`live.py`), Qt window (`ui/shell.py`) and the layouts `inspector` and `panel` are done. Some checks with real hardware are open (`specs/ui.md`, section 13). Date: 2026-11-10.

## 1. Decisions made

| Topic | Decision |
|---|---|
| Stack | Python core. OpenCV for vision. PyTorch (Apple MPS) for ML. |
| Machine | This Mac, M4 Pro, 48 GB. |
| Camera | Xiaomi 13T phone, main lens (24 mm equiv.), 4K, fed to the Mac through Camo. Arm or tripod above board center. App must accept any camera source with enough pixels per mm. |
| Display | Monitor or laptop next to table. One window: video with overlay, side panel. |
| UI | Modular: many layouts on one game core, and a setting changes the layout while the app runs. Each layout maps to the same actions and state. `specs/ui.md` has the design. Each priority includes its UI part (section 2). |
| Board | MVP: Kill Team "Tomb World" board, 606 x 703 mm. Later: Volkus open board, 559 x 760 mm. |
| Grid | Map dependent, from config. Tomb World: the 7 x 6 printed sections (97 mm) inside a 12 mm margin; an inch grid does not agree with its print. Volkus: 1 in grid. |
| Board frame | One square AprilTag card under each board corner. The tag sits in the card's outer quadrant, diagonal to the corner. The board corner rests on the inner quadrant, both edges against a printed L that stays visible just outside the board, so the tag's inner margin corner is the board corner. Nothing covers the play area. Board coordinate origin is one corner. Grid offset is configurable. |
| Geometry | All measurement is 2D in the board plane. No vantages, no heights. |
| Walls | Not detected in the image. Each layout is a map file, recorded by an agent from a photo of the layout card, and drawn as an overlay. Not used in measurements yet. |
| Doors | Hatchways and breach points are part of the map file: a marker next to a wall gives an opening, the wall key gives its kind (A1 and B2: breach point, other walls: hatchway). No clicks, no vision detection. |
| Bases | Round bases 25, 28, 32, 40, 50 mm, measured at the bottom, where the base touches the board. Later: 35 x 60 mm oval. |
| Tokens | Round tokens 20 and 40 mm. |
| Measure mode | Snapshot first. Key press triggers detect and measure on a clean frame. Live tracking later. |
| Interaction | Click an operative or token on screen. Hotkeys for ring radius, grid, snapshot. |
| Operative ID | All models visually distinct. Vision ID with few-shot gallery. No markers on models. |

## 2. Priorities and phases

Order from the owner. Each phase has a done test.

| # | Priority | Approach | Done when |
|---|---|---|---|
| 1 | Find board corners | AprilTag 36h11 card under each corner, tag outside the board. Detect every frame. Warn when a tag is lost or moved. | Four corners found in every clean frame. Drift of a bumped board detected within a second. |
| 2 | Undistort and rectify | One-time lens calibration. Per-frame homography from the 4 tags. Correct for the board thickness (tags lie on the table, play surface is higher). Warp to a top-down image at a fixed scale (mm per pixel). Optional map grid overlay (Tomb World sections, Volkus 1 in). | Board edges straight, corners at 90 deg. Grid lines land on the printed board grid across the whole board. Known 1 in distance measures 25.4 mm plus or minus 1 mm anywhere. |
| 3 | Find bases and positions | Detection is separate from identity. Background subtraction against an empty-board reference, then edge points, then circle fit with the known diameter set. Later: a YOLO segmentation model trained on base rims. | All bases found on a Tomb World board with walls. Center error under 2 mm. Correct diameter class. |
| 4 | Identify operatives | Crop around each found base. DINOv2 embedding. Nearest neighbor in an enrolled gallery (10 to 20 photos per operative from this camera). User confirms or fixes with one click. Fixes feed the gallery. | 9 of 10 bases labeled right on first try. |
| 5 | Rings and distances | Select operative. Hotkey picks N in 1 to 9. Draw circle with radius = base radius + N in. Select a second operative: draw and print rim-to-rim distance. | Ring and distance match a tape measure within 2 mm. |
| 6 | Terrain layouts | One map file per layout card (`data/maps/<terrain>/<layout>.toml`): walls with their card key, pillars, wall ends, hatchways and breach points, objective markers, teleporter pads, drop zones. An agent records the file from a photo of the card (skill `record-map`); the loader checks it. The UI chooses terrain type and layout, and toggles the layout, the drop zones and the terrain keys. Vision detection rejected: models hide walls, and the card already holds the layout. | Map file of a card loads, and its overlay sits on the real terrain of a photo of that layout. |
| 7 | Tokens | Detect 20 and 40 mm round tokens with the same circle fit, different diameter set. Rings 1 to 9 in. Mark every operative whose base rim lies within the selected distance. | Tokens found. In-range list matches tape measure. |

Answer to the question in priority 3: yes, bases come before identity. Detection is pure geometry and needs no training. Identity runs on the crop around each found base. Keeping them separate means a new team only needs gallery photos, never a retrained detector.

### UI work per priority

The UI grows with the priorities. `specs/ui.md` has the design: section 8 for the contract and the layouts, section 12 for the foundation steps.

A priority is done only when its "Done when" test above and its UI part below are both done. For each priority:

1. Put the logic in a module with no Qt, and add a pipeline stage when the priority finds something in the image.
2. Add the actions to `live.ACTIONS` and the new values to `LiveSession.state()`. Add the board overlays to the core image. Status text, lists and popups belong to the layouts.
3. Keys and the generated menu then make the feature available in all layouts. Add the controls to the `inspector` layout. The `panel` layout generates its controls from the action list.
4. Add the new states to the recorded list of the contract test (`specs/ui.md`, section 11).
5. Update `specs/ui.md` (sections 3 and 8), `README.md` ("How to use", "What you see"), and `docs/cli.md` (keys).

| Work | Actions | State | `inspector` layout | UI done when |
|---|---|---|---|---|
| Priorities 1 to 4 | Exist: `view.*`, `reference.*`, `snapshot.*`, `board.click`, `base.choose`, `names.*`, `frame.save`. | Exist: `drift`, `lens`, `reference_set`, `bases`, `selected`, `candidates`. | Live, Snapshot and Base selected panels. | Part of the UI foundation. |
| UI foundation, before priority 5 | Add `app.layout` (`l`). | Add fps and detect time to `debug`. | First version. | `specs/ui.md` section 11 passes. Steps 2 to 5 of `specs/ui.md` section 12 are done. |
| 5 Rings and distances | `ring.1` to `ring.9` (keys `1` to `9`), `ring.clear` (`0`). | `target` (second base), `ring_in`, `distance` (rim to rim, in mm and in). | Base selected: ring buttons and the distance. | In each layout, a key draws the ring on the selected base, and a second base shows the distance in the status area. |
| 6 Terrain layouts (done) | `map.terrain_type(name)`, `map.layout(name)`, `view.terrain` ("Layout"), `view.zones` ("Drop zones"), `view.keys` ("Terrain keys"). No keys. | `terrain_types`, `terrain_type`, `map_layouts`, `map_layout`, `terrain`, `zones`, `keys`. | Terrain type and Layout drop-down lists at the top of the panel (both layouts), the toggles with the view controls. | In each layout, a user chooses a layout and sees it on the board; the choice stays for the next start. |
| 7 Tokens | Token selection through `board.click`. The ring actions also apply to a token. | `tokens`, `selected_token`, `in_range`. | New Token selected panel with the in-range list. | In each layout, a selected token shows its ring and the list of operatives in range. |
| Later: wounds and statuses | `operative.wound`, `operative.status`. | `roster`. | Roster list in the Snapshot panel. | Build the `rosters` layout (`specs/ui.md`, Option 3). |

Priority 5 changed the meaning of a click. The first left click selects a base, and a click on another base sets `target`. The name list is in the Base selected panel of `inspector`, and a right click opens the name popup in the other layouts.

## 3. Feasibility verdict

Feasible. Risk per priority:

| # | Risk | Note |
|---|---|---|
| 1, 2 | Low | Standard OpenCV. |
| 3 | Medium | Known diameters make circle fit robust under partial occlusion by the model body. Dark board and wall shadows lower contrast. |
| 4 | Medium | Works when models look different from above. Enrollment photos must come from the real camera. |
| 5 | Low | Pure math after 3. |
| 6 | Low | No vision. Risk is a wrong reading of the card; the loader checks and the drawn map is compared with the photo. |
| 7 | Low to medium | Metallic or transparent tokens reflect light. Printed ring on the token helps. |

## 4. Camera geometry

Board size is a config value. Tomb World: 606 x 703 mm. Volkus: 559 x 760 mm.

Camera: Xiaomi 13T. Main lens 50 MP at 24 mm equivalent, f/1.9, OIS. 4K at 30 fps. Field of view in 16:9 video: about 74 deg x 46 deg.

Sample photos from this phone show the board filling about 95% of the frame width. At 4K that is about 4 px per mm. The tag cards add 50 mm outside each corner (board plus cards: 706 x 803 mm for Tomb World), so the mount sits higher and density drops to about 3.5 px per mm. Target is 1.5 px per mm, so density is not a problem. 16:9 video crops the short side by 25% compared with the 4:3 photos, so the mount sits a little higher than for the photos.

Parallax at 1.0 m, at a board corner (about 465 mm from camera nadir):

- Base top, 3 mm high: shifts about 1.4 mm outward. Correctable, since camera pose is known.
- Model body, 40 mm high: shifts about 19 mm outward. Body covers the outer side of its base. Inner side of the base rim stays visible.

- Tag cards on the table, board surface t mm higher: the homography from the tags maps the table plane. A point on the play surface shifts outward by r * t / h. For t = 3 mm at a board corner: about 1.4 mm. Priority 2 corrects this with the known camera pose and a `board.thickness_mm` config value. Fallback: put each card on a spacer as thick as the board.

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
- UI: `PySide6-Essentials` (Qt). Video widget, layouts with panels, mouse clicks (`specs/ui.md`). The commands other than the live app keep OpenCV `imshow` windows.
- Config and data: JSON or TOML. Calibration, gallery embeddings, token and door sizes.

Later option: FastAPI plus browser front end over WebSocket for a tablet at the table. It uses the same actions and state as the Qt layouts. Core must not depend on Qt.

## 7. Proposed repository layout

```
battle-sight/
  specs/              # this plan, about-project, images
  src/battlesight/
    camera.py         # capture, reconnect
    calib.py          # tag detection, drift check, lens model, ChArUco calibration
    rectify.py        # homography, thickness correction, warp to board mm space, edge and grid checks
    circles.py        # background subtraction, known-radius circle fit (bases and tokens)
    ident.py          # embeddings, gallery, kNN
    terrain.py        # map files: walls, pillars, hatchways, breach points, objectives, zones
    measure.py        # rim-to-rim distances, rings, bands, in-range tests
    overlay.py        # drawing in board coordinates
    live.py           # game core of the live app: LiveSession, actions, state (no Qt)
    ui/               # PySide6: shell.py, parts.py, layouts/ (panel, inspector, later rosters)
  data/
    ui.toml           # selected layout and layout options (git-ignored)
    print/            # printable sheets: corner tags, ChArUco target
    teams/            # gallery per team: enrollment images and embeddings (git-ignored)
    maps/             # terrain map files per layout (agent-recorded from the card photo)
    sizes.toml        # base and token dimensions (terrain sizes: [terrain] in board.toml)
  tools/
    make_charuco.py   # print sheet for lens calibration (`battlesight calibrate`)
    make_tags.py      # print sheet for AprilTags
    enroll_operative.py
```

## 8. Risks

- **Dark board and wall shadows** reduce contrast at base rims. Mitigation: diffuse light, background subtraction against the reference frame, known-radius fit.
- **Metallic or transparent tokens** reflect light. Mitigation: detect by printed ring, or user places token by click.
- **Camera bumped mid-game.** Mitigation: tag drift check every frame, auto re-homography, warning in the UI.
- **Board slides over still cards.** The drift check sees the cards, not the board. The board's weight on the cards makes them move together in most bumps. Mitigation: the L on each card shows where the corner belongs, so the user puts it back. Later: check the board edge against the expected edge lines (dark board on a lighter table) and warn on a mismatch.
- **Few-shot ID confuses similar paint schemes.** Mitigation: confirmation step, corrections go into gallery, upgrade to trained classifier.
- **Phone stream drops or heats up.** Mitigation: USB, charger, screen off, app reconnects on its own.
- **UI falls behind the priorities, or layouts drift apart.** Mitigation: the UI part is in the done test of each priority (section 2), keys and a generated menu make each action available in all layouts, and a contract test runs each layout.

## 9. Open implementation details

Resolved inside the phase that needs them.

- Priority 1: tag size and print sheet done (40 mm tag, 5 mm margin = 1 module quiet zone, 90 mm square cards with a 40 mm L corner mark, all four on one A4 page). L length does not change accuracy: the board corner is the tag's inner margin corner at any card rotation. Install Camo, confirm 4K reaches OpenCV.
- Priority 2: done. Tomb World board thickness 1.0 mm (caliper), board size 703 x 606 mm confirmed. Width is the edge from tag 0 to tag 1.
- Priority 2 results on the two test photos (nominal lens, thickness 1.0 mm): tag fit rms 1.0 mm, max 1.7 mm. Board edges tilt under 0.15 deg, corners 90 deg within 0.21 deg. Printed section lines within 3.1 mm of nominal (card placement: the board right edge is 2.5 to 3.3 mm inside the L), spacing of neighbouring lines within 1.4 mm over 97 mm (0.37 mm per inch).
- Priority 3: in code (`circles.py`, stage `bases`, command `bases`). Known-radius circle fit on the dark rim, 120 rays, robust fit, size class from the free fit, rim coverage at least 0.5. Empty-board reference = board with terrain and tokens set up, before the operatives, same camera and light; the reference finds its own tags. The diameter is the base bottom on the play surface, so no parallax correction. Results on the two test photos (single-frame mode, no reference exists for them): all 9 bases with a clearly visible rim found, centre on the rim, right class (25/28/32/40/50). Bases with most of the rim hidden by a model leaning over it or a wall are missed (5 of 7). False bases: tokens with a dark ring, a round barricade, wall gratings (2 and 5 per photo). A reference from another photo of the same board does not work: the light changed and 45 % of the board differs. Open: test photo pair (empty board, then with operatives) for the done test. Then decide whether the reference is enough or a trained base segmenter is needed. Benchmark (`tools/bench_bases.py`, `specs/bases-benchmark.md`), first baseline 2026-11-03 on the 3 test photos, single-frame mode, schema 1 ground truth: 16 true bases, 19 found, 11 matched (clear 9/9, hidden 2/7), 1 wrong size, 8 false (2 inside a base, 6 other), recall 0.69, precision 0.58, f1 0.63, done 0/3. 2026-11-07: negative photo `tomb-world-empty.jpg` removed (not a valid reference, grid spacing fail), new clicked ground truth for `tomb-world-2`; baseline on the 2 game photos: 16 true bases, 18 found, 10 matched (clear 10/12, hidden 0/4), 8 false, recall 0.62, precision 0.56, f1 0.59, done 0/2. 2026-11-10: the stage uses the base sizes of the teams (roster: gallery `team.toml`, `roster_mm` in the test ground truth) and then keeps the radius of each size in the fit; with a reference too, the coverage limit is `reference_min_coverage` 0.4. Baseline on the 4 photos, 34 true bases: before 25 matched, 10 false, f1 0.72, done 0/4; after 30 matched (clear 28/29, hidden 2/5), 1 false, 0 wrong size, recall 0.88, precision 0.97, f1 0.92, max centre error 0.93 mm, done 2/4 (both photos with a reference: 19 of 19, 0 false). The 4 bases not found are on the 2 photos without a reference.
- Priority 4: in code (`ident.py`, stage `ident`, commands `ident`, `enroll`, `gallery`, `image --team`). Crop = base diameter + 25 mm board each side from the bases image, grey outside a circle. DINOv2 small (`facebook/dinov2-small`, `transformers`, MPS), class token + mean patch token. Gallery per team in `data/teams/<team>/`, base size per operative in `team.toml`; only operatives of the base size class can match. Each gallery crop at 8 rotations. Score = mean of the 3 best photos. Hungarian one-to-one assignment, because each operative is on the board once. Fix = click base, click name; the crop goes into the gallery at once. 2026-11-07 first check on the 2 test photos (no real gallery): about 90 ms for 10 bases; crops of different operatives score 0.5 to 0.87, one model in both photos 0.75, so the board around the base weighs much. Marked done by the owner on 2026-11-10. Baseline (`tests/images/ident_baseline.json`, real gallery, 2 teams, 554 files): on the 2 4K photos with reference, all 15 correctly detected bases have the right name, 15 of 19 bases in total (1 wrong size, 3 missed). On all 4 photos: 22 of 34 bases right (accuracy 0.65), naming accuracy 0.88, 1 of 4 photos passes the "9 of 10" test. The 4K losses come from base detection, not from naming. The 2 older photos have no reference (7 of 15 right). 2026-11-10: 17 gallery photos were crops of the 2 4K test photos (name fixes on `--image-source`), so those scores were too good; the benchmark now leaves such copies out (`ident_bench.copies_of`), `Gallery.add` refuses the same crop two times, 13 identical extra files moved out of the gallery. Baseline with the roster and fixed-radius detection (gallery 586 files): 28 of 34 right (accuracy 0.82), naming accuracy 0.93, 0 wrong sizes, 4 missed, 2 of 4 photos pass (both 4K photos 19 of 19). Naming alone on the true base positions: 31 of 34. All 6 losses are on the 2 photos without a reference.
- Priority 5: in code (`measure.py`, `ring.*`, `board.click`, `board.select`). Ring radius and rim-to-rim distance come from the found base circles in board mm. Works after `space` and after `b`. A ring belongs to the selected base, so a new selection starts with no ring. The synthetic test: ring radius within 1 mm, distance within 2 mm. Open: the done test with a tape measure on the real board.
- UI foundation: open decisions are in `specs/ui.md`, section 10.
- Priority 6: done 2026-11-10. `terrain.py`, `data/maps/<terrain>/<layout>.toml`, written by an agent from a photo of the layout card (skill `record-map`), not clicked. Walls with their card key (A1..A4, B1..B4), pillars (A walls except A2 have a middle pillar), wall ends, hatchways and breach points (at the centre between two pillars; A1 and B2 walls have breach points), objectives, teleporter pads, deployment zones. Sizes in `[terrain]` of `data/board.toml`. Overlay chosen in the UI (terrain type and layout drop-down lists), with toggles for the layout, the drop zones and the terrain keys (wall keys, `T` on teleporter pads). A dashed circle 1 in around each objective marker. Command `battlesight map` checks a file and draws it. Done test: Layout 6 drawn on `tomb-world-4k-1.empty.png` (the same layout), walls, pillars, objective markers and their 1 in circles on the real ones.
- Later: roster format, wounds and statuses, live tracking, oval bases, Volkus board.
