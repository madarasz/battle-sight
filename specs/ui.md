# Battle Sight: UI options

Status: research only, nothing implemented. Date: 2026-11-03.

## 1. Goal

One window with the camera view and a control panel on the right side. The panel has:

- Rectify on/off (top-down view or camera view).
- Grid on/off.
- Tag and corner overlay on/off.
- Buttons for re-lock reference and snapshot.
- A debug area at the bottom of the panel. It shows the text that the banner on the frame shows now: drift status (OK, MOVED, LOST), tags found, fit max error in mm, fps, detect time, lens source.

The plan (`specs/plan.md`, section 1 "Display" and section 6) already asks for "one window: video with overlay, side panel" and selects PySide6. The core must not depend on Qt, because a FastAPI and browser front end can come later.

## 2. Current state

- `corners` and `rectify` in `src/battlesight/app.py` each have their own `cv2.imshow` loop. Keys control them: `v` view, `g` grid, `r` re-lock, `s` snapshot, `q` quit.
- `draw_text_banner` and `draw_banner` in `src/battlesight/overlay.py` draw the status banner into the frame.
- Camera read, detection, rectify and drawing are in the same loop as the window code. Before you add a UI, you must move them out of the loop.

## 3. Options

| Option | How | Good | Bad |
|---|---|---|---|
| A. OpenCV only | Draw the panel as an image, put it to the right of the frame, read clicks with `cv2.setMouseCallback`. | No new dependency. Small change. | You draw and hit-test the buttons yourself. You must convert click positions through `fit_to_width`. `cv2.createButton` needs an OpenCV built with Qt, and the pip wheel does not have it. It is a bad base for the later click work (doors, operative ID). |
| B. PySide6 | `QMainWindow`. Video widget in the center, `QDockWidget` on the right with check boxes and buttons, status text at the bottom of the panel. | The plan selects it. Real widgets. Good mouse support for priority 7 (doors) and for picking operatives. Low delay. | New dependency (about 100 MB). Needs a camera thread. Possible Qt conflict with OpenCV (section 6). |
| C. FastAPI and browser | Server sends the video as MJPEG or over a WebSocket. The panel is HTML and JS. The toggles go through REST or WebSocket messages. | Works on a tablet or phone at the table. No Qt install problem. Easy tests with FastAPI `TestClient`, no display needed. | Most work. The plan marks it "later". |

## 4. Size of B and C

The Qt-free core (section 5, step 1) is the same for B and C. It is about half of the total work. The front end is different:

| Part | B (PySide6) | C (FastAPI and browser) |
|---|---|---|
| Window or server | about 150 lines Python | about 100 lines Python |
| Video to the screen | Qt signal and `QImage`, about 30 lines | JPEG encode and MJPEG or WebSocket stream, about 60 lines |
| Panel controls | Qt widgets, about 100 lines | HTML, JS and CSS, about 200 to 300 lines |
| Sending the toggles | Direct method call | REST or WebSocket messages, state kept in sync in both directions |
| Languages | 1 (Python) | 3 (Python, JS, HTML/CSS) |
| Processes | 1 | Server and browser, a port, a browser to open |
| Total | about 300 lines | about 500 to 600 lines |

C is about 1.5 to 2 times the work of B. C costs more because of these items:

- Video transport. Each frame must be JPEG encoded. At 4K this uses a lot of CPU, so make the frame smaller first. You also get about 50 to 150 ms more delay than with Qt.
- Two sides to debug: Python and JS. Problems at the boundary (old state, reconnects) are harder to find.
- Click mapping. You must convert browser pixels back to frame pixels. B has the same problem, but Qt gives the positions more easily.
- Lifecycle. You must start the server, open the browser, and handle a closed tab or a lost connection.

## 5. Recommendation and implementation points

Recommendation: B (PySide6) if you only use a laptop or monitor next to the table. If the tablet at the table becomes a real goal, start with C. In both cases, build the Qt-free core first. Then a change from B to C later replaces only the front end.

Steps for B:

1. Qt-free core. Move the live loop into a `LiveSession` class, for example in `src/battlesight/live.py`. It holds `TagDetector`, `DriftMonitor`, the lens and the last `BoardView`. It has:
   - A settings object: `top_down`, `grid`, `show_tags`, `show_debug`.
   - `step(frame) -> (image, status)`. `status` is a JSON-safe dict with the debug values, so Qt and a later web front end can both show it.
   - `relock()` and `snapshot()`.
   Keep `corners` and `rectify` as plain `imshow` fallbacks on top of the same core.
2. `ui/` package (already in the plan layout, section 7). `QMainWindow` and a worker `QThread` that runs camera read and `LiveSession.step`. The thread sends each image and status to the GUI thread with a signal.
3. Right panel. A `QDockWidget` with check boxes for rectify, grid and tag overlay, and buttons for re-lock and snapshot. A changed check box changes the `LiveSession` settings.
4. Debug area at the bottom of the panel. It shows the `status` dict. The video stays clean. Keep the on-frame banner as an option.
5. Keys. `QShortcut` for `q`, `v`, `g`, `r`, `s`, so the keys stay the same as now.
6. Speed. Resize the 4K frame to the widget size with `cv2.resize` before the conversion to `QImage`.
7. Subcommand. Add one `battlesight ui` subcommand in `app.py` that replaces `corners` and `rectify` as the main mode. The panel controls the mode. This is still an open decision (section 7).
8. Docs. Add the new mode, the panel and its controls to README "How to use" and "What you see", and to `docs/cli.md`.

## 6. Risks

- Qt conflict. On Linux, `opencv-python` comes with its own Qt plugins, and they break PySide6. macOS uses Cocoa, so the risk there is low. If the conflict occurs, change to `opencv-python-headless`. Then `image --show` and the live `calibrate` need a Qt viewer, because `cv2.imshow` stops working.
- Thread rules. Do all Qt painting in the GUI thread. Send frames only with signals.
- Camera down. `Camera.read()` returns `None` while the camera reconnects. The worker thread must handle this and send a status, not stop.

## 7. Open decisions

- One `battlesight ui` mode that replaces `corners` and `rectify`, or a panel in each mode. Recommendation: one `ui` mode, and keep the old modes as plain fallbacks.
- B now and C later, or C now. It depends on whether a tablet at the table is a goal for the MVP.
