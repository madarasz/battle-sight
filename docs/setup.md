# Setup and calibration

Do steps 1 to 6 one time before your first game. Do steps 5 and 6 again only
when you change the board or the camera. Do step 7 before each game.

## 1. Install the app (macOS)

1. Open the Terminal app.
2. Install `uv` (the tool that installs and runs Battle Sight), if you do not
   have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Go to the Battle Sight folder and install the app:

   ```bash
   uv sync
   ```

   This step downloads all the parts the app needs. It can take some minutes
   the first time.

## 2. Print the corner cards

The app finds the board with four printed cards, one under each board corner.

1. Make the print sheet:

   ```bash
   uv run tools/make_tags.py
   ```

   This writes `data/print/corner_tags.pdf` (A4, all four cards on one page).
2. Print the sheet at 100 %. Do not use "fit to page".
3. Measure the 100 mm ruler on the print. It must be 100 mm. If not, print
   again and check the printer scale.
4. Cut the cards on the grey lines. Do not cut into the white border around
   the black square. The app needs this white border to see the tag.

Each card is a 90 mm square. The black square pattern (the tag) is in one
corner. A black L in the opposite corner shows where the board corner goes.

## 3. Put the cards and the board on the table

1. Put each card on the table at its board corner. The label on the card
   tells which corner: 0 = top-left, 1 = top-right, 2 = bottom-right,
   3 = bottom-left, as you see the board on the screen.
2. Put the board corner on the card. Both board edges must touch the inner
   side of the black L.
3. Check the two lines of the L:
   - Both lines are fully visible and touch the board: correct.
   - The board covers a line: the board is too far out. Move it in.
   - There is a white gap between line and board: the board is too far in.
     Move it out.

The cards are under the board, so they do not cover the play area. The
weight of the board holds the cards.

## 4. Connect the camera

Put the camera above the board, looking down. All four cards must be in the
image.

You set up the camera one time. `tools/list_cameras.py` checks the camera
and saves it in `data/camera.toml`. This file holds all camera settings:
the camera source and image size (this step), the nominal focal length
(`focal_35mm`, 24 mm for the Xiaomi 13T main lens), and the lens
calibration (step 6). After that, all live commands
(`corners`, `rectify`, `calibrate`) use the saved camera, and you do not
type `--source`, `--width`, or `--height` again.

There are two ways to connect a phone. Use Camo if it works on your Mac.
Use a Wi-Fi stream if Camo cannot install its camera (for example, on a
company-managed Mac).

| | A. Camo (USB) | B. Wi-Fi stream |
|---|---|---|
| Saved source | camera number, for example `1` | stream URL, for example `http://192.168.1.20:8080/video` |
| Needs | Camo Camera system extension on the Mac | phone and Mac on the same network |
| Image size | requested by the app (saved) | set in the phone app |
| Lag and image quality | low lag, little compression | more lag, more compression |

### A. Camo (USB)

1. Install Camo on the Mac (reincubate.com/camo) and on the phone. Connect
   the phone with a USB cable.
2. In Camo on the Mac, select the main lens and 4K. Lock the focus and the
   exposure.
3. List the cameras:

   ```bash
   uv run tools/list_cameras.py
   ```

   Each camera shows as an index with its image size. Camo is usually
   index 1, and it gives the size that you selected in Camo (for example
   3840x2160). If the script shows only index 0, read
   [Troubleshooting](#troubleshooting).
4. Save the Camo index (here: 1) and the image size:

   ```bash
   uv run tools/list_cameras.py --use 1 --width 3840 --height 2160
   ```

   If the camera gives a different size, the script tells you and saves the
   size that the camera gives.

On the first start, macOS asks if the Terminal app can use the camera.
Click "Allow".

### B. Wi-Fi stream

The phone sends its video over the network. The app opens the stream URL
with FFmpeg. No camera extension is necessary on the Mac, and Camo Studio
does not need to run.

1. Install a camera stream app on the phone. On Android, "IP Webcam" works.
   On iPhone, use an app that serves an MJPEG (`http://`) or RTSP
   (`rtsp://`) stream. Close Camo on the phone: only one app at a time can
   use the phone camera.
2. In the app settings, set:
   - the main (back) camera,
   - the largest video size the network can carry (4K gives the best
     accuracy, 1080p gives less lag),
   - focus and exposure locked, if the app has these settings.
3. Start the stream. The app shows the phone address. In IP Webcam, tap
   "Start server". The address shows on the video, for example
   `http://192.168.1.20:8080`. The stream URL is this address plus
   `/video`. You can also find the address in the phone settings: Wi-Fi >
   the network you are connected to > IP address.
4. Put the phone and the Mac on the same Wi-Fi network. Open the stream URL
   in a browser on the Mac. You must see the video.
5. Check and save the stream:

   ```bash
   uv run tools/list_cameras.py --url http://192.168.1.20:8080/video
   ```

   The script shows the image size and the measured frame rate, and saves
   the URL.

On the first start, macOS asks if the Terminal app can find devices on the
local network. Click "Allow".

The phone can get a new address after a restart. Then do step 5 again with
the new address. To keep the address, set a DHCP reservation for the phone
in the router settings.

### Built-in camera (first test only)

The built-in camera cannot look down on the board, but you can use it to
test the app with a card in your hand:

```bash
uv run tools/list_cameras.py --use 0 --width 1920 --height 1080
```

### Start the app

```bash
uv run battlesight corners
```

The first output line tells which settings file the app uses. When the
banner shows green "4 corners OK", the setup is correct.

Command options override the saved settings for one run, for example
`uv run battlesight corners --source 0`. To change the saved camera, run
`tools/list_cameras.py --use` or `--url` again. To only check a camera and
not save it, add `--no-save`.

Calibrate the lens again (step 6) when you change between Camo and the
stream, or change the image size.

## 5. Set your board and map

The file `data/board.toml` holds the board settings. Open it in a text
editor. All lengths are in millimetres.

- `[board]`:
  - `width_mm`: the edge from card 0 (top-left) to card 1 (top-right).
  - `height_mm`: the other edge.
  - `thickness_mm`: the height of the play surface above the table. Measure
    it with a caliper or a ruler.

  If width and height are the wrong way round, the app shows an error that
  tells you so. Change the two values.
- `[grid]`: the map grid that the app draws.
  - Tomb World: `margin_mm = 12.0` and `sections = [7, 6]` (the printed
    sections on the board).
  - Volkus: `margin_mm = 0.0`, `pitch_mm = 25.4` (a 1 inch grid), and remove
    the `sections` line.
  - `printed = true`: the grid lines are printed on the board. The app then
    also checks its accuracy against the printed lines.
- `[drift]`, `tolerance_mm`: how far a card can move before the app shows
  "BOARD MOVED" (default 2 mm).
- `[bases]`, `diameters_mm`: the base sizes in mm that the app looks for.
  Measure a base at its bottom, where it touches the board. The default list
  is `[25, 28, 32, 40, 50]`. Remove the sizes that your teams do not use:
  fewer sizes give fewer mistakes.

## 6. Calibrate the lens (optional)

Most phones correct the lens themselves, so the app works without this step.
Do it for better accuracy at the board edges.

1. Make the calibration sheet:

   ```bash
   uv run tools/make_charuco.py
   ```

2. Print `data/print/charuco.pdf`. You can print at 100 % or use "fit to
   page". The size of the print does not change the result. The scale must
   be the same in both directions (no stretch). Glue the print to something
   flat and stiff (for example, cardboard or a clipboard).
3. Start the calibration. It uses the camera that you saved in step 4:

   ```bash
   uv run battlesight calibrate
   ```

4. Hold the sheet in front of the camera. Press `space` to grab a view.
   Grab 10 to 20 views. Move the sheet to all areas of the image. Tilt it in
   some views.
5. Press `c` to finish. The app writes the lens values to the `[lens]`
   part of `data/camera.toml` and uses them from then on. The camera
   settings from step 4 stay in the same file.

Do the calibration again when you change the camera, the lens, or the image
size.

## 7. Take the empty-board photo

The app finds the bases best when it can compare the board with a photo of
the same board without operatives. Do this step before each game, because
the terrain and the light change.

1. Set up the terrain and the tokens on the board. Do not put operatives on
   the board yet.
2. Make sure that the light is as it will be in the game. Close the curtains
   or turn on the lamps first.
3. Start the bases mode:

   ```bash
   uv run battlesight bases
   ```

4. Keep your hands and other objects out of the image. Push `e`. The app
   saves the photo to `data/snapshots/empty_board.png` and uses it until you
   take a new one. The next start of `battlesight bases` loads it again.
5. Put the operatives on the board and push `space`.

If you move the camera or the light changes much, take the photo again. A
small move of the board is not a problem: the app finds the corner cards in
the empty-board photo too.

### Photo pairs for the tests (optional)

The developers need photo pairs to test and improve the base detection.
A pair is one empty-board photo and one game photo. Obey these rules:

- Take both photos in the same session. Set the terrain and the tokens,
  take the empty photo, put the operatives on the board, then take the game
  photo. Do not move tokens or terrain between the two photos.
- You can hold the phone in your hand. Hold it at about the same place for
  both photos, straight above the board centre. Tall walls look different
  when the camera moves.
- Lock the exposure and the white balance if the camera app can do it. Do
  not change the lamps between the two photos.
- Give the files the same name: `<name>.jpg` (game) and `<name>.empty.jpg`
  (empty board).

## 8. Teach the app your operatives

The app can name each operative on the board. First it must learn how each
operative looks from your camera. Do this step one time for each team. You
need 10 to 20 photos of each operative. The app takes them for you with the
game camera. Read [Good gallery photos](#good-gallery-photos) first.

1. Select a short name for the team, for example `kommandos`. Use the same
   name in each command.
2. Put one operative on the board. Start the enroll mode with its name:

   ```bash
   uv run battlesight enroll --team kommandos --operative "Boss Nob"
   ```

3. Push `space`. The app finds the base and keeps a photo of it. The banner
   shows the number of photos.
4. Turn the model a little and move it to another place on the board. Push
   `space` again. Do this 10 to 20 times. Use places near the center and
   near the edges, because the model looks different there.
5. If the app finds more than one base, click the base of the operative.
   If you took a bad photo, push `u` to remove it.
6. Push `q`. Do steps 2 to 6 for each operative of the team.

### Good gallery photos

The app compares the operatives in a game with the gallery photos. Thus the
gallery photos must look like the game: the same camera, the same board, the
same light.

- **Use the game camera, in its game position.** Do not take close-up
  photos. The app must see the four corner cards to find the board, and it
  measures the base in board millimetres. A close-up photo, or a photo
  without the cards, gives no gallery photo.
- **Put the operative on the board, not on a desk.** The app keeps a circle
  around the base, with about 25 mm of board around the base edge. This
  board is part of each photo. A desk around the model teaches the app the
  wrong background.
- **The empty board and the board with terrain are both correct.** Keep
  about 30 mm of free board around the base. A wall or a token near the
  base gets into the photo, and the app can learn it as a part of the
  operative. A wall that covers the base edge also stops the app from
  finding the base.
- **Take the empty-board photo first (step 7)** for the board as it is now.
  The enroll mode uses it to find the base. An old empty-board photo of
  other terrain can give wrong bases.
- **Use the game light.** Lock the focus and the exposure of the camera, as
  for a game.
- **You do not cut the photos.** The app finds the base and keeps only the
  circle around it. The rest of the camera image is not kept.
- **Move the operative and turn it.** The camera sees the model from a
  different side at each place on the board: near an edge, the model leans
  away from the board centre in the image. Use the centre, all four edges,
  and the corners. At each place, turn the model to a new direction. The
  app also turns each photo itself, but a real turn changes the shadows and
  the visible sides of the model.
- **One operative in the image is the easiest.** If more operatives are on
  the board, click the base of the correct operative after `space`.
- **Keep your hands out of the image** when you push `space`.

| How many photos | When |
|---|---|
| 10 | Operatives of the team look very different (colour, shape, weapon). |
| 15 to 20 | Two or more operatives look alike, for example the same body with a different weapon. |
| More of two operatives | `battlesight gallery` shows that the app mixes up these two. |

A fast way to get many photos: enroll each operative with about 5 photos.
Then use `battlesight ident` in your games. When a name is wrong, click the
base and the correct name. When all names are correct, push `a`. Each
correction and each `a` adds photos from real games to the gallery.

The app keeps the photos in `data/teams/kommandos/`, one folder for each
operative. It also keeps the base size of each operative in
`data/teams/kommandos/team.toml`. An operative gets a name only on a base
of this size.

To check the photos, run:

```bash
uv run battlesight gallery --team kommandos
```

The app compares each photo with the other photos and shows how many it
names correctly. If it mixes up two operatives, add more photos of these
two operatives.

The first start downloads the recognition model (about 90 MB). This needs
the internet one time only.

## Troubleshooting

### `list_cameras.py` shows only index 0

The script prints warnings like these:

```
OpenCV: out device of bound (0-0): 1
OpenCV: camera failed to properly initialize!
```

These warnings do no harm. `(0-0)` means that OpenCV found only one camera
(index 0, the built-in camera). The script then tries indices 1 to 5, and
each try prints a warning. The problem is that macOS does not see the Camo
camera.

1. Check the cameras that macOS sees:

   ```bash
   system_profiler SPCameraDataType
   ```

   If "Camo" is not in the list, no app on the Mac can use it. Camo Studio
   can show the phone video in its own window and still not give it to
   other apps.
2. Check the Camo system extension:

   ```bash
   systemextensionsctl list | grep -i reincubate
   ```

   No output means that the extension is not active.
3. On a personal Mac, open Camo Studio. macOS asks you to approve the
   extension in System Settings > Privacy & Security. Approve it, then
   restart Camo Studio.
4. Check if your Mac is company-managed (MDM):

   ```bash
   profiles status -type enrollment
   ```

   If it shows `MDM enrollment: Yes`, the company can allow only a list of
   system extensions. macOS then does not ask you to approve Camo, and the
   settings show no Camo extension. You cannot change this. Do one of these:
   - Ask your IT department to allow the extension: team ID `Q248YREB53`,
     bundle ID `com.reincubate.macos.cam.avextension`.
   - Use a Wi-Fi stream ([B. Wi-Fi stream](#b-wi-fi-stream)).
   - Use a personal Mac.

### The camera number opens the wrong camera

The index order can change when you connect or disconnect a camera. Run
`uv run tools/list_cameras.py` again. The Camo index gives the size that you
selected in Camo (for example 3840x2160). The built-in camera gives a
smaller size. Save the correct index again with `--use`.

### The stream URL does not open

`cannot open camera source 'http://...'`:

1. Open the URL in a browser on the Mac.
   - The browser shows the video: go to step 3.
   - The browser does not show the video: go to step 2.
2. Check the network:
   - The phone and the Mac must be on the same Wi-Fi network. Guest
     networks often block traffic between devices.
   - A VPN (for example, GlobalProtect) can send local traffic through the
     VPN. Disconnect it, or use a network where the VPN allows local
     traffic.
   - Use the address that the phone app shows now. The phone can get a new
     address after a restart. Save the new URL with
     `uv run tools/list_cameras.py --url ...`.
3. Allow local network access: System Settings > Privacy & Security >
   Local Network. Turn on the Terminal app (or the app that runs `uv`).
   Then quit the Terminal app fully (Cmd+Q) and open it again.

   This is the cause when the error is `No route to host`, but the browser
   shows the video and `ping <phone-address>` works. macOS blocks apps
   without this permission from the local network and gives this error,
   not a permission error.
4. Check the full URL with the path, for example `/video`. The address
   alone (`http://192.168.1.20:8080`) is the web page of the phone app, not
   the stream.

### The stream lags or the image breaks up

- Make the video size or the quality smaller in the phone app.
- Move the phone and the Mac nearer to the router, or use 5 GHz Wi-Fi.
- For RTSP, the app uses TCP. To change FFmpeg options, set
  `OPENCV_FFMPEG_CAPTURE_OPTIONS` before you start the app, for example
  `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp"`.
- Plug the phone into a charger. Streaming uses a lot of power, and some
  phones make the video slower when the battery is low or the phone is hot.
