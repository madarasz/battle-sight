# Issue List

Fixes for common problems.

- **The board slid but there is no warning.** The app measures from the
  cards, not from the board. Put the board corner back against the black L
  of the card.
- **Error that width and height are the wrong way round.** Change
  `width_mm` and `height_mm` in `data/board.toml`. See
  [setup.md](setup.md), step 5.
- **The camera does not start.** Make sure that the Terminal app has camera
  permission (System Settings > Privacy & Security > Camera). Find the
  correct camera number with `uv run tools/list_cameras.py`. For Camo, a
  Wi-Fi stream or scrcpy, see [setup.md](setup.md),
  Troubleshooting.
- **The terrain drawing does not sit on the real walls.** Turn the board or
  the cards.
- **A base is not found.** The app must see at least half of the black edge
  of the base. Move the camera nearer to the top of the board, or turn the
  model. Take an empty-board reference.
- **A base of a new operative is not found, or the name list is empty.**
  Teach the app this operative (setup step 8) with its base size.
- **A token or a terrain part shows as a base.** Take the empty-board
  reference after you set up the terrain and the tokens.
- **An operative gets the wrong name frequently.** Add more photos of the
  operatives (setup step 8).
- **The inspector shows "has no maximum wounds".** Add the wounds of the
  operative: `uv run battlesight enroll --team kommandos --operative "Boss Nob" --wounds 12`,
  then start the app again.
- **The size of a base is wrong.** Check `base_mm` of each operative in
  `data/teams/<team>/team.toml`.
- **The first "Identify all operatives" is slow.** The app downloads the
  recognition model one time.
- **The grid does not sit on the printed lines.** Check the map settings
  ([setup.md](setup.md), step 5). Then calibrate the lens (step 6).
