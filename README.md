# Halloween Skeleton Party Cam 💀🕺

Turns everyone on your dance floor into dancing skeletons, live, so you can AirPlay it to a TV
somewhere else in the house and lure people onto the floor.

In the default mode the **people disappear entirely**: the app paints the empty dance floor over
them, so all that's left is skeletons dancing in your real room. A rotating banner
("THE DEAD ARE DANCING. WHAT'S YOUR EXCUSE?") nudges the audience to come join.

Built for a Mac with Apple Silicon and a TP-Link Tapo camera, but any RTSP camera or webcam works.

## How it works

1. **Pose detection** (YOLO11 pose) finds each dancer's joints, then tracks and smooths them so the
   skeletons move naturally. Skeletons fade in and out instead of popping, left/right swaps are
   undone, the skull stays attached to the shoulders, and feet stay planted. It's tuned for a camera
   mounted high in a corner with most people facing *away* from it: the skull is placed from the
   ears or shoulders when the face isn't visible.
2. **Person segmentation** (YOLO11 seg) outlines each person so they can be erased.
3. **Invisibility cloak.** The camera never moves, so the app keeps learning what the floor looks
   like whenever nobody is standing on a spot. People are replaced with that empty floor, relit to
   match the current party-light colors.
4. **Skeleton rendering.** Ivory bones: skull, ribcage, vertebrae, pelvis, paired forearm and shin
   bones, and finger bones. They're lit by the room (purple party light makes purple bones, dim
   corners make darker bones), matched to the camera's softness and grain, and cast a soft shadow
   on the floor, so they look like they're in the room rather than pasted on. Bones scale with
   distance from the camera. A glowing neon tube style is also available.

5. **30 fps output from a 20 fps camera.** With people erased, the room barely changes between
   frames, so the app draws in-between poses for each skeleton, blending its last two detected
   poses. The picture runs about 0.1 s behind the camera, which AirPlay's own 1-2 s lag hides.

On an M2 Pro it shows a steady 30 fps with a handful of dancers. With 20 or more skeletons on
screen it draws about 18 fps, still evenly paced. Pose detection and person segmentation each
run in their own process on the GPU at the same time. People are erased on a background thread
while the GPU works on the next frame, and distant dancers are drawn with less detail. Between
camera frames the app keeps the GPU lightly busy, because macOS otherwise lowers the GPU clock
and every frame gets about 30% slower. Every 10 seconds it prints a `[perf]` line to the
terminal (camera, analyzed and shown fps, plus timings) so you can check how it's doing.

## Setup

Requires macOS on Apple Silicon (it also runs on CPU/CUDA, just slower) and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd halloween-video-feed
uv sync
cp .env.example .env   # then fill in your camera details
```

### Tapo camera

Tapo cameras only open their RTSP stream once you create a **Camera Account**:
Tapo app → your camera → ⚙️ → *Advanced Settings* → *Camera Account*. On newer firmware, also
turn on *Third-Party Compatibility* if you see it. Put that username and password, plus the camera's
IP, in `.env`. They're separate from your TP-Link cloud login.

Model weights download automatically into `models/` on first run.

## Running it

```bash
uv run party-cam                 # camera from .env
uv run party-cam --source 0      # Mac webcam, for a quick test
```

### Getting it onto the TV

**HDMI cable (recommended).** AirPlay compresses the whole picture to stream it over Wi-Fi, so
fine textures like a shaggy rug break into blocks, and it gets worse when a house full of phones
shares the Wi-Fi. A cable has none of that. It also lets the Mac run with the lid closed: macOS
keeps a closed MacBook awake when it's on **power** with a **wired** display attached, and the TV
becomes its only screen.

1. Plug in the charger and the HDMI cable. Set the TV to that HDMI input.
2. Start the party cam fullscreen:

   ```bash
   uv run party-night --fullscreen
   ```

3. Close the lid if you're short on space. Give the Mac some airflow: keep the vents on the bottom
   sides and behind the hinge clear, don't enclose it, and use a hard surface or a vertical stand.
   The `[perf]` line shows macOS's thermal state; anything past `nominal` means it's getting warm
   and will start slowing down.

**AirPlay.** Control Center → **Screen Mirroring** → your TV → **Use As Separate Display**. Then run
`uv run party-night`, drag the window onto the TV, and press `f`. The lid has to stay open.

`party-night` is the launcher for the actual party:
- It runs `party-cam` (any options are passed through). `party-cam` keeps the Mac and the TV
  awake on its own while it runs; the launcher also keeps them awake between restarts.
- If the app crashes, or freezes for 45 seconds, it restarts it. With `--fullscreen` it comes back
  fullscreen on its own. That's on the TV when the lid is closed, since the TV is the only screen.
- Quitting with `q` or Ctrl-C stops it for good. With the lid closed, use an external keyboard,
  or open the lid.
- Everything is logged to `local/logs/` so you can check the morning after.

### Before guests arrive

- **Let it learn the empty floor.** Start the app with nobody on the dance floor for a few seconds.
  The empty-floor image is saved to `local/` and reused next time. Press `b` to re-learn it if the
  camera gets bumped.
- **Ignore the skeleton DJ prop.** If the model detects your decorative skeleton as a person, press
  `z`, drag a box around it, and press Enter. Zones are saved in `local/`. Press `x` to clear them.
- Press `h` to see FPS, how many dancers are detected, and how much of the floor has been learned.

### On-screen messages

About every 30 seconds, a funny message fades in at the bottom of the screen for 8 seconds,
like "TIBIA HONEST, YOU SHOULD BE DANCING". There are over 220 messages, and each list is
shuffled, so a message typically doesn't come back for over an hour.

Half the time, the message fits the moment:

- **How busy the floor is.** An empty floor, one lone dancer, a few people, a crowd, or a packed floor of 20 or more.
- **The time of night.** Early evening, the witching hour around midnight, after 1 AM, and last call.
- **Live numbers.** "17 SKELETONS ON THE FLOOR. BE NUMBER 18." or "IT'S 11:42 PM. DO YOU KNOW WHERE YOUR BONES ARE?"

Edit or add your own in [partycam/messages.py](partycam/messages.py). Use `--banner-every 45` to
show them less often, or press `j` to turn them off.

### Surprise cameos

Every few minutes (about 3 on average, never while a message is up), something brief happens:

- **Bats.** One bat, or a small group, flits across the top of the picture along the ceiling in a
  few seconds. They're soft dark silhouettes, so they read as something in the room.
- **A skeleton dog.** It fades in at one end of the dance floor, trots partway across, stops to
  look up at the dancers and bark with its tail wagging, then trots on and fades out. It's lit by the
  room like the dancers, casts a shadow, and walks behind or in front of people depending on where
  they stand. It learns where the floor is and how big things look there from where dancers'
  feet have been, so it walks on the floor at the right size.

Bats come up about twice as often as the dog, and the dog never appears twice in a row. Press `v`
or `d` to bring one on right away. Use `--cameo-every 300` to make them rarer, or `--no-cameos` to
turn them off (`v` and `d` still work).

### Missing-person posters

Put poster images (JPG, PNG or WebP) in `local/missing/`. About every 10 minutes, one cuts in
like an emergency broadcast: a burst of static, then the poster for 15 seconds under a red
"MISSING PERSON ALERT" bar, with "FULL CASE FILE POSTED IN THE RESTROOMS" along the bottom. It
zooms in slowly while it's up, then cuts back out through static. The first one shows up 1½ to 3
minutes after starting, so you can check it worked.

Every poster shows once before any repeats. Posters, banner messages and cameos wait for each
other, so only one is on screen at a time. Press `p` to cut to the next poster right away, for
example when someone whose poster it is walks in. Use `--poster-every 900` to show them less
often, or `--no-posters` to turn them off (`p` still works). `local/` is gitignored, so your
guests' photos never get committed.

### Recording a test clip

To tune the look without a live party, record a raw clip from the camera and replay it:

```bash
uv run party-cam-record --seconds 90
uv run party-cam --source local/clips/clip_<timestamp>.mp4
```

Clips are saved in `local/clips/` (gitignored) and loop when replayed. Recording uses `ffmpeg` if
it's installed (`brew install ffmpeg`), which saves the camera's stream as-is with almost no CPU.

### Hotkeys

| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | Background: invisible dancers / dimmed room / black void / spectral edges |
| `c` | Skeleton style: bone, multi-color neon, ecto green, phantom, inferno |
| `s` / `r` | Toggle skulls / ribcages |
| `t` | Phantom trail length for the current style (medium → long → short → off). Off by default for bone, medium for neon |
| `j` | Toggle the "join the dance floor" banner |
| `v` / `d` | Send in the bats / the skeleton dog now |
| `p` | Cut to the next missing-person poster now |
| `b` | Re-learn the empty floor |
| `z` / `x` | Add an ignore zone / clear zones |
| `m` | Mirror |
| `f` | Fullscreen (on whichever display the window is on) |
| `h` | Help and FPS overlay |
| `q` / `Esc` | Quit |

### Useful flags

- `--imgsz 960 --seg-imgsz 640`: faster, for a small party. The defaults (1280 and 960) are tuned
  for a packed floor, where they find and erase many more partly hidden people.
- `--room-brightness 0.6`: darker room, so the skeletons pop more.
- `--fps 0`: draw only on new camera frames, with no in-between poses.
- `--seg-every 1`: run person segmentation on every frame instead of every other one. Erases
  fast movers slightly better, at the cost of a few FPS with a big crowd.
- `--width 1280`: lighter on the CPU if you need more FPS.
- `--no-gpu-warm`: let the GPU idle between frames. Saves power on battery, but frames get
  slower after about a minute. Keep the Mac plugged in for the party instead.

## Tests

```bash
uv run pytest
```

About 80 tests that run in a few seconds, with no camera or GPU needed. They cover the logic that's
easy to break without noticing: credential redaction in logs, the ignore-zone rule, skeleton
tracking (fades, left/right swaps, smoothing), the in-between poses, message rotation, cameo
and poster timing, the dog's walk, and the launcher's restart behavior. How it looks and how fast it runs
are checked by eye and by the `[perf]` line instead.

## Privacy

Nothing is uploaded, and nothing is recorded unless you run `party-cam-record`. Video stays on
your Mac and goes only to your TV over AirPlay.
Your camera credentials stay in `.env`. The learned floor image, ignore zones, missing-person posters,
and any test clips stay in `local/`. Both are gitignored.

## License

MIT
