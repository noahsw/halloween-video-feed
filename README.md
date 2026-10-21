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
uv run party-cam --list-displays
```

### Getting it onto the TV (AirPlay)

1. On the Mac: Control Center → **Screen Mirroring** → pick your TV (Apple TV or an AirPlay 2 TV).
2. In the same menu (or System Settings → Displays), choose **Use As Separate Display**, so the TV
   shows only the party cam while you keep using your laptop screen.
3. Start it fullscreen on the TV, matching the TV by name:

   ```bash
   uv run party-night --display "Living Room"
   ```

   `party-night` is the launcher for the actual party:
   - It runs `party-cam` (any options are passed through) and keeps the Mac awake.
   - If the app crashes, or freezes for 45 seconds, it restarts it, fullscreen on the TV again.
   - Quitting with `q` or Ctrl-C stops it for good.
   - Everything is logged to `local/logs/` so you can check the morning after.

   You can also press `d` in the app to move the fullscreen window to the next display.

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
| `b` | Re-learn the empty floor |
| `z` / `x` | Add an ignore zone / clear zones |
| `m` | Mirror |
| `f` / `d` | Fullscreen / move fullscreen to next display |
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

## Privacy

Nothing is uploaded, and nothing is recorded unless you run `party-cam-record`. Video stays on
your Mac and goes only to your TV over AirPlay.
Your camera credentials stay in `.env`. The learned floor image, ignore zones, and any test clips
stay in `local/`. Both are gitignored.

## License

MIT
