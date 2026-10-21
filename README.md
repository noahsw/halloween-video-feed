# Halloween Skeleton Party Cam 💀🕺

Turns everyone on your dance floor into dancing skeletons, live, so you can AirPlay it to a TV
somewhere else in the house and lure people onto the floor.

In the default mode the **people disappear entirely**: the app paints the empty dance floor over
them, so all that's left is skeletons dancing in your real room. A rotating banner
("THE DEAD ARE DANCING. WHAT'S YOUR EXCUSE?") nudges the audience to come join.

Built for a Mac with Apple Silicon and a TP-Link Tapo camera, but any RTSP camera or webcam works.

## How it works

1. **Pose detection** (YOLO11 pose) finds each dancer's joints, then tracks and smooths them so the skeletons don't jitter.
   It's tuned for a camera mounted high in a corner with most people facing *away* from it: the
   skull is placed from the ears or shoulders when the face isn't visible.
2. **Person segmentation** (YOLO11 seg) outlines each person so they can be erased.
3. **Invisibility cloak.** The camera never moves, so the app keeps learning what the floor looks
   like whenever nobody is standing on a spot. People are replaced with that empty floor, relit to
   match the current party-light colors.
4. **Skeleton rendering.** Ivory bones with a colored aura: skull, ribcage, vertebrae, pelvis,
   paired forearm and shin bones, and finger bones. Bones scale with distance from the camera.
   A neon tube style is also available.

On an M2 Pro it keeps up with a 15 fps camera at 1080p output.

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
3. Start fullscreen on the TV, matching it by name:

   ```bash
   caffeinate -dis uv run party-cam --display "Living Room"
   ```

   `caffeinate` keeps the Mac from sleeping mid-party. You can also press `d` to move the
   fullscreen window to the next display.

### Before guests arrive

- **Let it learn the empty floor.** Start the app with nobody on the dance floor for a few seconds.
  The empty-floor image is saved to `local/` and reused next time. Press `b` to re-learn it if the
  camera gets bumped.
- **Ignore the skeleton DJ prop.** If the model detects your decorative skeleton as a person, press
  `z`, drag a box around it, and press Enter. Zones are saved in `local/`. Press `x` to clear them.
- Press `h` to see FPS, how many dancers are detected, and how much of the floor has been learned.

### Hotkeys

| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | Background: invisible dancers / dimmed room / black void / spectral edges |
| `c` | Skeleton style: bone, multi-color neon, ecto green, phantom, inferno |
| `s` / `r` | Toggle skulls / ribcages |
| `t` | Phantom trail length (medium → long → short → off) |
| `j` | Toggle the "join the dance floor" banner |
| `b` | Re-learn the empty floor |
| `z` / `x` | Add an ignore zone / clear zones |
| `m` | Mirror |
| `f` / `d` | Fullscreen / move fullscreen to next display |
| `h` | Help and FPS overlay |
| `q` / `Esc` | Quit |

### Useful flags

- `--imgsz 1280`: finds smaller, farther dancers, at some cost to FPS.
- `--pose-model yolo11m-pose.pt`: more accurate pose model, about half the speed.
- `--room-brightness 0.6`: darker room, so the skeletons pop more.
- `--width 1280`: lighter on the CPU if you need more FPS.

## Privacy

Nothing is recorded or uploaded. Video stays on your Mac and goes only to your TV over AirPlay.
Your camera credentials stay in `.env`, and the learned floor image and ignore zones stay in
`local/`. Both are gitignored.

## License

MIT
