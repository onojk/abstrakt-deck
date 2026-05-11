# abstrakt-deck

[![CI](https://github.com/onojk/abstrakt-deck/actions/workflows/ci.yml/badge.svg)](https://github.com/onojk/abstrakt-deck/actions/workflows/ci.yml)

Native Linux desktop kaleidoscope visualizer and 4K music video generator built with Rust + wgpu. Load any audio file (or use live mic), configure the visualizer with parameter sliders, locks, and three autonomous chaos modes — then export a 4K UHD music video with synced audio in a single click. Multi-shape 3D geometry wrapped with procedural or image painters, folded through configurable kaleido symmetry, with audio-driven beat detection and bass-energy reactivity. Sister project to [abstrakt-engine](https://github.com/onojk/abstrakt-engine) (the Android version).

## Example output

Music videos generated with abstrakt-deck:
- [Velvet Numbers - Made with abstrakt-deck 4K Visualizer](https://www.youtube.com/watch?v=be_Pq0-iFvk)

More examples on the [official artist channel](https://www.youtube.com/channel/UCED3b70ET1LOQu83Jp8w1Ig).

## Features

### Geometry
- 4 shape types: Cylinder, Sphere, Cube, Tetrahedron
- Configurable kaleido fold count (2–24)
- Variable kaleido zoom (0.3–1.5) with bass-reactive modulation
- Variable rotation speed (0–4×, freezable)

### Frame
- 8 frame variants: None, Circle, Square, Rounded, Hexagon, Octagon, Flower, Star (keys 1–8)
- Variable frame size (0.4–1.0)
- Hue-shiftable frame color

### Painters
- **HueStripe**: scrolling rainbow vertical bands
- **Spiral**: radial color spiral with arm rotation
- **Plasma**: flowing demoscene-style color blobs
- **Skin**: user-loaded image (File → Open Skin…), auto-cropped center 16:1 strip, resized to 4096×256 with mipmaps. The Skin section of the parameter panel (press `M` to toggle panel) shows a thumbnail with a draggable yellow rectangle for adjusting vertical crop offset.

### Effects
- Color invert
- Colorize tint with intensity blend
- Sine-wave UV distortion (animated, amplitude + frequency control)
- **Contrast** (0–2.0) and **Saturation** (0–2.0) post-processing
- **Contrast passes** (1–6 iterations) — multiple clamped passes produce hard color edges from gradients, a posterization effect

### Autonomous modes

Three independent modes that automate parameter changes:

- **Random Mode** (`N`): timer-based parameter cycling. Aggressiveness slider controls timer frequency.
- **Reactive Mode** (`B`): audio-triggered painter/parameter changes on detected beats. Subtle and musical.
- **Party Mode** (`Y`): aggressive full-parameter reroll on strong beats — the "shuffle everything" option. Aggressiveness controls trigger threshold: lower values fire only on big musical changes (verse-to-chorus transitions), higher values on minor variations.

Modes can be combined. All three default OFF.

### Parameter locks

Click the 🔒/🔓 icon next to any parameter slider or dropdown to lock that parameter from being changed by Random/Party modes. Useful for skin-dominant Party Mode: lock the Painter to "Skin" and chaos modes will randomize everything else while your skin stays visible. Locks are saved with presets.

### Audio reactivity
- System audio capture via PipeWire monitor source (falls back to default input device)
- FFT-based beat detection → shape shake
- Bass energy (60–200 Hz) → kaleido zoom pulse
- Load an audio file for playback-synced export (see below)

### Input
- Keyboard parameter control (see Controls below)
- Parameter panel with sliders, dropdowns, and mode toggles (`M` to toggle)
- MIDI Control Change routing (works with VMPK or any USB MIDI controller)
- Save/load preset to `~/.config/abstrakt-deck/preset.json`
- Press `F12` to record the live visualizer to MP4 (saves to `~/Videos/abstrakt-deck/`)

## Music video export

Generate complete 4K music videos with one click:

1. **Load audio**: File → Open Audio… (supports WAV, FLAC, MP3, OGG via Symphonia)
2. **Configure visualizer**: set shape, painter, frame, effects, modes, locks
3. **Set export resolution**: 480p / 720p / 1080p / 4K UHD (in the Export section of the panel)
4. **Set framerate**: 30 fps or 60 fps
5. **Click Export…** in the panel (disabled until audio is loaded)
6. **Choose output path** when the save dialog appears
7. Visualizer renders all frames offline at the chosen resolution, then ffmpeg muxes with audio
8. Final output: H.264 + AAC MP4, YouTube-ready

Export details:
- Frame PNGs are stored in `~/.cache/abstrakt-deck/exports/` during render and auto-deleted on success
- **Headless mode** (checkbox in the Export panel) skips the live preview blit for ~15% faster 4K render
- Audio energies are EMA-smoothed during export to eliminate FFT window-shift jitter
- ffmpeg flags: `-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest`
- Requires `ffmpeg` installed (included in the apt dependencies below)

Typical export times on Intel Lunar Lake iGPU:

| Resolution | Framerate | Render (4:30 song) | Mux |
|------------|-----------|-------------------|-----|
| 720p | 30 fps | ~3 min | ~3 min |
| 1080p | 60 fps | ~5 min | ~5 min |
| 4K UHD | 30 fps | ~7 min | ~10 min |
| 4K UHD | 60 fps | ~10 min | ~15 min |

## Requirements

- Linux with PipeWire (Ubuntu 25.10+ recommended)
- Vulkan-capable GPU (Intel integrated graphics work fine)
- Rust 1.83+ (install via [rustup](https://rustup.rs/))

System libraries:

    sudo apt install -y \
        build-essential pkg-config \
        libasound2-dev libudev-dev \
        libx11-dev libxi-dev libxrandr-dev libxcursor-dev libxinerama-dev \
        libwayland-dev libxkbcommon-dev \
        pipewire-alsa \
        ffmpeg

## Build and run

    git clone https://github.com/onojk/abstrakt-deck.git
    cd abstrakt-deck
    cargo run --release

The visualizer opens in a 1280×720 window. Click the window for focus before using keyboard controls.

## Controls

### Parameter panel

Press `M` to toggle the parameter panel (right sidebar). All sliders, dropdowns, mode toggles, and the Export section live here. Lock icons next to each parameter protect them from Random/Party mode randomization.

### Keyboard

| Key | Action |
|-----|--------|
| `?` | toggle cheat sheet overlay |
| `Shift+Tab` | cycle 3D shape (Cylinder → Sphere → Cube → Tetrahedron) |
| `P` | cycle painter (HueStripe → Spiral → Plasma → Skin) |
| `[` / `]` | decrease / increase kaleido fold count (2 to 24) |
| `Z` / `X` | decrease / increase kaleido zoom (0.3 to 1.5) |
| `,` / `.` | decrease / increase rotation speed (0 to 4×) |
| `1` – `8` | frame shape (None, Circle, Square, Rounded, Hexagon, Octagon, Flower, Star) |
| `-` / `=` | decrease / increase frame size |
| `R` / `G` | shift frame color hue |
| `I` | toggle color invert |
| `T` | toggle colorize tint |
| `;` | cycle colorize hue (+30°) |
| `9` / `0` | decrease / increase colorize intensity |
| `D` | toggle distortion |
| `Q` / `W` | decrease / increase distortion amplitude |
| `E` / `F` | decrease / increase distortion frequency |
| `/` / `'` | decrease / increase bass-zoom intensity |
| `Space` | toggle beat-reactive shake |
| `M` | toggle parameter panel |
| `N` | toggle Random Mode |
| `B` | toggle Reactive Mode |
| `Y` | toggle Party Mode |
| `F11` | toggle fullscreen (borderless, current monitor) |
| `F12` | toggle live video recording (saves to `~/Videos/abstrakt-deck/`) |
| `Ctrl+S` | save preset to `~/.config/abstrakt-deck/preset.json` |
| `Ctrl+L` | load preset |
| `Esc` | exit |

Export is via **File → Export…** button in the parameter panel (not a keybinding).

### MIDI

Plug in any USB MIDI controller, or use a virtual one like [VMPK](https://vmpk.sourceforge.io/). The first connected port is auto-selected. Default CC mappings:

| CC | Action | Notes |
|----|--------|-------|
| 1 | fold count | mod wheel |
| 5 | bass-zoom intensity | |
| 7 | kaleido zoom | volume |
| 10 | rotation speed | pan |
| 64 | cycle frame shape (≥ 64 triggers) | sustain |
| 65 | cycle 3D shape | portamento |
| 66 | cycle painter | sostenuto |
| 71 | frame size | resonance |
| 74 | frame color hue | cutoff |
| 76 | colorize hue | vibrato rate |
| 80 / 81 / 82 | distortion toggle / amplitude / frequency | |
| 91 / 93 / 92 | invert toggle / colorize toggle / colorize intensity | |
| Note On | trigger shake | velocity = strength |

## Architecture

The render pipeline is five passes per frame:

1. **Painter pass** — selected painter shader writes to a 4096×256 RGBA offscreen texture
2. **Shape pass** — 3D mesh rendered with the painter texture wrapped on its surface, output to a screen-resolution shape FBO. Distortion, invert, colorize, contrast, and saturation effects apply here.
3. **Kaleido pass** — fragment shader samples the shape FBO with polar-coordinate folding. Zoom is modulated by smoothed bass energy.
4. **Frame pass** — SDF-based frame mask composited over the kaleido FBO, writing to the scene texture.
5. **Blit pass** — copies the scene texture to the swapchain. When recording is active, the scene texture is also copied to a CPU-readable buffer and piped to an `ffmpeg` subprocess as raw RGBA frames.

For music video export, per-frame FBOs are created at the target export resolution (up to 3840×2160) and the scene is rendered offline frame-by-frame. PNG frames are saved by a worker thread; after all frames are written, a second thread calls `ffmpeg` to mux with audio.

Audio runs on a separate thread (cpal callback). Beat events flow to the render thread via crossbeam-channel. MIDI runs on its own thread (midir callback) with the same pattern.

## Status

Slices 1–24l complete (including 24f-hotfix). The visualizer is feature-complete for music video generation: audio loading, 3D kaleidoscope with full parameter control, three autonomous modes with per-parameter locks, and a full offline 4K export pipeline with ffmpeg muxing.

Possible future slices: multi-monitor support, additional painters (Audio Paint, Print Head), additional shapes.

## Related

- [abstrakt-engine](https://github.com/onojk/abstrakt-engine) — Android version

## License

MIT
