# abstrakt-deck

[![CI](https://github.com/onojk/abstrakt-deck/actions/workflows/ci.yml/badge.svg)](https://github.com/onojk/abstrakt-deck/actions/workflows/ci.yml)

Native Linux desktop audio-reactive kaleidoscope visualizer built with Rust + wgpu. Multi-shape 3D geometry wrapped with procedural painters, folded through configurable kaleido symmetry, with audio-driven beat detection and bass-energy reactivity. Real-time control via keyboard, MIDI, or audio input. Sister project to [abstrakt-engine](https://github.com/onojk/abstrakt-engine) (the Android version).

## Features

### Geometry
- 4 shape types: Cylinder, Sphere, Cube, Tetrahedron
- Configurable kaleido fold count (2–24)
- Variable kaleido zoom (0.3–1.5) with bass-reactive modulation
- Variable rotation speed (0–4×, freezable)

### Frame
- 7 frame variants: None, Circle, Square, Rounded, Hexagon, Octagon, Star
- Variable frame size (0.4–1.0)
- Hue-shiftable frame color

### Painters
- **HueStripe**: scrolling rainbow vertical bands
- **Spiral**: radial color spiral with arm rotation
- **Plasma**: flowing demoscene-style color blobs
- **Skin**: user-loaded image (File → Open Skin…), auto-cropped center 16:1 strip, resized to 4096×256. Press `C` after load to enter crop mode and scroll the strip up/down with `[`/`]`, `Enter` to commit, `Esc` to cancel

### Effects
- Color invert
- Colorize tint with intensity blend
- Sine-wave UV distortion (animated, amplitude + frequency control)

### Audio reactivity
- System audio capture via PipeWire monitor source
- FFT-based beat detection → shape shake
- Bass energy (60–200 Hz) → kaleido zoom pulse

### Input
- Keyboard parameter control
- MIDI Control Change routing (works with VMPK or any USB MIDI controller)
- Save/load preset to `~/.config/abstrakt-deck/preset.json`
- Press `F12` to record the visualizer to MP4 (saves to `~/Videos/abstrakt-deck/`)

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

### Keyboard

| Key | Action |
|-----|--------|
| `?` | toggle cheat sheet overlay (animated slide-in) |
| `Shift+Tab` | cycle 3D shape (Cylinder → Sphere → Cube → Tetrahedron) |
| `P` | cycle painter (HueStripe → Spiral → Plasma → Skin) |
| `[` / `]` | decrease / increase kaleido fold count (2 to 24) |
| `Z` / `X` | decrease / increase kaleido zoom (0.3 to 1.5) |
| `,` / `.` | decrease / increase rotation speed (0 to 4×) |
| `1` – `7` | frame shape (None, Circle, Square, Rounded, Hexagon, Octagon, Star) |
| `-` / `=` | decrease / increase frame size |
| `R` / `G` / `B` | shift frame color hue |
| `I` | toggle color invert |
| `T` | toggle colorize tint |
| `;` | cycle colorize hue (+30°) |
| `9` / `0` | decrease / increase colorize intensity |
| `D` | toggle distortion |
| `Q` / `W` | decrease / increase distortion amplitude |
| `E` / `F` | decrease / increase distortion frequency |
| `/` / `'` | decrease / increase bass-zoom intensity |
| `Space` | toggle beat-reactive shake |
| `F11` | toggle fullscreen (borderless, current monitor) |
| `F12` | toggle video recording (saves to `~/Videos/abstrakt-deck/`) |
| `Ctrl+S` | save preset to `~/.config/abstrakt-deck/preset.json` |
| `Ctrl+L` | load preset |
| `Esc` | exit |

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

1. **Painter pass** — selected painter shader writes to a 2048×1024 RGBA8 offscreen texture
2. **Shape pass** — 3D mesh is rendered with the painter texture wrapped on its surface, output to a screen-resolution shape FBO. Distortion, invert, and colorize effects apply here in the fragment shader.
3. **Kaleido pass** — fragment shader samples the shape FBO with polar-coordinate folding. Zoom is modulated by smoothed bass energy from the audio analyzer.
4. **Frame pass** — fragment shader samples the kaleido FBO and applies an SDF-based frame mask, writing to an intermediate scene texture (not directly to the swapchain).
5. **Blit pass** — copies the scene texture to the swapchain. When recording is active, the scene texture is also copied to a CPU-readable buffer in the same encoder submit, then mapped, stripped of row padding, and piped to an `ffmpeg` subprocess as raw RGBA frames.

Audio runs on a separate thread (cpal callback). Beat events flow to the render thread via crossbeam-channel. MIDI runs on its own thread (midir callback) with the same pattern.

## Status

Slices 1–19 complete. The visualizer is feature-complete for the experimental version goals (audio + MIDI + keyboard control over a multi-shape multi-painter kaleido pipeline with video export).

Possible future slices: multi-monitor support, additional painters (Audio Paint, Print Head), additional shapes.

## License

MIT
