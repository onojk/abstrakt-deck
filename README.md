# abstrakt-deck

[![CI](https://github.com/onojk/abstrakt-deck/actions/workflows/ci.yml/badge.svg)](https://github.com/onojk/abstrakt-deck/actions/workflows/ci.yml)

Native Linux desktop audio-reactive kaleidoscope visualizer. Sister project to [onojk/abstrakt-engine](https://github.com/onojk/abstrakt-engine), which is the Android version.

Built with Rust + wgpu. Targets Linux (Vulkan).

## Run

    cargo run --release

Press Escape or close the window to exit.

## Keyboard controls

| Key | Action |
|-----|--------|
| `[ ]` | Fold count (2–24) |
| `Z X` | Kaleido zoom (0.30–1.50) |
| `, .` | Rotation speed (0–4×) |
| `1–7` | Frame shape (None/Circle/Square/Rounded/Hexagon/Octagon/Star) |
| `- =` | Frame size |
| `R G B` | Cycle frame color hue |
| `Space` | Toggle beat-reactive shake |
| `Tab` | Cycle shape (Cylinder → Sphere → Cube → Tetrahedron) |
| `/ '` | Bass-zoom intensity |
| `I` | Toggle color invert |
| `T` | Toggle colorize tint |
| `;` | Cycle colorize hue (+30°) |
| `9 0` | Colorize intensity |
| `D` | Toggle distortion |
| `Q W` | Distortion amplitude (0–0.5) |
| `E F` | Distortion frequency (0.5–8) |
| `Ctrl+S` | Save preset to `~/.config/abstrakt-deck/preset.json` |
| `Ctrl+L` | Load preset from same file |
| `Esc` | Exit |

## Slices completed

- [x] Slice 1: window + wgpu clear color + FPS in title
- [x] Slice 2: fullscreen triangle + WGSL shader + uniforms
- [x] Slice 3: Hue Stripe painter shader
- [x] Slice 4: painter FBO (2048×1024) + composite blit pass
- [x] Slice 5: 3D cylinder mesh + painter surface texture + depth + MVP
- [x] Slice 6: kaleido fold pass (polar-coordinate fold)
- [x] Slice 7: frame mask overlay (hexagon SDF, neon cyan, anti-aliased)
- [x] Slice 8: audio capture (cpal/ALSA/PipeWire) + FFT beat detection + shake
- [x] Slice 9: keyboard-driven runtime control
- [x] Slice 10: MIDI input via midir (VMPK CC→params, Note On→shake)
- [x] Slice 11: multi-shape (cylinder, sphere, cube, tetrahedron)
- [x] Slice 12: bass-reactive kaleido zoom (attack/decay envelope)
- [x] Slice 13: color invert + colorize tint in shape shader
- [x] Slice 13.5: colorize intensity blend
- [x] Slice 14: distortion effect (sine-wave UV warp)
- [x] Slice 15: preset save/load (`~/.config/abstrakt-deck/preset.json`)
- [x] Slice 16: unit tests + GitHub Actions CI

## License

MIT
