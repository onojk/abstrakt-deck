# abstrakt-deck

Native Linux desktop audio-reactive kaleidoscope visualizer. Sister project to [onojk/abstrakt-engine](https://github.com/onojk/abstrakt-engine), which is the Android version.

Built with Rust + wgpu. Targets Linux (Vulkan), with future cross-platform potential.

## Status

Slice 1: window opens, wgpu renders a clear color, FPS displayed in title.

## Run

    cargo run --release

Press Escape or close the window to exit.

## Roadmap

- [x] Slice 1: window + clear color + FPS
- [ ] Slice 2: fullscreen quad + WGSL shader
- [ ] Slice 3: painter texture (port from abstrakt-engine)
- [ ] Slice 4: audio capture via cpal + ALSA
- [ ] Slice 5: 3D shape rendering (cylinder + sphere)
- [ ] Slice 6: kaleido fold pass
- [ ] Slice 7: frame mask
- [ ] Slice 8: MIDI input via midir
- [ ] Slice 9: multi-monitor support

## License

MIT
