# abstrakt-deck — development conventions

## WGSL / Rust uniform buffer alignment

WGSL uniform address space imposes strict alignment rules that differ from
Rust's `#[repr(C)]`. Mismatches produce a GPU validation crash at runtime —
`cargo check`, `clippy`, and `cargo test` do **not** catch them.

### Rule 1 — never use `vec2<f32>` in a uniform struct on the Rust side

`vec2<f32>` in WGSL has 8-byte alignment. A struct `{ f32, vec2<f32>, f32 }`
lays out at offsets 0 / 8 / 16 (24 bytes total) in WGSL, but the equivalent
Rust `#[repr(C)] struct { f32, f32, f32, f32 }` is 16 bytes. This mismatch
silently corrupts uniform reads.

**Fix:** use flat `f32` fields on both sides; reconstruct vectors in the shader.

### Rule 2 — never use `array<f32, N>` in a uniform struct

WGSL requires each array element in uniform address space to have stride ≥ 16
bytes (one `vec4` slot). `array<f32, N>` has stride 4 and fails validation:

```
Shader validation error: Alignment requirements for address space Uniform
are not met — the array stride 4 is not a multiple of the required alignment 16
```

**Fix:** pack into `array<vec4<f32>, ceil(N/4)>` and use a helper to extract
individual lanes. For N=8 (two vec4s):

```wgsl
struct Foo {
    // ...other f32 fields...
    bands: array<vec4<f32>, 2>,  // bands[0..4] in .xyzw of [0], bands[4..8] in .xyzw of [1]
};

fn band(i: i32) -> f32 {
    let v = foo.bands[i / 4];
    let lane = i % 4;
    if      lane == 0 { return v.x; }
    else if lane == 1 { return v.y; }
    else if lane == 2 { return v.z; }
    else              { return v.w; }
}
```

On the Rust side `[f32; 8]` maps identically (same 32-byte footprint at the
same offset) — no Rust changes needed when fixing the WGSL side.

### Rule 3 — verify alignment after every uniform struct change

Always run `cargo run --release` (or `cargo run`) after adding or modifying a
uniform struct. Static analysis cannot catch GPU validation errors.
