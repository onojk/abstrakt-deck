/// Quantized settings signature + JSON-persisted bad-combo blocklist.
///
/// `SettingsSignature` stores only quantized primitives (ints + bools) so it
/// derives `Eq`/`Hash` cleanly and raw floats from VisualParams never recur
/// exactly.  The caller in `main.rs` builds a signature from VisualParams;
/// this module has no dependency on VisualParams or wgpu.
///
/// Persistence: `dirs::config_dir()/abstrakt-deck/blocklist.json`
/// (same directory as `preset.json`, mirrors `preset_path()` in main.rs).
/// Written synchronously on every `add()` — records are small and writes are
/// rare (one per detected collapse), so the overhead is negligible.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── Bucket constants (floats are quantized before storage) ───────────────────

/// Step size for fold_count buckets.  fold_count ∈ [2, 24]; step 2 → ~11 buckets.
pub const FOLD_BUCKET: f32 = 2.0;
/// Step size for zoom buckets.  zoom ∈ [0.3, 1.5]; step 0.1 → ~12 buckets.
pub const ZOOM_BUCKET: f32 = 0.1;
/// Step size for colorize_hue buckets.  hue ∈ [0, 360); step 30 → 12 buckets.
pub const HUE_BUCKET: f32 = 30.0;

// ── Settings signature ────────────────────────────────────────────────────────

/// Quantized description of the visual settings that most influence whether a
/// frame collapses to black or white.  All fields are primitive so the type
/// derives `Eq` and `Hash` without any special treatment of floats.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SettingsSignature {
    /// `ShapeKind as u32` — caller casts the discriminant.
    pub shape: u32,
    /// `PainterKind as u32`.
    pub painter: u32,
    /// `FrameShape as u32`.
    pub frame_shape: u32,
    /// `PaletteMode as u32`.
    pub palette_mode: u32,
    /// `(fold_count / FOLD_BUCKET).floor() as i32`
    pub fold_bucket: i32,
    /// `(zoom / ZOOM_BUCKET).floor() as i32`
    pub zoom_bucket: i32,
    /// `(colorize_hue / HUE_BUCKET).floor() as i32`
    pub hue_bucket: i32,
    pub invert: bool,
}

impl SettingsSignature {
    /// Build a signature from already-extracted primitive values.
    /// Raw floats are bucketed here; enum variants must be cast to u32 by the caller.
    pub fn new(
        shape: u32,
        painter: u32,
        frame_shape: u32,
        palette_mode: u32,
        fold_count: f32,
        zoom: f32,
        colorize_hue: f32,
        invert: bool,
    ) -> Self {
        Self {
            shape,
            painter,
            frame_shape,
            palette_mode,
            invert,
            fold_bucket: (fold_count / FOLD_BUCKET).floor() as i32,
            zoom_bucket: (zoom / ZOOM_BUCKET).floor() as i32,
            hue_bucket: (colorize_hue / HUE_BUCKET).floor() as i32,
        }
    }
}

// ── Blocklist record ──────────────────────────────────────────────────────────

/// One entry in the blocklist JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlocklistRecord {
    pub signature: SettingsSignature,
    /// `"BLACK"` or `"WHITE"` — matches `Degeneracy` variant names.
    pub reason: String,
    /// RFC-3339 timestamp (`chrono::Utc::now().to_rfc3339()` in the caller).
    pub timestamp: String,
    pub mean_l: f32,
    pub degenerate_frac: f32,
}

// ── Persistent store ──────────────────────────────────────────────────────────

/// In-memory + on-disk blocklist.  Load once at startup; `add()` persists
/// immediately so a crash after detection still preserves the record.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Blocklist {
    records: Vec<BlocklistRecord>,
}

impl Blocklist {
    /// Path: `~/.config/abstrakt-deck/blocklist.json` (platform-appropriate via `dirs`).
    pub fn path() -> Option<PathBuf> {
        let mut p = dirs::config_dir()?;
        p.push("abstrakt-deck");
        p.push("blocklist.json");
        Some(p)
    }

    /// Load from disk, returning an empty blocklist on any error (missing file,
    /// parse failure, etc.).  Logs a warning on parse failure so bad data is
    /// visible but never fatal.
    pub fn load() -> Self {
        let path = match Self::path() {
            Some(p) => p,
            None => {
                log::warn!("[blocklist] cannot determine config dir — blocklist disabled");
                return Self::default();
            }
        };
        match std::fs::read_to_string(&path) {
            Ok(json) => match serde_json::from_str::<Self>(&json) {
                Ok(bl) => {
                    log::info!("[blocklist] loaded {} record(s) from {}", bl.records.len(), path.display());
                    bl
                }
                Err(e) => {
                    log::warn!("[blocklist] parse error ({e}) — starting with empty list");
                    Self::default()
                }
            },
            Err(_) => {
                // Missing file is the normal first-run case; not a warning.
                Self::default()
            }
        }
    }

    /// `true` if any stored record matches `sig` exactly (after quantization).
    pub fn is_blocked(&self, sig: &SettingsSignature) -> bool {
        self.records.iter().any(|r| &r.signature == sig)
    }

    /// How many records are currently stored.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Add a record (deduped by signature) and persist immediately.
    ///
    /// Returns `Err` if the file write fails, but **always** keeps the record
    /// in RAM so detection continues to work for the rest of the session even
    /// if the disk is full / read-only.
    pub fn add(&mut self, record: BlocklistRecord) -> Result<(), String> {
        if !self.is_blocked(&record.signature) {
            log::info!(
                "[blocklist] adding {reason} record for sig shape={s} painter={p}",
                reason = record.reason,
                s = record.signature.shape,
                p = record.signature.painter,
            );
            self.records.push(record);
        }
        self.save()
    }

    /// All records (read-only slice), newest last.
    pub fn records(&self) -> &[BlocklistRecord] {
        &self.records
    }

    /// Persist the full in-memory state to disk.  Called automatically by `add()`.
    fn save(&self) -> Result<(), String> {
        let path = Self::path().ok_or_else(|| "no config dir".to_string())?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, json).map_err(|e| e.to_string())
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn sig_a() -> SettingsSignature {
        SettingsSignature::new(1, 2, 3, 0, 12.0, 1.0, 180.0, true)
    }

    #[test]
    fn buckets_absorb_small_float_variation() {
        // Values that differ by less than one bucket step must hash to the same sig.
        let a = SettingsSignature::new(1, 2, 3, 0, 12.0, 1.00, 180.0, true);
        let b = SettingsSignature::new(1, 2, 3, 0, 13.5, 1.09, 199.0, true);
        // fold: 12/2=6 vs 13.5/2=6.75→6  same bucket
        // zoom: 1.0/0.1=10 vs 1.09/0.1=10.9→10  same bucket
        // hue:  180/30=6   vs 199/30=6.63→6      same bucket
        assert_eq!(a, b, "near values should land in the same bucket");
    }

    #[test]
    fn distinct_bucket_values_differ() {
        let a = SettingsSignature::new(1, 2, 3, 0, 12.0, 1.0, 180.0, true);
        let b = SettingsSignature::new(1, 2, 3, 0, 12.0, 1.0, 210.0, true); // hue crosses 30° boundary
        assert_ne!(a, b, "hue in next bucket should produce different sig");
    }

    #[test]
    fn invert_flag_distinguishes_sigs() {
        let a = SettingsSignature::new(1, 2, 3, 0, 12.0, 1.0, 180.0, true);
        let b = SettingsSignature::new(1, 2, 3, 0, 12.0, 1.0, 180.0, false);
        assert_ne!(a, b);
    }

    #[test]
    fn is_blocked_in_ram_only() {
        let mut bl = Blocklist::default();
        assert!(!bl.is_blocked(&sig_a()));

        // Manually push without saving (avoids touching the filesystem in tests).
        bl.records.push(BlocklistRecord {
            signature: sig_a(),
            reason: "BLACK".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            mean_l: 0.01,
            degenerate_frac: 0.98,
        });
        assert!(bl.is_blocked(&sig_a()));
    }

    #[test]
    fn dedup_prevents_duplicate_records() {
        let mut bl = Blocklist::default();
        // Manually push first record so we don't hit the filesystem.
        bl.records.push(BlocklistRecord {
            signature: sig_a(),
            reason: "BLACK".into(),
            timestamp: "t".into(),
            mean_l: 0.0,
            degenerate_frac: 1.0,
        });
        let before = bl.len();
        // `add()` must NOT insert a duplicate (it will still call save() and may
        // fail because we have no real config dir in CI, but the dedup check
        // is observable via len()).  We ignore the save error here.
        let _ = bl.add(BlocklistRecord {
            signature: sig_a(),
            reason: "BLACK".into(),
            timestamp: "t2".into(),
            mean_l: 0.0,
            degenerate_frac: 1.0,
        });
        assert_eq!(bl.len(), before, "duplicate sig must not be inserted");
    }

    #[test]
    fn buckets_are_floor_not_round() {
        // fold_count = 11.9 → 11.9/2 = 5.95 → floor = 5
        // fold_count = 12.0 → 12.0/2 = 6.0  → floor = 6
        let a = SettingsSignature::new(0, 0, 0, 0, 11.9, 1.0, 0.0, false);
        let b = SettingsSignature::new(0, 0, 0, 0, 12.0, 1.0, 0.0, false);
        assert_ne!(a.fold_bucket, b.fold_bucket, "11.9 and 12.0 must land in different fold buckets");
    }
}
