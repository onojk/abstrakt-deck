//! Lyric corpus: stores lines/words for the text overlay.
//!
//! Slice 1: deterministic accessor `fragment(i)`.
//! Slice 2 will add an audio-reactive sampler on top.

pub struct LyricCorpus {
    lines: Vec<String>,
}

impl LyricCorpus {
    pub fn from_text(text: &str) -> Self {
        let lines = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        Self { lines }
    }

    /// Returns the i-th line (wraps around if i ≥ len).
    /// Returns an empty string if the corpus is empty.
    pub fn fragment(&self, i: usize) -> &str {
        if self.lines.is_empty() {
            return "";
        }
        &self.lines[i % self.lines.len()]
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}
