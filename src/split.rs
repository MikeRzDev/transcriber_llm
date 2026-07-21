//! Client-side audio chunking: for engines that cannot take arbitrarily
//! long input, and as an opt-in strategy for long recordings. whisper.cpp
//! windows internally with word-safe seeking, so Auto mode passes audio
//! through whole; Silence/Fixed force client-side chunks. All splitting
//! happens on the decoded f32 samples — nothing is re-encoded, so there
//! is no quality loss.

pub const TARGET_CHUNK_SECS: f32 = 60.0;
/// How far back from a target boundary to hunt for a speech pause
const SEARCH_ZONE_SECS: f32 = 15.0;
/// Overlap used when speech is continuous and the cut may land mid-word
const OVERLAP_SECS: f32 = 0.5;
const FRAME_MS: usize = 25;
/// Frame RMS below this counts as a speech pause
const SILENCE_RMS: f32 = 3e-3;
/// Leftovers shorter than this merge into the previous chunk
const MIN_TAIL_SECS: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SplitMode {
    /// Let the engine window internally (whisper.cpp is word-safe);
    /// chunk only when the engine declares a hard input limit
    #[default]
    Auto,
    /// Chunk near the target length, cutting at the quietest speech pause
    Silence,
    /// Chunk at exactly the target length with a small overlap
    Fixed,
}

impl SplitMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "silence" => Some(Self::Silence),
            "fixed" => Some(Self::Fixed),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Silence => "silence",
            Self::Fixed => "fixed",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "auto — engine-managed windows",
            Self::Silence => "silence — cut at speech pauses (~60 s chunks)",
            Self::Fixed => "fixed — 60 s chunks, 0.5 s overlap",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            Self::Auto => Self::Silence,
            Self::Silence => Self::Fixed,
            Self::Fixed => Self::Auto,
        }
    }
}

/// What an inference engine accepts. The whisper.cpp engine takes 16 kHz
/// mono f32 of any length (it windows internally); a future engine with a
/// hard input limit sets `max_window_secs` and chunking becomes mandatory
/// regardless of the configured mode.
pub struct EngineCaps {
    pub sample_rate: u32,
    pub max_window_secs: Option<f32>,
}

impl EngineCaps {
    pub fn whisper() -> Self {
        Self {
            sample_rate: 16_000,
            max_window_secs: None,
        }
    }
}

/// One planned chunk, in sample indices. `emit_from` marks where fresh
/// content starts: segments ending at or before it were already produced
/// by the previous chunk (overlap) and must be dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
    pub emit_from: usize,
}

pub fn plan_chunks(samples: &[f32], caps: &EngineCaps, mode: SplitMode) -> Vec<Chunk> {
    let n = samples.len();
    let rate = caps.sample_rate as usize;
    let window_secs = match (mode, caps.max_window_secs) {
        (SplitMode::Auto, None) => {
            return vec![Chunk {
                start: 0,
                end: n,
                emit_from: 0,
            }];
        }
        (SplitMode::Auto, Some(max)) => max,
        (_, Some(max)) => TARGET_CHUNK_SECS.min(max),
        (_, None) => TARGET_CHUNK_SECS,
    };
    let window = (window_secs * rate as f32) as usize;
    let zone = (SEARCH_ZONE_SECS * rate as f32) as usize;
    let overlap = (OVERLAP_SECS * rate as f32) as usize;
    let min_tail = (MIN_TAIL_SECS * rate as f32) as usize;
    // A soft target may swallow a short tail; a hard engine limit cannot
    let soft = caps.max_window_secs.is_none();

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut emit_from = 0usize;
    loop {
        let remaining = n - start;
        if remaining <= window || (soft && remaining < window + min_tail) {
            chunks.push(Chunk {
                start,
                end: n,
                emit_from,
            });
            return chunks;
        }
        let hard_end = start + window;
        // Never cut in the first half of a chunk, even if it is all silence
        let zone_start = hard_end.saturating_sub(zone).max(start + window / 2);
        let silence_cut = match mode {
            SplitMode::Fixed => None,
            _ => quietest_point(samples, zone_start, hard_end, rate),
        };
        let (end, next_start, next_emit) = match silence_cut {
            // Clean cut inside a pause: contiguous, nothing duplicated
            Some(cut) => (cut, cut, cut),
            // Continuous speech: cut at the target and re-decode a little
            // overlap so a word straddling the boundary isn't lost
            None => (hard_end, hard_end.saturating_sub(overlap), hard_end),
        };
        chunks.push(Chunk {
            start,
            end,
            emit_from,
        });
        start = next_start;
        emit_from = next_emit;
    }
}

/// Center of the quietest FRAME_MS frame in [from, to), if genuinely quiet.
fn quietest_point(samples: &[f32], from: usize, to: usize, rate: usize) -> Option<usize> {
    let frame = rate * FRAME_MS / 1000;
    if frame == 0 {
        return None;
    }
    let mut best: Option<(f32, usize)> = None;
    let mut i = from;
    while i + frame <= to.min(samples.len()) {
        let rms =
            (samples[i..i + frame].iter().map(|s| s * s).sum::<f32>() / frame as f32).sqrt();
        if best.map(|(b, _)| rms < b).unwrap_or(true) {
            best = Some((rms, i + frame / 2));
        }
        i += frame;
    }
    best.and_then(|(rms, at)| (rms < SILENCE_RMS).then_some(at))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = 16_000;

    fn secs(s: f32) -> usize {
        (s * RATE as f32) as usize
    }

    /// `spans` of (start_sec, end_sec) hold speech; silence elsewhere.
    fn audio_with_speech(total_secs: f32, spans: &[(f32, f32)]) -> Vec<f32> {
        let mut samples = vec![0.0f32; secs(total_secs)];
        for &(a, b) in spans {
            for (idx, sample) in samples[secs(a)..secs(b)].iter_mut().enumerate() {
                *sample = 0.1 * ((idx as f32) * 0.3).sin();
            }
        }
        samples
    }

    #[test]
    fn auto_without_engine_limit_is_one_chunk() {
        let samples = audio_with_speech(120.0, &[(0.0, 120.0)]);
        let chunks = plan_chunks(&samples, &EngineCaps::whisper(), SplitMode::Auto);
        assert_eq!(
            chunks,
            vec![Chunk {
                start: 0,
                end: samples.len(),
                emit_from: 0
            }]
        );
    }

    #[test]
    fn silence_mode_cuts_in_the_pause() {
        // pause at 50–52 s, inside the 45–60 s search zone
        let samples = audio_with_speech(120.0, &[(0.0, 50.0), (52.0, 120.0)]);
        let chunks = plan_chunks(&samples, &EngineCaps::whisper(), SplitMode::Silence);
        assert!(chunks.len() >= 2);
        let cut = chunks[0].end;
        assert!(
            cut >= secs(50.0) && cut <= secs(52.0),
            "cut at {}s, not in the pause",
            cut / RATE
        );
        // clean cut: contiguous, nothing deduped
        assert_eq!(chunks[1].start, cut);
        assert_eq!(chunks[1].emit_from, cut);
    }

    #[test]
    fn continuous_speech_falls_back_to_overlap() {
        let samples = audio_with_speech(120.0, &[(0.0, 120.0)]);
        let chunks = plan_chunks(&samples, &EngineCaps::whisper(), SplitMode::Silence);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].end, secs(60.0));
        assert_eq!(chunks[1].start, secs(60.0) - secs(0.5));
        // the overlap region re-decodes but is deduped on emit
        assert_eq!(chunks[1].emit_from, secs(60.0));
    }

    #[test]
    fn fixed_mode_ignores_pauses() {
        let samples = audio_with_speech(120.0, &[(0.0, 50.0), (52.0, 120.0)]);
        let chunks = plan_chunks(&samples, &EngineCaps::whisper(), SplitMode::Fixed);
        assert_eq!(chunks[0].end, secs(60.0));
        assert_eq!(chunks[1].start, secs(60.0) - secs(0.5));
    }

    #[test]
    fn short_tail_merges_into_the_last_chunk() {
        let samples = audio_with_speech(61.0, &[(0.0, 61.0)]);
        let chunks = plan_chunks(&samples, &EngineCaps::whisper(), SplitMode::Fixed);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end, samples.len());
    }

    #[test]
    fn hard_engine_limit_caps_every_chunk_with_full_coverage() {
        let caps = EngineCaps {
            sample_rate: 16_000,
            max_window_secs: Some(30.0),
        };
        let samples = audio_with_speech(100.0, &[(0.0, 100.0)]);
        let chunks = plan_chunks(&samples, &caps, SplitMode::Auto);
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert!(
                chunk.end - chunk.start <= secs(30.0),
                "chunk exceeds the engine limit"
            );
        }
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks.last().unwrap().end, samples.len());
        for pair in chunks.windows(2) {
            assert!(pair[1].start <= pair[0].end, "gap between chunks");
            assert_eq!(pair[1].emit_from, pair[0].end, "emit must resume exactly");
        }
    }

    #[test]
    fn split_mode_parse_label_and_cycle() {
        assert_eq!(SplitMode::parse("silence"), Some(SplitMode::Silence));
        assert_eq!(SplitMode::parse("bogus"), None);
        assert_eq!(SplitMode::Auto.next(), SplitMode::Silence);
        assert_eq!(SplitMode::Fixed.next(), SplitMode::Auto);
        for mode in [SplitMode::Auto, SplitMode::Silence, SplitMode::Fixed] {
            assert_eq!(SplitMode::parse(mode.as_str()), Some(mode));
        }
    }
}
