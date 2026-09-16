//! Transcript pane state: streamed segments and scroll/follow behavior.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::transcribe::Segment;

/// Wrapped history is prepared in the update phase, once per new segment.
/// Rendering clones only visible lines, regardless of recording duration.
#[derive(Default)]
pub(crate) struct TranscriptLayout {
    pub width: usize,
    pub names: BTreeMap<u8, String>,
    pub segment_count: usize,
    pub committed: Vec<ratatui::text::Line<'static>>,
    pub partial_source: Option<Segment>,
    pub partial: Vec<ratatui::text::Line<'static>>,
    #[cfg(test)]
    pub wrapped_segments: usize,
}

pub struct TranscriptState {
    pub segments: Vec<Segment>,
    pub partial: Option<Segment>,
    pub scroll: usize,
    /// Stick to the newest line while segments stream in
    pub follow: bool,
    /// Source media file of the current transcript
    pub source: Option<PathBuf>,
    pub duration_secs: Option<f32>,
    /// Language whisper detected (or was told), for the export header
    pub language: Option<String>,
    /// Model that produced the transcript, for the export header
    pub model_name: Option<String>,
    /// How speaker labels were produced (diarization method), for the
    /// export header; None when diarization was off
    pub diarization: Option<String>,
    /// Human names assigned to speaker indices via the naming dialog
    /// (`n`); empty = anonymous A/B/C labels
    pub speaker_names: BTreeMap<u8, String>,
    pub(crate) layout: TranscriptLayout,
}

impl TranscriptState {
    pub(crate) fn new() -> Self {
        Self {
            segments: Vec::new(),
            partial: None,
            scroll: 0,
            follow: true,
            source: None,
            duration_secs: None,
            language: None,
            model_name: None,
            diarization: None,
            speaker_names: BTreeMap::new(),
            layout: TranscriptLayout::default(),
        }
    }

    /// Reset for a new transcription job.
    pub fn begin(&mut self, source: PathBuf, model_name: String, diarization: Option<String>) {
        self.segments.clear();
        self.partial = None;
        self.scroll = 0;
        self.follow = true;
        self.duration_secs = None;
        self.language = None;
        self.model_name = Some(model_name);
        self.diarization = diarization;
        self.speaker_names.clear();
        self.source = Some(source);
        self.invalidate_layout();
    }

    /// Required when replacing/editing committed history rather than appending.
    pub(crate) fn invalidate_layout(&mut self) {
        self.layout = TranscriptLayout::default();
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.follow = false;
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.follow = false;
        self.scroll += lines;
    }

    pub fn scroll_top(&mut self) {
        self.follow = false;
        self.scroll = 0;
    }

    pub fn follow_tail(&mut self) {
        self.follow = true;
    }

    /// Pin the scroll position to the content, called once per frame
    /// before rendering: follow snaps to the end, manual scroll clamps.
    pub fn clamp(&mut self, total_lines: usize, viewport: usize) {
        let max_scroll = total_lines.saturating_sub(viewport);
        if self.follow {
            self.scroll = max_scroll;
        } else {
            self.scroll = self.scroll.min(max_scroll);
        }
    }
}
