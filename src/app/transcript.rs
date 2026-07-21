//! Transcript pane state: streamed segments and scroll/follow behavior.

use std::path::PathBuf;

use crate::transcribe::Segment;

pub struct TranscriptState {
    pub segments: Vec<Segment>,
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
}

impl TranscriptState {
    pub(crate) fn new() -> Self {
        Self {
            segments: Vec::new(),
            scroll: 0,
            follow: true,
            source: None,
            duration_secs: None,
            language: None,
            model_name: None,
        }
    }

    /// Reset for a new transcription job.
    pub fn begin(&mut self, source: PathBuf, model_name: String) {
        self.segments.clear();
        self.scroll = 0;
        self.follow = true;
        self.duration_secs = None;
        self.language = None;
        self.model_name = Some(model_name);
        self.source = Some(source);
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
