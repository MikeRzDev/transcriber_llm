//! The job log pane: a timestamped record of everything that happened
//! since the first file was loaded — file selection, model load, audio
//! extraction, engine output, segments, exports, errors. Shown in the
//! right pane in place of the transcript (toggled with `l`).

use std::collections::VecDeque;

/// Hard cap so a very long session cannot grow without bound; beyond it
/// the oldest lines fall off the front.
const MAX_LINES: usize = 10_000;

pub struct JobLog {
    pub lines: VecDeque<String>,
    pub scroll: usize,
    pub follow: bool,
    /// Tag of the most recent `push_tagged` line — rapid progress
    /// updates with the same tag replace their predecessor instead of
    /// flooding the log with one line per percent.
    last_tag: Option<&'static str>,
}

impl JobLog {
    pub(crate) fn new() -> Self {
        Self {
            lines: VecDeque::new(),
            scroll: 0,
            follow: true,
            last_tag: None,
        }
    }

    /// Append a timestamped line.
    pub fn push(&mut self, msg: impl AsRef<str>) {
        self.last_tag = None;
        self.push_line(msg.as_ref());
    }

    /// Append a timestamped line that replaces the previous one when it
    /// carries the same tag (used for percent progress updates).
    pub fn push_tagged(&mut self, tag: &'static str, msg: &str) {
        if self.last_tag == Some(tag) {
            self.lines.pop_back();
        }
        self.last_tag = Some(tag);
        self.push_line(msg);
    }

    fn push_line(&mut self, msg: &str) {
        if self.lines.len() >= MAX_LINES {
            self.lines.pop_front();
        }
        self.lines
            .push_back(format!("[{}] {}", crate::format::clock_now(), msg));
    }

    pub fn scroll_up(&mut self, lines: usize) {
        self.follow = false;
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        // clamp() re-engages follow once the tail comes into view
        self.scroll += lines;
    }

    pub fn scroll_top(&mut self) {
        self.follow = false;
        self.scroll = 0;
    }

    pub fn follow_tail(&mut self) {
        self.follow = true;
    }

    /// Pre-draw clamp, mirroring the transcript pane: follow pins to the
    /// tail; manual scroll clamps into range and re-engages follow when
    /// it reaches the end.
    pub fn clamp(&mut self, viewport: usize) {
        let max = self.lines.len().saturating_sub(viewport);
        if self.follow || self.scroll >= max {
            self.scroll = max;
            self.follow = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_timestamped_and_capped() {
        let mut log = JobLog::new();
        log.push("hello");
        assert_eq!(log.lines.len(), 1);
        let line = &log.lines[0];
        // "[HH:MM:SS] hello"
        assert_eq!(line.len(), "[00:00:00] hello".len(), "line: {line}");
        assert!(line.ends_with("] hello"));

        for i in 0..MAX_LINES + 5 {
            log.push(format!("line {i}"));
        }
        assert_eq!(log.lines.len(), MAX_LINES);
        assert!(log.lines.back().unwrap().ends_with(&format!(
            "line {}",
            MAX_LINES + 4
        )));
    }

    #[test]
    fn tagged_lines_coalesce_until_interrupted() {
        let mut log = JobLog::new();
        log.push_tagged("pct", "transcribing: 10%");
        log.push_tagged("pct", "transcribing: 20%");
        assert_eq!(log.lines.len(), 1);
        assert!(log.lines[0].ends_with("transcribing: 20%"));

        // a different line breaks the run; the next tag starts fresh
        log.push("segment arrived");
        log.push_tagged("pct", "transcribing: 30%");
        assert_eq!(log.lines.len(), 3);
    }

    #[test]
    fn scroll_and_follow_behave_like_the_transcript() {
        let mut log = JobLog::new();
        for i in 0..100 {
            log.push(format!("l{i}"));
        }
        log.clamp(10);
        assert!(log.follow);
        assert_eq!(log.scroll, 90); // pinned to the tail

        log.scroll_up(5);
        log.clamp(10);
        assert!(!log.follow);
        assert_eq!(log.scroll, 85);

        // scrolling back down to the end re-engages follow
        log.scroll_down(50);
        log.clamp(10);
        assert!(log.follow);
        assert_eq!(log.scroll, 90);
    }
}
