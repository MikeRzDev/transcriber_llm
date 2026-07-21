//! The job log pane: a timestamped record of everything that happened
//! since the first file was loaded — file selection, model load, audio
//! extraction, engine output, segments, exports, errors. Shown in the
//! right pane in place of the transcript (toggled with `l`).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

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

    /// Wipe the log and reset scroll/follow (the `x` hotkey).
    pub fn clear(&mut self) {
        self.lines.clear();
        self.scroll = 0;
        self.follow = true;
        self.last_tag = None;
    }

    /// Write every line to `<dir>/log_<YYYYMMDD_HHMMSS>.log` and return
    /// the path. An empty log is an error, not an empty file.
    pub fn export_to(&self, dir: &Path) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(!self.lines.is_empty(), "log is empty — nothing to export");
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("log_{}.log", crate::format::file_timestamp()));
        let mut contents = String::with_capacity(self.lines.iter().map(|l| l.len() + 1).sum());
        for line in &self.lines {
            contents.push_str(line);
            contents.push('\n');
        }
        std::fs::write(&path, contents)?;
        Ok(path)
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
    fn export_writes_timestamped_file_and_rejects_empty() {
        let dir = std::env::temp_dir().join(format!(
            "transcribe-stt-joblog-{}",
            std::process::id()
        ));
        let empty = JobLog::new();
        assert!(empty.export_to(&dir).is_err());
        assert!(!dir.exists(), "empty export must not create the folder");

        let mut log = JobLog::new();
        log.push("first line");
        log.push("second line");
        let path = log.export_to(&dir).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            name.starts_with("log_") && name.ends_with(".log"),
            "name: {name}"
        );
        assert_eq!(name.len(), "log_YYYYMMDD_HHMMSS.log".len());
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("] first line\n"));
        assert!(contents.ends_with("] second line\n"));
        std::fs::remove_dir_all(&dir).unwrap();
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
