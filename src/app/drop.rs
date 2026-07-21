//! Drag-and-drop support: terminals deliver dropped files as pasted text
//! in several shapes (quoted, backslash-escaped, file:// URLs).

use std::path::PathBuf;
use std::time::Instant;

use crate::app::browser::file_name;
use crate::app::{App, Focus};
use crate::audio;

/// A drop-burst continues while consecutive chars arrive this quickly.
const BURST_CONTINUE_MS: u128 = 150;
/// A quiet period this long ends the burst and flushes it as a path.
const BURST_FLUSH_MS: u128 = 250;
/// Flushed bursts must be longer than this to count as a dropped path
/// (filters out someone just typing '/' or '~' then pausing).
const MIN_DROP_LEN: usize = 2;

/// Fallback drop detection for terminals without bracketed paste: a
/// dropped file arrives as a rapid burst of key events starting with
/// '/' or '~'. The caller decides when feeding is allowed (not while a
/// modal captures typing, no CONTROL modifier).
pub struct DropDetector {
    buf: String,
    last_char: Instant,
}

impl DropDetector {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            last_char: Instant::now(),
        }
    }

    /// Offer a printable char; true when it was consumed by a burst.
    pub fn feed_char(&mut self, c: char) -> bool {
        let burst_continues =
            !self.buf.is_empty() && self.last_char.elapsed().as_millis() < BURST_CONTINUE_MS;
        let burst_starts = self.buf.is_empty() && (c == '/' || c == '~');
        if burst_continues || burst_starts {
            self.buf.push(c);
            self.last_char = Instant::now();
            true
        } else {
            false
        }
    }

    /// Offer an Enter key; Some(text) when it terminates a live burst
    /// (some terminals end a drop with a newline).
    pub fn feed_enter(&mut self) -> Option<String> {
        if !self.buf.is_empty() && self.last_char.elapsed().as_millis() < BURST_CONTINUE_MS {
            Some(std::mem::take(&mut self.buf))
        } else {
            None
        }
    }

    /// Call once per frame: a quiet period ends the burst. Discards
    /// too-short bursts instead of returning them.
    pub fn poll(&mut self) -> Option<String> {
        if !self.buf.is_empty() && self.last_char.elapsed().as_millis() > BURST_FLUSH_MS {
            let text = std::mem::take(&mut self.buf);
            if text.len() > MIN_DROP_LEN {
                return Some(text);
            }
        }
        None
    }
}

impl Default for DropDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Handle a path dropped onto the terminal window (arrives as pasted
    /// text, possibly quoted or backslash-escaped by the terminal).
    pub fn handle_dropped_text(&mut self, text: &str) {
        let Some(path) = parse_dropped_path(text) else {
            self.status = "Dropped text doesn't look like a file path".into();
            return;
        };
        if path.is_dir() {
            self.browser.set_cwd(path);
            self.focus = Focus::Files;
            return;
        }
        if !path.exists() {
            self.status = format!("Not found: {}", path.display());
            return;
        }
        if !audio::is_media_file(&path) {
            self.status = format!("Unsupported file type: {}", file_name(&path));
            return;
        }
        self.start_transcription(path);
    }
}

/// Terminals paste dropped files as text: possibly 'single-quoted',
/// "double-quoted", or with backslash-escaped spaces and parens.
fn parse_dropped_path(text: &str) -> Option<PathBuf> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let unquoted = trimmed
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
        .or_else(|| trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')))
        .unwrap_or(trimmed);

    // Some terminals paste drops as file:// URLs with percent-encoding
    let unescaped = if let Some(url_path) = unquoted.strip_prefix("file://") {
        percent_decode(url_path.trim_start_matches("localhost"))
    } else {
        // Undo backslash escaping (e.g. "My\ File.mp4")
        let mut plain = String::with_capacity(unquoted.len());
        let mut chars = unquoted.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    plain.push(next);
                }
            } else {
                plain.push(c);
            }
        }
        plain
    };

    let expanded = if let Some(rest) = unescaped.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home).join(rest)
        } else {
            PathBuf::from(unescaped)
        }
    } else {
        PathBuf::from(unescaped)
    };

    if expanded.is_absolute() || expanded.exists() {
        Some(expanded)
    } else {
        None
    }
}

fn percent_decode(s: &str) -> String {
    // Malformed sequences pass through unchanged; invalid UTF-8 is lossy —
    // matching what terminals need for file:// drops.
    percent_encoding::percent_decode_str(s)
        .decode_utf8_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn burst_starts_only_on_path_chars() {
        let mut d = DropDetector::new();
        assert!(!d.feed_char('a'));
        assert!(!d.feed_char('x'));
        assert!(d.feed_char('/'));
        // once started, any char continues the burst
        assert!(d.feed_char('t'));
        assert!(d.feed_char('m'));
        assert!(d.feed_char('p'));
    }

    #[test]
    fn enter_terminates_a_live_burst() {
        let mut d = DropDetector::new();
        assert_eq!(d.feed_enter(), None);
        for c in "/tmp/a.wav".chars() {
            assert!(d.feed_char(c));
        }
        assert_eq!(d.feed_enter().as_deref(), Some("/tmp/a.wav"));
        // the burst is consumed
        assert_eq!(d.feed_enter(), None);
    }

    #[test]
    fn quiet_period_flushes_and_short_bursts_are_discarded() {
        let mut d = DropDetector::new();
        assert!(d.feed_char('/'));
        // too fresh to flush
        assert_eq!(d.poll(), None);
        std::thread::sleep(Duration::from_millis(BURST_FLUSH_MS as u64 + 60));
        // a bare '/' is below MIN_DROP_LEN: dropped, not returned
        assert_eq!(d.poll(), None);
        assert_eq!(d.feed_enter(), None);

        for c in "/tmp/a.wav".chars() {
            assert!(d.feed_char(c));
        }
        std::thread::sleep(Duration::from_millis(BURST_FLUSH_MS as u64 + 60));
        assert_eq!(d.poll().as_deref(), Some("/tmp/a.wav"));
    }

    #[test]
    fn stale_burst_does_not_claim_enter() {
        let mut d = DropDetector::new();
        assert!(d.feed_char('/'));
        assert!(d.feed_char('x'));
        std::thread::sleep(Duration::from_millis(BURST_CONTINUE_MS as u64 + 60));
        assert_eq!(d.feed_enter(), None);
    }

    #[test]
    fn plain_absolute_path() {
        assert_eq!(
            parse_dropped_path("/tmp/file.mp4"),
            Some(PathBuf::from("/tmp/file.mp4"))
        );
    }

    #[test]
    fn backslash_escaped_spaces() {
        assert_eq!(
            parse_dropped_path("/tmp/my\\ file\\ (1).mp4 "),
            Some(PathBuf::from("/tmp/my file (1).mp4"))
        );
    }

    #[test]
    fn quoted_paths() {
        assert_eq!(
            parse_dropped_path("'/tmp/my file.mp4'"),
            Some(PathBuf::from("/tmp/my file.mp4"))
        );
        assert_eq!(
            parse_dropped_path("\"/tmp/a.wav\""),
            Some(PathBuf::from("/tmp/a.wav"))
        );
    }

    #[test]
    fn tilde_expansion() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(
            parse_dropped_path("~/x.wav"),
            Some(PathBuf::from(home).join("x.wav"))
        );
    }

    #[test]
    fn rejects_non_paths() {
        assert_eq!(parse_dropped_path(""), None);
        assert_eq!(parse_dropped_path("   "), None);
        assert_eq!(parse_dropped_path("hello world"), None);
    }

    #[test]
    fn file_url_with_percent_encoding() {
        assert_eq!(
            parse_dropped_path("file:///tmp/my%20file.mp4"),
            Some(PathBuf::from("/tmp/my file.mp4"))
        );
        assert_eq!(
            parse_dropped_path("file://localhost/tmp/a.wav"),
            Some(PathBuf::from("/tmp/a.wav"))
        );
    }
}
