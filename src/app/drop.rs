//! Drag-and-drop support: terminals deliver dropped files as pasted text
//! in several shapes (quoted, backslash-escaped, file:// URLs).

use std::path::PathBuf;

use crate::app::browser::file_name;
use crate::app::{App, Focus};
use crate::audio;

impl App {
    /// Handle a path dropped onto the terminal window (arrives as pasted
    /// text, possibly quoted or backslash-escaped by the terminal).
    pub fn handle_dropped_text(&mut self, text: &str) {
        let Some(path) = parse_dropped_path(text) else {
            self.status = "Dropped text doesn't look like a file path".into();
            return;
        };
        if path.is_dir() {
            self.cwd = path;
            self.file_selected = 0;
            self.refresh_entries();
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
