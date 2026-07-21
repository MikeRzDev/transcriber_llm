//! The file-browser pane: directory listing and entry navigation.

use std::path::{Path, PathBuf};

use crate::app::App;
use crate::audio;

pub enum FileEntry {
    Parent,
    Dir(PathBuf),
    Media(PathBuf),
}

impl FileEntry {
    pub fn label(&self) -> String {
        match self {
            FileEntry::Parent => "../".into(),
            FileEntry::Dir(p) => format!("{}/", file_name(p)),
            FileEntry::Media(p) => {
                if audio::is_video_file(p) {
                    format!("{} ⧉", file_name(p))
                } else {
                    file_name(p)
                }
            }
        }
    }
}

pub(crate) fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

/// State of the left pane: the directory being browsed and the selection.
pub struct FileBrowser {
    pub cwd: PathBuf,
    pub entries: Vec<FileEntry>,
    pub selected: usize,
    /// First visible row of the virtualized list. A Cell because the
    /// definitive value depends on the viewport height, which is only
    /// known at render time (where the App is borrowed immutably).
    pub scroll: std::cell::Cell<usize>,
}

impl FileBrowser {
    pub(crate) fn new(cwd: PathBuf) -> Self {
        let mut browser = Self {
            cwd,
            entries: Vec::new(),
            selected: 0,
            scroll: std::cell::Cell::new(0),
        };
        browser.refresh();
        browser
    }

    pub fn refresh(&mut self) {
        let mut dirs: Vec<PathBuf> = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.cwd) {
            for entry in rd.flatten() {
                let path = entry.path();
                let name = file_name(&path);
                if name.starts_with('.') {
                    continue;
                }
                if path.is_dir() {
                    dirs.push(path);
                } else if audio::is_media_file(&path) {
                    files.push(path);
                }
            }
        }
        dirs.sort();
        files.sort();

        self.entries.clear();
        if self.cwd.parent().is_some() {
            self.entries.push(FileEntry::Parent);
        }
        self.entries.extend(dirs.into_iter().map(FileEntry::Dir));
        self.entries.extend(files.into_iter().map(FileEntry::Media));
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn select_next(&mut self) {
        if !self.entries.is_empty() {
            self.selected = (self.selected + 1).min(self.entries.len() - 1);
        }
    }

    pub fn set_cwd(&mut self, dir: PathBuf) {
        self.cwd = dir;
        self.selected = 0;
        self.refresh();
    }

    /// Enter on the selection: directories are entered in place; a media
    /// file is returned for the caller to act on.
    pub fn enter(&mut self) -> Option<PathBuf> {
        match self.entries.get(self.selected) {
            Some(FileEntry::Parent) => {
                if let Some(parent) = self.cwd.parent() {
                    self.set_cwd(parent.to_path_buf());
                }
                None
            }
            Some(FileEntry::Dir(p)) => {
                let dir = p.clone();
                self.set_cwd(dir);
                None
            }
            Some(FileEntry::Media(p)) => Some(p.clone()),
            None => None,
        }
    }
}

impl App {
    pub fn enter_selected(&mut self) {
        if let Some(path) = self.browser.enter() {
            self.request_transcription(path);
        }
    }
}

/// RecyclerView-style windowing for a virtualized list: keep the
/// previous scroll offset, moving it only when the selection would
/// leave the viewport. Only rows in [offset, offset+viewport) exist as
/// widgets; everything else stays as raw data.
pub fn scroll_window(offset: usize, selected: usize, len: usize, viewport: usize) -> usize {
    if viewport == 0 || len == 0 {
        return 0;
    }
    let max_offset = len.saturating_sub(viewport);
    let mut off = offset.min(max_offset);
    if selected < off {
        off = selected; // selection moved above the window → snap up
    } else if selected >= off + viewport {
        off = selected + 1 - viewport; // below the window → snap down
    }
    off.min(max_offset)
}

#[cfg(test)]
mod tests {
    use super::scroll_window;

    #[test]
    fn window_follows_selection_minimally() {
        // selection inside the window: offset unchanged
        assert_eq!(scroll_window(5, 7, 100, 10), 5);
        // selection walked below the window: scroll just enough
        assert_eq!(scroll_window(5, 15, 100, 10), 6);
        // selection jumped above the window: snap to it
        assert_eq!(scroll_window(50, 3, 100, 10), 3);
        // offset never exceeds len - viewport
        assert_eq!(scroll_window(999, 99, 100, 10), 90);
        // list shorter than the viewport never scrolls
        assert_eq!(scroll_window(4, 2, 5, 10), 0);
        // degenerate cases
        assert_eq!(scroll_window(3, 0, 0, 10), 0);
        assert_eq!(scroll_window(3, 5, 100, 0), 0);
    }
}
