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
}

impl FileBrowser {
    pub(crate) fn new(cwd: PathBuf) -> Self {
        let mut browser = Self {
            cwd,
            entries: Vec::new(),
            selected: 0,
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
            self.start_transcription(path);
        }
    }
}
