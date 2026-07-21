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

impl App {
    pub fn refresh_entries(&mut self) {
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
        self.file_selected = self.file_selected.min(self.entries.len().saturating_sub(1));
    }

    pub fn enter_selected(&mut self) {
        match self.entries.get(self.file_selected) {
            Some(FileEntry::Parent) => {
                if let Some(parent) = self.cwd.parent() {
                    self.cwd = parent.to_path_buf();
                    self.file_selected = 0;
                    self.refresh_entries();
                }
            }
            Some(FileEntry::Dir(p)) => {
                self.cwd = p.clone();
                self.file_selected = 0;
                self.refresh_entries();
            }
            Some(FileEntry::Media(p)) => {
                let path = p.clone();
                self.start_transcription(path);
            }
            None => {}
        }
    }
}
