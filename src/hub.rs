//! Hugging Face Hub integration for Model management: search repos, list
//! their GGML/GGUF files (`api`), and download models (`download`) — each
//! on a background thread reporting back over a channel (same pattern as
//! the transcribe worker).

mod api;
mod download;

pub use api::{list_files, search};
pub use download::download;

use std::path::{Path, PathBuf};

use serde::Deserialize;

const SUGGESTED_JSON: &str = include_str!("../assets/suggested_models.json");

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SuggestedModel {
    pub name: String,
    pub repo: String,
    #[serde(default)]
    pub file: String,
    #[serde(default)]
    pub size: String,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub note: String,
}

fn default_format() -> String {
    "ggml".into()
}

impl SuggestedModel {
    /// Only ggml-family formats are loadable by the whisper.cpp engine;
    /// anything else (e.g. MLX) is listed greyed out.
    pub fn supported(&self) -> bool {
        matches!(self.format.as_str(), "ggml" | "gguf")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepoHit {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HubFile {
    pub name: String,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub enum HubEvent {
    SearchResults {
        query: String,
        hits: Vec<RepoHit>,
    },
    SearchFailed {
        query: String,
        error: String,
    },
    Files {
        repo: String,
        files: Vec<HubFile>,
    },
    FilesFailed {
        repo: String,
        error: String,
    },
    Progress {
        file: String,
        got: u64,
        total: u64,
    },
    Done {
        file: String,
        path: PathBuf,
    },
    Cancelled {
        file: String,
    },
    Failed {
        file: String,
        error: String,
    },
    /// A background models-folder move finished (not hub-originated, but it
    /// rides the same channel into the render loop)
    ModelsMoved {
        moved: usize,
        skipped: usize,
        failed: usize,
    },
}

/// The Metal backend runs the same GGML files as the CPU backend; this only
/// decides which capability badge the UI shows.
pub fn metal_available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

pub fn backend_label() -> &'static str {
    if metal_available() {
        "Metal GPU (Apple Silicon detected)"
    } else {
        "CPU"
    }
}

pub fn suggested_models() -> Vec<SuggestedModel> {
    parse_suggested(SUGGESTED_JSON)
}

fn parse_suggested(json: &str) -> Vec<SuggestedModel> {
    #[derive(Default, Deserialize)]
    struct SuggestedFile {
        #[serde(default)]
        models: Vec<serde_json::Value>,
    }
    let file: SuggestedFile = serde_json::from_str(json).unwrap_or_default();
    file.models
        .into_iter()
        // per-entry: a malformed entry is dropped, the rest still load
        .filter_map(|m| serde_json::from_value(m).ok())
        .collect()
}

/// The on-disk name for a repo file: its final path component.
pub fn dest_name(rfilename: &str) -> Option<String> {
    Path::new(rfilename)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::is_model_file;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn suggested_json_parses_and_supported_entries_are_models() {
        let models = suggested_models();
        assert_eq!(models.len(), 3);
        // the one runnable entry: whisper-large-v3 mapped to its GGML build
        assert!(models
            .iter()
            .any(|m| m.repo == "ggerganov/whisper.cpp" && m.file == "ggml-large-v3.bin"));
        for m in &models {
            if m.supported() {
                assert!(is_model_file(Path::new(&m.file)), "{} not a model", m.file);
            } else {
                assert_eq!(m.format, "mlx");
            }
        }
        // the two MLX-only requests stay visible but greyed out
        assert_eq!(models.iter().filter(|m| !m.supported()).count(), 2);
    }

    #[test]
    fn dest_name_strips_repo_subdirs() {
        assert_eq!(dest_name("ggml-tiny.bin").as_deref(), Some("ggml-tiny.bin"));
        assert_eq!(dest_name("sub/dir/model.bin").as_deref(), Some("model.bin"));
        assert_eq!(dest_name(""), None);
    }

    /// Live network test: search → list files → download the smallest real
    /// whisper model into a temp dir. Run with: cargo test -- --ignored hub
    #[test]
    #[ignore = "hits the Hugging Face API and downloads ~75 MB"]
    fn hub_search_list_download_end_to_end() {
        use std::sync::mpsc::channel;
        let (tx, rx) = channel();

        search("whisper.cpp".into(), tx.clone());
        let hits = loop {
            match rx.recv_timeout(Duration::from_secs(30)).unwrap() {
                HubEvent::SearchResults { hits, .. } => break hits,
                HubEvent::SearchFailed { error, .. } => panic!("search failed: {error}"),
                _ => {}
            }
        };
        assert!(hits.iter().any(|h| h.id == "ggerganov/whisper.cpp"));

        list_files("ggerganov/whisper.cpp".into(), tx.clone());
        let files = loop {
            match rx.recv_timeout(Duration::from_secs(30)).unwrap() {
                HubEvent::Files { files, .. } => break files,
                HubEvent::FilesFailed { error, .. } => panic!("list failed: {error}"),
                _ => {}
            }
        };
        assert!(files.iter().any(|f| f.name == "ggml-tiny.bin"));

        let dir = std::env::temp_dir().join(format!("transcribe-stt-hub-{}", std::process::id()));
        download(
            "ggerganov/whisper.cpp".into(),
            "ggml-tiny.bin".into(),
            dir.clone(),
            Arc::new(AtomicBool::new(false)),
            tx,
        );
        let path = loop {
            match rx.recv_timeout(Duration::from_secs(300)).unwrap() {
                HubEvent::Done { path, .. } => break path,
                HubEvent::Failed { error, .. } => panic!("download failed: {error}"),
                HubEvent::Cancelled { .. } => panic!("unexpected cancel"),
                _ => {}
            }
        };
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 50_000_000, "suspiciously small: {size}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
