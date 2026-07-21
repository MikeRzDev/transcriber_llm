//! Hugging Face Hub integration for Model management: search repos, list
//! their GGML/GGUF files (`api`), and download models (`download`) —
//! single files for whisper.cpp, whole repos for directory (MLX) models —
//! each on a background thread reporting back over a channel (same
//! pattern as the transcribe worker).

mod api;
mod download;

pub use api::{list_files, search};
pub use download::{download, download_dir};

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
    /// Whether an engine in this build can run the model: GGML/GGUF
    /// always (whisper.cpp); MLX directory models on Apple Silicon.
    /// Running MLX additionally needs mlx-audio, but that is checked at
    /// job time — downloading ahead of the install is allowed.
    pub fn supported(&self) -> bool {
        match self.format.as_str() {
            "ggml" | "gguf" => true,
            "mlx" => metal_available(),
            _ => false,
        }
    }

    /// MLX entries name a whole repo, downloaded as a directory.
    pub fn is_dir_model(&self) -> bool {
        self.format == "mlx"
    }

    /// The on-disk name once downloaded: the file's base name, or the
    /// repo's for directory models.
    pub fn dest_name(&self) -> String {
        if self.file.is_empty() {
            repo_dir_name(&self.repo)
        } else {
            dest_name(&self.file).unwrap_or_default()
        }
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

/// A directory-model variant inside a repo: the repo root or a subfolder
/// that holds its own `config.json` + safetensors weights. Each downloads
/// as an individual folder model.
#[derive(Clone, Debug, PartialEq)]
pub struct DirVariant {
    /// Repo-relative folder ("" = the repo root)
    pub subdir: String,
    /// Summed size of every file under the variant
    pub size_bytes: u64,
    pub file_count: usize,
}

impl DirVariant {
    /// Row label: the subfolder name, or the repo basename for the root.
    pub fn label(&self, repo: &str) -> String {
        if self.subdir.is_empty() {
            format!("{} (whole repo)", repo_dir_name(repo))
        } else {
            format!("{}/", self.subdir)
        }
    }
}

/// Local folder name for a repo (sub)download: the repo basename, plus
/// the variant subdir when one is chosen (`owner/repo` + `8bit` →
/// `repo-8bit`). Distinct variants stay distinct on disk.
pub fn variant_dir_name(repo: &str, subdir: &str) -> String {
    let base = repo_dir_name(repo);
    if subdir.is_empty() {
        base
    } else {
        format!("{base}-{}", subdir.replace('/', "-"))
    }
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
        /// Directory-model variants found alongside (or instead of) the
        /// single-file models
        variants: Vec<DirVariant>,
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
    /// A running transfer stopped on request; the `.part` file is retained at
    /// `got` bytes so it can be resumed.
    Paused {
        file: String,
        got: u64,
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

/// The on-disk folder name for a whole-repo download: the repo id's
/// basename (`mlx-community/parakeet-tdt-0.6b-v3` → `parakeet-tdt-0.6b-v3`).
pub fn repo_dir_name(repo: &str) -> String {
    repo.rsplit('/').next().unwrap_or(repo).to_string()
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
        // whisper-large-v3 mapped to its GGML build, as a single file
        let whisper = models
            .iter()
            .find(|m| m.repo == "ggerganov/whisper.cpp")
            .unwrap();
        assert_eq!(whisper.file, "ggml-large-v3.bin");
        assert!(whisper.supported());
        assert!(!whisper.is_dir_model());
        assert_eq!(whisper.dest_name(), "ggml-large-v3.bin");
        for m in &models {
            if m.format == "mlx" {
                // MLX entries are whole-repo directory models: no single
                // file, named after the repo, Apple-Silicon-gated.
                assert!(m.is_dir_model());
                assert!(m.file.is_empty());
                assert_eq!(m.dest_name(), repo_dir_name(&m.repo));
                assert_eq!(m.supported(), metal_available());
            } else {
                assert!(is_model_file(Path::new(&m.file)), "{} not a model", m.file);
            }
        }
        assert_eq!(models.iter().filter(|m| m.is_dir_model()).count(), 2);
    }

    #[test]
    fn dest_name_strips_repo_subdirs() {
        assert_eq!(dest_name("ggml-tiny.bin").as_deref(), Some("ggml-tiny.bin"));
        assert_eq!(dest_name("sub/dir/model.bin").as_deref(), Some("model.bin"));
        assert_eq!(dest_name(""), None);
    }

    #[test]
    fn repo_dir_name_is_the_repo_basename() {
        assert_eq!(
            repo_dir_name("mlx-community/parakeet-tdt-0.6b-v3"),
            "parakeet-tdt-0.6b-v3"
        );
        assert_eq!(repo_dir_name("bare-name"), "bare-name");
    }

    #[test]
    fn variant_dir_names_stay_distinct_per_variant() {
        assert_eq!(variant_dir_name("owner/repo", ""), "repo");
        assert_eq!(variant_dir_name("owner/repo", "8bit"), "repo-8bit");
        assert_eq!(variant_dir_name("owner/repo", "sub/4bit"), "repo-sub-4bit");
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

    /// Live network test: download a whole directory model (the ~23 MB
    /// 4-bit whisper-tiny MLX conversion) and verify the folder, its
    /// integrity manifest, and that the scan accepts it.
    /// Run with: cargo test -- --ignored dir_download
    #[test]
    #[ignore = "hits the Hugging Face API and downloads ~23 MB"]
    fn hub_dir_download_end_to_end() {
        use std::sync::mpsc::channel;
        let (tx, rx) = channel();
        let dir =
            std::env::temp_dir().join(format!("transcribe-stt-hubdir-{}", std::process::id()));

        download_dir(
            "mlx-community/whisper-tiny-4bit".into(),
            String::new(),
            dir.clone(),
            Arc::new(AtomicBool::new(false)),
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
        assert_eq!(path, dir.join("whisper-tiny-4bit"));
        assert!(path.join("config.json").is_file());
        assert!(path.join("model.safetensors").is_file());
        // The manifest is written and satisfied
        let manifest = crate::models::DirManifest::load(&path).expect("manifest exists");
        assert_eq!(manifest.repo, "mlx-community/whisper-tiny-4bit");
        assert!(crate::models::manifest_gaps(&path).is_empty());
        // And the scan treats the folder as one model with summed size
        let models = crate::models::scan_models(&dir);
        assert_eq!(models.len(), 1);
        assert!(models[0].is_dir);
        assert!(models[0].size_bytes > 22_000_000, "{}", models[0].size_bytes);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
