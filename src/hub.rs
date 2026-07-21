//! Hugging Face Hub integration for Model management: search repos, list
//! their GGML/GGUF files, and download models — each on a background thread
//! reporting back over a channel (same pattern as the transcribe worker).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use crate::models::is_model_file;

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
    SearchResults { query: String, hits: Vec<RepoHit> },
    SearchFailed { query: String, error: String },
    Files { repo: String, files: Vec<HubFile> },
    FilesFailed { repo: String, error: String },
    Progress { file: String, got: u64, total: u64 },
    Done { file: String, path: PathBuf },
    Cancelled { file: String },
    Failed { file: String, error: String },
    /// A background models-folder move finished (not hub-originated, but it
    /// rides the same channel into the render loop)
    ModelsMoved { moved: usize, skipped: usize, failed: usize },
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

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(30))
        .user_agent(concat!(
            env!("CARGO_PKG_NAME"),
            "/",
            env!("CARGO_PKG_VERSION")
        ))
        .build()
}

/// Search the Hub for repos matching `query`, most-downloaded first,
/// restricted to the speech-to-text (automatic-speech-recognition)
/// category. Untagged conversion repos won't appear. The receiver should
/// drop results whose `query` no longer matches the input.
pub fn search(query: String, tx: Sender<HubEvent>) {
    std::thread::spawn(move || {
        let url = format!(
            "https://huggingface.co/api/models?search={}&pipeline_tag=automatic-speech-recognition&limit=25&sort=downloads&direction=-1",
            urlencode(&query)
        );
        let result = (|| -> Result<Vec<RepoHit>, String> {
            let resp = agent().get(&url).call().map_err(|e| e.to_string())?;
            let v: serde_json::Value =
                serde_json::from_reader(resp.into_reader()).map_err(|e| e.to_string())?;
            Ok(parse_search(&v))
        })();
        let _ = tx.send(match result {
            Ok(hits) => HubEvent::SearchResults { query, hits },
            Err(error) => HubEvent::SearchFailed { query, error },
        });
    });
}

fn parse_search(v: &serde_json::Value) -> Vec<RepoHit> {
    /// Search hits arrive with either `id` or the older `modelId`.
    #[derive(Deserialize)]
    struct RawHit {
        id: Option<String>,
        #[serde(rename = "modelId")]
        model_id: Option<String>,
        #[serde(default)]
        downloads: u64,
        #[serde(default)]
        likes: u64,
    }
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|entry| serde_json::from_value::<RawHit>(entry.clone()).ok())
        .filter_map(|hit| {
            Some(RepoHit {
                id: hit.id.or(hit.model_id)?,
                downloads: hit.downloads,
                likes: hit.likes,
            })
        })
        .collect()
}

/// List the GGML/GGUF files (with sizes) inside a repo.
pub fn list_files(repo: String, tx: Sender<HubEvent>) {
    std::thread::spawn(move || {
        let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
        let result = (|| -> Result<Vec<HubFile>, String> {
            let resp = agent().get(&url).call().map_err(|e| e.to_string())?;
            let v: serde_json::Value =
                serde_json::from_reader(resp.into_reader()).map_err(|e| e.to_string())?;
            Ok(parse_files(&v))
        })();
        let _ = tx.send(match result {
            Ok(files) => HubEvent::Files { repo, files },
            Err(error) => HubEvent::FilesFailed { repo, error },
        });
    });
}

fn parse_files(v: &serde_json::Value) -> Vec<HubFile> {
    #[derive(Deserialize)]
    struct RawSibling {
        rfilename: String,
        // `size` may be absent or null → unknown
        size: Option<u64>,
    }
    let mut files: Vec<HubFile> = v["siblings"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|s| serde_json::from_value::<RawSibling>(s.clone()).ok())
                .filter(|s| is_model_file(Path::new(&s.rfilename)))
                .map(|s| HubFile {
                    name: s.rfilename,
                    size_bytes: s.size.unwrap_or(0),
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort_by(|a, b| a.name.cmp(&b.name));
    files
}

/// The on-disk name for a repo file: its final path component.
pub fn dest_name(rfilename: &str) -> Option<String> {
    Path::new(rfilename)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
}

/// Stream `repo/file` into `dest_dir` as `<name>.part`, renaming into place
/// when complete. Progress is reported every couple of MB; `cancel` stops
/// the transfer and removes the partial file.
pub fn download(
    repo: String,
    file: String,
    dest_dir: PathBuf,
    cancel: Arc<AtomicBool>,
    tx: Sender<HubEvent>,
) {
    std::thread::spawn(move || {
        let display = dest_name(&file).unwrap_or_else(|| file.clone());
        let result = download_blocking(&repo, &file, &dest_dir, &cancel, &display, &tx);
        let _ = tx.send(match result {
            Ok(Some(path)) => HubEvent::Done {
                file: display,
                path,
            },
            Ok(None) => HubEvent::Cancelled { file: display },
            Err(error) => HubEvent::Failed {
                file: display,
                error,
            },
        });
    });
}

fn download_blocking(
    repo: &str,
    file: &str,
    dest_dir: &Path,
    cancel: &AtomicBool,
    display: &str,
    tx: &Sender<HubEvent>,
) -> Result<Option<PathBuf>, String> {
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let base = dest_name(file).ok_or("bad file name")?;
    let final_path = dest_dir.join(&base);
    let part_path = dest_dir.join(format!("{base}.part"));

    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    // The HF CDN throws transient 503s; retry those (and transport errors)
    // with backoff, but fail 4xx immediately — they won't get better.
    let mut attempt = 0;
    let resp = loop {
        attempt += 1;
        match agent().get(&url).call() {
            Ok(resp) => break resp,
            Err(ureq::Error::Status(code, _)) if code < 500 => {
                return Err(format!("status code {code} for {url}"));
            }
            Err(e) => {
                if attempt >= 3 || cancel.load(Ordering::Relaxed) {
                    return Err(e.to_string());
                }
                std::thread::sleep(Duration::from_secs(2 * attempt));
            }
        }
    };
    let total: u64 = resp
        .header("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(&part_path).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 1 << 16];
    let mut got: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(out);
            let _ = std::fs::remove_file(&part_path);
            return Ok(None);
        }
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        got += n as u64;
        if got - last_report >= 2 * 1024 * 1024 {
            last_report = got;
            let _ = tx.send(HubEvent::Progress {
                file: display.to_string(),
                got,
                total,
            });
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    if total > 0 && got < total {
        let _ = std::fs::remove_file(&part_path);
        return Err(format!("connection closed early ({got} of {total} bytes)"));
    }
    std::fs::rename(&part_path, &final_path).map_err(|e| e.to_string())?;
    Ok(Some(final_path))
}

/// Percent-encode everything outside the RFC 3986 unreserved set.
fn urlencode(s: &str) -> String {
    const QUERY: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
        .remove(b'-')
        .remove(b'.')
        .remove(b'_')
        .remove(b'~');
    percent_encoding::utf8_percent_encode(s, QUERY).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn search_response_parses_both_id_shapes() {
        let v: serde_json::Value = serde_json::from_str(
            r#"[
                {"id": "ggerganov/whisper.cpp", "downloads": 123, "likes": 7},
                {"modelId": "x/y"},
                {"downloads": 5}
            ]"#,
        )
        .unwrap();
        let hits = parse_search(&v);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, "ggerganov/whisper.cpp");
        assert_eq!(hits[0].downloads, 123);
        assert_eq!(hits[1].id, "x/y");
        assert_eq!(hits[1].downloads, 0);
    }

    #[test]
    fn file_listing_keeps_only_ggml_family_sorted() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"siblings": [
                {"rfilename": "README.md", "size": 10},
                {"rfilename": "z.gguf", "size": 2},
                {"rfilename": "ggml-tiny.bin", "size": 75000000},
                {"rfilename": "encoder.mlmodelc.zip", "size": 9},
                {"rfilename": "sub/dir/model.bin"}
            ]}"#,
        )
        .unwrap();
        let files = parse_files(&v);
        let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["ggml-tiny.bin", "sub/dir/model.bin", "z.gguf"]);
        assert_eq!(files[0].size_bytes, 75_000_000);
        assert_eq!(files[1].size_bytes, 0); // size missing → unknown
    }

    #[test]
    fn dest_name_strips_repo_subdirs() {
        assert_eq!(dest_name("ggml-tiny.bin").as_deref(), Some("ggml-tiny.bin"));
        assert_eq!(dest_name("sub/dir/model.bin").as_deref(), Some("model.bin"));
        assert_eq!(dest_name(""), None);
    }

    #[test]
    fn urlencode_escapes_query_text() {
        assert_eq!(urlencode("whisper large-v3"), "whisper%20large-v3");
        assert_eq!(urlencode("a&b=c"), "a%26b%3Dc");
        assert_eq!(urlencode("safe-._~09AZ"), "safe-._~09AZ");
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
