//! Hugging Face API calls: repo search and file listing, each on a
//! background thread reporting over the hub event channel.

use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Duration;

use serde::Deserialize;

use super::{DirVariant, HubEvent, HubFile, RepoHit};
use crate::models::is_model_file;

pub(super) fn agent() -> ureq::Agent {
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
        let result = (|| -> anyhow::Result<Vec<RepoHit>> {
            let resp = agent().get(&url).call()?;
            let v: serde_json::Value = serde_json::from_reader(resp.into_reader())?;
            Ok(parse_search(&v))
        })();
        let _ = tx.send(match result {
            Ok(hits) => HubEvent::SearchResults { query, hits },
            // Errors cross the thread boundary as display strings
            Err(error) => HubEvent::SearchFailed {
                query,
                error: format!("{error:#}"),
            },
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

/// List what a repo offers for download: its GGML/GGUF files (with
/// sizes) plus any directory-model variants (root or subfolders holding
/// config.json + safetensors).
pub fn list_files(repo: String, tx: Sender<HubEvent>) {
    std::thread::spawn(move || {
        let result = fetch_repo_json(&repo).map(|v| {
            let files = parse_files(&v);
            let variants = find_dir_variants(&parse_all_files(&v));
            (files, variants)
        });
        let _ = tx.send(match result {
            Ok((files, variants)) => HubEvent::Files {
                repo,
                files,
                variants,
            },
            Err(error) => HubEvent::FilesFailed {
                repo,
                error: format!("{error:#}"),
            },
        });
    });
}

/// Find directory-model variants in a repo's full file list: every folder
/// (including the root, "") that directly holds a `config.json` and MLX
/// weights (safetensors / npz). Sizes sum everything under the folder.
pub(super) fn find_dir_variants(files: &[HubFile]) -> Vec<DirVariant> {
    let parent = |name: &str| {
        Path::new(name)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let mut roots: Vec<String> = files
        .iter()
        .filter(|f| Path::new(&f.name).file_name().is_some_and(|n| n == "config.json"))
        .map(|f| parent(&f.name))
        .filter(|dir| {
            files.iter().any(|f| {
                parent(&f.name) == *dir && crate::models::is_mlx_weights_file(&f.name)
            })
        })
        .collect();
    roots.sort();
    roots
        .into_iter()
        .map(|subdir| {
            let under: Vec<&HubFile> = files
                .iter()
                .filter(|f| {
                    subdir.is_empty() || f.name.starts_with(&format!("{subdir}/"))
                })
                .collect();
            DirVariant {
                size_bytes: under.iter().map(|f| f.size_bytes).sum(),
                file_count: under.len(),
                subdir,
            }
        })
        .collect()
}

/// Every file a whole-repo (directory model) download must fetch, with
/// sizes. Blocking — call from a worker thread.
pub(super) fn fetch_repo_files(repo: &str) -> anyhow::Result<Vec<HubFile>> {
    Ok(parse_all_files(&fetch_repo_json(repo)?))
}

fn fetch_repo_json(repo: &str) -> anyhow::Result<serde_json::Value> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let resp = agent().get(&url).call()?;
    Ok(serde_json::from_reader(resp.into_reader())?)
}

fn parse_siblings(v: &serde_json::Value, keep: impl Fn(&str) -> bool) -> Vec<HubFile> {
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
                .filter(|s| keep(&s.rfilename))
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

fn parse_files(v: &serde_json::Value) -> Vec<HubFile> {
    parse_siblings(v, |name| is_model_file(Path::new(name)))
}

/// Everything except git internals and other dotfiles.
fn parse_all_files(v: &serde_json::Value) -> Vec<HubFile> {
    parse_siblings(v, |name| {
        !Path::new(name)
            .components()
            .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
    })
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
    fn full_file_listing_keeps_everything_but_dotfiles() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"siblings": [
                {"rfilename": ".gitattributes", "size": 1519},
                {"rfilename": "README.md", "size": 10},
                {"rfilename": "config.json", "size": 244093},
                {"rfilename": "model.safetensors", "size": 2508288736},
                {"rfilename": "sub/.hidden", "size": 5},
                {"rfilename": "tokenizer.model", "size": 360916}
            ]}"#,
        )
        .unwrap();
        let files = parse_all_files(&v);
        let names: Vec<&str> = files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["README.md", "config.json", "model.safetensors", "tokenizer.model"]
        );
    }

    #[test]
    fn dir_variants_found_at_root_and_in_subfolders() {
        let file = |name: &str, size| HubFile {
            name: name.into(),
            size_bytes: size,
        };
        // A root-level MLX model
        let root = [
            file("README.md", 10),
            file("config.json", 100),
            file("model.safetensors", 5000),
            file("tokenizer.model", 40),
        ];
        let variants = find_dir_variants(&root);
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].subdir, "");
        assert_eq!(variants[0].size_bytes, 5150);
        assert_eq!(variants[0].file_count, 4);

        // Quantization variants in subfolders, each its own download
        let quants = [
            file("README.md", 10),
            file("4bit/config.json", 100),
            file("4bit/model.safetensors", 2000),
            file("8bit/config.json", 100),
            file("8bit/model.safetensors", 4000),
            file("8bit/tokenizer.model", 40),
        ];
        let variants = find_dir_variants(&quants);
        let labels: Vec<(&str, u64, usize)> = variants
            .iter()
            .map(|v| (v.subdir.as_str(), v.size_bytes, v.file_count))
            .collect();
        assert_eq!(labels, vec![("4bit", 2100, 2), ("8bit", 4140, 3)]);

        // Legacy npz conversions can't be run by mlx-audio → not offered
        let npz = [file("config.json", 262), file("weights.npz", 74418540)];
        assert!(find_dir_variants(&npz).is_empty());

        // config.json without weights (or vice versa) is not a variant
        let none = [
            file("config.json", 100),
            file("weights.bin", 900),
            file("other/model.safetensors", 500),
        ];
        assert!(find_dir_variants(&none).is_empty());
    }

    #[test]
    fn urlencode_escapes_query_text() {
        assert_eq!(urlencode("whisper large-v3"), "whisper%20large-v3");
        assert_eq!(urlencode("a&b=c"), "a%26b%3Dc");
        assert_eq!(urlencode("safe-._~09AZ"), "safe-._~09AZ");
    }
}
