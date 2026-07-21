//! Hugging Face API calls: repo search and file listing, each on a
//! background thread reporting over the hub event channel.

use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Duration;

use serde::Deserialize;

use super::{HubEvent, HubFile, RepoHit};
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

/// List the GGML/GGUF files (with sizes) inside a repo.
pub fn list_files(repo: String, tx: Sender<HubEvent>) {
    std::thread::spawn(move || {
        let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
        let result = (|| -> anyhow::Result<Vec<HubFile>> {
            let resp = agent().get(&url).call()?;
            let v: serde_json::Value = serde_json::from_reader(resp.into_reader())?;
            Ok(parse_files(&v))
        })();
        let _ = tx.send(match result {
            Ok(files) => HubEvent::Files { repo, files },
            Err(error) => HubEvent::FilesFailed {
                repo,
                error: format!("{error:#}"),
            },
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
    fn urlencode_escapes_query_text() {
        assert_eq!(urlencode("whisper large-v3"), "whisper%20large-v3");
        assert_eq!(urlencode("a&b=c"), "a%26b%3Dc");
        assert_eq!(urlencode("safe-._~09AZ"), "safe-._~09AZ");
    }
}
