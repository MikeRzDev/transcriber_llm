//! Streaming model downloads: `.part` file + rename-into-place, retry
//! with backoff on transient CDN errors, cancellation cleanup.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use super::api::agent;
use super::{dest_name, HubEvent};

/// Transfer buffer size.
const STREAM_BUF_BYTES: usize = 1 << 16;
/// Progress events fire at most once per this many bytes.
const PROGRESS_STEP_BYTES: u64 = 2 * 1024 * 1024;
/// Transient-error retries before giving up.
const MAX_ATTEMPTS: u64 = 3;

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
                error: format!("{error:#}"),
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
) -> anyhow::Result<Option<PathBuf>> {
    std::fs::create_dir_all(dest_dir)?;
    let base = dest_name(file).ok_or_else(|| anyhow::anyhow!("bad file name"))?;
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
                anyhow::bail!("status code {code} for {url}");
            }
            Err(e) => {
                if attempt >= MAX_ATTEMPTS || cancel.load(Ordering::Relaxed) {
                    return Err(e.into());
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
    let mut out = std::fs::File::create(&part_path)?;
    let mut buf = [0u8; STREAM_BUF_BYTES];
    let mut got: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(out);
            let _ = std::fs::remove_file(&part_path);
            return Ok(None);
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        got += n as u64;
        if got - last_report >= PROGRESS_STEP_BYTES {
            last_report = got;
            let _ = tx.send(HubEvent::Progress {
                file: display.to_string(),
                got,
                total,
            });
        }
    }
    out.flush()?;
    drop(out);
    if total > 0 && got < total {
        let _ = std::fs::remove_file(&part_path);
        anyhow::bail!("connection closed early ({got} of {total} bytes)");
    }
    std::fs::rename(&part_path, &final_path)?;
    Ok(Some(final_path))
}
