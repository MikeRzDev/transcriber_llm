//! Streaming model downloads: `.part` file + rename-into-place, retry
//! with backoff on transient CDN errors, pause (resume via HTTP Range) and
//! cancellation cleanup. Single-file GGML models and whole-repo directory
//! models (MLX) share the same streaming core.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Duration;

use super::api::agent;
use super::{dest_name, variant_dir_name, HubEvent, HubFile};

/// Transfer buffer size.
const STREAM_BUF_BYTES: usize = 1 << 16;
/// Progress events fire at most once per this many bytes.
const PROGRESS_STEP_BYTES: u64 = 2 * 1024 * 1024;
/// Transient-error retries before giving up.
const MAX_ATTEMPTS: u64 = 3;

/// How a transfer ended.
enum Outcome {
    Done(PathBuf),
    Cancelled,
    /// Stopped on request; the partial data is kept at this many bytes.
    Paused(u64),
}

/// How one file's stream ended (a directory download runs many of
/// these; the diarizer fetches its ONNX models through the same core).
pub(crate) enum FileOutcome {
    Done,
    Cancelled,
    Paused(u64),
}

/// Stream `repo/file` into `dest_dir` as `<name>.part`, renaming into place
/// when complete. Progress is reported every couple of MB. `cancel` stops the
/// transfer and removes the partial file; `pause` stops it but keeps the
/// partial so a later call resumes from where it left off (HTTP Range).
pub fn download(
    repo: String,
    file: String,
    dest_dir: PathBuf,
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    tx: Sender<HubEvent>,
) {
    let save_as = dest_name(&file).unwrap_or_else(|| file.clone());
    download_as(repo, file, save_as, dest_dir, cancel, pause, tx);
}

/// `download`, but saved under an explicit local name instead of the
/// remote file's base name — used by the diarization catalog, whose
/// repos name their models generically (`model.onnx`).
pub fn download_as(
    repo: String,
    file: String,
    save_as: String,
    dest_dir: PathBuf,
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    tx: Sender<HubEvent>,
) {
    std::thread::spawn(move || {
        let result = download_blocking(&repo, &file, &save_as, &dest_dir, &cancel, &pause, &tx);
        let _ = tx.send(finish_event(result, save_as));
    });
}

/// Download a whole directory model — every (non-dot) file of `repo`
/// under `subdir` ("" = the whole repo, else one variant subfolder) —
/// into `<dest_dir>/<name>.part/`, renaming the directory to `<name>`
/// when all files are complete. `scan_models` ignores `.part` names, so
/// a partial dir is never a model. Progress aggregates bytes across all
/// files. Pause keeps the partial dir (a later call resumes, skipping
/// complete files); cancel removes it.
pub fn download_dir(
    repo: String,
    subdir: String,
    dest_dir: PathBuf,
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    tx: Sender<HubEvent>,
) {
    std::thread::spawn(move || {
        let name = variant_dir_name(&repo, &subdir);
        let result =
            download_dir_blocking(&repo, &subdir, &name, &dest_dir, &cancel, &pause, &tx);
        let _ = tx.send(finish_event(result, name));
    });
}

fn finish_event(result: anyhow::Result<Outcome>, display: String) -> HubEvent {
    match result {
        Ok(Outcome::Done(path)) => HubEvent::Done {
            file: display,
            path,
        },
        Ok(Outcome::Cancelled) => HubEvent::Cancelled { file: display },
        Ok(Outcome::Paused(got)) => HubEvent::Paused { file: display, got },
        Err(error) => HubEvent::Failed {
            file: display,
            error: format!("{error:#}"),
        },
    }
}

fn download_blocking(
    repo: &str,
    file: &str,
    save_as: &str,
    dest_dir: &Path,
    cancel: &AtomicBool,
    pause: &AtomicBool,
    tx: &Sender<HubEvent>,
) -> anyhow::Result<Outcome> {
    anyhow::ensure!(!save_as.is_empty(), "bad file name");
    std::fs::create_dir_all(dest_dir)?;
    let final_path = dest_dir.join(save_as);
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    let mut report = |got, total| {
        let _ = tx.send(HubEvent::Progress {
            file: save_as.to_string(),
            got,
            total,
        });
    };
    Ok(
        match stream_file(&url, &final_path, cancel, pause, &mut report)? {
            FileOutcome::Done => Outcome::Done(final_path),
            FileOutcome::Cancelled => Outcome::Cancelled,
            FileOutcome::Paused(got) => Outcome::Paused(got),
        },
    )
}

fn download_dir_blocking(
    repo: &str,
    subdir: &str,
    name: &str,
    dest_dir: &Path,
    cancel: &AtomicBool,
    pause: &AtomicBool,
    tx: &Sender<HubEvent>,
) -> anyhow::Result<Outcome> {
    // File names are local from here on: the variant subdir prefix is
    // stripped for disk paths and re-attached when building URLs.
    let files = variant_files(super::api::fetch_repo_files(repo)?, subdir);
    anyhow::ensure!(!files.is_empty(), "no downloadable files in {repo}/{subdir}");
    let part_dir = dest_dir.join(format!("{name}.part"));
    let final_dir = dest_dir.join(name);
    // A model that failed its integrity manifest comes back through here:
    // demote it to `.part` and only its missing files get fetched below.
    if final_dir.exists() && !part_dir.exists() {
        std::fs::rename(&final_dir, &part_dir)?;
    }
    std::fs::create_dir_all(&part_dir)?;
    // The integrity manifest: what a complete copy of this model holds.
    // Written first (and shipped inside the model dir) so any later scan
    // can prove the files are all present and full-size.
    crate::models::DirManifest {
        repo: repo.to_string(),
        subdir: subdir.to_string(),
        files: files
            .iter()
            .map(|f| crate::models::ManifestEntry {
                name: f.name.clone(),
                size: f.size_bytes,
            })
            .collect(),
    }
    .write(&part_dir)?;

    let plan = plan_dir_download(&files, &part_dir);
    let mut done = plan.done_bytes;
    let total = plan.total_bytes;
    // Report the resume offset at once so the gauge starts truthful
    let _ = tx.send(HubEvent::Progress {
        file: name.to_string(),
        got: done,
        total,
    });

    for file in &plan.remaining {
        let dest = part_dir.join(&file.name);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let remote = if subdir.is_empty() {
            file.name.clone()
        } else {
            format!("{subdir}/{}", file.name)
        };
        let url = format!("https://huggingface.co/{repo}/resolve/main/{remote}");
        // Per-file progress rides on top of the bytes already banked
        let mut report = |got, _file_total| {
            let _ = tx.send(HubEvent::Progress {
                file: name.to_string(),
                got: done + got,
                total,
            });
        };
        match stream_file(&url, &dest, cancel, pause, &mut report)? {
            FileOutcome::Done => {
                done += std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            }
            FileOutcome::Cancelled => {
                let _ = std::fs::remove_dir_all(&part_dir);
                return Ok(Outcome::Cancelled);
            }
            FileOutcome::Paused(got) => return Ok(Outcome::Paused(done + got)),
        }
    }

    std::fs::rename(&part_dir, &final_dir)?;
    Ok(Outcome::Done(final_dir))
}

/// The files belonging to one variant, with the subdir prefix stripped
/// (their local, on-disk names).
fn variant_files(files: Vec<HubFile>, subdir: &str) -> Vec<HubFile> {
    if subdir.is_empty() {
        return files;
    }
    let prefix = format!("{subdir}/");
    files
        .into_iter()
        .filter_map(|f| {
            f.name.strip_prefix(&prefix).map(|local| HubFile {
                name: local.to_string(),
                size_bytes: f.size_bytes,
            })
        })
        .collect()
}

/// What a directory download still has to fetch: files not already
/// complete (present at full size) in the partial dir from an earlier
/// paused run.
struct DirPlan {
    remaining: Vec<HubFile>,
    done_bytes: u64,
    total_bytes: u64,
}

fn plan_dir_download(files: &[HubFile], part_dir: &Path) -> DirPlan {
    let mut plan = DirPlan {
        remaining: Vec::new(),
        done_bytes: 0,
        total_bytes: 0,
    };
    for file in files {
        plan.total_bytes += file.size_bytes;
        let on_disk = std::fs::metadata(part_dir.join(&file.name))
            .map(|m| m.len())
            .unwrap_or(0);
        if file.size_bytes > 0 && on_disk == file.size_bytes {
            plan.done_bytes += on_disk;
        } else {
            plan.remaining.push(file.clone());
        }
    }
    plan
}

/// Stream `url` into `<final_path>.part` — resuming any existing partial
/// via HTTP Range — then rename into place. `on_progress` gets (got,
/// total) byte counts, throttled. Cancel removes the partial; pause keeps
/// it for a later resume.
pub(crate) fn stream_file(
    url: &str,
    final_path: &Path,
    cancel: &AtomicBool,
    pause: &AtomicBool,
    on_progress: &mut dyn FnMut(u64, u64),
) -> anyhow::Result<FileOutcome> {
    let part_path = final_path.with_file_name(format!(
        "{}.part",
        final_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));

    // A partial file left by an earlier pause is resumed with a Range request.
    let resume_from = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);

    // The HF CDN throws transient 503s; retry those (and transport errors)
    // with backoff, but fail 4xx immediately — they won't get better.
    let mut attempt = 0;
    let resp = loop {
        attempt += 1;
        let mut req = agent().get(url);
        if resume_from > 0 {
            req = req.set("Range", &format!("bytes={resume_from}-"));
        }
        match req.call() {
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

    // A 206 honours our Range and continues the file; a 200 means the server
    // ignored it and is sending the whole thing, so start over from the top.
    let resuming = resume_from > 0 && resp.status() == 206;
    let total = if resuming {
        resp.header("content-range")
            .and_then(total_from_content_range)
            .unwrap_or(0)
    } else {
        resp.header("content-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    };

    let mut reader = resp.into_reader();
    let (mut out, mut got) = if resuming {
        (
            std::fs::OpenOptions::new().append(true).open(&part_path)?,
            resume_from,
        )
    } else {
        (std::fs::File::create(&part_path)?, 0)
    };
    let mut buf = [0u8; STREAM_BUF_BYTES];
    let mut last_report = got;
    // Emit the starting offset so a resumed transfer shows real progress at once.
    on_progress(got, total);

    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(out);
            let _ = std::fs::remove_file(&part_path);
            return Ok(FileOutcome::Cancelled);
        }
        if pause.load(Ordering::Relaxed) {
            out.flush()?;
            drop(out);
            // Keep the .part file — a later start resumes from its length.
            return Ok(FileOutcome::Paused(got));
        }
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        got += n as u64;
        if got - last_report >= PROGRESS_STEP_BYTES {
            last_report = got;
            on_progress(got, total);
        }
    }
    out.flush()?;
    drop(out);
    if total > 0 && got < total {
        let _ = std::fs::remove_file(&part_path);
        anyhow::bail!("connection closed early ({got} of {total} bytes)");
    }
    std::fs::rename(&part_path, final_path)?;
    Ok(FileOutcome::Done)
}

/// Pull the total size out of a `Content-Range: bytes 200-1000/1001` header.
fn total_from_content_range(header: &str) -> Option<u64> {
    header.rsplit('/').next()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> PathBuf {
        use std::sync::atomic::AtomicUsize;
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "transcribe-stt-dl-test-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn file(name: &str, size_bytes: u64) -> HubFile {
        HubFile {
            name: name.into(),
            size_bytes,
        }
    }

    #[test]
    fn dir_plan_skips_complete_files_and_counts_bytes() {
        let part_dir = tempdir();
        // config.json fully downloaded, model.safetensors partially (as
        // its own .part, which the plan must not mistake for the file)
        std::fs::write(part_dir.join("config.json"), vec![0u8; 100]).unwrap();
        std::fs::write(part_dir.join("model.safetensors.part"), vec![0u8; 30]).unwrap();
        let files = [
            file("config.json", 100),
            file("model.safetensors", 5000),
            file("tokenizer.model", 40),
        ];

        let plan = plan_dir_download(&files, &part_dir);
        assert_eq!(plan.total_bytes, 5140);
        assert_eq!(plan.done_bytes, 100);
        let remaining: Vec<&str> = plan.remaining.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(remaining, vec!["model.safetensors", "tokenizer.model"]);
        std::fs::remove_dir_all(&part_dir).unwrap();
    }

    #[test]
    fn variant_files_strip_the_subdir_prefix() {
        let files = vec![
            file("README.md", 10),
            file("8bit/config.json", 100),
            file("8bit/model.safetensors", 4000),
            file("4bit/config.json", 100),
        ];
        let local = variant_files(files.clone(), "8bit");
        let names: Vec<&str> = local.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["config.json", "model.safetensors"]);
        // the root variant keeps everything as-is
        assert_eq!(variant_files(files.clone(), "").len(), 4);
    }

    #[test]
    fn dir_plan_refetches_size_mismatches_and_unknown_sizes() {
        let part_dir = tempdir();
        // truncated earlier download (wrong size) must be refetched
        std::fs::write(part_dir.join("vocab.txt"), vec![0u8; 10]).unwrap();
        // unknown upstream size (0) can never be proven complete
        std::fs::write(part_dir.join("README.md"), vec![0u8; 5]).unwrap();
        let files = [file("vocab.txt", 46772), file("README.md", 0)];

        let plan = plan_dir_download(&files, &part_dir);
        assert_eq!(plan.done_bytes, 0);
        assert_eq!(plan.remaining.len(), 2);
        std::fs::remove_dir_all(&part_dir).unwrap();
    }
}
