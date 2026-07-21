//! The MLX engine: runs directory-shaped models (Parakeet, Qwen3-ASR,
//! Canary, Whisper-MLX, …) by shelling out to mlx-audio's STT CLI
//! (`python -m mlx_audio.stt.generate`) and parsing the JSON it writes.
//! The runtime is self-provisioning: if no Python with mlx_audio is
//! found, the engine installs mlx-audio into an app-managed venv on
//! first use. The subprocess loads the model each job (cold start);
//! residency between jobs is a whisper.cpp-only concern.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Context;

use super::Engine;
use crate::audio;
use crate::transcribe::{Event, Job, Segment};

/// Shown whenever the mlx-audio runtime cannot be set up automatically.
const INSTALL_HINT: &str = "install it manually with: pip3 install mlx-audio (needs Python ≥ 3.10)";

/// How often child processes are polled for exit / cancellation.
const POLL_INTERVAL: Duration = Duration::from_millis(120);

/// Python interpreters probed, in order — both for an existing
/// mlx_audio install and as the base interpreter for creating the
/// app-managed venv.
const PYTHON_CANDIDATES: [&str; 4] = [
    "python3",
    "python",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
];

pub(super) struct MlxEngine {
    /// Interpreter that passed the mlx_audio probe. Only success is
    /// cached: a failed probe re-runs next job, so an install done
    /// outside the app mid-session is picked up without restarting.
    python: Option<PathBuf>,
}

impl MlxEngine {
    pub(super) fn new() -> Self {
        Self { python: None }
    }

    /// The interpreter to run mlx-audio with, setting the runtime up on
    /// first use if it is missing. `Ok(None)` means the user cancelled
    /// mid-install.
    fn ensure_runtime(
        &mut self,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> anyhow::Result<Option<PathBuf>> {
        anyhow::ensure!(
            cfg!(all(target_os = "macos", target_arch = "aarch64")),
            "MLX models need an Apple Silicon Mac — pick a GGML/GGUF model instead"
        );
        if let Some(python) = &self.python {
            return Ok(Some(python.clone()));
        }
        let python = match find_mlx_python() {
            Some(python) => python,
            // Missing component → the app provisions it itself, once
            None => match install_runtime(events, cancel)? {
                Some(python) => python,
                None => return Ok(None),
            },
        };
        self.python = Some(python.clone());
        Ok(Some(python))
    }
}

/// The app-managed Python environment. A private venv keeps the install
/// out of the system/Homebrew Python (whose PEP 668 guard rejects plain
/// `pip install`) and survives OS Python upgrades untouched.
fn venv_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("TRANSCRIBE_STT_MLX_VENV") {
        return Some(PathBuf::from(dir));
    }
    Some(
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/Application Support/transcribe-stt/mlx-venv"),
    )
}

fn venv_python() -> Option<PathBuf> {
    let python = venv_dir()?.join("bin/python3");
    python.is_file().then_some(python)
}

/// Does this interpreter have the mlx_audio package? find_spec instead
/// of a real import: answers in ~50 ms without mlx's import cost.
fn has_mlx_audio(python: &Path) -> bool {
    Command::new(python)
        .args([
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('mlx_audio') else 1)",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// An interpreter that can import mlx_audio: the app venv first, then
/// any Python the user installed it into themselves.
fn find_mlx_python() -> Option<PathBuf> {
    venv_python()
        .filter(|p| has_mlx_audio(p))
        .or_else(|| {
            PYTHON_CANDIDATES
                .iter()
                .map(PathBuf::from)
                .find(|p| has_mlx_audio(p))
        })
}

/// A Python ≥ 3.10 (mlx-audio's floor) that can create the venv.
fn find_base_python() -> Option<PathBuf> {
    PYTHON_CANDIDATES.iter().map(PathBuf::from).find(|p| {
        Command::new(p)
            .args(["-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// One-time runtime setup: create the app venv and pip-install mlx-audio
/// into it. Several hundred MB of packages, so this can take minutes —
/// progress is the indeterminate load spinner, and cancel kills the
/// child (`Ok(None)`). A partial venv is safe: the next attempt resumes
/// (venv creation is idempotent, pip skips what's already there).
fn install_runtime(
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<PathBuf>> {
    let base = find_base_python().ok_or_else(|| {
        anyhow::anyhow!(
            "no Python ≥ 3.10 found to set up the MLX runtime — brew install python, or {INSTALL_HINT}"
        )
    })?;
    let venv = venv_dir()
        .ok_or_else(|| anyhow::anyhow!("no home directory to hold the MLX runtime"))?;
    let _ = events.send(Event::LoadingModel(
        "mlx-audio runtime (one-time setup, ~2 min)".into(),
    ));
    let _ = events.send(Event::LoadProgress(-1)); // indeterminate

    if let Some(parent) = venv.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let Some((status, stderr)) =
        run_cancellable(Command::new(&base).arg("-m").arg("venv").arg(&venv), cancel)?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        status.success(),
        "creating the MLX runtime venv failed: {}",
        error_summary(&stderr)
    );

    let python = venv.join("bin").join("python3");
    let Some((status, stderr)) = run_cancellable(
        Command::new(&python).args(["-m", "pip", "install", "--quiet", "mlx-audio"]),
        cancel,
    )?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        status.success(),
        "installing mlx-audio failed: {} — {INSTALL_HINT}",
        error_summary(&stderr)
    );
    anyhow::ensure!(
        has_mlx_audio(&python),
        "mlx-audio installed but does not import — {INSTALL_HINT}"
    );
    Ok(Some(python))
}

/// Whether the MLX runtime is ready, for UI badges. The venv check is a
/// cheap stat — so an in-app install flips the badge live — while the
/// system-Python probe (a subprocess) runs once per process.
pub fn mlx_audio_available() -> bool {
    static SYSTEM_PROBE: OnceLock<bool> = OnceLock::new();
    venv_python().is_some() || *SYSTEM_PROBE.get_or_init(|| find_mlx_python().is_some())
}

/// Run a command to completion, polling `cancel` (the child is killed on
/// request → `Ok(None)`) and draining stderr on a thread so a chatty
/// child can't fill the pipe and deadlock. Stdout is discarded.
fn run_cancellable(
    cmd: &mut Command,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<(ExitStatus, String)>> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", cmd.get_program().to_string_lossy()))?;
    let stderr = child.stderr.take();
    let drain = std::thread::spawn(move || {
        use std::io::Read;
        let mut text = String::new();
        if let Some(mut pipe) = stderr {
            let _ = pipe.read_to_string(&mut text);
        }
        text
    });
    let status = loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = drain.join();
            return Ok(None);
        }
        match child.try_wait()? {
            Some(status) => break status,
            None => std::thread::sleep(POLL_INTERVAL),
        }
    };
    let stderr_text = drain.join().unwrap_or_default();
    Ok(Some((status, stderr_text)))
}

impl Engine for MlxEngine {
    fn run(
        &mut self,
        job: &Job,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        let Some(python) = self.ensure_runtime(events, cancel)? else {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        };

        let _ = events.send(Event::Decoding);
        let Some(decoded) = audio::load_media_with(&job.audio, Some(cancel.as_ref()), |p| {
            let _ = events.send(Event::DecodeProgress(p));
        })?
        else {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        };
        let _ = events.send(Event::AudioInfo {
            duration_secs: decoded.duration_secs,
        });
        // mlx-audio reports no fine-grained progress — indeterminate spinner
        let _ = events.send(Event::Progress(-1));

        // Hand the subprocess our decoded 16 kHz mono audio so video and
        // exotic codecs work exactly like they do on the whisper path.
        let scratch = Scratch::new();
        audio::write_wav_16k_mono(&scratch.wav, &decoded.samples)?;

        let started = Instant::now();
        let mut cmd = Command::new(&python);
        cmd.args(["-m", "mlx_audio.stt.generate", "--model"])
            .arg(&job.model)
            .arg("--audio")
            .arg(&scratch.wav)
            .arg("--output-path")
            .arg(&scratch.out_base)
            .args(["--format", "json"]);
        // Only forward an explicit hint; each family has its own default
        // (auto-detection or English) that a blind flag would override.
        if let Some(lang) = &job.language {
            cmd.args(["--language", lang]);
        }
        // run_cancellable keeps the child's chatter off the TUI's
        // terminal and kills it on cancel.
        let Some((status, stderr_text)) = run_cancellable(&mut cmd, cancel)? else {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        };
        anyhow::ensure!(
            status.success(),
            "mlx-audio failed on {}: {}",
            job.model
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            error_summary(&stderr_text)
        );

        let segments = scratch.read_output(decoded.duration_secs)?;
        for segment in &segments {
            let _ = events.send(Event::Segment(segment.clone()));
        }
        let _ = events.send(Event::Done {
            elapsed_secs: started.elapsed().as_secs_f32(),
            audio_secs: decoded.duration_secs,
            language: None,
        });
        Ok(())
    }

    fn loaded(&self) -> bool {
        false // the model lives on disk; nothing stays resident
    }

    fn unload(&mut self) {}
}

/// Temp files for one job: the input WAV plus mlx-audio's output (it
/// appends `.json` — or `.txt` when a model yields no segments — to the
/// base path we pass). Removed on drop, so every exit path cleans up.
struct Scratch {
    wav: PathBuf,
    out_base: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir().join(format!(
            "transcribe-stt-mlx-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::Relaxed)
        ));
        Self {
            wav: base.with_extension("wav"),
            out_base: base,
        }
    }

    fn json_path(&self) -> PathBuf {
        self.out_base.with_extension("json")
    }

    fn txt_path(&self) -> PathBuf {
        self.out_base.with_extension("txt")
    }

    fn read_output(&self, audio_secs: f32) -> anyhow::Result<Vec<Segment>> {
        if let Ok(text) = std::fs::read_to_string(self.json_path()) {
            return parse_output_json(&text, audio_secs);
        }
        // mlx-audio saves plain text instead when a model returns no
        // segment timing — surface it as one whole-file segment.
        if let Ok(text) = std::fs::read_to_string(self.txt_path()) {
            return Ok(whole_file_segment(&text, audio_secs).into_iter().collect());
        }
        anyhow::bail!("mlx-audio exited successfully but wrote no output")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for path in [self.wav.clone(), self.json_path(), self.txt_path()] {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// The last meaningful stderr line — for Python that is the exception
/// message at the end of the traceback.
fn error_summary(stderr: &str) -> String {
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("no error output")
        .to_string()
}

fn whole_file_segment(text: &str, audio_secs: f32) -> Option<Segment> {
    let text = text.trim();
    (!text.is_empty()).then(|| Segment {
        start_ms: 0,
        end_ms: (audio_secs * 1000.0).round() as i64,
        text: text.to_string(),
        speaker: None,
    })
}

/// Map mlx-audio's JSON to `Segment`s. Whisper-family models write a
/// `segments` list, Parakeet-family a `sentences` list; both use `start`
/// / `end` in float seconds. `speaker_id` appears only for diarizing
/// models. A file with neither list falls back to its whole-file `text`.
fn parse_output_json(text: &str, audio_secs: f32) -> anyhow::Result<Vec<Segment>> {
    let v: serde_json::Value =
        serde_json::from_str(text).context("parsing mlx-audio JSON output")?;
    let list = ["segments", "sentences"]
        .iter()
        .find_map(|key| v[*key].as_array().filter(|a| !a.is_empty()));
    let Some(list) = list else {
        return Ok(whole_file_segment(v["text"].as_str().unwrap_or(""), audio_secs)
            .into_iter()
            .collect());
    };
    let seconds = |item: &serde_json::Value, key: &str, alt: &str| {
        item[key].as_f64().or_else(|| item[alt].as_f64()).unwrap_or(0.0)
    };
    Ok(list
        .iter()
        .filter_map(|item| {
            let text = item["text"].as_str().unwrap_or("").to_string();
            if text.trim().is_empty() {
                return None;
            }
            Some(Segment {
                start_ms: (seconds(item, "start", "start_time") * 1000.0).round() as i64,
                end_ms: (seconds(item, "end", "end_time") * 1000.0).round() as i64,
                text,
                speaker: item.get("speaker_id").and_then(speaker_index),
            })
        })
        .collect())
}

/// `speaker_id` arrives as an integer or a string like "speaker_1".
fn speaker_index(v: &serde_json::Value) -> Option<u8> {
    if let Some(n) = v.as_u64() {
        return Some(n.min(u8::MAX as u64) as u8);
    }
    let digits: String = v.as_str()?.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u64>().ok().map(|n| n.min(u8::MAX as u64) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `segments` shape (Whisper-family), fields as save_as_json
    /// writes them — including extras our mapping must ignore.
    #[test]
    fn parses_segments_shape_with_speakers_and_word_extras() {
        let json = r#"{
            "text": "hello world again",
            "segments": [
                {"text": " hello world", "start": 0.0, "end": 2.48, "duration": 2.48,
                 "words": [{"word": "hello", "start": 0.0, "end": 1.0}]},
                {"text": "again", "start": 2.48, "end": 4.0, "duration": 1.52, "speaker_id": 1},
                {"text": "   ", "start": 4.0, "end": 4.2, "duration": 0.2}
            ]
        }"#;
        let segments = parse_output_json(json, 4.2).unwrap();
        assert_eq!(segments.len(), 2); // blank segment dropped
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 2480);
        assert_eq!(segments[0].text, " hello world");
        assert_eq!(segments[0].speaker, None);
        assert_eq!(segments[1].speaker, Some(1));
    }

    /// The `sentences` shape (Parakeet-family), with token sub-lists and a
    /// string speaker id.
    #[test]
    fn parses_sentences_shape() {
        let json = r#"{
            "text": "guten tag",
            "sentences": [
                {"text": "guten tag", "start": 0.5, "end": 1.75, "duration": 1.25,
                 "tokens": [{"text": "guten", "start": 0.5, "end": 1.0, "duration": 0.5}],
                 "speaker_id": "speaker_2"}
            ]
        }"#;
        let segments = parse_output_json(json, 2.0).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 500);
        assert_eq!(segments[0].end_ms, 1750);
        assert_eq!(segments[0].speaker, Some(2));
    }

    #[test]
    fn text_only_output_becomes_one_whole_file_segment() {
        let json = r#"{"text": "  just text  ", "segments": []}"#;
        let segments = parse_output_json(json, 12.5).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 12500);
        assert_eq!(segments[0].text, "just text");

        assert!(parse_output_json(r#"{"text": ""}"#, 1.0).unwrap().is_empty());
        assert!(parse_output_json("not json", 1.0).is_err());
    }

    #[test]
    fn tolerates_legacy_start_time_keys() {
        let json = r#"{"segments": [{"text": "hi", "start_time": 1.0, "end_time": 2.0}]}"#;
        let segments = parse_output_json(json, 2.0).unwrap();
        assert_eq!(segments[0].start_ms, 1000);
        assert_eq!(segments[0].end_ms, 2000);
    }

    #[test]
    fn error_summary_takes_the_last_traceback_line() {
        let stderr = "Traceback (most recent call last):\n  File x\nValueError: bad model\n\n";
        assert_eq!(error_summary(stderr), "ValueError: bad model");
        assert_eq!(error_summary(""), "no error output");
    }

    #[test]
    fn run_cancellable_reports_status_and_stderr_and_honours_cancel() {
        // success with stderr captured
        let (status, stderr) = run_cancellable(
            Command::new("sh").args(["-c", "echo oops >&2; exit 3"]),
            &Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
        .expect("not cancelled");
        assert_eq!(status.code(), Some(3));
        assert_eq!(error_summary(&stderr), "oops");

        // a pre-set cancel flag kills the child immediately
        let cancelled = run_cancellable(
            Command::new("sleep").arg("30"),
            &Arc::new(AtomicBool::new(true)),
        )
        .unwrap();
        assert!(cancelled.is_none());
    }
}
