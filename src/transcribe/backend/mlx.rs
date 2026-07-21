//! The MLX engine: runs directory-shaped models (Parakeet, Qwen3-ASR,
//! Canary, Whisper-MLX, …) by shelling out to mlx-audio's STT CLI
//! (`python -m mlx_audio.stt.generate`) and parsing the JSON it writes.
//! The runtime is self-provisioning: if no Python with mlx_audio is
//! found, the engine installs mlx-audio into an app-managed venv on
//! first use. The subprocess loads the model each job (cold start);
//! residency between jobs is a whisper.cpp-only concern.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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

/// While a child runs, a liveness heartbeat is emitted this often…
const HEARTBEAT_EVERY: Duration = Duration::from_secs(5);
/// …but only after this much output silence — a chatty child needs none.
const QUIET_BEFORE_HEARTBEAT: Duration = Duration::from_secs(3);

/// Instrumented runner: instead of only reading what the stock CLI
/// happens to print, this script drives mlx-audio's Python API directly
/// and reports every internal stage — interpreter up, mlx import, model
/// load (timed separately), generation (with `verbose=True`, which makes
/// whisper-family models print each decoded segment live, plus token
/// stats and peak memory), and save. Any import/API drift in a future
/// mlx-audio version falls back to the stock CLI so jobs keep working.
///
/// `import mlx_audio.stt.models` must come first: generate.py sits in an
/// import cycle with the models package (glmasr imports back into it),
/// and only the models-first order lets the cycle complete — the same
/// order `-m mlx_audio.stt.generate` produces implicitly.
const PY_RUNNER: &str = r#"
import logging
import sys
import time

T0 = time.time()

def mark(msg):
    print(f"[stage +{time.time() - T0:5.1f}s] {msg}", flush=True)

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s: %(message)s")

flags = dict(zip(sys.argv[1::2], sys.argv[2::2]))

def _install_tqdm_probe():
    # Model implementations track their decode loops with tqdm (whisper:
    # frames, parakeet: chunks, …) but disable or bury the bar depending
    # on verbosity. Replacing tqdm with this subclass surfaces every
    # update as a machine-readable "@@pct N" line — the host turns those
    # into its real progress gauge — regardless of the disable flag.
    # Must run BEFORE mlx_audio imports (`from tqdm import tqdm` binds).
    try:
        import tqdm as _tqdm_mod
    except ImportError:
        return
    _real = _tqdm_mod.tqdm

    class ProbeTqdm(_real):
        def __init__(self, *args, **kwargs):
            self._probe_total = kwargs.get("total")
            if self._probe_total is None and args:
                self._probe_total = getattr(args[0], "__len__", lambda: None)()
            self._probe_n = 0
            self._probe_last = -1
            kwargs["disable"] = True  # we report; no raw bar spam
            super().__init__(*args, **kwargs)

        def _probe_emit(self):
            if not self._probe_total:
                return
            pct = min(100, int(self._probe_n * 100 / self._probe_total))
            if pct != self._probe_last:
                self._probe_last = pct
                print(f"@@pct {pct}", flush=True)

        def update(self, n=1):
            self._probe_n += n or 0
            self._probe_emit()
            return super().update(n)

        def __iter__(self):
            for item in super().__iter__():
                yield item
                self._probe_n += 1
                self._probe_emit()

    _tqdm_mod.tqdm = ProbeTqdm
    for name in ("tqdm.auto", "tqdm.std"):
        mod = sys.modules.get(name)
        if mod is not None:
            mod.tqdm = ProbeTqdm

def run_instrumented():
    import platform
    mark(f"python {platform.python_version()} up · importing mlx-audio (loads Metal)…")
    import mlx_audio.stt.models  # first: completes generate.py's import cycle
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model
    import mlx.core as mx
    mark(f"mlx-audio imported in {time.time() - T0:.1f}s")

    t = time.time()
    mark(f"loading model {flags['--model']}…")
    model = load_model(flags["--model"])
    mark(f"model loaded in {time.time() - t:.1f}s")

    kwargs = {}
    if "--language" in flags:
        kwargs["language"] = flags["--language"]
    t = time.time()
    mark("transcribing…")
    generate_transcription(
        model=model,
        audio=flags["--audio"],
        output_path=flags["--output-path"],
        format=flags.get("--format", "json"),
        verbose=True,
        **kwargs,
    )
    mark(
        f"transcription + save done in {time.time() - t:.1f}s"
        f" · peak memory {mx.get_peak_memory() / 1e9:.2f} GB"
    )

# ---- execution ----
_install_tqdm_probe()
try:
    run_instrumented()
except (ImportError, AttributeError, KeyError, TypeError) as e:
    mark(f"instrumented path failed ({type(e).__name__}: {e}); falling back to stock CLI")
    import runpy
    sys.argv = ["mlx_audio.stt.generate"] + sys.argv[1:]
    runpy.run_module("mlx_audio.stt.generate", run_name="__main__")
"#;

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
    let _ = events.send(Event::EngineLog(format!(
        "creating venv at {}",
        venv.display()
    )));
    let Some((status, stderr)) = run_cancellable(
        Command::new(&base).arg("-m").arg("venv").arg(&venv),
        cancel,
        Some(events),
    )?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        status.success(),
        "creating the MLX runtime venv failed: {}",
        error_summary(&stderr)
    );

    let python = venv.join("bin").join("python3");
    let _ = events.send(Event::EngineLog(
        "installing mlx-audio via pip (several hundred MB, can take minutes)…".into(),
    ));
    // No --quiet: pip's collect/download lines stream to the TUI so the
    // multi-minute install always shows what it is doing
    let Some((status, stderr)) = run_cancellable(
        Command::new(&python).args(["-m", "pip", "install", "mlx-audio"]),
        cancel,
        Some(events),
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

/// Printable text only: ANSI escape sequences and control bytes from a
/// child's progress bars would corrupt the ratatui status line.
fn sanitize_line(bytes: &[u8]) -> String {
    let raw = String::from_utf8_lossy(bytes);
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        match c {
            // Skip CSI sequences: ESC [ params… final-byte
            '\x1b' => {
                if chars.next() == Some('[') {
                    for c in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&c) {
                            break;
                        }
                    }
                }
            }
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out.trim().to_string()
}

/// Forward one completed line to the TUI, skipping blanks and repeats
/// (progress bars redraw the same line many times a second). Returns
/// whether a line was actually sent.
fn emit_line(line: &[u8], log: &Option<Sender<Event>>, last_sent: &mut String) -> bool {
    let Some(log) = log else { return false };
    let text = sanitize_line(line);
    if text.is_empty() || text == *last_sent {
        return false;
    }
    last_sent.clone_from(&text);
    // "@@pct N" comes from the runner's tqdm probe inside the model's
    // decode loop — it drives the real progress gauge, not the log
    if let Some(pct) = text
        .strip_prefix("@@pct ")
        .and_then(|p| p.trim().parse::<i32>().ok())
    {
        let _ = log.send(Event::Progress(pct.clamp(0, 100)));
        return true;
    }
    let _ = log.send(Event::EngineLog(text));
    true
}

/// Drain a child pipe on a thread, forwarding each line (\n or \r
/// terminated — tqdm/pip progress bars redraw with bare \r) as a live
/// `Event::EngineLog`, stamping `activity` (ms since `started`) on every
/// sent line so the heartbeat knows when the child went quiet. Join
/// returns the full accumulated text.
fn stream_lines<R: std::io::Read + Send + 'static>(
    mut pipe: R,
    log: Option<Sender<Event>>,
    activity: Arc<AtomicU64>,
    started: Instant,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut all: Vec<u8> = Vec::new();
        let mut line: Vec<u8> = Vec::new();
        let mut last_sent = String::new();
        let mut chunk = [0u8; 4096];
        let flush = |line: &[u8], last_sent: &mut String| {
            if emit_line(line, &log, last_sent) {
                activity.store(started.elapsed().as_millis() as u64, Ordering::Relaxed);
            }
        };
        loop {
            let n = match pipe.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            all.extend_from_slice(&chunk[..n]);
            for &b in &chunk[..n] {
                if b == b'\n' || b == b'\r' {
                    flush(&line, &mut last_sent);
                    line.clear();
                } else {
                    line.push(b);
                }
            }
        }
        flush(&line, &mut last_sent);
        String::from_utf8_lossy(&all).into_owned()
    })
}

/// Run a command to completion, polling `cancel` (the child is killed on
/// request → `Ok(None)`). Both pipes are drained on threads so a chatty
/// child can't fill one and deadlock; with `log` set, every output line
/// streams to the TUI live, and a heartbeat fires during output silence
/// so the log never looks stalled. Returns the exit status and stderr.
fn run_cancellable(
    cmd: &mut Command,
    cancel: &Arc<AtomicBool>,
    log: Option<&Sender<Event>>,
) -> anyhow::Result<Option<(ExitStatus, String)>> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Python block-buffers stdout into a pipe; unbuffered keeps the
        // live log real-time instead of arriving in late bursts
        .env("PYTHONUNBUFFERED", "1");
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", cmd.get_program().to_string_lossy()))?;
    let started = Instant::now();
    // ms-since-start of the child's most recent output line
    let activity = Arc::new(AtomicU64::new(0));
    let out_drain = stream_lines(
        child.stdout.take().expect("stdout piped"),
        log.cloned(),
        activity.clone(),
        started,
    );
    let err_drain = stream_lines(
        child.stderr.take().expect("stderr piped"),
        log.cloned(),
        activity.clone(),
        started,
    );
    let mut last_beat = Instant::now();
    let status = loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = out_drain.join();
            let _ = err_drain.join();
            return Ok(None);
        }
        match child.try_wait()? {
            Some(status) => break status,
            None => {
                if let Some(log) = log {
                    let elapsed = started.elapsed();
                    let quiet_ms = (elapsed.as_millis() as u64)
                        .saturating_sub(activity.load(Ordering::Relaxed));
                    if last_beat.elapsed() >= HEARTBEAT_EVERY
                        && Duration::from_millis(quiet_ms) >= QUIET_BEFORE_HEARTBEAT
                    {
                        let _ = log.send(Event::EngineHeartbeat(elapsed.as_secs()));
                        last_beat = Instant::now();
                    }
                }
                std::thread::sleep(POLL_INTERVAL)
            }
        }
    };
    let _ = out_drain.join();
    let stderr_text = err_drain.join().unwrap_or_default();
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
        let _ = events.send(Event::EngineLog(format!(
            "mlx runtime: {}",
            python.display()
        )));

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
        let _ = events.send(Event::EngineLog(format!(
            "wrote handoff WAV: {} ({:.1}s @ 16 kHz mono)",
            scratch.wav.display(),
            decoded.duration_secs
        )));

        let model_name = job
            .model
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        std::fs::write(&scratch.runner, PY_RUNNER)?;
        let started = Instant::now();
        let mut cmd = Command::new(&python);
        cmd.arg(&scratch.runner)
            .arg("--model")
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
        // mlx imports silently for several seconds before the first line
        // of child output — say so instead of appearing stalled
        let _ = events.send(Event::EngineLog(format!(
            "spawning instrumented mlx-audio runner · model {model_name} (loads per job)"
        )));
        // run_cancellable streams the child's chatter into the status
        // line as a live log and kills the child on cancel.
        let Some((status, stderr_text)) = run_cancellable(&mut cmd, cancel, Some(events))? else {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        };
        let _ = events.send(Event::EngineLog(format!(
            "mlx-audio exited with {status} after {:.1}s",
            started.elapsed().as_secs_f32()
        )));
        anyhow::ensure!(
            status.success(),
            "mlx-audio failed on {model_name}: {}",
            error_summary(&stderr_text)
        );

        let segments = scratch.read_output(decoded.duration_secs)?;
        let _ = events.send(Event::EngineLog(format!(
            "parsed {} segment(s) from mlx-audio output",
            segments.len()
        )));
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

/// Temp files for one job: the input WAV, the instrumented runner
/// script, plus mlx-audio's output (it appends `.json` — or `.txt` when
/// a model yields no segments — to the base path we pass). Removed on
/// drop, so every exit path cleans up.
struct Scratch {
    wav: PathBuf,
    runner: PathBuf,
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
            runner: base.with_extension("py"),
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
        for path in [
            self.wav.clone(),
            self.runner.clone(),
            self.json_path(),
            self.txt_path(),
        ] {
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
            None,
        )
        .unwrap()
        .expect("not cancelled");
        assert_eq!(status.code(), Some(3));
        assert_eq!(error_summary(&stderr), "oops");

        // a pre-set cancel flag kills the child immediately
        let cancelled = run_cancellable(
            Command::new("sleep").arg("30"),
            &Arc::new(AtomicBool::new(true)),
            None,
        )
        .unwrap();
        assert!(cancelled.is_none());
    }

    #[test]
    fn run_cancellable_streams_both_pipes_as_log_events() {
        let (tx, rx) = std::sync::mpsc::channel();
        // \r-terminated updates (progress-bar style) and \n lines, on
        // both pipes, with a repeated line that must be deduplicated
        let (status, stderr) = run_cancellable(
            Command::new("sh").args([
                "-c",
                "printf 'downloading 10%%\\rdownloading 99%%\\r'; echo done; echo 'warn: x' >&2; echo 'warn: x' >&2",
            ]),
            &Arc::new(AtomicBool::new(false)),
            Some(&tx),
        )
        .unwrap()
        .expect("not cancelled");
        assert!(status.success());
        assert!(stderr.contains("warn: x"));

        let mut lines: Vec<String> = Vec::new();
        while let Ok(Event::EngineLog(line)) = rx.try_recv() {
            lines.push(line);
        }
        // stdout/stderr drain on separate threads → order is not fixed
        assert!(lines.contains(&"downloading 10%".to_string()));
        assert!(lines.contains(&"downloading 99%".to_string()));
        assert!(lines.contains(&"done".to_string()));
        assert_eq!(lines.iter().filter(|l| *l == "warn: x").count(), 1);
    }

    #[test]
    fn pct_lines_become_progress_events_not_log_lines() {
        let (tx, rx) = std::sync::mpsc::channel();
        run_cancellable(
            Command::new("sh").args(["-c", "echo '@@pct 40'; echo '@@pct 85'; echo other"]),
            &Arc::new(AtomicBool::new(false)),
            Some(&tx),
        )
        .unwrap()
        .expect("not cancelled");

        let mut progress = Vec::new();
        let mut logs = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                Event::Progress(p) => progress.push(p),
                Event::EngineLog(line) => logs.push(line),
                _ => {}
            }
        }
        assert_eq!(progress, vec![40, 85]);
        assert_eq!(logs, vec!["other"]);
    }

    #[test]
    fn sanitize_line_strips_ansi_and_control_bytes() {
        assert_eq!(sanitize_line(b"\x1b[32mgreen\x1b[0m  "), "green");
        assert_eq!(sanitize_line(b"a\x07b\tc"), "abc");
        assert_eq!(sanitize_line(b"   "), "");
    }
}
