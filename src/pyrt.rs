//! The app-managed Python runtime and its subprocess plumbing, shared by
//! everything that shells out to Python: the MLX engine (mlx-audio) and
//! the embedding diarizer (sherpa-onnx). One private venv holds every
//! package — a venv keeps installs out of the system/Homebrew Python
//! (whose PEP 668 guard rejects plain `pip install`) and survives OS
//! Python upgrades untouched. Helpers here probe interpreters, install
//! packages on first use, and run children cancellably with their output
//! streamed live into the TUI log.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;

use crate::transcribe::Event;

/// How often child processes are polled for exit / cancellation.
const POLL_INTERVAL: Duration = Duration::from_millis(120);

/// While a child runs, a liveness heartbeat is emitted this often…
const HEARTBEAT_EVERY: Duration = Duration::from_secs(5);
/// …but only after this much output silence — a chatty child needs none.
const QUIET_BEFORE_HEARTBEAT: Duration = Duration::from_secs(3);

/// Python interpreters probed, in order — both for an existing package
/// install and as the base interpreter for creating the app venv.
const PYTHON_CANDIDATES: [&str; 4] = [
    "python3",
    "python",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
];

/// The app-managed Python environment. The env var (and the on-disk
/// name) predate the venv holding more than mlx-audio and are kept for
/// compatibility.
pub fn venv_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("TRANSCRIBE_STT_MLX_VENV") {
        return Some(PathBuf::from(dir));
    }
    Some(
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/Application Support/transcribe-stt/mlx-venv"),
    )
}

pub fn venv_python() -> Option<PathBuf> {
    let python = venv_dir()?.join("bin/python3");
    python.is_file().then_some(python)
}

/// Does this interpreter have `module`? find_spec instead of a real
/// import: answers in ~50 ms without the module's import cost.
pub fn has_module(python: &Path, module: &str) -> bool {
    Command::new(python)
        .args([
            "-c",
            &format!(
                "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('{module}') else 1)"
            ),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// An interpreter that can import `module`: the app venv first, then
/// any Python the user installed it into themselves.
pub fn find_python_with(module: &str) -> Option<PathBuf> {
    venv_python()
        .filter(|p| has_module(p, module))
        .or_else(|| {
            PYTHON_CANDIDATES
                .iter()
                .map(PathBuf::from)
                .find(|p| has_module(p, module))
        })
}

/// A Python ≥ 3.10 (the packages' floor) that can create the venv.
pub fn find_base_python() -> Option<PathBuf> {
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

/// One-time package setup: create the app venv if missing and
/// pip-install `packages` into it. Progress is the indeterminate load
/// spinner plus pip's own streamed output; cancel kills the child
/// (`Ok(None)`). A partial venv is safe: the next attempt resumes (venv
/// creation is idempotent, pip skips what's already there).
pub fn install_packages(
    packages: &[&str],
    setup_label: &str,
    size_note: &str,
    hint: &str,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<PathBuf>> {
    let base = find_base_python().ok_or_else(|| {
        anyhow::anyhow!("no Python ≥ 3.10 found — brew install python, or {hint}")
    })?;
    let venv =
        venv_dir().ok_or_else(|| anyhow::anyhow!("no home directory to hold the Python runtime"))?;
    let _ = events.send(Event::LoadingModel(setup_label.to_string()));
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
        "creating the Python venv failed: {}",
        error_summary(&stderr)
    );

    let python = venv.join("bin").join("python3");
    let _ = events.send(Event::EngineLog(format!(
        "installing {} via pip ({size_note})…",
        packages.join(" ")
    )));
    // No --quiet: pip's collect/download lines stream to the TUI so a
    // long install always shows what it is doing
    let Some((status, stderr)) = run_cancellable(
        Command::new(&python).args(["-m", "pip", "install"]).args(packages),
        cancel,
        Some(events),
    )?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        status.success(),
        "installing {} failed: {} — {hint}",
        packages.join(" "),
        error_summary(&stderr)
    );
    Ok(Some(python))
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
    // "@@pct N" comes from a runner's progress probe inside the model's
    // compute loop — it drives the real progress gauge, not the log
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
pub fn run_cancellable(
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

/// The last meaningful stderr line — for Python that is the exception
/// message at the end of the traceback.
pub fn error_summary(stderr: &str) -> String {
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("no error output")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
