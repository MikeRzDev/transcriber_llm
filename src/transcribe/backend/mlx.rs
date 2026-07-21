//! The MLX engine: runs directory-shaped models (Parakeet, Qwen3-ASR,
//! Canary, Whisper-MLX, …) by shelling out to mlx-audio's STT CLI
//! (`python -m mlx_audio.stt.generate`) and parsing the JSON it writes.
//! The runtime is self-provisioning: if no Python with mlx_audio is
//! found, the engine installs mlx-audio into the app-managed venv
//! (`crate::pyrt`) on first use. The subprocess loads the model each job
//! (cold start); residency between jobs is a whisper.cpp-only concern.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::Context;

use super::Engine;
use crate::audio;
use crate::pyrt;
use crate::transcribe::{Event, Job, Segment};

/// Shown whenever the mlx-audio runtime cannot be set up automatically.
const INSTALL_HINT: &str = "install it manually with: pip3 install mlx-audio (needs Python ≥ 3.10)";

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
    import inspect
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
    # Some model families need per-job decode limits the mlx-audio
    # defaults get wrong on long recordings. getattr, not type(model):
    # a few wrapper classes only expose generate via delegation, and an
    # AttributeError here would silently demote the job to the stock CLI.
    gen_params = {}
    gen = getattr(model, "generate", None)
    if gen is not None:
        try:
            gen_params = inspect.signature(gen).parameters
        except (TypeError, ValueError):
            gen_params = {}
    # chunk_duration=None default (parakeet) attends over the entire
    # file in one pass — on long recordings that is a multi-GB Metal
    # allocation that aborts the job. Bound those; models that declare
    # their own chunking default keep it. The stock CLI fallback below
    # chunks at 30s on its own.
    chunk_param = gen_params.get("chunk_duration")
    if chunk_param is not None and chunk_param.default is None:
        kwargs["chunk_duration"] = 120.0
        mark("model attends whole-file by default; bounding chunks to 120s")
    elif chunk_param is not None and (chunk_param.default or 0) > 300:
        # LLM-style models default to 20-minute chunks: one degenerate
        # chunk can eat the whole token budget (mlx-audio then silently
        # drops the remaining chunks), and every chunk lands in the
        # exports as a single 20-minute segment. Five-minute chunks
        # bound both; the splitter cuts at silence, not mid-word.
        kwargs["chunk_duration"] = 300.0
        mark("capping model chunking to 300s for stable long-audio output")
    # Greedy LLM decoders can degenerate into endless filler (". . .")
    # that burns the token budget; a mild penalty breaks such loops
    # without touching normal speech.
    if "repetition_penalty" in gen_params:
        kwargs["repetition_penalty"] = 1.1
    # max_tokens is a TOTAL text budget across all chunks (qwen3-asr:
    # 8192 ≈ 35-40 min of dense speech) — once spent, the rest of the
    # audio is silently dropped. Scale it to the handoff WAV's length;
    # ~10 tokens/s is ~2.5x the densest rate observed in real calls.
    if "max_tokens" in gen_params:
        try:
            import wave
            with wave.open(flags["--audio"], "rb") as w:
                dur_secs = w.getnframes() / (w.getframerate() or 16000)
        except (OSError, EOFError, wave.Error):
            dur_secs = 0
        if dur_secs > 0:
            kwargs["max_tokens"] = max(8192, int(dur_secs * 10))
            mark(f"token budget {kwargs['max_tokens']} for {dur_secs:.0f}s of audio")
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
        let python = match pyrt::find_python_with("mlx_audio") {
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

/// One-time runtime setup: pip-install mlx-audio into the app venv.
/// Several hundred MB of packages, so this can take minutes.
fn install_runtime(
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(python) = pyrt::install_packages(
        &["mlx-audio"],
        "mlx-audio runtime (one-time setup, ~2 min)",
        "several hundred MB, can take minutes",
        INSTALL_HINT,
        events,
        cancel,
    )?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        pyrt::has_module(&python, "mlx_audio"),
        "mlx-audio installed but does not import — {INSTALL_HINT}"
    );
    Ok(Some(python))
}

/// Whether the MLX runtime is ready, for UI badges. The venv check is a
/// cheap stat — so an in-app install flips the badge live — while the
/// system-Python probe (a subprocess) runs once per process.
pub fn mlx_audio_available() -> bool {
    static SYSTEM_PROBE: OnceLock<bool> = OnceLock::new();
    pyrt::venv_python().is_some()
        || *SYSTEM_PROBE.get_or_init(|| pyrt::find_python_with("mlx_audio").is_some())
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
        let Some((status, stderr_text)) = pyrt::run_cancellable(&mut cmd, cancel, Some(events))?
        else {
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
            pyrt::error_summary(&stderr_text)
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
}
