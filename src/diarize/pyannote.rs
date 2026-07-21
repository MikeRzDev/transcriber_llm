//! The pyannote.audio strategy: `speaker-diarization-community-1`, the
//! open-accuracy SOTA, run through PyTorch in the app-managed venv. The
//! costs the other strategies avoid live here: a ~2 GB torch install and
//! a gated model — a Hugging Face token plus accepted terms at
//! hf.co/pyannote/speaker-diarization-community-1 — both reported as
//! requirements before anything is set up. Engine-agnostic like the
//! embedding pipeline: same worker post-pass, same speaker-turn JSON.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use anyhow::Context;

use super::{Requirement, SpeakerTurn};
use crate::audio;
use crate::pyrt;
use crate::transcribe::Event;

/// The gated pipeline this strategy runs.
pub const MODEL: &str = "pyannote/speaker-diarization-community-1";

/// Shown whenever the runtime cannot be set up automatically.
const INSTALL_HINT: &str =
    "install it manually with: pip3 install pyannote.audio (needs Python ≥ 3.10)";

/// Approximate pip footprint (torch dominates).
const RUNTIME_SIZE: &str = "~2 GB";

/// The runner: loads the waveform from the 16 kHz mono handoff WAV
/// itself (sidestepping pyannote's audio-IO backends entirely), runs the
/// pipeline on MPS when available, and writes the shared speaker-turn
/// JSON. Tolerates both the 3.x return shape (an Annotation) and 4.x
/// (an output object carrying `.speaker_diarization`).
const PY_DIARIZE: &str = r#"
import json
import sys
import time
import wave

T0 = time.time()

def mark(msg):
    print(f"[stage +{time.time() - T0:5.1f}s] {msg}", flush=True)

flags = dict(zip(sys.argv[1::2], sys.argv[2::2]))

mark("importing torch + pyannote.audio…")
import numpy as np
import torch
from pyannote.audio import Pipeline

mark(f"loading {flags['--model']} (downloads to the HF cache on first run)…")
pipeline = Pipeline.from_pretrained(flags["--model"])
if pipeline is None:
    raise RuntimeError(
        "pipeline load returned None — accept the model terms at "
        f"hf.co/{flags['--model']} and set HF_TOKEN (or run: hf auth login)"
    )
device = "mps" if torch.backends.mps.is_available() else "cpu"
try:
    pipeline.to(torch.device(device))
except Exception as e:
    mark(f"{device} unavailable for this pipeline ({e}); using cpu")
    device = "cpu"
    pipeline.to(torch.device(device))
mark(f"pipeline ready on {device}")

with wave.open(flags["--wav"]) as f:
    assert f.getnchannels() == 1 and f.getsampwidth() == 2, "expected 16-bit mono WAV"
    sample_rate = f.getframerate()
    samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0
waveform = torch.from_numpy(samples).unsqueeze(0)  # (channel, time)

kwargs = {}
n = int(flags.get("--num-speakers", "0"))
if n > 0:
    kwargs["num_speakers"] = n
mark("diarizing…")
output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
annotation = getattr(output, "speaker_diarization", output)

speakers = {}
turns = []
for segment, _, label in annotation.itertracks(yield_label=True):
    idx = speakers.setdefault(label, len(speakers))
    turns.append({"start": segment.start, "end": segment.end, "speaker": idx})
with open(flags["--out"], "w") as f:
    json.dump(turns, f)
mark(f"done: {len(speakers)} speaker(s) across {len(turns)} turn(s)")
"#;

/// Whether a Hugging Face token is reachable: the env vars the HF stack
/// reads, or the file `hf auth login` writes.
pub fn hf_token_available() -> bool {
    if std::env::var_os("HF_TOKEN").is_some()
        || std::env::var_os("HUGGING_FACE_HUB_TOKEN").is_some()
    {
        return true;
    }
    let Some(home) = std::env::var_os("HOME") else {
        return false;
    };
    let home = PathBuf::from(home);
    home.join(".cache/huggingface/token").is_file() || home.join(".huggingface/token").is_file()
}

/// Interpreter with pyannote.audio importable. Success cached
/// process-wide; failure re-probes so a mid-session install is seen.
fn find_runtime_python() -> Option<PathBuf> {
    static FOUND: Mutex<Option<PathBuf>> = Mutex::new(None);
    let mut found = FOUND.lock().unwrap_or_else(|p| p.into_inner());
    if found.is_none() {
        *found = pyrt::find_python_with("pyannote.audio");
    }
    found.clone()
}

/// Probes a subprocess on first call — call on user actions, not per
/// rendered frame.
pub fn runtime_available() -> bool {
    find_runtime_python().is_some()
}

/// What the strategy needs: the PyTorch runtime and HF access. The
/// pipeline weights themselves fetch into the HF cache on first run —
/// possible only once both requirements hold, so they are the gate.
pub fn requirements() -> Vec<Requirement> {
    vec![
        Requirement {
            name: "pyannote.audio + PyTorch runtime (pip)".into(),
            size: RUNTIME_SIZE,
            satisfied: runtime_available(),
        },
        Requirement {
            name: format!(
                "Hugging Face token with accepted terms for {MODEL} \
                 (Settings → HF token, hf auth login, or the HF_TOKEN env var)"
            ),
            size: "free",
            satisfied: hf_token_available(),
        },
    ]
}

/// Scratch files for one run: the handoff WAV, the runner script, and
/// the JSON turns. Removed on drop.
struct Scratch {
    wav: PathBuf,
    runner: PathBuf,
    out: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir().join(format!(
            "transcribe-stt-pyannote-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::Relaxed)
        ));
        Self {
            wav: base.with_extension("wav"),
            runner: base.with_extension("py"),
            out: base.with_extension("json"),
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for path in [&self.wav, &self.runner, &self.out] {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Diarize `audio` into speaker turns with community-1, provisioning the
/// PyTorch runtime on first use. `Ok(None)` = cancelled.
pub fn run(
    audio_path: &std::path::Path,
    speakers: Option<u8>,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<Vec<SpeakerTurn>>> {
    // The token gate comes first: never install 2 GB of torch only to
    // fail on a missing token afterwards.
    anyhow::ensure!(
        hf_token_available(),
        "{MODEL} is gated — accept its terms at hf.co/{MODEL} and set your token in \
         Settings → HF token (or hf auth login / HF_TOKEN); alternatively press d for \
         the ungated embeddings strategy"
    );
    let python = match find_runtime_python() {
        Some(python) => python,
        None => {
            let Some(python) = pyrt::install_packages(
                &["pyannote.audio"],
                "pyannote.audio + PyTorch runtime (one-time setup, ~2 GB)",
                RUNTIME_SIZE,
                INSTALL_HINT,
                events,
                cancel,
            )?
            else {
                return Ok(None);
            };
            anyhow::ensure!(
                pyrt::has_module(&python, "pyannote.audio"),
                "pyannote.audio installed but does not import — {INSTALL_HINT}"
            );
            python
        }
    };

    let Some(decoded) = audio::load_media_with(audio_path, Some(cancel.as_ref()), |_| {})? else {
        return Ok(None);
    };
    let scratch = Scratch::new();
    audio::write_wav_16k_mono(&scratch.wav, &decoded.samples)?;
    std::fs::write(&scratch.runner, PY_DIARIZE)?;

    let speakers_note = match speakers {
        Some(n) => format!(", {n} known speakers"),
        None => ", auto speaker count".into(),
    };
    let _ = events.send(Event::EngineLog(format!(
        "diarizing {:.1}s of audio ({MODEL}{speakers_note})…",
        decoded.duration_secs
    )));
    // No fine-grained progress from the pipeline — the heartbeat keeps
    // the log alive through the silent stretch
    let _ = events.send(Event::Progress(-1));
    let mut cmd = Command::new(&python);
    cmd.arg(&scratch.runner)
        .arg("--model")
        .arg(MODEL)
        .arg("--wav")
        .arg(&scratch.wav)
        .arg("--out")
        .arg(&scratch.out);
    if let Some(n) = speakers {
        cmd.args(["--num-speakers", &n.to_string()]);
    }
    let Some((status, stderr)) = pyrt::run_cancellable(&mut cmd, cancel, Some(events))? else {
        return Ok(None);
    };
    anyhow::ensure!(
        status.success(),
        "pyannote diarizer failed: {} — if this is a 401/403, accept the terms at \
         hf.co/{MODEL} and refresh your token",
        pyrt::error_summary(&stderr)
    );

    let text = std::fs::read_to_string(&scratch.out)
        .context("pyannote diarizer exited successfully but wrote no output")?;
    Ok(Some(super::parse_turns(&text)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate is the runtime plus HF access — both named so the UI can
    /// spell out what stands between the user and the model.
    #[test]
    fn requirements_name_the_runtime_and_the_token() {
        let reqs = requirements();
        assert_eq!(reqs.len(), 2);
        assert!(reqs[0].name.contains("PyTorch"), "{}", reqs[0].name);
        assert_eq!(reqs[0].size, RUNTIME_SIZE);
        assert!(reqs[1].name.contains(MODEL), "{}", reqs[1].name);
        assert!(reqs[1].name.contains("HF_TOKEN"), "{}", reqs[1].name);
    }
}
