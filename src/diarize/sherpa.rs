//! The embedding diarization pipeline: pyannote segmentation-3.0 for
//! "who speaks when" plus speaker-embedding clustering for "who is who",
//! run fully offline by sherpa-onnx's Python wheel (ONNX Runtime, no
//! PyTorch, no account/token). Engine-agnostic by construction — it
//! decodes the job's media itself and returns speaker turns the worker
//! maps onto whatever segments the transcription engine produced.
//!
//! Self-provisioning like the MLX engine: the wheel pip-installs into
//! the app venv and the two ONNX models download from public Hugging
//! Face repos into `<models>/diarization` on first use — each announced
//! with its size before any bytes move.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use anyhow::Context;

use super::{Requirement, SpeakerTurn};
use crate::audio;
use crate::hub;
use crate::pyrt;
use crate::transcribe::Event;

/// Shown whenever the sherpa-onnx runtime cannot be set up automatically.
const INSTALL_HINT: &str =
    "install it manually with: pip3 install sherpa-onnx numpy (needs Python ≥ 3.10)";

/// Approximate pip footprint of the sherpa-onnx wheel (+ numpy).
const RUNTIME_SIZE: &str = "~25 MB";

/// A component's place in the pipeline: exactly one of each role runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiarizeRole {
    /// Who speaks when (speech + speaker-change detection)
    Segmentation,
    /// Who is who (speaker-identity embeddings for clustering)
    Embedding,
}

impl DiarizeRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::Segmentation => "segmentation",
            Self::Embedding => "embedding",
        }
    }
}

/// One downloadable diarization component: an ONNX model fetched from a
/// public HF repo into `dir(models_dir)/<local>`. The catalog below is
/// the single source of truth — the hub's diarization category, the
/// requirement reports, and the runner all read it.
pub struct DiarizeModel {
    pub role: DiarizeRole,
    /// Display name in the hub and requirement lists.
    pub label: &'static str,
    /// On-disk file name under the diarization folder (also the config
    /// token selecting it as active).
    pub local: &'static str,
    pub repo: &'static str,
    pub remote: &'static str,
    pub size: &'static str,
    /// One-line description for the hub row.
    pub note: &'static str,
}

impl DiarizeModel {
    pub fn installed(&self, models_dir: &Path) -> bool {
        dir(models_dir).join(self.local).is_file()
    }
}

/// Everything the pipeline can run, defaults first per role.
pub const CATALOG: [DiarizeModel; 6] = [
    DiarizeModel {
        role: DiarizeRole::Segmentation,
        label: "pyannote segmentation-3.0",
        local: "pyannote-segmentation-3.onnx",
        repo: "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        remote: "model.onnx",
        size: "6 MB",
        note: "speaker-change detection · default",
    },
    DiarizeModel {
        role: DiarizeRole::Segmentation,
        label: "reverb-diarization-v1",
        local: "reverb-diarization-v1.onnx",
        repo: "csukuangfj/sherpa-onnx-reverb-diarization-v1",
        remote: "model.onnx",
        size: "10 MB",
        note: "Rev WavLM segmentation · English · non-commercial license",
    },
    DiarizeModel {
        role: DiarizeRole::Embedding,
        label: "titanet-small",
        local: "nemo-titanet-small.onnx",
        repo: "csukuangfj/speaker-embedding-models",
        remote: "nemo_en_titanet_small.onnx",
        size: "40 MB",
        note: "NeMo TitaNet-S speaker embeddings · default",
    },
    DiarizeModel {
        role: DiarizeRole::Embedding,
        label: "titanet-large",
        local: "nemo-titanet-large.onnx",
        repo: "csukuangfj/speaker-embedding-models",
        remote: "nemo_en_titanet_large.onnx",
        size: "101 MB",
        note: "NeMo TitaNet-L · higher quality, slower",
    },
    DiarizeModel {
        role: DiarizeRole::Embedding,
        label: "campplus-voxceleb",
        local: "wespeaker-campplus-voxceleb.onnx",
        repo: "csukuangfj/speaker-embedding-models",
        remote: "wespeaker_en_voxceleb_CAM++.onnx",
        size: "29 MB",
        note: "WeSpeaker CAM++ · VoxCeleb-trained",
    },
    DiarizeModel {
        role: DiarizeRole::Embedding,
        label: "campplus-zh-en",
        local: "campplus-zh-en.onnx",
        repo: "csukuangfj/speaker-embedding-models",
        remote: "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        size: "28 MB",
        note: "3D-Speaker CAM++ · Chinese + English",
    },
];

/// The catalog entry `configured` names, else the role's default (the
/// first catalog entry of that role). An unknown or stale token falls
/// back to the default rather than failing the job.
pub fn pick(role: DiarizeRole, configured: Option<&str>) -> &'static DiarizeModel {
    let of_role = || CATALOG.iter().filter(|m| m.role == role);
    configured
        .and_then(|local| of_role().find(|m| m.local == local))
        .or_else(|| of_role().next())
        .expect("catalog has every role")
}

/// The runner: reads the 16 kHz mono handoff WAV, runs the sherpa-onnx
/// offline diarization pipeline (progress as machine-readable "@@pct N"
/// lines the host turns into its gauge), and writes the speaker turns as
/// JSON `[{"start": s, "end": s, "speaker": n}, …]`.
const PY_DIARIZE: &str = r#"
import json
import sys
import wave

flags = dict(zip(sys.argv[1::2], sys.argv[2::2]))

import numpy as np
import sherpa_onnx

with wave.open(flags["--wav"]) as f:
    assert f.getnchannels() == 1 and f.getsampwidth() == 2, "expected 16-bit mono WAV"
    sample_rate = f.getframerate()
    samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0

config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=flags["--segmentation"]
        ),
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=flags["--embedding"]),
    # A known speaker count pins the clustering exactly; -1 infers the
    # count from the threshold instead.
    clustering=sherpa_onnx.FastClusteringConfig(
        num_clusters=int(flags.get("--num-speakers", "-1")),
        threshold=0.5,
    ),
    min_duration_on=0.3,
    min_duration_off=0.5,
)
sd = sherpa_onnx.OfflineSpeakerDiarization(config)
assert sd.sample_rate == sample_rate, f"pipeline wants {sd.sample_rate} Hz, WAV is {sample_rate}"

def progress(processed, total):
    print(f"@@pct {processed * 100 // max(total, 1)}", flush=True)
    return 0

try:
    result = sd.process(samples, callback=progress)
except TypeError:
    # older sherpa-onnx without the callback kwarg: run without progress
    result = sd.process(samples)
result = result.sort_by_start_time()
turns = [{"start": r.start, "end": r.end, "speaker": r.speaker} for r in result]
with open(flags["--out"], "w") as f:
    json.dump(turns, f)
print(
    f"diarization: {len(set(t['speaker'] for t in turns))} speaker(s)"
    f" across {len(turns)} turn(s)",
    flush=True,
)
"#;

/// Where the diarization models live, under the models folder.
pub fn dir(models_dir: &Path) -> PathBuf {
    models_dir.join("diarization")
}

/// Interpreter with sherpa_onnx importable. Success is cached
/// process-wide (the probe is a subprocess); failure re-probes, so an
/// install made mid-session is picked up without restarting.
fn find_runtime_python() -> Option<PathBuf> {
    static FOUND: Mutex<Option<PathBuf>> = Mutex::new(None);
    let mut found = FOUND.lock().unwrap_or_else(|p| p.into_inner());
    if found.is_none() {
        *found = pyrt::find_python_with("sherpa_onnx");
    }
    found.clone()
}

/// Whether the sherpa-onnx runtime is importable right now. Probes a
/// subprocess on the first call (~50 ms per candidate interpreter until
/// one is found) — call on user actions, not per rendered frame.
pub fn runtime_available() -> bool {
    find_runtime_python().is_some()
}

/// What the pipeline needs to run with `choice`: the Python runtime plus
/// the active segmentation/embedding pair.
pub fn requirements(models_dir: &Path, choice: &super::DiarizeModelChoice) -> Vec<Requirement> {
    let mut reqs = vec![Requirement {
        name: "sherpa-onnx runtime (pip)".into(),
        size: RUNTIME_SIZE,
        satisfied: runtime_available(),
    }];
    for model in active_pair(choice) {
        reqs.push(Requirement {
            name: format!("{} {} model", model.label, model.role.label()),
            size: model.size,
            satisfied: model.installed(models_dir),
        });
    }
    reqs
}

/// The segmentation + embedding pair `choice` selects.
fn active_pair(choice: &super::DiarizeModelChoice) -> [&'static DiarizeModel; 2] {
    [
        pick(DiarizeRole::Segmentation, choice.segmentation.as_deref()),
        pick(DiarizeRole::Embedding, choice.embedding.as_deref()),
    ]
}

/// Download any missing active ONNX model into the diarization folder,
/// announcing each with its size first. `Ok(None)` = cancelled.
fn ensure_models(
    models: &[&'static DiarizeModel],
    models_dir: &Path,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<()>> {
    let dir = dir(models_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    for model in models {
        let dest = dir.join(model.local);
        if dest.is_file() {
            continue;
        }
        let _ = events.send(Event::EngineLog(format!(
            "downloading {} ({}) from huggingface.co/{}…",
            model.label, model.size, model.repo
        )));
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model.repo, model.remote
        );
        let mut last_reported: u64 = 0;
        let outcome = hub::stream_file(
            &url,
            &dest,
            cancel,
            &AtomicBool::new(false), // diarization downloads have no pause
            &mut |got, total| {
                // A log line every few MB keeps the multi-MB fetches visibly alive
                if got - last_reported >= 4 * 1024 * 1024 || got == total {
                    last_reported = got;
                    let _ = events.send(Event::EngineLog(format!(
                        "{}: {} / {}",
                        model.local,
                        crate::format::human_size(got),
                        crate::format::human_size(total)
                    )));
                }
            },
        )
        .with_context(|| format!("downloading {}", model.label))?;
        match outcome {
            hub::FileOutcome::Done => {
                let _ = events.send(Event::EngineLog(format!("downloaded {}", model.local)));
            }
            hub::FileOutcome::Cancelled | hub::FileOutcome::Paused(_) => return Ok(None),
        }
    }
    Ok(Some(()))
}

/// The interpreter to diarize with, installing the runtime on first use.
/// `Ok(None)` = cancelled mid-install.
fn ensure_runtime(
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<PathBuf>> {
    if let Some(python) = find_runtime_python() {
        return Ok(Some(python));
    }
    let Some(python) = pyrt::install_packages(
        &["sherpa-onnx", "numpy"],
        "sherpa-onnx diarization runtime (one-time setup)",
        RUNTIME_SIZE,
        INSTALL_HINT,
        events,
        cancel,
    )?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        pyrt::has_module(&python, "sherpa_onnx"),
        "sherpa-onnx installed but does not import — {INSTALL_HINT}"
    );
    Ok(Some(python))
}

/// Scratch files for one run: the 16 kHz handoff WAV, the runner script,
/// and the JSON turns it writes. Removed on drop.
struct Scratch {
    wav: PathBuf,
    runner: PathBuf,
    out: PathBuf,
}

impl Scratch {
    fn new() -> Self {
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir().join(format!(
            "transcribe-stt-diarize-{}-{}",
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

/// Diarize `audio` into speaker turns using the segmentation/embedding
/// pair `choice` selects, provisioning the runtime and models on first
/// use. `Ok(None)` = cancelled. Progress and phase lines stream over
/// `events` like any engine's.
pub fn run(
    audio_path: &Path,
    models_dir: &Path,
    choice: &super::DiarizeModelChoice,
    speakers: Option<u8>,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<Vec<SpeakerTurn>>> {
    let Some(python) = ensure_runtime(events, cancel)? else {
        return Ok(None);
    };
    let [segmentation, embedding] = active_pair(choice);
    if ensure_models(&[segmentation, embedding], models_dir, events, cancel)?.is_none() {
        return Ok(None);
    }

    // The diarizer consumes the same decoded 16 kHz mono audio the
    // engines do, so video and exotic codecs behave identically.
    let Some(decoded) = audio::load_media_with(audio_path, Some(cancel.as_ref()), |_| {})? else {
        return Ok(None);
    };
    let scratch = Scratch::new();
    audio::write_wav_16k_mono(&scratch.wav, &decoded.samples)?;
    std::fs::write(&scratch.runner, PY_DIARIZE)?;

    let dir = dir(models_dir);
    let speakers_note = match speakers {
        Some(n) => format!(", {n} known speakers"),
        None => ", auto speaker count".into(),
    };
    let _ = events.send(Event::EngineLog(format!(
        "diarizing {:.1}s of audio ({} segmentation + {} embeddings{speakers_note})…",
        decoded.duration_secs, segmentation.label, embedding.label
    )));
    let _ = events.send(Event::Progress(0));
    let mut cmd = Command::new(&python);
    cmd.arg(&scratch.runner)
        .arg("--wav")
        .arg(&scratch.wav)
        .arg("--segmentation")
        .arg(dir.join(segmentation.local))
        .arg("--embedding")
        .arg(dir.join(embedding.local))
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
        "diarizer failed: {}",
        pyrt::error_summary(&stderr)
    );

    let text = std::fs::read_to_string(&scratch.out)
        .context("diarizer exited successfully but wrote no output")?;
    Ok(Some(super::parse_turns(&text)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requirements_track_model_files_on_disk() {
        let base = std::env::temp_dir().join(format!("transcribe-stt-sherpa-{}", std::process::id()));
        let dir_path = dir(&base);
        std::fs::create_dir_all(&dir_path).unwrap();
        let default_seg = pick(DiarizeRole::Segmentation, None);
        std::fs::write(dir_path.join(default_seg.local), b"onnx").unwrap();

        let reqs = requirements(&base, &super::super::DiarizeModelChoice::default());
        assert_eq!(reqs.len(), 3); // runtime + segmentation + embedding
        let seg = reqs.iter().find(|r| r.name.contains("segmentation")).unwrap();
        assert!(seg.satisfied);
        let emb = reqs.iter().find(|r| r.name.contains("embedding")).unwrap();
        assert!(!emb.satisfied);
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn pick_honours_the_choice_and_falls_back_to_defaults() {
        // defaults: the first catalog entry of each role
        assert_eq!(
            pick(DiarizeRole::Segmentation, None).local,
            "pyannote-segmentation-3.onnx"
        );
        assert_eq!(pick(DiarizeRole::Embedding, None).local, "nemo-titanet-small.onnx");
        // an explicit choice wins…
        assert_eq!(
            pick(DiarizeRole::Embedding, Some("campplus-zh-en.onnx")).local,
            "campplus-zh-en.onnx"
        );
        // …but a stale token (uninstalled catalog removal, typo) falls back
        assert_eq!(
            pick(DiarizeRole::Embedding, Some("gone.onnx")).local,
            "nemo-titanet-small.onnx"
        );
        // a choice can never cross roles
        assert_eq!(
            pick(DiarizeRole::Segmentation, Some("nemo-titanet-small.onnx")).local,
            "pyannote-segmentation-3.onnx"
        );
    }

    #[test]
    fn catalog_locals_are_unique_and_each_role_has_a_default() {
        let mut locals: Vec<&str> = CATALOG.iter().map(|m| m.local).collect();
        locals.sort();
        locals.dedup();
        assert_eq!(locals.len(), CATALOG.len(), "duplicate local file names");
        assert!(CATALOG.iter().any(|m| m.role == DiarizeRole::Segmentation));
        assert!(CATALOG.iter().any(|m| m.role == DiarizeRole::Embedding));
    }
}
