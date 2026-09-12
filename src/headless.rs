//! Non-TUI entry points: `--headless` transcription and
//! `--download-test-model`, printing progress to stderr and segments to
//! stdout.

use std::path::PathBuf;
use std::sync::mpsc::channel;

use anyhow::{bail, Result};

use crate::cli::Args;
use crate::diarize::{self, DiarizeMethod, DiarizeStrategy};
use crate::format::{clock_time, human_size};
use crate::transcribe::{Event, Job};
use crate::{config, export, hub, models, transcribe};

pub fn run(args: &Args) -> Result<()> {
    let Some(audio) = args.path.clone() else {
        bail!("--headless requires an audio file path");
    };
    let strategy = match args.diarize.as_deref() {
        None => DiarizeStrategy::Off,
        Some(s) => DiarizeStrategy::parse(s).ok_or_else(|| {
            anyhow::anyhow!("unknown diarization strategy: {s} (use off, auto, tdrz, or embedding)")
        })?,
    };

    let mut cfg = config::load();
    // The configured HF token reaches the diarizers via the environment
    config::apply_hf_token(&cfg);
    // A CLI speaker count overrides the configured one for this run
    if args.speakers.is_some() {
        cfg.diarize_speakers = args.speakers.filter(|n| *n > 0);
    }
    let models_dir = args
        .models_dir
        .clone()
        .unwrap_or_else(|| config::resolve_models_dir(&cfg));
    let found = models::scan_models(&models_dir);
    let model = args.model.clone().or_else(|| {
        models::pick_default(&found, cfg.default_model.as_deref()).map(|m| m.path.clone())
    });
    let Some(model) = model else {
        bail!(
            "no model found in {} — download one via Model management (s) or --download-test-model",
            models_dir.display()
        );
    };
    let model_name = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Resolve the strategy against the model in use; the tdrz method
    // transcribes with the tdrz model itself.
    let method = strategy.resolve(Some(&model_name));
    let model = if method == DiarizeMethod::Tdrz && !models::is_tdrz(&model_name) {
        found
            .iter()
            .find(|m| models::is_tdrz(&m.name))
            .map(|m| m.path.clone())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "diarization ({}) needs {} ({}) from huggingface.co/{} in {}",
                    strategy.label(),
                    diarize::TDRZ_FILE,
                    diarize::TDRZ_SIZE,
                    diarize::TDRZ_REPO,
                    models_dir.display()
                )
            })?
    } else {
        model
    };
    if method != DiarizeMethod::None {
        eprintln!("diarization: {}", method.label());
        if let Some(note) = diarize::download_note(method, &found, &models_dir, &cfg.diarize_models)
        {
            eprintln!("diarization: {note}");
        }
    }
    run_headless(
        model,
        audio,
        method,
        &cfg,
        args.language.clone().or_else(|| cfg.language.clone()),
        args.serve,
    )
}

/// Fetch the smallest whisper model into the models folder — enough for a
/// quick end-to-end smoke test without committing to a multi-GB download.
pub fn download_test_model() -> Result<()> {
    const REPO: &str = "ggerganov/whisper.cpp";
    const FILE: &str = "ggml-tiny.bin";

    let dir = config::resolve_models_dir(&config::load());
    let dest = dir.join(FILE);
    if dest.exists() {
        eprintln!("already there: {}", dest.display());
        return Ok(());
    }
    eprintln!(
        "downloading {FILE} (~75 MB, smallest whisper model) to {}",
        dir.display()
    );

    let (tx, rx) = channel();
    hub::download(
        REPO.into(),
        FILE.into(),
        dir,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        tx,
    );
    for event in rx {
        match event {
            hub::HubEvent::Progress { got, total, .. } => {
                if let Some(pct) = (got * 100).checked_div(total) {
                    eprint!(
                        "\r{pct:>3}%  {} / {}   ",
                        human_size(got),
                        human_size(total)
                    );
                }
            }
            hub::HubEvent::Done { path, .. } => {
                eprintln!("\rdone: {}                         ", path.display());
                return Ok(());
            }
            hub::HubEvent::Failed { error, .. } => bail!("download failed: {error}"),
            hub::HubEvent::Cancelled { .. } => bail!("download cancelled"),
            _ => {}
        }
    }
    Ok(())
}

fn run_headless(
    model: PathBuf,
    audio: PathBuf,
    diarize: DiarizeMethod,
    cfg: &config::Config,
    language: Option<String>,
    serve: Option<u16>,
) -> Result<()> {
    eprintln!("model: {}", model.display());
    eprintln!("audio: {}", audio.display());

    let (tx, rx) = channel();
    let mut transcriber = transcribe::spawn(tx);
    let server = serve
        .map(|port| crate::stream_service::StreamServer::start(port, transcriber.text_stream()))
        .transpose()?;
    if let Some(server) = &server {
        eprintln!(
            "text service: http://{}/events · ws://{}/ws",
            server.address(),
            server.address()
        );
    }
    transcriber.submit(Job {
        model,
        audio,
        diarize,
        diarize_models: cfg.diarize_models.clone(),
        diarize_speakers: cfg.diarize_speakers,
        language,
        split_mode: cfg.split_mode,
    });

    let result = run_headless_loop(rx);
    transcriber.shutdown();
    drop(server);
    result
}

fn run_headless_loop(rx: std::sync::mpsc::Receiver<Event>) -> Result<()> {
    for event in rx {
        match event {
            Event::SessionStarted { .. } => {}
            Event::LoadingModel(name) => eprintln!("loading {name}…"),
            Event::LoadProgress(_) => {}
            Event::ModelReady { load_secs } => eprintln!("model ready in {load_secs:.1}s"),
            Event::Unloading => eprintln!("releasing model…"),
            Event::Unloaded => eprintln!("model released"),
            Event::Decoding => eprintln!("decoding audio…"),
            Event::DecodeProgress(_) => {}
            Event::AudioInfo { duration_secs } => eprintln!("audio: {duration_secs:.1}s"),
            Event::Progress(_) => {}
            Event::RecordingStarted { .. }
            | Event::RecordingLevel { .. }
            | Event::RecordingStopped
            | Event::LivePartial(_)
            | Event::LiveProgress { .. } => {}
            Event::EngineLog(line) => eprintln!("{line}"),
            Event::EngineHeartbeat(secs) => eprintln!("engine working… {secs}s elapsed"),
            Event::Segment(seg) => {
                println!(
                    "[{} → {}] {}",
                    clock_time(seg.start_ms),
                    clock_time(seg.end_ms),
                    seg.text.trim()
                );
            }
            Event::SegmentsFinal(segments) => {
                println!("--- diarized ---");
                for seg in segments {
                    let speaker = seg
                        .speaker
                        .map(|s| format!("{}: ", export::speaker_label(s)))
                        .unwrap_or_default();
                    println!(
                        "[{} → {}] {}{}",
                        clock_time(seg.start_ms),
                        clock_time(seg.end_ms),
                        speaker,
                        seg.text.trim()
                    );
                }
            }
            Event::Done {
                elapsed_secs,
                audio_secs,
                language,
            } => {
                eprintln!(
                    "done in {elapsed_secs:.1}s ({:.2}x realtime, language: {})",
                    elapsed_secs / audio_secs.max(0.001),
                    language.as_deref().unwrap_or("?")
                );
                return Ok(());
            }
            Event::Cancelled => return Ok(()),
            Event::Error(msg) => bail!("{msg}"),
        }
    }
    Ok(())
}
