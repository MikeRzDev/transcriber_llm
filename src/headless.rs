//! Non-TUI entry points: `--headless` transcription and
//! `--download-test-model`, printing progress to stderr and segments to
//! stdout.

use std::path::PathBuf;
use std::sync::mpsc::channel;

use anyhow::{bail, Result};

use crate::cli::Args;
use crate::format::{clock_time, human_size};
use crate::transcribe::{Event, Job};
use crate::{config, export, hub, models, split, transcribe};

fn default_model(diarize: bool) -> Option<PathBuf> {
    let cfg = config::load();
    let found = models::scan_models(&config::resolve_models_dir(&cfg));
    if diarize {
        return found
            .iter()
            .find(|m| models::is_tdrz(&m.name))
            .map(|m| m.path.clone());
    }
    models::pick_default(&found, cfg.default_model.as_deref()).map(|m| m.path.clone())
}

pub fn run(args: &Args) -> Result<()> {
    let Some(audio) = args.path.clone() else {
        bail!("--headless requires an audio file path");
    };
    let Some(model) = args.model.clone().or_else(|| default_model(args.diarize)) else {
        bail!(
            "no suitable model found in {} — {}",
            config::resolve_models_dir(&config::load()).display(),
            if args.diarize {
                "get ggml-small.en-tdrz.bin from huggingface.co/akashmjn/tinydiarize-whisper.cpp"
            } else {
                "download one via Model management (s) or --download-test-model"
            }
        );
    };
    run_headless(
        model,
        audio,
        args.diarize,
        args.language.clone(),
        config::load().split_mode,
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
    diarize: bool,
    language: Option<String>,
    split_mode: split::SplitMode,
) -> Result<()> {
    eprintln!("model: {}", model.display());
    eprintln!("audio: {}", audio.display());

    let (tx, rx) = channel();
    let mut transcriber = transcribe::spawn(tx);
    transcriber.submit(Job {
        model,
        audio,
        diarize,
        language,
        split_mode,
    });

    let result = run_headless_loop(rx);
    transcriber.shutdown();
    result
}

fn run_headless_loop(rx: std::sync::mpsc::Receiver<Event>) -> Result<()> {
    for event in rx {
        match event {
            Event::LoadingModel(name) => eprintln!("loading {name}…"),
            Event::LoadProgress(_) => {}
            Event::ModelReady { load_secs } => eprintln!("model ready in {load_secs:.1}s"),
            Event::Unloading => eprintln!("releasing model…"),
            Event::Unloaded => eprintln!("model released"),
            Event::Decoding => eprintln!("decoding audio…"),
            Event::DecodeProgress(_) => {}
            Event::AudioInfo { duration_secs } => eprintln!("audio: {duration_secs:.1}s"),
            Event::Progress(_) => {}
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
