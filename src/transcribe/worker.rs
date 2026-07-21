//! The long-lived worker thread. It owns the backends (whisper.cpp keeps
//! its resident context lazy-loaded between jobs; MLX is stateless),
//! turns `WorkerMsg`s into streamed `Event`s, and runs the embedding
//! diarization post-pass over any engine's output when a job asks for it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use super::backend::{Backends, Engine};
use super::{Event, Job, Segment};
use crate::diarize::{self, DiarizeMethod};

enum WorkerMsg {
    Job(Job),
    /// Drop the resident model context (the user switched models); the
    /// next job loads its model fresh
    Unload,
}

pub struct Transcriber {
    jobs: Option<Sender<WorkerMsg>>,
    handle: Option<std::thread::JoinHandle<()>>,
    pub cancel: Arc<AtomicBool>,
}

impl Transcriber {
    pub fn submit(&self, job: Job) {
        self.cancel.store(false, Ordering::SeqCst);
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(WorkerMsg::Job(job));
        }
    }

    /// Release the loaded model. Queued behind any running job, so it is
    /// safe to call mid-transcription.
    pub fn request_unload(&self) {
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(WorkerMsg::Unload);
        }
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// Abort any running job and wait for the worker to drop the whisper
    /// context. Exiting while the context is alive trips a GGML Metal
    /// assertion in an atexit destructor.
    pub fn shutdown(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.jobs = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Run the engine with its events teed, then diarize as a post-pass: the
/// streamed segments are collected while being forwarded, `Done` is held
/// back, and once the engine finishes the embedding diarizer labels the
/// transcript before `Done` is released. Diarization failure (or cancel
/// mid-pass) downgrades to an unlabeled transcript — the finished
/// transcription is never thrown away.
fn run_with_diarize_postpass(
    engine: &mut dyn Engine,
    job: &Job,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let (inner_tx, inner_rx) = channel::<Event>();
    let outer = events.clone();
    let tee = std::thread::spawn(move || {
        let mut segments: Vec<Segment> = Vec::new();
        let mut done: Option<Event> = None;
        for event in inner_rx {
            match event {
                Event::Segment(seg) => {
                    segments.push(seg.clone());
                    let _ = outer.send(Event::Segment(seg));
                }
                Event::SegmentsFinal(segs) => {
                    segments.clone_from(&segs);
                    let _ = outer.send(Event::SegmentsFinal(segs));
                }
                held @ Event::Done { .. } => done = Some(held),
                other => {
                    let _ = outer.send(other);
                }
            }
        }
        (segments, done)
    });

    let run_result = engine.run(job, &inner_tx, cancel);
    drop(inner_tx); // ends the tee loop
    let (segments, done) = tee.join().expect("tee thread never panics");
    run_result?;
    // No held Done means the engine cancelled or errored — nothing to label
    let Some(done) = done else { return Ok(()) };

    // A model that labels speakers natively needs no second opinion, and
    // an empty transcript has nothing to attribute
    if segments.is_empty() || segments.iter().all(|s| s.speaker.is_some()) {
        let _ = events.send(done);
        return Ok(());
    }

    // Diarization models live beside the transcription models
    let models_dir = job
        .model
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let turns = match job.diarize {
        DiarizeMethod::Pyannote => {
            diarize::pyannote::run(&job.audio, job.diarize_speakers, events, cancel)
        }
        _ => diarize::sherpa::run(
            &job.audio,
            &models_dir,
            &job.diarize_models,
            job.diarize_speakers,
            events,
            cancel,
        ),
    };
    match turns {
        Ok(Some(turns)) => {
            let labeled = diarize::assign_speakers(&segments, &turns);
            let _ = events.send(Event::SegmentsFinal(labeled));
        }
        Ok(None) => {
            let _ = events.send(Event::EngineLog(
                "diarization cancelled — transcript kept unlabeled".into(),
            ));
        }
        Err(e) => {
            let _ = events.send(Event::EngineLog(format!(
                "diarization failed: {e:#} — transcript kept unlabeled"
            )));
        }
    }
    let _ = events.send(done);
    Ok(())
}

pub fn spawn(events: Sender<Event>) -> Transcriber {
    let (tx_job, rx_job): (Sender<WorkerMsg>, Receiver<WorkerMsg>) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();

    let handle = std::thread::spawn(move || {
        // Route whisper.cpp/ggml chatter through `log` so it can't
        // write to stderr and corrupt the TUI.
        whisper_rs::install_logging_hooks();

        // Lazy by design: nothing is loaded until the first job arrives,
        // then whisper's context stays resident until an Unload or a job
        // that needs a different model. MLX holds nothing between jobs.
        let mut backends = Backends::new();

        while let Ok(msg) = rx_job.recv() {
            match msg {
                WorkerMsg::Job(job) => {
                    let engine = backends.for_model(&job.model);
                    let post_pass = matches!(
                        job.diarize,
                        DiarizeMethod::Embedding | DiarizeMethod::Pyannote
                    );
                    let result = if post_pass {
                        run_with_diarize_postpass(engine, &job, &events, &worker_cancel)
                    } else {
                        engine.run(&job, &events, &worker_cancel)
                    };
                    if let Err(e) = result {
                        let _ = events.send(Event::Error(format!("{e:#}")));
                    }
                }
                WorkerMsg::Unload => {
                    if backends.loaded() {
                        let _ = events.send(Event::Unloading);
                        backends.unload_all();
                        let _ = events.send(Event::Unloaded);
                    }
                }
            }
        }
    });

    Transcriber {
        jobs: Some(tx_job),
        handle: Some(handle),
        cancel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Emits `segments` segments (optionally pre-labeled) and a Done.
    struct FakeEngine {
        segments: usize,
        labeled: bool,
    }

    impl Engine for FakeEngine {
        fn run(
            &mut self,
            _job: &Job,
            events: &Sender<Event>,
            _cancel: &Arc<AtomicBool>,
        ) -> anyhow::Result<()> {
            let _ = events.send(Event::AudioInfo { duration_secs: 1.0 });
            for i in 0..self.segments {
                let _ = events.send(Event::Segment(Segment {
                    start_ms: i as i64 * 1000,
                    end_ms: (i as i64 + 1) * 1000,
                    text: format!("s{i}"),
                    speaker: self.labeled.then_some(i as u8),
                }));
            }
            let _ = events.send(Event::Done {
                elapsed_secs: 0.1,
                audio_secs: 1.0,
                language: None,
            });
            Ok(())
        }

        fn loaded(&self) -> bool {
            false
        }

        fn unload(&mut self) {}
    }

    fn embedding_job() -> Job {
        Job {
            model: PathBuf::from("/models/model.bin"),
            audio: PathBuf::from("/audio/a.wav"),
            diarize: DiarizeMethod::Embedding,
            diarize_models: Default::default(),
            diarize_speakers: None,
            language: None,
            split_mode: crate::split::SplitMode::Auto,
        }
    }

    fn run_postpass(engine: &mut FakeEngine) -> Vec<Event> {
        let (tx, rx) = channel();
        run_with_diarize_postpass(engine, &embedding_job(), &tx, &Arc::new(AtomicBool::new(false)))
            .unwrap();
        drop(tx);
        rx.iter().collect()
    }

    /// Natively-labeled segments need no second opinion: the post-pass
    /// forwards everything unchanged, Done last, without invoking the
    /// diarizer (which would hit the filesystem/network).
    #[test]
    fn postpass_skips_diarizer_when_model_labeled_speakers_natively() {
        let events = run_postpass(&mut FakeEngine {
            segments: 2,
            labeled: true,
        });
        let segments: Vec<&Segment> = events
            .iter()
            .filter_map(|e| match e {
                Event::Segment(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, Some(0));
        assert!(matches!(events.last(), Some(Event::Done { .. })));
        // no relabeling pass ran
        assert!(!events.iter().any(|e| matches!(e, Event::SegmentsFinal(_))));
    }

    /// An empty transcript has nothing to attribute: Done still arrives.
    #[test]
    fn postpass_forwards_held_done_when_no_segments() {
        let events = run_postpass(&mut FakeEngine {
            segments: 0,
            labeled: false,
        });
        assert!(matches!(events.last(), Some(Event::Done { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::AudioInfo { .. })));
    }
}
