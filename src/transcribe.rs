use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio;
use crate::split::{self, EngineCaps, SplitMode};

pub struct Job {
    pub model: PathBuf,
    pub audio: PathBuf,
    /// Label speaker turns (only effective with a tdrz model)
    pub diarize: bool,
    /// ISO 639-1 hint; None = auto-detect
    pub language: Option<String>,
    /// Client-side chunking strategy for long audio
    pub split_mode: SplitMode,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// Speaker index (0 = A, 1 = B) from tinydiarize turn detection;
    /// None when the model doesn't diarize.
    pub speaker: Option<u8>,
}

/// Assign alternating speaker indices from per-segment "next segment is a
/// new speaker" flags. Assumes a two-person conversation: every detected
/// turn flips between speaker 0 and speaker 1.
pub fn alternate_speakers(turn_after: &[bool]) -> Vec<u8> {
    let mut current = 0u8;
    turn_after
        .iter()
        .map(|&turn| {
            let speaker = current;
            if turn {
                current = 1 - current;
            }
            speaker
        })
        .collect()
}

#[derive(Debug)]
pub enum Event {
    LoadingModel(String),
    /// 0–100 while the model file is read and initialized
    LoadProgress(i32),
    ModelReady {
        load_secs: f32,
    },
    /// The resident model is being released (user switched models)
    Unloading,
    Unloaded,
    Decoding,
    AudioInfo {
        duration_secs: f32,
    },
    Progress(i32),
    Segment(Segment),
    /// Re-issued full transcript with speaker labels, sent after a
    /// tinydiarize run completes (streamed segments carry no speaker).
    SegmentsFinal(Vec<Segment>),
    Done {
        elapsed_secs: f32,
        audio_secs: f32,
        language: Option<String>,
    },
    Cancelled,
    Error(String),
}

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

pub fn spawn(events: Sender<Event>) -> Transcriber {
    let (tx_job, rx_job): (Sender<WorkerMsg>, Receiver<WorkerMsg>) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = cancel.clone();

    let handle = std::thread::spawn(move || {
        // Route whisper.cpp/ggml chatter through `log` so it can't
        // write to stderr and corrupt the TUI.
        whisper_rs::install_logging_hooks();

        // Lazy by design: nothing is loaded until the first job arrives,
        // then the context stays resident until an Unload or a job that
        // needs a different model.
        let mut loaded: Option<(PathBuf, WhisperContext)> = None;

        while let Ok(msg) = rx_job.recv() {
            match msg {
                WorkerMsg::Job(job) => {
                    if let Err(e) = run_job(&job, &mut loaded, &events, &worker_cancel) {
                        let _ = events.send(Event::Error(format!("{e:#}")));
                    }
                }
                WorkerMsg::Unload => {
                    if loaded.is_some() {
                        let _ = events.send(Event::Unloading);
                        loaded = None;
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

/// Read the model file in chunks, mapping bytes read onto 0–80% of the
/// load gauge (the remaining 20% is ggml parse + Metal upload). Returns
/// None if cancelled mid-read.
fn read_model_with_progress(
    path: &std::path::Path,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<Option<Vec<u8>>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let total = file.metadata()?.len();
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::with_capacity(total as usize);
    let mut chunk = vec![0u8; 8 * 1024 * 1024];
    let mut read_total: u64 = 0;
    let mut last_pct = -1;
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..n]);
        read_total += n as u64;
        let pct = if total > 0 {
            (read_total * 80 / total) as i32
        } else {
            0
        };
        if pct != last_pct {
            last_pct = pct;
            let _ = events.send(Event::LoadProgress(pct));
        }
    }
    Ok(Some(buffer))
}

fn run_job(
    job: &Job,
    loaded: &mut Option<(PathBuf, WhisperContext)>,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let model_name = job
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    if loaded
        .as_ref()
        .map(|(p, _)| p != &job.model)
        .unwrap_or(true)
    {
        // Drop the old context first so both models are never resident at once
        *loaded = None;
        let _ = events.send(Event::LoadingModel(model_name.clone()));
        let started = Instant::now();
        // Read the file ourselves so load progress is real (byte-level) and
        // cancellable; whisper then initializes from the buffer. Peak memory
        // is transiently ~2× the model while ggml copies it out.
        let Some(buffer) = read_model_with_progress(&job.model, events, cancel)? else {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        };
        let _ = events.send(Event::LoadProgress(85));
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);
        let ctx = WhisperContext::new_from_buffer_with_params(&buffer, ctx_params)?;
        drop(buffer);
        let _ = events.send(Event::LoadProgress(100));
        let _ = events.send(Event::ModelReady {
            load_secs: started.elapsed().as_secs_f32(),
        });
        *loaded = Some((job.model.clone(), ctx));
    }
    let (_, ctx) = loaded.as_ref().unwrap();

    let _ = events.send(Event::Decoding);
    let decoded = audio::load_media(&job.audio)?;
    let _ = events.send(Event::AudioInfo {
        duration_secs: decoded.duration_secs,
    });

    let mut state = ctx.create_state()?;

    // Diarization is opt-in and only tdrz models emit turn markers
    let diarize = job.diarize && crate::models::is_tdrz(&model_name);

    // Priority: English-only models are always "en" (they have no
    // multilingual tokens and auto-detection returns garbage), then the
    // user-configured language, then auto-detection.
    let language: &str = if crate::models::is_english_only(&model_name) {
        "en"
    } else {
        job.language.as_deref().unwrap_or("auto")
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);

    // Chunk plan: one whole-file chunk in Auto mode (whisper.cpp windows
    // internally, word-safe); Silence/Fixed modes chunk client-side, and
    // an engine-declared max window forces chunking in any mode.
    let caps = EngineCaps::whisper();
    let chunks = split::plan_chunks(&decoded.samples, &caps, job.split_mode);
    let n_chunks = chunks.len();

    let started = Instant::now();
    let mut labeled: Vec<Segment> = Vec::new();
    let mut turn_flags: Vec<bool> = Vec::new();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        }
        let rate = caps.sample_rate as i64;
        let offset_ms = chunk.start as i64 * 1000 / rate;
        let emit_from_ms = chunk.emit_from as i64 * 1000 / rate;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_tdrz_enable(diarize);
        params.set_language(Some(language));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_n_threads(threads);

        let progress_tx = events.clone();
        params.set_progress_callback_safe(move |p: i32| {
            let overall = (chunk_idx as i32 * 100 + p) / n_chunks as i32;
            let _ = progress_tx.send(Event::Progress(overall));
        });

        let segment_tx = events.clone();
        params.set_segment_callback_safe(move |data: whisper_rs::SegmentCallbackData| {
            // whisper timestamps are in centiseconds, relative to the chunk
            let end_ms = data.end_timestamp * 10 + offset_ms;
            // segments entirely inside the overlap were already emitted
            // by the previous chunk
            if end_ms <= emit_from_ms {
                return;
            }
            let _ = segment_tx.send(Event::Segment(Segment {
                start_ms: data.start_timestamp * 10 + offset_ms,
                end_ms,
                text: data.text,
                speaker: None,
            }));
        });

        // whisper-rs 0.16.0's set_abort_callback_safe instantiates its C
        // trampoline for the bare closure type while storing a double-boxed
        // trait object, so a plain closure gets reinterpreted as garbage and
        // randomly aborts mid-transcription (whisper errors -6/-9). Passing a
        // pre-boxed Box<dyn FnMut() -> bool> makes F the trait-object box and
        // the layers line up.
        let abort_flag = cancel.clone();
        let abort_cb: Box<dyn FnMut() -> bool> =
            Box::new(move || abort_flag.load(Ordering::SeqCst));
        params.set_abort_callback_safe(abort_cb);

        let result = state.full(params, &decoded.samples[chunk.start..chunk.end]);
        if cancel.load(Ordering::SeqCst) {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        }
        result?;

        if diarize {
            for seg in state.as_iter() {
                let end_ms = seg.end_timestamp() * 10 + offset_ms;
                if end_ms <= emit_from_ms {
                    continue;
                }
                labeled.push(Segment {
                    start_ms: seg.start_timestamp() * 10 + offset_ms,
                    end_ms,
                    text: seg
                        .to_str_lossy()
                        .map(|t| t.into_owned())
                        .unwrap_or_default(),
                    speaker: None,
                });
                turn_flags.push(seg.next_segment_speaker_turn());
            }
        }
    }

    if diarize {
        // Alternation runs across the concatenated chunks, so speaker
        // identity carries over chunk boundaries
        let speakers = alternate_speakers(&turn_flags);
        for (seg, speaker) in labeled.iter_mut().zip(speakers) {
            seg.speaker = Some(speaker);
        }
        let _ = events.send(Event::SegmentsFinal(labeled));
    }

    let language = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(|s| s.to_string());

    let _ = events.send(Event::Done {
        elapsed_secs: started.elapsed().as_secs_f32(),
        audio_secs: decoded.duration_secs,
        language,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unload with nothing loaded is a no-op, and the worker must still
    /// shut down cleanly afterwards (no hang, no panic).
    #[test]
    fn unload_is_safe_with_no_model_loaded() {
        let (tx, _rx) = channel();
        let mut transcriber = spawn(tx);
        transcriber.request_unload();
        transcriber.request_unload();
        transcriber.shutdown();
    }

    #[test]
    fn no_turns_is_all_speaker_a() {
        assert_eq!(alternate_speakers(&[false, false, false]), vec![0, 0, 0]);
    }

    #[test]
    fn turns_alternate_between_two_speakers() {
        // turn flag means the NEXT segment has a new speaker
        assert_eq!(
            alternate_speakers(&[true, false, true, true, false]),
            vec![0, 1, 1, 0, 1]
        );
    }
}
