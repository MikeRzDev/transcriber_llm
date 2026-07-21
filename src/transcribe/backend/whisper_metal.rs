//! The whisper.cpp backend (Metal-accelerated via whisper-rs): ensure
//! the model is resident, decode the media, run whisper over the planned
//! chunks, and finalize diarization labels.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::Engine;
use crate::audio;
use crate::split::{self, EngineCaps};
use crate::transcribe::{alternate_speakers, Event, Job, Segment};

/// The whisper.cpp engine: loads GGML/GGUF files lazily and keeps the
/// context resident between jobs until unloaded or a different model is
/// requested.
pub(super) struct WhisperEngine {
    loaded: Option<(PathBuf, WhisperContext)>,
}

impl WhisperEngine {
    pub(super) fn new() -> Self {
        Self { loaded: None }
    }
}

impl Engine for WhisperEngine {
    fn run(
        &mut self,
        job: &Job,
        events: &Sender<Event>,
        cancel: &Arc<AtomicBool>,
    ) -> anyhow::Result<()> {
        run_job(job, &mut self.loaded, events, cancel)
    }

    fn loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn unload(&mut self) {
        self.loaded = None;
    }
}

/// Reading the model file maps onto 0–80% of the load gauge; the
/// remaining 20% is ggml parse + Metal upload.
const READ_PROGRESS_SPAN: u64 = 80;
/// Gauge position while ggml parses the buffer.
const PARSE_PROGRESS: i32 = 85;
const LOAD_DONE_PROGRESS: i32 = 100;
/// Read granularity for the model file (progress resolution).
const READ_CHUNK_BYTES: usize = 8 * 1024 * 1024;
/// whisper stops scaling well past this many threads.
const MAX_THREADS: i32 = 8;
/// whisper timestamps are centiseconds; the app speaks milliseconds.
const CENTIS_TO_MS: i64 = 10;

/// Read the model file in chunks, mapping bytes read onto the load
/// gauge. Returns None if cancelled mid-read.
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
    let mut chunk = vec![0u8; READ_CHUNK_BYTES];
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
        let pct = (read_total * READ_PROGRESS_SPAN)
            .checked_div(total)
            .unwrap_or(0) as i32;
        if pct != last_pct {
            last_pct = pct;
            let _ = events.send(Event::LoadProgress(pct));
        }
    }
    Ok(Some(buffer))
}

/// Make `job.model` the resident context, loading it if a different one
/// (or none) is resident. Returns false when cancelled mid-load.
fn ensure_model_loaded(
    job: &Job,
    model_name: &str,
    loaded: &mut Option<(PathBuf, WhisperContext)>,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> anyhow::Result<bool> {
    if loaded
        .as_ref()
        .map(|(p, _)| p == &job.model)
        .unwrap_or(false)
    {
        return Ok(true);
    }
    // Drop the old context first so both models are never resident at once
    *loaded = None;
    let _ = events.send(Event::LoadingModel(model_name.to_string()));
    let size = std::fs::metadata(&job.model).map(|m| m.len()).unwrap_or(0);
    let _ = events.send(Event::EngineLog(format!(
        "reading model file: {} ({})",
        job.model.display(),
        crate::format::human_size(size)
    )));
    let started = Instant::now();
    // Read the file ourselves so load progress is real (byte-level) and
    // cancellable; whisper then initializes from the buffer. Peak memory
    // is transiently ~2× the model while ggml copies it out.
    let Some(buffer) = read_model_with_progress(&job.model, events, cancel)? else {
        return Ok(false);
    };
    let _ = events.send(Event::LoadProgress(PARSE_PROGRESS));
    let _ = events.send(Event::EngineLog(
        "parsing ggml and uploading weights to Metal (GPU)…".into(),
    ));
    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.use_gpu(true);
    let ctx = WhisperContext::new_from_buffer_with_params(&buffer, ctx_params)?;
    drop(buffer);
    let _ = events.send(Event::LoadProgress(LOAD_DONE_PROGRESS));
    let _ = events.send(Event::ModelReady {
        load_secs: started.elapsed().as_secs_f32(),
    });
    *loaded = Some((job.model.clone(), ctx));
    Ok(true)
}

/// Priority: English-only models are always "en" (they have no
/// multilingual tokens and auto-detection returns garbage), then the
/// user-configured language, then auto-detection.
fn resolve_language<'a>(model_name: &str, job: &'a Job) -> &'a str {
    if crate::models::is_english_only(model_name) {
        "en"
    } else {
        job.language.as_deref().unwrap_or("auto")
    }
}

/// Whisper params for one chunk: greedy sampling, live progress and
/// segment streaming rebased to absolute time, cancel via abort callback.
#[allow(clippy::too_many_arguments)]
fn build_params<'a>(
    diarize: bool,
    language: &'a str,
    threads: i32,
    chunk_idx: usize,
    n_chunks: usize,
    offset_ms: i64,
    emit_from_ms: i64,
    events: &Sender<Event>,
    cancel: &Arc<AtomicBool>,
) -> FullParams<'a, 'static> {
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
        let end_ms = data.end_timestamp * CENTIS_TO_MS + offset_ms;
        // segments entirely inside the overlap were already emitted
        // by the previous chunk
        if end_ms <= emit_from_ms {
            return;
        }
        let _ = segment_tx.send(Event::Segment(Segment {
            start_ms: data.start_timestamp * CENTIS_TO_MS + offset_ms,
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
    let abort_cb: Box<dyn FnMut() -> bool> = Box::new(move || abort_flag.load(Ordering::SeqCst));
    params.set_abort_callback_safe(abort_cb);

    params
}

/// After a diarized chunk, pull the chunk's segments (with turn flags)
/// out of the whisper state for the final labeled pass.
fn collect_diarized(
    state: &whisper_rs::WhisperState,
    offset_ms: i64,
    emit_from_ms: i64,
    labeled: &mut Vec<Segment>,
    turn_flags: &mut Vec<bool>,
) {
    for seg in state.as_iter() {
        let end_ms = seg.end_timestamp() * CENTIS_TO_MS + offset_ms;
        if end_ms <= emit_from_ms {
            continue;
        }
        labeled.push(Segment {
            start_ms: seg.start_timestamp() * CENTIS_TO_MS + offset_ms,
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

    if !ensure_model_loaded(job, &model_name, loaded, events, cancel)? {
        let _ = events.send(Event::Cancelled);
        return Ok(());
    }
    let (_, ctx) = loaded.as_ref().unwrap();

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

    let mut state = ctx.create_state()?;
    let _ = events.send(Event::EngineLog("whisper inference state ready".into()));

    // Diarization is opt-in and only tdrz models emit turn markers
    let diarize = job.diarize && crate::models::is_tdrz(&model_name);
    let language = resolve_language(&model_name, job);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(MAX_THREADS);

    // Chunk plan: one whole-file chunk in Auto mode (whisper.cpp windows
    // internally, word-safe); Silence/Fixed modes chunk client-side, and
    // an engine-declared max window forces chunking in any mode.
    let caps = EngineCaps::whisper();
    let chunks = split::plan_chunks(&decoded.samples, &caps, job.split_mode);
    let n_chunks = chunks.len();
    let _ = events.send(Event::EngineLog(format!(
        "chunk plan: {n_chunks} chunk(s) · split mode {} · {threads} threads · \
         language {language}{}",
        job.split_mode.label(),
        if diarize { " · diarization on" } else { "" }
    )));

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
        let _ = events.send(Event::EngineLog(format!(
            "chunk {}/{n_chunks}: {:.0}s → {:.0}s",
            chunk_idx + 1,
            chunk.start as f32 / caps.sample_rate as f32,
            chunk.end as f32 / caps.sample_rate as f32
        )));

        let params = build_params(
            diarize,
            language,
            threads,
            chunk_idx,
            n_chunks,
            offset_ms,
            emit_from_ms,
            events,
            cancel,
        );

        let result = state.full(params, &decoded.samples[chunk.start..chunk.end]);
        if cancel.load(Ordering::SeqCst) {
            let _ = events.send(Event::Cancelled);
            return Ok(());
        }
        result?;

        if diarize {
            collect_diarized(
                &state,
                offset_ms,
                emit_from_ms,
                &mut labeled,
                &mut turn_flags,
            );
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
