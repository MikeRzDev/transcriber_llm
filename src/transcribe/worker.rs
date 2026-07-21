//! The long-lived worker thread. It owns the resident `WhisperContext`
//! (lazy-loaded, kept between jobs, released on model switch) and turns
//! `WorkerMsg`s into streamed `Event`s.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use whisper_rs::WhisperContext;

use super::run::run_job;
use super::{Event, Job};

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
