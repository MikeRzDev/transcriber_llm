//! State and key handling for the Model management modal (Hugging Face
//! hub): suggested list → search results → repo file view → download.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::KeyCode;

use crate::app::App;
use crate::hub::{self, HubEvent, HubFile, RepoHit, SuggestedModel};

/// State of the Model management modal (search + download from Hugging Face).
pub struct HubState {
    pub open: bool,
    /// Search bar contents; empty shows the suggested list
    pub input: String,
    pub selected: usize,
    pub searching: bool,
    /// Some(repo) while its file list is being fetched
    pub listing_repo: Option<String>,
    /// None → suggested list; Some → search results
    pub results: Option<Vec<RepoHit>>,
    /// Some((repo, files)) → file view of one repo
    pub files: Option<(String, Vec<HubFile>)>,
    pub suggested: Vec<SuggestedModel>,
    /// Some((file, got_bytes, total_bytes)) while a download runs
    pub download: Option<(String, u64, u64)>,
    pub cancel: Arc<AtomicBool>,
    /// Modal-local status/info line
    pub info: String,
    /// Debounce marker: search fires shortly after typing pauses
    pub(crate) last_edit: Option<Instant>,
}

/// Whichever of the three hub lists is currently visible.
pub enum HubList<'a> {
    Files(&'a [HubFile]),
    Results(&'a [RepoHit]),
    Suggested(&'a [SuggestedModel]),
}

impl HubList<'_> {
    pub fn len(&self) -> usize {
        match self {
            HubList::Files(files) => files.len(),
            HubList::Results(results) => results.len(),
            HubList::Suggested(suggested) => suggested.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl HubState {
    /// Priority order: a repo's file view, then search results, then the
    /// suggested list.
    pub fn visible_list(&self) -> HubList<'_> {
        if let Some((_, files)) = &self.files {
            HubList::Files(files)
        } else if let Some(results) = &self.results {
            HubList::Results(results)
        } else {
            HubList::Suggested(&self.suggested)
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            open: false,
            input: String::new(),
            selected: 0,
            searching: false,
            listing_repo: None,
            results: None,
            files: None,
            suggested: hub::suggested_models(),
            download: None,
            cancel: Arc::new(AtomicBool::new(false)),
            info: String::new(),
            last_edit: None,
        }
    }
}

impl App {
    pub fn open_hub(&mut self) {
        self.refresh_models();
        self.hub.open = true;
        self.hub.selected = 0;
        self.hub.info.clear();
    }

    /// Called every render loop: fires the debounced search and drains
    /// events from hub worker threads.
    pub fn hub_pump(&mut self) {
        if let Some(t) = self.hub.last_edit {
            if t.elapsed() >= Duration::from_millis(450) {
                self.hub.last_edit = None;
                let query = self.hub.input.trim().to_string();
                if query.len() >= 2 && self.hub.files.is_none() {
                    self.hub.searching = true;
                    hub::search(query, self.hub_tx.clone());
                }
            }
        }
        let events: Vec<HubEvent> = std::iter::from_fn(|| self.hub_rx.try_recv().ok()).collect();
        for event in events {
            self.handle_hub_event(event);
        }
    }

    pub fn hub_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                if self.hub.download.is_some() {
                    self.hub.cancel.store(true, Ordering::Relaxed);
                    self.hub.info = "Cancelling download…".into();
                } else if self.hub.files.is_some() {
                    self.hub.files = None;
                    self.hub.selected = 0;
                } else if !self.hub.input.is_empty() {
                    self.hub.input.clear();
                    self.hub.results = None;
                    self.hub.searching = false;
                    self.hub.last_edit = None;
                    self.hub.selected = 0;
                    self.hub.info.clear();
                } else {
                    self.hub.open = false;
                }
            }
            KeyCode::Up => self.hub.selected = self.hub.selected.saturating_sub(1),
            KeyCode::Down => {
                let len = self.hub.visible_list().len();
                if len > 0 {
                    self.hub.selected = (self.hub.selected + 1).min(len - 1);
                }
            }
            KeyCode::Backspace if self.hub.files.is_none() => {
                self.hub.input.pop();
                self.hub_input_edited();
            }
            KeyCode::Char(c) if self.hub.files.is_none() => {
                self.hub.input.push(c);
                self.hub_input_edited();
            }
            KeyCode::Enter => self.hub_enter(),
            _ => {}
        }
    }

    fn hub_input_edited(&mut self) {
        self.hub.last_edit = Some(Instant::now());
        self.hub.selected = 0;
        if self.hub.input.trim().is_empty() {
            self.hub.results = None;
            self.hub.searching = false;
        }
    }

    fn hub_enter(&mut self) {
        if self.hub.download.is_some() {
            self.hub.info = "A download is already running — Esc cancels it".into();
            return;
        }
        // File view: download the selected file
        if let Some((repo, files)) = &self.hub.files {
            if let Some(f) = files.get(self.hub.selected) {
                let (repo, file) = (repo.clone(), f.name.clone());
                self.hub_start_download(repo, file);
            }
            return;
        }
        // Search results: open the repo's file list
        if let Some(results) = &self.hub.results {
            if let Some(hit) = results.get(self.hub.selected) {
                let repo = hit.id.clone();
                self.hub.listing_repo = Some(repo.clone());
                self.hub.info = format!("Fetching file list for {repo}…");
                hub::list_files(repo, self.hub_tx.clone());
            }
            return;
        }
        // Suggested list: download directly (if the format is runnable)
        if let Some(s) = self.hub.suggested.get(self.hub.selected).cloned() {
            if !s.supported() {
                self.hub.info = format!(
                    "{} is {}-format — whisper.cpp can only run GGML/GGUF models",
                    s.name, s.format
                );
                return;
            }
            self.hub_start_download(s.repo, s.file);
        }
    }

    fn hub_start_download(&mut self, repo: String, file: String) {
        let Some(base) = hub::dest_name(&file) else {
            self.hub.info = "Bad file name".into();
            return;
        };
        if self.library.models.iter().any(|m| m.name == base) {
            self.hub.info = format!("{base} is already in the models folder");
            return;
        }
        self.hub.cancel = Arc::new(AtomicBool::new(false));
        self.hub.download = Some((base.clone(), 0, 0));
        self.hub.info = format!("Downloading {base}…");
        hub::download(
            repo,
            file,
            self.library.dir.clone(),
            self.hub.cancel.clone(),
            self.hub_tx.clone(),
        );
    }

    pub(crate) fn handle_hub_event(&mut self, event: HubEvent) {
        match event {
            HubEvent::SearchResults { query, hits } => {
                self.hub.searching = false;
                // Drop stale results the input has moved past
                if query == self.hub.input.trim() {
                    self.hub.info = if hits.is_empty() {
                        format!("No repos match '{query}'")
                    } else {
                        String::new()
                    };
                    self.hub.results = Some(hits);
                    self.hub.selected = 0;
                }
            }
            HubEvent::SearchFailed { query, error } => {
                self.hub.searching = false;
                self.hub.info = format!("Search '{query}' failed: {error}");
            }
            HubEvent::Files { repo, files } => {
                if self.hub.listing_repo.as_deref() == Some(repo.as_str()) {
                    self.hub.listing_repo = None;
                    if files.is_empty() {
                        self.hub.info = format!("No GGML/GGUF files in {repo}");
                    } else {
                        self.hub.files = Some((repo, files));
                        self.hub.selected = 0;
                        self.hub.info.clear();
                    }
                }
            }
            HubEvent::FilesFailed { repo, error } => {
                self.hub.listing_repo = None;
                self.hub.info = format!("Listing {repo} failed: {error}");
            }
            HubEvent::Progress { file, got, total } => {
                self.hub.download = Some((file, got, total));
            }
            HubEvent::Done { file, path } => {
                self.hub.download = None;
                self.refresh_models();
                self.hub.info = format!("Downloaded {file} ✓");
                self.status = format!("Downloaded {file} to {}", self.library.dir.display());
                if self.library.selected.is_none() {
                    self.library.selected =
                        self.library.models.iter().find(|m| m.path == path).cloned();
                }
            }
            HubEvent::Cancelled { file } => {
                self.hub.download = None;
                self.hub.info = format!("Cancelled {file}");
            }
            HubEvent::Failed { file, error } => {
                self.hub.download = None;
                self.hub.info = format!("Download of {file} failed: {error}");
            }
            HubEvent::ModelsMoved {
                moved,
                skipped,
                failed,
            } => {
                self.refresh_models();
                self.adopt_selection();
                let mut parts = vec![format!(
                    "Moved {moved} model(s) to {}",
                    self.library.dir.display()
                )];
                if skipped > 0 {
                    parts.push(format!("{skipped} already existed"));
                }
                if failed > 0 {
                    parts.push(format!("{failed} FAILED"));
                }
                self.status = parts.join(" · ");
            }
        }
    }
}
