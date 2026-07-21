//! State and key handling for the Model management modal (Hugging Face
//! hub): suggested list → search results → repo file view → download.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::KeyCode;

use crate::app::App;
use crate::config;
use crate::diarize::sherpa::{self, DiarizeRole};
use crate::diarize::{self, DiarizeMethod, DiarizeStrategy};
use crate::hub::{self, DirVariant, HubEvent, HubFile, RepoHit, SuggestedModel};
use crate::models;

/// One repo's downloadables: directory-model variants first, then its
/// single GGML/GGUF files. Rows are indexed across both lists.
pub struct RepoView {
    pub repo: String,
    pub variants: Vec<DirVariant>,
    pub files: Vec<HubFile>,
}

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
    /// Some → the downloadables of one repo
    pub files: Option<RepoView>,
    pub suggested: Vec<SuggestedModel>,
    /// Some while a download is running or paused
    pub download: Option<Download>,
    /// Modal-local status/info line
    pub info: String,
    /// Some while confirming deletion of a downloaded model
    pub delete_prompt: Option<DeletePrompt>,
    /// The download target (dest name) whose memory-misfit warning was
    /// shown; Enter on the same target again downloads anyway
    pub hw_ack: Option<String>,
    /// Debounce marker: search fires shortly after typing pauses
    pub(crate) last_edit: Option<Instant>,
}

/// Yes/No confirmation shown before a downloaded model is deleted from disk.
pub struct DeletePrompt {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub yes_selected: bool,
}

/// A model download that is running or paused. A pause stops the worker but
/// keeps the `.part` data; resuming spawns a fresh worker that continues it.
pub struct Download {
    /// On-disk base name (progress label + `.part` matching)
    pub file: String,
    /// Repo id, kept so a paused transfer can resume
    pub repo: String,
    /// Repo-relative path used to build the download URL (single-file
    /// downloads only; empty for directory models)
    pub remote_file: String,
    /// Whole-repo directory model (MLX) rather than a single file
    pub is_dir: bool,
    /// Variant subfolder for directory downloads ("" = the whole repo)
    pub subdir: String,
    /// Where the download lands: the models folder, or its diarization
    /// subfolder for diarization components
    pub dest_dir: PathBuf,
    pub got: u64,
    pub total: u64,
    pub paused: bool,
    pub cancel: Arc<AtomicBool>,
    pub pause: Arc<AtomicBool>,
}

/// Whichever of the three hub lists is currently visible.
pub enum HubList<'a> {
    /// One repo's downloadables (variant folders first, then files)
    Files(&'a RepoView),
    Results(&'a [RepoHit]),
    /// The pre-search view: downloaded models (checkmarked) merged with the
    /// curated suggestions that aren't downloaded yet.
    Default(Vec<DefaultEntry<'a>>),
}

impl HubList<'_> {
    pub fn len(&self) -> usize {
        match self {
            HubList::Files(view) => view.variants.len() + view.files.len(),
            HubList::Results(results) => results.len(),
            HubList::Default(entries) => entries.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a default-view row is: a transcription model, the diarization
/// section header, or a diarization component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Model,
    Section,
    Diarize,
}

/// One row of the default hub view: a model already on disk, a curated
/// suggestion still to be downloaded, or a diarization catalog entry.
pub struct DefaultEntry<'a> {
    pub kind: EntryKind,
    /// Diarize rows only: this component is the one its role uses
    pub active: bool,
    /// Friendly display name (the suggestion's name, else the file name)
    pub name: &'a str,
    /// On-disk name — the download destination and library match key
    /// (a file's base name, or the folder name for directory models)
    pub file: String,
    /// Repo to download from; None once the model is on disk
    pub repo: Option<&'a str>,
    /// Real size when downloaded, else the suggestion's estimate
    pub size: String,
    pub note: &'a str,
    pub format: &'a str,
    /// Directory model (whole-repo download, runs on the MLX engine)
    pub is_dir: bool,
    /// An engine in this build can run it
    pub supported: bool,
    /// Present in the models folder
    pub installed: bool,
    /// Fits this machine's memory budget (see [`crate::hw::fits`])
    pub hw_ok: bool,
}

impl HubState {
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
            info: String::new(),
            delete_prompt: None,
            hw_ack: None,
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
        self.hub.hw_ack = None;
        // An unusable models folder explains itself immediately instead
        // of failing the first download with a bare OS error
        if let Some(reason) = crate::models::dir_unavailable(&self.library.dir) {
            self.hub.info = format!("Models folder unavailable: {reason}");
        }
    }

    /// Priority order: a repo's file view, then search results, then the
    /// default view (downloaded models + curated suggestions).
    pub fn hub_visible_list(&self) -> HubList<'_> {
        if let Some(view) = &self.hub.files {
            HubList::Files(view)
        } else if let Some(results) = &self.hub.results {
            HubList::Results(results)
        } else {
            HubList::Default(self.hub_default_entries())
        }
    }

    /// The pre-search rows: every downloaded model first (checkmarked, real
    /// folder size), then the curated suggestions not yet on disk (greyed).
    pub fn hub_default_entries(&self) -> Vec<DefaultEntry<'_>> {
        let mut entries = Vec::new();
        for m in &self.library.models {
            let sug = self.hub.suggested.iter().find(|s| s.dest_name() == m.name);
            entries.push(DefaultEntry {
                kind: EntryKind::Model,
                active: false,
                name: sug.map_or(m.name.as_str(), |s| s.name.as_str()),
                file: m.name.clone(),
                repo: None,
                size: crate::format::human_size(m.size_bytes),
                note: sug.map_or("", |s| s.note.as_str()),
                format: sug.map_or("", |s| s.format.as_str()),
                is_dir: m.is_dir,
                supported: true,
                installed: true,
                hw_ok: crate::hw::fits(m.size_bytes),
            });
        }
        for s in &self.hub.suggested {
            let dest = s.dest_name();
            if self.library.models.iter().any(|m| m.name == dest) {
                continue; // already listed among the downloaded models above
            }
            entries.push(DefaultEntry {
                kind: EntryKind::Model,
                active: false,
                name: &s.name,
                file: dest,
                repo: Some(&s.repo),
                size: s.size.clone(),
                note: &s.note,
                format: &s.format,
                is_dir: s.is_dir_model(),
                supported: s.supported(),
                installed: false,
                hw_ok: crate::hw::fits(crate::hw::parse_human_size(&s.size)),
            });
        }
        // The diarization category: the tdrz whisper build (the
        // TinyDiarize strategy) and the embedding pipeline's components
        // (usable with any transcription model, stored under
        // <models>/diarization). Enter downloads a missing one or makes
        // an installed one active — the tdrz row selects its strategy,
        // the others become their role's choice.
        entries.push(DefaultEntry {
            kind: EntryKind::Section,
            active: false,
            name: "diarization — ✓ = in use · Enter downloads / selects",
            file: String::new(),
            repo: None,
            size: String::new(),
            note: "",
            format: "",
            is_dir: false,
            supported: true,
            installed: false,
            hw_ok: true,
        });
        entries.push(DefaultEntry {
            kind: EntryKind::Diarize,
            active: self.resolved_diarize_method() == DiarizeMethod::Tdrz,
            name: "tinydiarize (tdrz)",
            file: diarize::TDRZ_FILE.to_string(),
            repo: Some(diarize::TDRZ_REPO),
            size: diarize::TDRZ_SIZE.to_string(),
            note: "whisper small.en with speaker-turn tokens · English · 2 speakers",
            format: "strategy",
            is_dir: false,
            supported: true,
            installed: self.library.models.iter().any(|m| models::is_tdrz(&m.name)),
            hw_ok: true,
        });
        for m in &sherpa::CATALOG {
            let configured = match m.role {
                DiarizeRole::Segmentation => self.config.diarize_models.segmentation.as_deref(),
                DiarizeRole::Embedding => self.config.diarize_models.embedding.as_deref(),
            };
            entries.push(DefaultEntry {
                kind: EntryKind::Diarize,
                active: sherpa::pick(m.role, configured).local == m.local,
                name: m.label,
                file: m.local.to_string(),
                repo: Some(m.repo),
                size: m.size.to_string(),
                note: m.note,
                format: m.role.label(),
                is_dir: false,
                supported: true,
                installed: m.installed(&self.library.dir),
                hw_ok: true,
            });
        }
        entries
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
        // A pending delete confirmation captures every key first.
        if self.hub.delete_prompt.is_some() {
            self.hub_delete_prompt_key(code);
            return;
        }
        match code {
            KeyCode::Esc => {
                if self.hub.download.is_some() {
                    self.hub_cancel_download();
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
            // Download controls (guarded so the letters still type into search
            // when nothing is downloading).
            KeyCode::Char('p') if self.hub.download.is_some() => self.hub_toggle_pause(),
            KeyCode::Char('c') if self.hub.download.is_some() => self.hub_cancel_download(),
            KeyCode::Up => self.hub.selected = self.hub.selected.saturating_sub(1),
            KeyCode::Down => {
                let len = self.hub_visible_list().len();
                if len > 0 {
                    self.hub.selected = (self.hub.selected + 1).min(len - 1);
                }
            }
            KeyCode::Delete => self.hub_request_delete(),
            KeyCode::Backspace if self.hub.files.is_none() => {
                if self.hub.input.is_empty() {
                    // Empty search box + a downloaded model selected → offer to
                    // delete it (the Mac "delete" key reports as Backspace).
                    self.hub_request_delete();
                } else {
                    self.hub.input.pop();
                    self.hub_input_edited();
                }
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

    /// Memory gate before a download: passes when the model fits this
    /// machine (or its size is unknown), or when the misfit warning for
    /// this exact target was already shown and Enter came again.
    fn hub_hw_gate(&mut self, dest: &str, name: &str, weights_bytes: u64) -> bool {
        if crate::hw::fits(weights_bytes) || self.hub.hw_ack.as_deref() == Some(dest) {
            self.hub.hw_ack = None;
            return true;
        }
        self.hub.hw_ack = Some(dest.to_string());
        self.hub.info = format!(
            "⚠ {name} {} — Enter again to download anyway",
            crate::hw::misfit_note(weights_bytes)
        );
        false
    }

    fn hub_enter(&mut self) {
        if self.hub.download.is_some() {
            self.hub.info = "A download is already running — p pauses · Esc cancels".into();
            return;
        }
        // File view: download the selected variant folder or single file
        if let Some(view) = &self.hub.files {
            if let Some(variant) = view.variants.get(self.hub.selected) {
                if !hub::metal_available() {
                    self.hub.info =
                        "MLX directory models need an Apple Silicon Mac".into();
                    return;
                }
                let repo = view.repo.clone();
                let subdir = variant.subdir.clone();
                let dest = hub::variant_dir_name(&repo, &subdir);
                if !self.hub_hw_gate(&dest, &dest, variant.size_bytes) {
                    return;
                }
                self.hub_start_dir_download(repo, subdir);
            } else if let Some(f) = view
                .files
                .get(self.hub.selected - view.variants.len())
            {
                let (repo, file, size) = (view.repo.clone(), f.name.clone(), f.size_bytes);
                if !self.hub_hw_gate(&file, &file, size) {
                    return;
                }
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
        // Default view: select a downloaded model as the default, start a
        // download for a suggestion that isn't on disk yet, or — in the
        // diarization section — download a component / make it active.
        let action = self.hub_default_entries().get(self.hub.selected).map(|e| {
            (
                e.kind,
                e.installed,
                e.supported,
                e.is_dir,
                e.repo.map(|r| r.to_string()),
                e.file.clone(),
                e.name.to_string(),
                e.format.to_string(),
                crate::hw::parse_human_size(&e.size),
            )
        });
        let Some((kind, installed, supported, is_dir, repo, file, name, format, weights)) = action
        else {
            return;
        };
        match kind {
            EntryKind::Section => {}
            EntryKind::Diarize => {
                // The tdrz row is a whole strategy, not a pipeline role:
                // its model lives in the models folder root (the whisper
                // engine transcribes with it) and Enter selects the
                // TinyDiarize strategy once it is installed.
                if file == diarize::TDRZ_FILE {
                    if installed {
                        self.config.diarize = DiarizeStrategy::Tdrz;
                        let _ = config::save(&self.config);
                        self.hub.info =
                            "Diarization strategy set to TinyDiarize (2 speakers, English)".into();
                    } else if let Some(repo) = repo {
                        self.hub_start_download(repo, file);
                    }
                } else if installed {
                    self.set_active_diarize_model(&file);
                } else if let Some(repo) = repo {
                    self.hub_start_diarize_download(repo, file);
                }
            }
            EntryKind::Model => {
                if installed {
                    if let Some(model) =
                        self.library.models.iter().find(|m| m.name == file).cloned()
                    {
                        let size_bytes = model.size_bytes;
                        self.choose_model(model);
                        self.hub.info = if crate::hw::fits(size_bytes) {
                            format!("Selected {file} as the default model")
                        } else {
                            format!(
                                "Selected {file} — ⚠ {}; jobs may abort out of memory",
                                crate::hw::misfit_note(size_bytes)
                            )
                        };
                    }
                } else if !supported {
                    self.hub.info = if format == "mlx" {
                        format!("{name} is an MLX model — it needs an Apple Silicon Mac")
                    } else {
                        format!("{name} is {format}-format — no engine here can run it")
                    };
                } else if let Some(repo) = repo {
                    if !self.hub_hw_gate(&file, &name, weights) {
                        return;
                    }
                    if is_dir {
                        self.hub_start_dir_download(repo, String::new());
                    } else {
                        self.hub_start_download(repo, file);
                    }
                }
            }
        }
    }

    /// Move the hub selection onto the tdrz row of the diarization
    /// section (leaving any search or file view — the row lives in the
    /// default view). Used by the tdrz download offer so the transfer it
    /// starts is in view.
    pub(crate) fn hub_select_tdrz_row(&mut self) {
        self.hub.files = None;
        self.hub.results = None;
        self.hub.input.clear();
        self.hub.searching = false;
        self.hub.last_edit = None;
        if let Some(index) = self
            .hub_default_entries()
            .iter()
            .position(|e| e.kind == EntryKind::Diarize && e.file == diarize::TDRZ_FILE)
        {
            self.hub.selected = index;
        }
    }

    /// Make an installed diarization component the one its role uses,
    /// and persist the choice.
    fn set_active_diarize_model(&mut self, local: &str) {
        let Some(model) = sherpa::CATALOG.iter().find(|m| m.local == local) else {
            return;
        };
        match model.role {
            DiarizeRole::Segmentation => {
                self.config.diarize_models.segmentation = Some(local.to_string())
            }
            DiarizeRole::Embedding => {
                self.config.diarize_models.embedding = Some(local.to_string())
            }
        }
        let _ = config::save(&self.config);
        self.hub.info = format!(
            "Diarization now uses {} for {}",
            model.label,
            model.role.label()
        );
    }

    /// A download can only land in a folder that exists (or can exist):
    /// a models folder on an unmounted external drive fails with a bare
    /// "permission denied" unless caught here with the real reason.
    fn hub_dest_unavailable(&mut self) -> bool {
        match crate::models::dir_unavailable(&self.library.dir) {
            Some(reason) => {
                self.hub.info = format!("Cannot download: {reason}");
                true
            }
            None => false,
        }
    }

    /// Download a diarization component into `<models>/diarization`
    /// under its catalog name.
    fn hub_start_diarize_download(&mut self, repo: String, local: String) {
        if self.hub_dest_unavailable() {
            return;
        }
        let Some(model) = sherpa::CATALOG.iter().find(|m| m.local == local) else {
            return;
        };
        let dest_dir = sherpa::dir(&self.library.dir);
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        self.hub.download = Some(Download {
            file: local.clone(),
            repo: repo.clone(),
            remote_file: model.remote.to_string(),
            is_dir: false,
            subdir: String::new(),
            dest_dir: dest_dir.clone(),
            got: 0,
            total: 0,
            paused: false,
            cancel: cancel.clone(),
            pause: pause.clone(),
        });
        self.hub.info = format!(
            "Downloading {local} ({})… — p pauses · Esc cancels",
            model.size
        );
        hub::download_as(
            repo,
            model.remote.to_string(),
            local,
            dest_dir,
            cancel,
            pause,
            self.hub_tx.clone(),
        );
    }

    pub(crate) fn hub_start_download(&mut self, repo: String, file: String) {
        if self.hub_dest_unavailable() {
            return;
        }
        let Some(base) = hub::dest_name(&file) else {
            self.hub.info = "Bad file name".into();
            return;
        };
        if self.library.models.iter().any(|m| m.name == base) {
            self.hub.info = format!("{base} is already in the models folder");
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        self.hub.download = Some(Download {
            file: base.clone(),
            repo: repo.clone(),
            remote_file: file.clone(),
            is_dir: false,
            subdir: String::new(),
            dest_dir: self.library.dir.clone(),
            got: 0,
            total: 0,
            paused: false,
            cancel: cancel.clone(),
            pause: pause.clone(),
        });
        self.hub.info = format!("Downloading {base}… — p pauses · Esc cancels");
        hub::download(repo, file, self.library.dir.clone(), cancel, pause, self.hub_tx.clone());
    }

    /// Download a directory model: every file of `repo` under `subdir`
    /// ("" = the whole repo) into its own folder under the models dir.
    /// Also the repair path: a directory failing its integrity manifest
    /// is absent from the scan, so it arrives back here and only its
    /// missing files are fetched.
    fn hub_start_dir_download(&mut self, repo: String, subdir: String) {
        if self.hub_dest_unavailable() {
            return;
        }
        let name = hub::variant_dir_name(&repo, &subdir);
        if self.library.models.iter().any(|m| m.name == name) {
            self.hub.info = format!("{name} is already in the models folder");
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        self.hub.download = Some(Download {
            file: name.clone(),
            repo: repo.clone(),
            remote_file: String::new(),
            is_dir: true,
            subdir: subdir.clone(),
            dest_dir: self.library.dir.clone(),
            got: 0,
            total: 0,
            paused: false,
            cancel: cancel.clone(),
            pause: pause.clone(),
        });
        self.hub.info = format!("Downloading {name} (model folder)… — p pauses · Esc cancels");
        hub::download_dir(
            repo,
            subdir,
            self.library.dir.clone(),
            cancel,
            pause,
            self.hub_tx.clone(),
        );
    }

    /// Toggle the running download between paused and resumed. Pausing asks the
    /// worker to stop (it keeps the `.part` file); resuming spawns a fresh
    /// worker that continues that partial file via an HTTP Range request.
    fn hub_toggle_pause(&mut self) {
        let Some(d) = &self.hub.download else {
            return;
        };
        if d.paused {
            let repo = d.repo.clone();
            let remote_file = d.remote_file.clone();
            let file = d.file.clone();
            let is_dir = d.is_dir;
            let subdir = d.subdir.clone();
            let dest_dir = d.dest_dir.clone();
            let cancel = Arc::new(AtomicBool::new(false));
            let pause = Arc::new(AtomicBool::new(false));
            if let Some(d) = &mut self.hub.download {
                d.paused = false;
                d.cancel = cancel.clone();
                d.pause = pause.clone();
            }
            self.hub.info = format!("Resuming {file}… — p pauses · Esc cancels");
            if is_dir {
                hub::download_dir(repo, subdir, dest_dir, cancel, pause, self.hub_tx.clone());
            } else {
                // download_as: the local name must survive the resume
                // (diarization components rename generic remote files)
                hub::download_as(
                    repo,
                    remote_file,
                    file,
                    dest_dir,
                    cancel,
                    pause,
                    self.hub_tx.clone(),
                );
            }
        } else {
            d.pause.store(true, Ordering::Relaxed);
            let name = d.file.clone();
            self.hub.info = format!("Pausing {name}…");
        }
    }

    /// Cancel the running or paused download and discard its partial file.
    fn hub_cancel_download(&mut self) {
        let Some(d) = &self.hub.download else {
            return;
        };
        if d.paused {
            // No worker is running — clean up the partial data directly.
            let name = d.file.clone();
            let part = d.dest_dir.join(format!("{name}.part"));
            if d.is_dir {
                let _ = std::fs::remove_dir_all(part);
            } else {
                let _ = std::fs::remove_file(part);
            }
            self.hub.download = None;
            self.hub.info = format!("Cancelled {name}");
        } else {
            d.cancel.store(true, Ordering::Relaxed);
            self.hub.info = "Cancelling download…".into();
        }
    }

    /// Open the delete confirmation for the selected row — downloaded
    /// models and diarization components (default view) can be deleted.
    fn hub_request_delete(&mut self) {
        if self.hub.files.is_some() || self.hub.results.is_some() || self.hub.download.is_some() {
            return;
        }
        let target = self
            .hub_default_entries()
            .get(self.hub.selected)
            .map(|e| (e.kind, e.installed, e.file.clone(), e.name.to_string()));
        let Some((kind, installed, file, name)) = target else {
            return;
        };
        if kind == EntryKind::Section {
            return;
        }
        if !installed {
            self.hub.info = format!("{name} isn't downloaded — nothing to delete");
            return;
        }
        // The tdrz build lives in the models folder root like any model;
        // only the pipeline components live under <models>/diarization.
        if kind == EntryKind::Diarize && file != diarize::TDRZ_FILE {
            let path = sherpa::dir(&self.library.dir).join(&file);
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            self.hub.delete_prompt = Some(DeletePrompt {
                name: file,
                path,
                size,
                yes_selected: false,
            });
            return;
        }
        if let Some(model) = self.library.models.iter().find(|m| m.name == file) {
            self.hub.delete_prompt = Some(DeletePrompt {
                name: model.name.clone(),
                path: model.path.clone(),
                size: model.size_bytes,
                // A destructive action defaults to "No".
                yes_selected: false,
            });
        }
    }

    fn hub_delete_prompt_key(&mut self, code: KeyCode) {
        let Some(prompt) = &mut self.hub.delete_prompt else {
            return;
        };
        match code {
            KeyCode::Esc | KeyCode::Char('n') => {
                self.hub.delete_prompt = None;
            }
            KeyCode::Left
            | KeyCode::Right
            | KeyCode::Tab
            | KeyCode::Char('h')
            | KeyCode::Char('l') => prompt.yes_selected = !prompt.yes_selected,
            KeyCode::Char('y') => {
                let prompt = self.hub.delete_prompt.take().unwrap();
                self.delete_model_file(prompt.path, prompt.name);
            }
            KeyCode::Enter => {
                let prompt = self.hub.delete_prompt.take().unwrap();
                if prompt.yes_selected {
                    self.delete_model_file(prompt.path, prompt.name);
                }
            }
            _ => {}
        }
    }

    /// Delete a model (file or directory), then rescan and keep the
    /// default selection valid.
    fn delete_model_file(&mut self, path: PathBuf, name: String) {
        let result = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match result {
            Ok(()) => {
                self.refresh_models();
                self.adopt_selection();
                // If the deleted model was the persisted default, move the
                // default on to whatever adopt_selection settled on.
                if self.config.default_model.as_deref() == Some(name.as_str()) {
                    self.config.default_model =
                        self.library.selected.as_ref().map(|m| m.name.clone());
                    let _ = config::save(&self.config);
                }
                self.hub.info = format!("Deleted {name} ✓");
                self.status = format!("Deleted {name} from {}", self.library.dir.display());
            }
            Err(e) => {
                self.hub.info = format!("Delete of {name} failed: {e}");
            }
        }
        let len = self.hub_visible_list().len();
        self.hub.selected = self.hub.selected.min(len.saturating_sub(1));
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
            HubEvent::Files {
                repo,
                files,
                variants,
            } => {
                if self.hub.listing_repo.as_deref() == Some(repo.as_str()) {
                    self.hub.listing_repo = None;
                    if files.is_empty() && variants.is_empty() {
                        self.hub.info =
                            format!("No GGML/GGUF files or MLX model folders in {repo}");
                    } else {
                        self.hub.files = Some(RepoView {
                            repo,
                            variants,
                            files,
                        });
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
                if let Some(d) = &mut self.hub.download {
                    if d.file == file {
                        d.got = got;
                        d.total = total;
                    }
                }
            }
            HubEvent::Paused { file, got } => {
                // Only honour the pause if we're still asking for it: a stale
                // event from a worker we've already resumed is ignored.
                let mut paused_now = false;
                if let Some(d) = &mut self.hub.download {
                    if d.file == file && d.pause.load(Ordering::Relaxed) {
                        d.paused = true;
                        d.got = got;
                        paused_now = true;
                    }
                }
                if paused_now {
                    self.hub.info = format!("Paused {file} — p resumes · Esc cancels");
                }
            }
            HubEvent::Done { file, path } => {
                self.hub.download = None;
                self.refresh_models();
                if let Some(model) = sherpa::CATALOG.iter().find(|m| m.local == file) {
                    // A finished diarization component becomes its role's
                    // active choice — it was downloaded to be used.
                    self.set_active_diarize_model(model.local);
                    self.hub.info =
                        format!("Downloaded {file} ✓ — now the active {}", model.role.label());
                    self.status = self.hub.info.clone();
                } else {
                    if models::is_tdrz(&file) {
                        // Same rule as the pipeline components: the tdrz
                        // build was downloaded to be used, so TinyDiarize
                        // becomes the active strategy.
                        self.config.diarize = DiarizeStrategy::Tdrz;
                        let _ = config::save(&self.config);
                        self.hub.info =
                            format!("Downloaded {file} ✓ — TinyDiarize is now the strategy");
                        self.status = self.hub.info.clone();
                    } else {
                        self.hub.info = format!("Downloaded {file} ✓");
                        self.status =
                            format!("Downloaded {file} to {}", self.library.dir.display());
                    }
                    if self.library.selected.is_none() {
                        self.library.selected =
                            self.library.models.iter().find(|m| m.path == path).cloned();
                    }
                }
            }
            HubEvent::Cancelled { file } => {
                self.hub.download = None;
                self.hub.info = format!("Cancelled {file}");
            }
            HubEvent::Failed { file, error } => {
                // A 401/403 means the repo is gated (or the token is bad):
                // open the consent page so the user can accept the terms,
                // and point at the settings row that stores a token.
                let gated_repo = self
                    .hub
                    .download
                    .as_ref()
                    .filter(|d| d.file == file)
                    .filter(|_| {
                        error.contains("status code 401") || error.contains("status code 403")
                    })
                    .map(|d| d.repo.clone());
                self.hub.download = None;
                if let Some(repo) = gated_repo {
                    hub::open_consent_page(&repo);
                    self.hub.info = format!(
                        "Download of {file} failed: access denied — accept the terms at \
                         hf.co/{repo} (opened in your browser), set your token in s → HF \
                         token, then retry"
                    );
                } else {
                    self.hub.info = format!("Download of {file} failed: {error}");
                }
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
