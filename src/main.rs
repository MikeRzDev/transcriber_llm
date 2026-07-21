mod app;
mod audio;
mod config;
mod export;
mod hub;
mod models;
mod split;
mod stats;
mod transcribe;
mod ui;

use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::Duration;

use anyhow::{bail, Result};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event as TermEvent, KeyCode, KeyEventKind,
    KeyModifiers,
};

use app::{App, DirTarget, Focus};
use transcribe::{Event, Job};

struct Args {
    path: Option<PathBuf>,
    model: Option<PathBuf>,
    headless: bool,
    diarize: bool,
    language: Option<String>,
    download_test_model: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        path: None,
        model: None,
        headless: false,
        diarize: false,
        language: None,
        download_test_model: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                args.model = Some(PathBuf::from(
                    iter.next()
                        .ok_or_else(|| anyhow::anyhow!("--model needs a path"))?,
                ));
            }
            "--headless" => args.headless = true,
            "--diarize" => args.diarize = true,
            "--download-test-model" => args.download_test_model = true,
            "-l" | "--language" => {
                args.language = Some(
                    iter.next()
                        .ok_or_else(|| anyhow::anyhow!("--language needs a code, e.g. en"))?,
                );
            }
            "-h" | "--help" => {
                println!(
                    "transcribe-stt — terminal speech-to-text client for whisper.cpp models (Metal)\n\n\
                     usage: transcribe-stt [OPTIONS] [PATH]\n\n\
                     PATH                 directory to browse, or audio file to transcribe\n\
                     -m, --model <FILE>   model to use (.bin / .gguf); default: auto-detect in the models folder\n\
                     --headless           transcribe PATH without the TUI, print segments to stdout\n\
                     --diarize            label speaker turns (requires a tdrz model, English)\n\
                     -l, --language <CODE> language hint (ISO 639-1, e.g. en, es); default: auto-detect\n\
                     --download-test-model fetch the smallest whisper model (ggml-tiny, ~75 MB)\n\
                     \x20                    into the models folder for quick testing"
                );
                std::process::exit(0);
            }
            other if !other.starts_with('-') => args.path = Some(PathBuf::from(other)),
            other => bail!("unknown flag: {other}"),
        }
    }
    Ok(args)
}

fn default_model(diarize: bool) -> Option<PathBuf> {
    let cfg = config::load();
    let found = models::scan_models(&config::resolve_models_dir(&cfg));
    if diarize {
        return found
            .iter()
            .find(|m| m.name.contains("tdrz"))
            .map(|m| m.path.clone());
    }
    cfg.default_model
        .as_ref()
        .and_then(|name| found.iter().find(|m| &m.name == name))
        .or_else(|| found.iter().find(|m| m.name.contains("large-v3")))
        .or_else(|| found.first())
        .map(|m| m.path.clone())
}

fn main() {
    let code = match real_main() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("Error: {e:#}");
            1
        }
    };
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // ggml registers Metal teardown in atexit handlers that can hang or
    // assert on macOS. Everything that needs cleanup (terminal state,
    // worker thread, whisper context) is already shut down explicitly,
    // so skip the C++ static destructors entirely.
    unsafe { libc::_exit(code) }
}

fn real_main() -> Result<()> {
    let args = parse_args()?;

    if args.download_test_model {
        return download_test_model();
    }

    if args.headless {
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
        return run_headless(
            model,
            audio,
            args.diarize,
            args.language.clone(),
            config::load().split_mode,
        );
    }

    run_tui(args)
}

/// Fetch the smallest whisper model into the models folder — enough for a
/// quick end-to-end smoke test without committing to a multi-GB download.
fn download_test_model() -> Result<()> {
    const REPO: &str = "ggerganov/whisper.cpp";
    const FILE: &str = "ggml-tiny.bin";

    let dir = config::resolve_models_dir(&config::load());
    let dest = dir.join(FILE);
    if dest.exists() {
        eprintln!("already there: {}", dest.display());
        return Ok(());
    }
    eprintln!("downloading {FILE} (~75 MB, smallest whisper model) to {}", dir.display());

    let (tx, rx) = channel();
    hub::download(
        REPO.into(),
        FILE.into(),
        dir,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        tx,
    );
    for event in rx {
        match event {
            hub::HubEvent::Progress { got, total, .. } => {
                if total > 0 {
                    eprint!(
                        "\r{:>3}%  {} / {}   ",
                        got * 100 / total,
                        models::human_size(got),
                        models::human_size(total)
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
            Event::AudioInfo { duration_secs } => eprintln!("audio: {duration_secs:.1}s"),
            Event::Progress(_) => {}
            Event::Segment(seg) => {
                println!(
                    "[{} → {}] {}",
                    export::clock_time(seg.start_ms),
                    export::clock_time(seg.end_ms),
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
                        export::clock_time(seg.start_ms),
                        export::clock_time(seg.end_ms),
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

fn run_tui(args: Args) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let (start_dir, auto_file) = match args.path {
        Some(p) if p.is_dir() => (p, None),
        Some(p) if p.is_file() => (p.parent().map(|d| d.to_path_buf()).unwrap_or(cwd), Some(p)),
        Some(p) => bail!("path not found: {}", p.display()),
        None => (cwd, None),
    };

    let (tx, rx) = channel();
    let transcriber = transcribe::spawn(tx);
    let mut app = App::new(start_dir, args.model, transcriber);
    if let Some(file) = auto_file {
        app.start_transcription(file);
    }

    let mut terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), EnableBracketedPaste);

    // Fallback for terminals without bracketed paste: a dropped file
    // arrives as a rapid burst of key events starting with '/' or '~'.
    let mut drop_buf = String::new();
    let mut last_drop_char = std::time::Instant::now();

    let result = (|| -> Result<()> {
        loop {
            app.stats.refresh_if_due();
            app.tick = app.tick.wrapping_add(1);
            terminal.draw(|frame| ui::draw(frame, &mut app))?;

            while let Ok(event) = rx.try_recv() {
                app.handle_event(event);
            }
            app.hub_pump();

            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    TermEvent::Key(key) if key.kind == KeyEventKind::Press => {
                        let mut consumed = false;
                        let typing_in_settings =
                            app.settings_open || app.model_picker_open || app.hub.open;
                        if let KeyCode::Char(c) = key.code {
                            let burst_continues =
                                !drop_buf.is_empty() && last_drop_char.elapsed().as_millis() < 150;
                            let burst_starts = drop_buf.is_empty() && (c == '/' || c == '~');
                            if !typing_in_settings
                                && !key.modifiers.contains(KeyModifiers::CONTROL)
                                && (burst_continues || burst_starts)
                            {
                                drop_buf.push(c);
                                last_drop_char = std::time::Instant::now();
                                consumed = true;
                            }
                        } else if key.code == KeyCode::Enter
                            && !drop_buf.is_empty()
                            && last_drop_char.elapsed().as_millis() < 150
                        {
                            // newline terminating a dropped path
                            let text = std::mem::take(&mut drop_buf);
                            app.handle_dropped_text(&text);
                            consumed = true;
                        }
                        if !consumed {
                            handle_key(&mut app, key.code, key.modifiers);
                        }
                    }
                    TermEvent::Paste(text) => {
                        app.handle_dropped_text(&text);
                    }
                    _ => {}
                }
            }

            // A quiet period ends the char-burst: treat it as a drop
            if !drop_buf.is_empty() && last_drop_char.elapsed().as_millis() > 250 {
                let text = std::mem::take(&mut drop_buf);
                if text.len() > 2 {
                    app.handle_dropped_text(&text);
                }
            }

            if app.should_quit {
                return Ok(());
            }
        }
    })();
    let _ = crossterm::execute!(std::io::stdout(), DisableBracketedPaste);
    ratatui::restore();
    app.transcriber.shutdown();
    result
}

fn handle_key(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
    if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
        app.should_quit = true;
        return;
    }

    if app.hub.open {
        if !modifiers.contains(KeyModifiers::CONTROL) {
            app.hub_key(code);
        }
        return;
    }

    if app.move_prompt.is_some() {
        app.move_prompt_key(code);
        return;
    }

    if app.dir_picker.is_some() {
        app.dir_picker_key(code);
        return;
    }

    if app.model_picker_open {
        match code {
            KeyCode::Esc | KeyCode::Char('m') | KeyCode::Char('q') => {
                app.model_picker_open = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.model_picker_selected = app.model_picker_selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if !app.models.is_empty() {
                    app.model_picker_selected =
                        (app.model_picker_selected + 1).min(app.models.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(model) = app.models.get(app.model_picker_selected).cloned() {
                    app.choose_model(model);
                }
                app.model_picker_open = false;
            }
            _ => {}
        }
        return;
    }

    if app.settings_open {
        if let Some(input) = &mut app.language_input {
            match code {
                KeyCode::Char(c) => input.push(c),
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Enter => {
                    let text = app.language_input.take().unwrap_or_default();
                    app.set_language(&text);
                }
                KeyCode::Esc => app.language_input = None,
                _ => {}
            }
            return;
        }
        match code {
            KeyCode::Esc | KeyCode::Char('s') | KeyCode::Char('q') => {
                app.settings_open = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                app.settings_selected = app.settings_selected.checked_sub(1).unwrap_or(6);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                app.settings_selected = (app.settings_selected + 1) % 7;
            }
            KeyCode::Enter => match app.settings_selected {
                0 => {
                    app.refresh_models();
                    app.model_picker_selected = app
                        .selected_model
                        .as_ref()
                        .and_then(|sel| app.models.iter().position(|m| m.path == sel.path))
                        .unwrap_or(0);
                    app.model_picker_open = true;
                }
                1 => app.open_dir_picker(DirTarget::Models),
                2 => app.open_dir_picker(DirTarget::Output),
                3 => {
                    app.settings_open = false;
                    app.open_hub();
                }
                4 => app.toggle_diarize(),
                5 => app.cycle_split_mode(),
                _ => app.language_input = Some(app.config.language.clone().unwrap_or_default()),
            },
            _ => {}
        }
        return;
    }

    match code {
        KeyCode::Char('q') => app.should_quit = true,
        KeyCode::Tab => {
            app.focus = match app.focus {
                Focus::Files => Focus::Transcript,
                Focus::Transcript => Focus::Files,
            };
        }
        KeyCode::Char('m') => {
            app.refresh_models();
            app.model_picker_selected = app
                .selected_model
                .as_ref()
                .and_then(|sel| app.models.iter().position(|m| m.path == sel.path))
                .unwrap_or(0);
            app.model_picker_open = true;
        }
        KeyCode::Char('s') => {
            app.settings_selected = 0;
            app.settings_open = true;
        }
        KeyCode::Char('d') => app.toggle_diarize(),
        KeyCode::Char('e') => {
            app.status = match app.export() {
                Ok(folder) => format!("Exported llm.md + json/txt/srt to {}", folder.display()),
                Err(e) => format!("Export failed: {e}"),
            };
        }
        KeyCode::Char('c') => {
            if app.busy() && app.work != app::WorkState::UnloadingModel {
                app.transcriber.request_cancel();
                app.status = "Cancelling…".into();
            }
        }
        KeyCode::Char('r') => app.refresh_entries(),
        KeyCode::Enter => {
            if app.focus == Focus::Files {
                app.enter_selected();
            }
        }
        KeyCode::Up | KeyCode::Char('k') => match app.focus {
            Focus::Files => app.file_selected = app.file_selected.saturating_sub(1),
            Focus::Transcript => {
                app.follow = false;
                app.transcript_scroll = app.transcript_scroll.saturating_sub(1);
            }
        },
        KeyCode::Down | KeyCode::Char('j') => match app.focus {
            Focus::Files => {
                if !app.entries.is_empty() {
                    app.file_selected = (app.file_selected + 1).min(app.entries.len() - 1);
                }
            }
            Focus::Transcript => {
                app.follow = false;
                app.transcript_scroll += 1;
            }
        },
        KeyCode::PageUp => {
            app.focus = Focus::Transcript;
            app.follow = false;
            app.transcript_scroll = app.transcript_scroll.saturating_sub(10);
        }
        KeyCode::PageDown => {
            app.focus = Focus::Transcript;
            app.follow = false;
            app.transcript_scroll += 10;
        }
        KeyCode::Char('g') => {
            app.follow = false;
            app.transcript_scroll = 0;
        }
        KeyCode::Char('G') => {
            app.follow = true;
        }
        _ => {}
    }
}
