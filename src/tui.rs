//! Terminal lifecycle and the render/event loop.

use std::sync::mpsc::channel;
use std::time::Duration;

use anyhow::{bail, Result};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event as TermEvent, KeyCode, KeyEventKind,
    KeyModifiers,
};

use crate::app::{self, App, DirTarget, Focus};
use crate::cli::Args;
use crate::{transcribe, ui};

pub fn run(args: Args) -> Result<()> {
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
