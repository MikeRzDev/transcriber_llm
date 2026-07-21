//! Terminal lifecycle and the render/event loop.

use std::sync::mpsc::channel;
use std::time::Duration;

use anyhow::{bail, Result};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event as TermEvent, KeyCode, KeyEventKind,
    KeyModifiers,
};

use crate::app::{App, DropDetector};
use crate::cli::Args;
use crate::{transcribe, ui};

pub fn run(args: Args) -> Result<()> {
    // Without an explicit path, browse from $HOME (mac and Linux): a
    // Finder/.app launch gives a useless cwd like /, and home is where
    // media lives. An explicit path argument still wins.
    let default_dir = || {
        std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    };
    let (start_dir, auto_file) = match args.path {
        Some(p) if p.is_dir() => (p, None),
        Some(p) if p.is_file() => (
            p.parent().map(|d| d.to_path_buf()).unwrap_or_else(default_dir),
            Some(p),
        ),
        Some(p) => bail!("path not found: {}", p.display()),
        None => (default_dir(), None),
    };

    let (tx, rx) = channel();
    let transcriber = transcribe::spawn(tx);
    let mut app = App::new(start_dir, args.model, transcriber);
    if let Some(file) = auto_file {
        // Confirmed like any other start — the TUI never begins a job
        // without asking
        app.request_transcription(file);
    }

    let mut terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), EnableBracketedPaste);

    // Fallback for terminals without bracketed paste
    let mut drops = DropDetector::new();

    let result = (|| -> Result<()> {
        loop {
            app.stats.refresh_if_due();
            app.tick = app.tick.wrapping_add(1);
            let size = terminal.size()?;
            let screen = ratatui::layout::Rect::new(0, 0, size.width, size.height);
            ui::clamp_transcript(&mut app, screen);
            ui::clamp_log(&mut app, screen);
            terminal.draw(|frame| ui::draw(frame, &app))?;

            while let Ok(event) = rx.try_recv() {
                app.handle_event(event);
            }
            app.hub_pump();

            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    TermEvent::Key(key) if key.kind == KeyEventKind::Press => {
                        let mut consumed = false;
                        let typing_in_modal = app.settings.open || app.picker.open || app.hub.open;
                        if let KeyCode::Char(c) = key.code {
                            if !typing_in_modal && !key.modifiers.contains(KeyModifiers::CONTROL) {
                                consumed = drops.feed_char(c);
                            }
                        } else if key.code == KeyCode::Enter {
                            if let Some(text) = drops.feed_enter() {
                                app.handle_dropped_text(&text);
                                consumed = true;
                            }
                        }
                        if !consumed {
                            app.on_key(key.code, key.modifiers);
                        }
                    }
                    TermEvent::Paste(text) => {
                        app.handle_dropped_text(&text);
                    }
                    _ => {}
                }
            }

            if let Some(text) = drops.poll() {
                app.handle_dropped_text(&text);
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
