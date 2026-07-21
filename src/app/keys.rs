//! Key routing. One small router hands each key to whichever modal owns
//! focus, in strict priority order; the base-screen bindings live here.

use crossterm::event::{KeyCode, KeyModifiers};

use crate::app::settings::SettingsRow;
use crate::app::{App, Focus, WorkState};

impl App {
    pub fn on_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }
        if self.hub.open {
            if !modifiers.contains(KeyModifiers::CONTROL) {
                self.hub_key(code);
            }
            return;
        }
        if self.settings.move_prompt.is_some() {
            self.move_prompt_key(code);
            return;
        }
        if self.settings.dir_picker.is_some() {
            self.dir_picker_key(code);
            return;
        }
        if self.picker.open {
            self.model_picker_key(code);
            return;
        }
        if self.settings.open {
            self.settings_key(code);
            return;
        }
        self.main_key(code);
    }

    fn main_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Files => Focus::Transcript,
                    Focus::Transcript => Focus::Files,
                };
            }
            KeyCode::Char('m') => self.open_model_picker(),
            KeyCode::Char('s') => {
                self.settings.selected = SettingsRow::DefaultModel;
                self.settings.open = true;
            }
            KeyCode::Char('d') => self.toggle_diarize(),
            KeyCode::Char('e') => {
                self.status = match self.export() {
                    Ok(folder) => {
                        format!("Exported llm.md + json/txt/srt to {}", folder.display())
                    }
                    Err(e) => format!("Export failed: {e}"),
                };
            }
            KeyCode::Char('c') => {
                if self.busy() && self.work != WorkState::UnloadingModel {
                    self.transcriber.request_cancel();
                    self.status = "Cancelling…".into();
                }
            }
            KeyCode::Char('r') => self.browser.refresh(),
            KeyCode::Enter => {
                if self.focus == Focus::Files {
                    self.enter_selected();
                }
            }
            KeyCode::Up | KeyCode::Char('k') => match self.focus {
                Focus::Files => self.browser.select_prev(),
                Focus::Transcript => self.transcript.scroll_up(1),
            },
            KeyCode::Down | KeyCode::Char('j') => match self.focus {
                Focus::Files => self.browser.select_next(),
                Focus::Transcript => self.transcript.scroll_down(1),
            },
            KeyCode::PageUp => {
                self.focus = Focus::Transcript;
                self.transcript.scroll_up(10);
            }
            KeyCode::PageDown => {
                self.focus = Focus::Transcript;
                self.transcript.scroll_down(10);
            }
            KeyCode::Char('g') => self.transcript.scroll_top(),
            KeyCode::Char('G') => self.transcript.follow_tail(),
            _ => {}
        }
    }
}
