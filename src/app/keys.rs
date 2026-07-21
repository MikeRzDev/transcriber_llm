//! Key routing. One small router hands each key to whichever modal owns
//! focus, in strict priority order; the base-screen bindings live here.

use crossterm::event::{KeyCode, KeyModifiers};

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
        if self.move_prompt.is_some() {
            self.move_prompt_key(code);
            return;
        }
        if self.dir_picker.is_some() {
            self.dir_picker_key(code);
            return;
        }
        if self.model_picker_open {
            self.model_picker_key(code);
            return;
        }
        if self.settings_open {
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
                self.settings_selected = 0;
                self.settings_open = true;
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
            KeyCode::Char('r') => self.refresh_entries(),
            KeyCode::Enter => {
                if self.focus == Focus::Files {
                    self.enter_selected();
                }
            }
            KeyCode::Up | KeyCode::Char('k') => match self.focus {
                Focus::Files => self.file_selected = self.file_selected.saturating_sub(1),
                Focus::Transcript => {
                    self.follow = false;
                    self.transcript_scroll = self.transcript_scroll.saturating_sub(1);
                }
            },
            KeyCode::Down | KeyCode::Char('j') => match self.focus {
                Focus::Files => {
                    if !self.entries.is_empty() {
                        self.file_selected = (self.file_selected + 1).min(self.entries.len() - 1);
                    }
                }
                Focus::Transcript => {
                    self.follow = false;
                    self.transcript_scroll += 1;
                }
            },
            KeyCode::PageUp => {
                self.focus = Focus::Transcript;
                self.follow = false;
                self.transcript_scroll = self.transcript_scroll.saturating_sub(10);
            }
            KeyCode::PageDown => {
                self.focus = Focus::Transcript;
                self.follow = false;
                self.transcript_scroll += 10;
            }
            KeyCode::Char('g') => {
                self.follow = false;
                self.transcript_scroll = 0;
            }
            KeyCode::Char('G') => {
                self.follow = true;
            }
            _ => {}
        }
    }
}
