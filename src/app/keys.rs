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
        // The start confirmation outranks every other modal: a dropped
        // file can open it while e.g. the hub is up, and it renders on top
        if self.start_prompt.is_some() {
            self.start_prompt_key(code);
            return;
        }
        if self.naming.is_some() {
            self.naming_key(code);
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

    /// Keys for the "transcribe this file?" confirmation dialog.
    fn start_prompt_key(&mut self, code: KeyCode) {
        let Some(prompt) = &mut self.start_prompt else {
            return;
        };
        match code {
            KeyCode::Esc | KeyCode::Char('n') => {
                self.start_prompt = None;
                self.status = "Transcription cancelled".into();
            }
            KeyCode::Left
            | KeyCode::Right
            | KeyCode::Tab
            | KeyCode::Char('h')
            | KeyCode::Char('l') => prompt.yes_selected = !prompt.yes_selected,
            KeyCode::Char('y') => {
                let prompt = self.start_prompt.take().unwrap();
                self.start_transcription(prompt.audio);
            }
            KeyCode::Enter => {
                let prompt = self.start_prompt.take().unwrap();
                if prompt.yes_selected {
                    self.start_transcription(prompt.audio);
                } else {
                    self.status = "Transcription cancelled".into();
                }
            }
            _ => {}
        }
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
            KeyCode::Char('d') => self.cycle_diarize(),
            KeyCode::Char('n') => self.open_speaker_naming(),
            KeyCode::Char('l') => {
                // Right pane: transcript ⇄ job log
                self.show_log = !self.show_log;
            }
            KeyCode::Char('e') => self.export_log(),
            KeyCode::Char('x') => self.clear_log(),
            KeyCode::Char('c') => {
                if self.busy() && self.work != WorkState::UnloadingModel {
                    self.job_log.push("cancel requested");
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
                Focus::Transcript if self.show_log => self.job_log.scroll_up(1),
                Focus::Transcript => self.transcript.scroll_up(1),
            },
            KeyCode::Down | KeyCode::Char('j') => match self.focus {
                Focus::Files => self.browser.select_next(),
                Focus::Transcript if self.show_log => self.job_log.scroll_down(1),
                Focus::Transcript => self.transcript.scroll_down(1),
            },
            KeyCode::PageUp => {
                self.focus = Focus::Transcript;
                if self.show_log {
                    self.job_log.scroll_up(10);
                } else {
                    self.transcript.scroll_up(10);
                }
            }
            KeyCode::PageDown => {
                self.focus = Focus::Transcript;
                if self.show_log {
                    self.job_log.scroll_down(10);
                } else {
                    self.transcript.scroll_down(10);
                }
            }
            KeyCode::Char('g') => {
                if self.show_log {
                    self.job_log.scroll_top();
                } else {
                    self.transcript.scroll_top();
                }
            }
            KeyCode::Char('G') => {
                if self.show_log {
                    self.job_log.follow_tail();
                } else {
                    self.transcript.follow_tail();
                }
            }
            _ => {}
        }
    }
}
