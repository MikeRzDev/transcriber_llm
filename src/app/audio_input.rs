//! Microphone selection stays separate from each recording session so the
//! same built-in, USB or Bluetooth input is used on the next recording.
use crossterm::event::KeyCode;

use super::App;

#[derive(Default)]
pub struct AudioInputState {
    /// None resolves the system default when recording starts.
    pub selected: Option<String>,
    pub open: bool,
    pub devices: Vec<String>,
    /// Row zero is System default, followed by the connected devices.
    pub cursor: usize,
    pub error: Option<String>,
}

impl AudioInputState {
    pub fn update_devices(&mut self, devices: Vec<String>) {
        self.devices = devices;
        self.cursor = self
            .selected
            .as_ref()
            .and_then(|name| self.devices.iter().position(|d| d == name))
            .map(|index| index + 1)
            .unwrap_or(0);
        self.error = None;
    }

    pub fn label(&self) -> &str {
        self.selected.as_deref().unwrap_or("System default")
    }
}

impl App {
    pub fn open_audio_inputs(&mut self) {
        if self.live.active {
            self.status = "Press R to stop recording before changing microphones".into();
            return;
        }
        self.audio_input.open = true;
        match crate::audio::capture::input_devices() {
            Ok(devices) => self.audio_input.update_devices(devices),
            Err(error) => {
                self.audio_input.devices.clear();
                self.audio_input.cursor = 0;
                self.audio_input.error = Some(format!("{error:#}"));
            }
        }
    }

    pub(crate) fn audio_input_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc | KeyCode::Char('a') => self.audio_input.open = false,
            KeyCode::Up | KeyCode::Char('k') => {
                self.audio_input.cursor = self.audio_input.cursor.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.audio_input.cursor =
                    (self.audio_input.cursor + 1).min(self.audio_input.devices.len())
            }
            KeyCode::Char('r') => self.open_audio_inputs(),
            KeyCode::Enter => {
                self.audio_input.selected = self
                    .audio_input
                    .cursor
                    .checked_sub(1)
                    .and_then(|index| self.audio_input.devices.get(index))
                    .cloned();
                self.audio_input.open = false;
                self.status = format!(
                    "Microphone: {} — press R to record",
                    self.audio_input.label()
                );
                self.job_log
                    .push(format!("microphone selected: {}", self.audio_input.label()));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refreshing_connected_devices_preserves_selection() {
        let mut input = AudioInputState {
            selected: Some("Bluetooth Headset".into()),
            ..Default::default()
        };
        input.update_devices(vec![
            "Built-in Microphone".into(),
            "Bluetooth Headset".into(),
        ]);
        assert_eq!(input.cursor, 2);
        input.update_devices(vec!["Built-in Microphone".into()]);
        assert_eq!(input.cursor, 0);
        // A disconnected selection stays explicit until the user changes it.
        assert_eq!(input.selected.as_deref(), Some("Bluetooth Headset"));
    }
}
