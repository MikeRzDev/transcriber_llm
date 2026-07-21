//! Rendering. `draw` composes the base screen and overlays whichever
//! modal is open; every widget lives in its own submodule and reads the
//! `App` state without mutating it.

mod dir_picker;
mod files;
mod header;
mod hub;
mod layout;
mod log;
mod model_picker;
mod move_prompt;
mod settings;
mod start_prompt;
mod status;
mod theme;
mod transcript;

pub use log::clamp_log;
pub use transcript::clamp_transcript;

use ratatui::Frame;

use crate::app::App;

pub fn draw(frame: &mut Frame, app: &App) {
    let key_rows = status::keys_rows(app, frame.area().width);
    let areas = layout::areas(frame.area(), key_rows);

    header::draw_header(frame, areas.header, app);
    files::draw_files(frame, areas.files, app);
    if app.show_log {
        log::draw_log(frame, areas.transcript, app);
    } else {
        transcript::draw_transcript(frame, areas.transcript, app);
    }
    status::draw_status(frame, areas.status, app);
    status::draw_keys(frame, areas.keys, app);

    if app.settings.open {
        settings::draw_settings(frame, app);
        if app.settings.formats_cursor.is_some() {
            settings::draw_export_formats(frame, app);
        }
    }
    if app.picker.open {
        model_picker::draw_model_picker(frame, app);
    }
    if app.hub.open {
        hub::draw_hub(frame, app);
        if app.hub.delete_prompt.is_some() {
            hub::draw_hub_delete_prompt(frame, app);
        }
    }
    if app.settings.dir_picker.is_some() {
        dir_picker::draw_dir_picker(frame, app);
    }
    if app.settings.move_prompt.is_some() {
        move_prompt::draw_move_prompt(frame, app);
    }
    // Topmost: it can open over any other modal (e.g. a file dropped
    // while the hub is up) and its keys take priority
    if app.start_prompt.is_some() {
        start_prompt::draw_start_prompt(frame, app);
    }
}
