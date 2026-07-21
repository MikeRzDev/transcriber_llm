use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Gauge, Paragraph};
use ratatui::Frame;

use crate::app::{App, WorkState};
use crate::ui::theme::{spinner_frame, ACCENT, DIM};

pub(super) fn draw_status(frame: &mut Frame, area: Rect, app: &App) {
    match &app.work {
        // A negative progress means the engine reports none (MLX) or the
        // media duration is unknown (ffmpeg extraction) — fall through to
        // the indeterminate spinner below.
        WorkState::Transcribing { progress }
        | WorkState::LoadingModel { progress, .. }
        | WorkState::Decoding { progress }
            if *progress >= 0 =>
        {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(30), Constraint::Min(10)])
                .split(area);
            let color = match app.work {
                WorkState::LoadingModel { .. } => Color::Yellow,
                WorkState::Decoding { .. } => Color::Cyan,
                _ => ACCENT,
            };
            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(color).bg(Color::Black))
                .ratio((*progress as f64 / 100.0).clamp(0.0, 1.0))
                .label(format!("{progress}%"));
            frame.render_widget(gauge, chunks[0]);
            frame.render_widget(Paragraph::new(format!(" {}", app.status)), chunks[1]);
        }
        WorkState::UnloadingModel
        | WorkState::Decoding { .. }
        | WorkState::Transcribing { .. }
        | WorkState::LoadingModel { .. } => {
            frame.render_widget(
                Paragraph::new(format!("{} {}", spinner_frame(app.tick), app.status))
                    .style(Style::default().fg(Color::Yellow)),
                area,
            );
        }
        WorkState::Idle => {
            let style = if app.status.starts_with("Error") {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Green)
            };
            frame.render_widget(Paragraph::new(app.status.as_str()).style(style), area);
        }
    }
}

pub(super) fn draw_keys(frame: &mut Frame, area: Rect, app: &App) {
    let keys: Vec<(&str, &str)> = if app.start_prompt.is_some() {
        vec![
            ("←→", "choose"),
            ("Enter", "confirm"),
            ("y/n", "shortcuts"),
            ("Esc", "cancel"),
        ]
    } else if app.hub.open {
        vec![
            ("type", "search"),
            ("↑↓", "select"),
            ("Enter", "open / download"),
            ("Esc", "back / cancel / close"),
        ]
    } else if app.settings.move_prompt.is_some() {
        vec![
            ("←→", "choose"),
            ("Enter", "confirm"),
            ("y/n", "shortcuts"),
            ("Esc", "leave them"),
        ]
    } else if app.settings.dir_picker.is_some() {
        vec![
            ("↑↓", "select"),
            ("Enter", "open / choose folder"),
            ("Esc", "cancel"),
        ]
    } else if app.picker.open {
        vec![
            ("↑↓", "select"),
            ("Enter", "set default model"),
            ("Esc", "close"),
        ]
    } else if app.settings.formats_cursor.is_some() {
        vec![
            ("↑↓", "select"),
            ("Space", "toggle format"),
            ("Esc", "done"),
        ]
    } else if app.settings.open {
        vec![("↑↓", "select"), ("Enter", "change"), ("Esc", "close")]
    } else {
        // The cancel key only exists while there is a job to cancel
        let cancellable = app.busy() && app.work != WorkState::UnloadingModel;
        let mut keys: Vec<(&str, &str)> = Vec::new();
        if cancellable {
            keys.push(("c", "cancel job"));
        }
        keys.extend([
            ("Tab", "pane"),
            ("Enter", "transcribe"),
            ("drop", "file → transcribe"),
            ("l", if app.show_log { "transcript" } else { "log" }),
            ("m", "models"),
            ("s", "settings"),
            ("d", "diarize"),
            ("q", "quit"),
        ]);
        keys
    };
    let mut spans: Vec<Span> = Vec::new();
    for (key, desc) in keys {
        spans.push(Span::styled(
            format!(" {key} "),
            Style::default().fg(Color::Black).bg(DIM),
        ));
        spans.push(Span::styled(format!(" {desc}  "), Style::default().fg(DIM)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}
