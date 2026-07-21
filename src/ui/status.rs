use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Gauge, Paragraph};
use ratatui::Frame;

use crate::app::{App, WorkState};
use crate::ui::theme::{spinner_frame, ACCENT, DIM};

pub(super) fn draw_status(frame: &mut Frame, area: Rect, app: &App) {
    match &app.work {
        WorkState::Transcribing { progress } | WorkState::LoadingModel { progress, .. } => {
            let loading = matches!(app.work, WorkState::LoadingModel { .. });
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(30), Constraint::Min(10)])
                .split(area);
            let color = if loading { Color::Yellow } else { ACCENT };
            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(color).bg(Color::Black))
                .ratio((*progress as f64 / 100.0).clamp(0.0, 1.0))
                .label(format!("{progress}%"));
            frame.render_widget(gauge, chunks[0]);
            frame.render_widget(Paragraph::new(format!(" {}", app.status)), chunks[1]);
        }
        WorkState::UnloadingModel | WorkState::Decoding => {
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
    let keys: &[(&str, &str)] = if app.hub.open {
        &[
            ("type", "search"),
            ("↑↓", "select"),
            ("Enter", "open / download"),
            ("Esc", "back / cancel / close"),
        ]
    } else if app.settings.move_prompt.is_some() {
        &[
            ("←→", "choose"),
            ("Enter", "confirm"),
            ("y/n", "shortcuts"),
            ("Esc", "leave them"),
        ]
    } else if app.settings.dir_picker.is_some() {
        &[
            ("↑↓", "select"),
            ("Enter", "open / choose folder"),
            ("Esc", "cancel"),
        ]
    } else if app.picker.open {
        &[
            ("↑↓", "select"),
            ("Enter", "set default model"),
            ("Esc", "close"),
        ]
    } else if app.settings.open {
        &[("↑↓", "select"), ("Enter", "change"), ("Esc", "close")]
    } else {
        &[
            ("Tab", "pane"),
            ("Enter", "transcribe"),
            ("drop", "file → transcribe"),
            ("m", "models"),
            ("s", "settings"),
            ("d", "diarize"),
            ("e", "export"),
            ("c", "cancel"),
            ("q", "quit"),
        ]
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
