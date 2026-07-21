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

/// The key hints for the current UI state.
fn key_hints(app: &App) -> Vec<(&'static str, &'static str)> {
    if app.start_prompt.is_some() {
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
            ("e", "export log"),
            ("x", "clear log"),
            ("m", "models"),
            ("s", "settings"),
            ("d", "diarize"),
            ("q", "quit"),
        ]);
        keys
    }
}

/// Rendered width of one hint: ` key ` + ` desc  `.
fn hint_width(key: &str, desc: &str) -> usize {
    key.chars().count() + desc.chars().count() + 5
}

/// Greedy-wrap the hints into rows that fit `width` columns. Always at
/// least one row; a hint wider than the window gets a row to itself.
fn wrap_hints<'a>(
    hints: &[(&'a str, &'a str)],
    width: usize,
) -> Vec<Vec<(&'a str, &'a str)>> {
    let mut rows: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
    let mut used = 0;
    for &(key, desc) in hints {
        let w = hint_width(key, desc);
        if used > 0 && used + w > width {
            rows.push(Vec::new());
            used = 0;
        }
        rows.last_mut().unwrap().push((key, desc));
        used += w;
    }
    rows
}

/// How many rows the key bar needs at this width — the layout reserves
/// exactly this many lines, so nothing is ever clipped.
pub(super) fn keys_rows(app: &App, width: u16) -> u16 {
    wrap_hints(&key_hints(app), width.max(1) as usize).len() as u16
}

pub(super) fn draw_keys(frame: &mut Frame, area: Rect, app: &App) {
    let hints = key_hints(app);
    let lines: Vec<Line> = wrap_hints(&hints, area.width.max(1) as usize)
        .into_iter()
        .map(|row| {
            let mut spans: Vec<Span> = Vec::new();
            for (key, desc) in row {
                spans.push(Span::styled(
                    format!(" {key} "),
                    Style::default().fg(Color::Black).bg(DIM),
                ));
                spans.push(Span::styled(format!(" {desc}  "), Style::default().fg(DIM)));
            }
            Line::from(spans)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
}

#[cfg(test)]
mod tests {
    use super::{hint_width, wrap_hints};

    #[test]
    fn hints_wrap_to_fit_and_never_drop() {
        let hints = [
            ("Tab", "pane"),
            ("Enter", "transcribe"),
            ("drop", "file → transcribe"),
            ("l", "log"),
            ("e", "export log"),
            ("x", "clear log"),
            ("m", "models"),
            ("s", "settings"),
            ("d", "diarize"),
            ("q", "quit"),
        ];
        // Wide window: everything on one row
        let total: usize = hints.iter().map(|(k, d)| hint_width(k, d)).sum();
        assert_eq!(wrap_hints(&hints, total).len(), 1);

        // Narrow window: multiple rows, each within width, nothing lost
        let rows = wrap_hints(&hints, 40);
        assert!(rows.len() > 1);
        let kept: usize = rows.iter().map(|r| r.len()).sum();
        assert_eq!(kept, hints.len());
        for row in &rows {
            let w: usize = row.iter().map(|(k, d)| hint_width(k, d)).sum();
            assert!(w <= 40, "row too wide: {w}");
        }

        // Degenerate width: one hint per row, still nothing lost
        let rows = wrap_hints(&hints, 1);
        assert_eq!(rows.iter().map(|r| r.len()).sum::<usize>(), hints.len());
    }
}
