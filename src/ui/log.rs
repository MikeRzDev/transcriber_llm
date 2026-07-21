use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::app::{App, Focus};
use crate::ui::layout::areas;
use crate::ui::theme::{ACCENT, DIM};

/// Pin the log scroll to its content for this frame. Runs in the update
/// phase (before `draw`) so rendering itself never mutates state.
pub fn clamp_log(app: &mut App, area: Rect) {
    let log_area = areas(area).transcript;
    let viewport = log_area.height.saturating_sub(2) as usize;
    app.job_log.clamp(viewport);
}

/// The job log in the right pane (in place of the transcript). Rendered
/// virtualized: only the lines inside the viewport become widgets, so a
/// 10k-line log costs the same per frame as an empty one.
pub(super) fn draw_log(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Transcript;
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };

    let title = format!(
        " job log ({}{}) ",
        app.job_log.lines.len(),
        if app.job_log.follow { " · live" } else { "" }
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.job_log.lines.is_empty() {
        frame.render_widget(
            Paragraph::new("Nothing logged yet.\n\nLoad a file to start — every step lands here.")
                .style(Style::default().fg(DIM))
                .alignment(ratatui::layout::Alignment::Center),
            inner,
        );
        return;
    }

    let viewport = inner.height as usize;
    let len = app.job_log.lines.len();
    let offset = app.job_log.scroll.min(len.saturating_sub(viewport));
    let lines: Vec<Line> = app
        .job_log
        .lines
        .iter()
        .skip(offset)
        .take(viewport)
        .map(|entry| log_line(entry))
        .collect();
    frame.render_widget(Paragraph::new(lines), inner);
}

/// One log entry: dim timestamp, message colored by severity.
fn log_line(entry: &str) -> Line<'_> {
    let (stamp, msg) = match entry.split_once("] ") {
        Some((stamp, msg)) => (format!("{stamp}] "), msg),
        None => (String::new(), entry),
    };
    let msg_style = if msg.starts_with("ERROR") {
        Style::default().fg(Color::Red)
    } else if msg.starts_with("Done in") || msg.starts_with("exported:") {
        Style::default().fg(Color::Green)
    } else {
        Style::default()
    };
    Line::from(vec![
        Span::styled(stamp, Style::default().fg(DIM)),
        Span::styled(msg, msg_style),
    ])
}
