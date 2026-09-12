use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use super::layout::centered_rect_rows;
use super::theme::{highlight_style, ACCENT, DIM};
use crate::app::App;

pub(super) fn draw_audio_inputs(frame: &mut Frame, app: &App) {
    let input = &app.audio_input;
    let area = centered_rect_rows(75, (input.devices.len() + 10).min(22) as u16, frame.area());
    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Microphone input · built-in / USB / Bluetooth ")
        .border_style(Style::default().fg(ACCENT));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let regions = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(1),
        Constraint::Length(4),
    ])
    .split(inner);
    frame.render_widget(
        Paragraph::new(format!("Selected: {}", input.label())).wrap(Wrap { trim: true }),
        regions[0],
    );
    let mut items = vec![ListItem::new("System default (macOS Sound input)")];
    items.extend(
        input
            .devices
            .iter()
            .map(|name| ListItem::new(name.as_str())),
    );
    let mut state = ListState::default().with_selected(Some(input.cursor));
    frame.render_stateful_widget(
        List::new(items)
            .highlight_style(highlight_style())
            .highlight_symbol("› "),
        regions[1],
        &mut state,
    );
    let note = input.error.as_deref().unwrap_or(if input.devices.is_empty() {
        "No microphones detected. Connect your microphone or Bluetooth headset, then press r to refresh."
    } else {
        "Connect Bluetooth microphones in macOS first. Press r to refresh this list."
    });
    frame.render_widget(
        Paragraph::new(format!(
            "{note}\nEnter selects · Esc closes · R records after closing"
        ))
        .style(Style::default().fg(if input.error.is_some() {
            Color::Yellow
        } else {
            DIM
        }))
        .wrap(Wrap { trim: true }),
        regions[2],
    );
}
