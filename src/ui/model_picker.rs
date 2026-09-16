use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::App;
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, ACCENT, DIM};

pub(super) fn draw_model_picker(frame: &mut Frame, app: &App) {
    let area = centered_rect(85, 75, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" models in {} ", app.library.dir.display()));

    if app.library.models.is_empty() {
        frame.render_widget(
            Paragraph::new("No GGML/GGUF or MLX models found.\n\nSet the models folder or download one in Settings (s).")
                .block(block)
                .style(Style::default().fg(Color::Yellow))
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    let items: Vec<ListItem> = app
        .library
        .models
        .iter()
        .map(|m| {
            let selected_marker = app
                .library
                .selected
                .as_ref()
                .map(|s| s.path == m.path)
                .unwrap_or(false);
            let marker = if selected_marker { "● " } else { "  " };
            let mut spans = vec![
                Span::styled(marker, Style::default().fg(ACCENT)),
                Span::raw(m.display_name()),
                Span::styled(format!("  {}", m.size_human()), Style::default().fg(DIM)),
            ];
            if !crate::hw::fits(m.size_bytes) {
                spans.push(Span::styled(
                    "  ⚠ exceeds this Mac's memory",
                    Style::default().fg(Color::Yellow),
                ));
            }
            let (live_label, color) = if app
                .picker
                .live_notes
                .get(&m.path)
                .is_some_and(|reason| reason.is_some())
            {
                ("Live: unavailable".into(), DIM)
            } else if let Some(fit) = app.picker.live_fit.get(&m.path) {
                use crate::live_benchmark::LiveFit;
                let color = match fit {
                    LiveFit::Good { .. } => Color::Green,
                    LiveFit::Borderline { .. } => Color::Yellow,
                    LiveFit::TooSlow { .. } => Color::Red,
                };
                (fit.label(), color)
            } else {
                ("Live: not benchmarked on this Mac".into(), DIM)
            };
            ListItem::new(vec![
                Line::from(spans),
                Line::from(Span::styled(
                    format!("  {live_label}"),
                    Style::default().fg(color),
                )),
            ])
        })
        .collect();

    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(3)]).split(inner);
    let list = List::new(items).highlight_style(highlight_style());

    let mut state = ListState::default();
    state.select(Some(app.picker.selected));
    frame.render_stateful_widget(list, rows[0], &mut state);
    frame.render_widget(
        Paragraph::new("Measured on this Mac · below 1s/audio s keeps up.\nLive ratings only; every model remains selectable.\n↑↓ select · Enter use · Esc close")
            .style(Style::default().fg(DIM))
            .wrap(Wrap { trim: true }),
        rows[1],
    );
}
