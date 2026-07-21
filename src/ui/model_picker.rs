use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::App;
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, ACCENT, DIM};

pub(super) fn draw_model_picker(frame: &mut Frame, app: &App) {
    let area = centered_rect(60, 40, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" models in {} ", app.library.dir.display()));

    if app.library.models.is_empty() {
        frame.render_widget(
            Paragraph::new("No .bin or .gguf models found.\n\nDownload one in Settings → Model management (s).")
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
            ListItem::new(Line::from(vec![
                Span::styled(marker, Style::default().fg(ACCENT)),
                Span::raw(m.name.clone()),
                Span::styled(format!("  {}", m.size_human()), Style::default().fg(DIM)),
            ]))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(highlight_style());

    let mut state = ListState::default();
    state.select(Some(app.picker.selected));
    frame.render_stateful_widget(list, area, &mut state);
}
