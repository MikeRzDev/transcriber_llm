use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};
use ratatui::Frame;

use crate::app::{App, DirRow};
use crate::ui::layout::{centered_rect, truncate_left};
use crate::ui::theme::{highlight_style, ACCENT, DIM};

pub(super) fn draw_dir_picker(frame: &mut Frame, app: &App) {
    let Some(picker) = &app.settings.dir_picker else {
        return;
    };
    let area = centered_rect(70, 60, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(format!(" choose {} folder ", picker.target.label()));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // current path
            Constraint::Min(3),    // rows
            Constraint::Length(1), // hint
        ])
        .split(inner);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" in: ", Style::default().fg(DIM)),
            Span::raw(truncate_left(
                &picker.cwd.display().to_string(),
                (rows[0].width as usize).saturating_sub(6).max(6),
            )),
        ])),
        rows[0],
    );

    let items: Vec<ListItem> = picker
        .rows()
        .into_iter()
        .map(|row| match row {
            DirRow::UseThis => ListItem::new(Line::from(Span::styled(
                " ✓ use this folder",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ))),
            DirRow::Parent => ListItem::new(Line::from(Span::styled(
                " ../",
                Style::default().fg(Color::Blue),
            ))),
            DirRow::Sub(path) => ListItem::new(Line::from(Span::styled(
                format!(
                    " {}/",
                    path.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.display().to_string())
                ),
                Style::default().fg(Color::Blue),
            ))),
        })
        .collect();

    let count = items.len();
    let list = List::new(items).highlight_style(highlight_style());
    let mut state = ListState::default();
    if count > 0 {
        state.select(Some(picker.selected.min(count - 1)));
    }
    frame.render_stateful_widget(list, rows[1], &mut state);

    frame.render_widget(
        Paragraph::new(" ↑↓ select · Enter open / choose · Esc cancel")
            .style(Style::default().fg(DIM)),
        rows[2],
    );
}
