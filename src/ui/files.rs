use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Frame;

use crate::app::{App, FileEntry, Focus};
use crate::ui::layout::truncate_left;
use crate::ui::theme::{ACCENT, DIM};

pub(super) fn draw_files(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Files;
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };

    let items: Vec<ListItem> = app
        .browser
        .entries
        .iter()
        .map(|e| {
            let style = match e {
                FileEntry::Parent | FileEntry::Dir(_) => Style::default().fg(Color::Blue),
                FileEntry::Media(_) => Style::default(),
            };
            ListItem::new(e.label()).style(style)
        })
        .collect();

    let title = format!(
        " {} ",
        truncate_left(
            &app.browser.cwd.display().to_string(),
            (area.width as usize).saturating_sub(4).max(4)
        )
    );
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style)
                .title(title),
        )
        .highlight_style(
            Style::default()
                .bg(if focused { ACCENT } else { DIM })
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        );

    let mut state = ListState::default();
    if !app.browser.entries.is_empty() {
        state.select(Some(app.browser.selected));
    }
    frame.render_stateful_widget(list, area, &mut state);
}
