use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Frame;

use crate::app::{scroll_window, App, FileEntry, Focus};
use crate::ui::layout::truncate_left;
use crate::ui::theme::{ACCENT, DIM};

pub(super) fn draw_files(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Files;
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };

    // Virtualized rows: only the entries inside the viewport become
    // widgets, so a directory with thousands of files costs the same
    // per frame as one with ten.
    let len = app.browser.entries.len();
    let viewport = area.height.saturating_sub(2) as usize; // minus borders
    let offset = scroll_window(app.browser.scroll.get(), app.browser.selected, len, viewport);
    app.browser.scroll.set(offset);
    let visible_end = (offset + viewport).min(len);

    let items: Vec<ListItem> = app.browser.entries[offset..visible_end]
        .iter()
        .map(|e| {
            let style = match e {
                FileEntry::Parent | FileEntry::Dir(_) => Style::default().fg(Color::Blue),
                FileEntry::Media(_) => Style::default(),
            };
            ListItem::new(e.label()).style(style)
        })
        .collect();

    let position = if len > viewport {
        format!(" {}/{} ", app.browser.selected + 1, len)
    } else {
        String::new()
    };
    let title = format!(
        " {} ",
        truncate_left(
            &app.browser.cwd.display().to_string(),
            (area.width as usize)
                .saturating_sub(4 + position.chars().count())
                .max(4)
        )
    );
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border_style)
                .title(title)
                .title_bottom(position),
        )
        .highlight_style(
            Style::default()
                .bg(if focused { ACCENT } else { DIM })
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        );

    let mut state = ListState::default();
    if !app.browser.entries.is_empty() {
        // Selection is relative to the rendered window
        state.select(Some(app.browser.selected - offset));
    }
    frame.render_stateful_widget(list, area, &mut state);
}
