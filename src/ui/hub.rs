//! The Model management modal: search bar, backend line, whichever of
//! the three lists is visible, and the download gauge / info line.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph};
use ratatui::Frame;

use crate::app::{App, HubList};
use crate::format::{fmt_count, human_size};
use crate::hub;
use crate::ui::layout::centered_rect;
use crate::ui::theme::{highlight_style, spinner_frame, ACCENT, DIM};

pub(super) fn draw_hub(frame: &mut Frame, app: &App) {
    let area = centered_rect(80, 70, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" model management — Hugging Face ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // search bar / breadcrumb
            Constraint::Length(1), // backend + destination
            Constraint::Length(1), // list header
            Constraint::Min(3),    // list
            Constraint::Length(1), // download gauge or info line
        ])
        .split(inner);

    draw_search_bar(frame, rows[0], app);
    draw_backend_line(frame, rows[1], app);

    let visible = app.hub.visible_list();
    let (header, items) = list_content(app, &visible);
    frame.render_widget(
        Paragraph::new(Span::styled(header, Style::default().fg(DIM))),
        rows[2],
    );
    let list = List::new(items).highlight_style(highlight_style());
    let mut state = ListState::default();
    if !visible.is_empty() {
        state.select(Some(app.hub.selected.min(visible.len() - 1)));
    }
    frame.render_stateful_widget(list, rows[3], &mut state);

    draw_bottom_line(frame, rows[4], app);
}

/// Search bar, or breadcrumb when inside a repo's file list.
fn draw_search_bar(frame: &mut Frame, area: Rect, app: &App) {
    if let Some((repo, _)) = &app.hub.files {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" repo: ", Style::default().fg(DIM)),
                Span::styled(repo.clone(), Style::default().add_modifier(Modifier::BOLD)),
            ])),
            area,
        );
        return;
    }
    let mut spans = vec![
        Span::styled(" search: ", Style::default().fg(DIM)),
        Span::styled(
            format!("{}▏", app.hub.input),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    if app.hub.searching || app.hub.listing_repo.is_some() {
        spans.push(Span::styled(
            format!("  {}", spinner_frame(app.tick)),
            Style::default().fg(Color::Yellow),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_backend_line(frame: &mut Frame, area: Rect, app: &App) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(
                " engine: whisper.cpp · {}  ·  downloads to {}",
                hub::backend_label(),
                app.library.dir.display()
            ),
            Style::default().fg(DIM),
        ))),
        area,
    );
}

/// Header text + rows for whichever list is showing.
fn list_content<'a>(app: &'a App, visible: &HubList<'a>) -> (String, Vec<ListItem<'a>>) {
    let metal_tag = if hub::metal_available() {
        "Metal ✓"
    } else {
        "CPU"
    };
    match visible {
        HubList::Files(files) => (
            " GGML/GGUF files — Enter downloads".into(),
            files
                .iter()
                .map(|f| {
                    let size = if f.size_bytes > 0 {
                        human_size(f.size_bytes)
                    } else {
                        "?".into()
                    };
                    ListItem::new(Line::from(vec![
                        Span::raw(format!(" {:<44}", f.name)),
                        Span::styled(format!("{size:>9}  "), Style::default().fg(DIM)),
                        Span::styled(metal_tag, Style::default().fg(Color::Green)),
                    ]))
                })
                .collect(),
        ),
        HubList::Results(results) => (
            format!(
                " {} speech-to-text repos — Enter lists their GGML/GGUF files",
                results.len()
            ),
            results
                .iter()
                .map(|r| {
                    ListItem::new(Line::from(vec![
                        Span::raw(format!(" {:<50}", r.id)),
                        Span::styled(
                            format!("↓ {}  ♥ {}", fmt_count(r.downloads), fmt_count(r.likes)),
                            Style::default().fg(DIM),
                        ),
                    ]))
                })
                .collect(),
        ),
        HubList::Suggested(suggested) => (
            " suggested models — Enter downloads · type to search speech-to-text on Hugging Face"
                .into(),
            suggested
                .iter()
                .map(|s| {
                    let installed = app.library.models.iter().any(|m| m.name == s.file);
                    let marker = if installed { "● " } else { "  " };
                    if s.supported() {
                        ListItem::new(Line::from(vec![
                            Span::styled(marker, Style::default().fg(ACCENT)),
                            Span::raw(format!("{:<26}", s.name)),
                            Span::styled(format!("{:>7}  ", s.size), Style::default().fg(DIM)),
                            Span::styled(metal_tag, Style::default().fg(Color::Green)),
                            Span::styled(format!("  {}", s.note), Style::default().fg(DIM)),
                        ]))
                    } else {
                        ListItem::new(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(format!("{:<26}", s.name), Style::default().fg(DIM)),
                            Span::styled(
                                format!("{:>7}  {}  {}", s.size, s.format.to_uppercase(), s.note),
                                Style::default().fg(DIM),
                            ),
                        ]))
                    }
                })
                .collect(),
        ),
    }
}

/// Bottom line: download progress wins over the info message.
fn draw_bottom_line(frame: &mut Frame, area: Rect, app: &App) {
    if let Some((file, got, total)) = &app.hub.download {
        let (ratio, label) = if *total > 0 {
            (
                (*got as f64 / *total as f64).clamp(0.0, 1.0),
                format!(
                    "{file}  {} / {}  {:.0}%",
                    human_size(*got),
                    human_size(*total),
                    *got as f64 / *total as f64 * 100.0
                ),
            )
        } else {
            (0.0, format!("{file}  {}…", human_size(*got)))
        };
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::default().fg(ACCENT).bg(Color::Black))
                .ratio(ratio)
                .label(label),
            area,
        );
    } else if !app.hub.info.is_empty() {
        let style = if app.hub.info.contains("failed") || app.hub.info.contains("can only run") {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Green)
        };
        frame.render_widget(
            Paragraph::new(format!(" {}", app.hub.info)).style(style),
            area,
        );
    }
}
