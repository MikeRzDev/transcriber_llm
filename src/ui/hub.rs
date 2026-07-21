//! The Model management modal: search bar, backend line, whichever of
//! the three lists is visible, and the download gauge / info line.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph};
use ratatui::Frame;

use crate::app::{App, Download, HubList};
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

    let visible = app.hub_visible_list();
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
    if let Some(view) = &app.hub.files {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" repo: ", Style::default().fg(DIM)),
                Span::styled(
                    view.repo.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
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
    let mlx = if hub::metal_available() {
        if crate::transcribe::mlx_audio_available() {
            " + MLX (mlx-audio ✓)"
        } else {
            " + MLX (auto-installs on first use)"
        }
    } else {
        ""
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(
                " engines: whisper.cpp · {}{mlx}  ·  downloads to {}",
                hub::backend_label(),
                app.library.dir.display()
            ),
            Style::default().fg(DIM),
        ))),
        area,
    );
}

/// The in-flight (or paused) download for `file`, if that model's row is the
/// one being transferred. The download tracks the on-disk base name, so
/// compare against each row's base name.
fn active_download<'a>(app: &'a App, file: &str) -> Option<&'a Download> {
    let d = app.hub.download.as_ref()?;
    let base = hub::dest_name(file)?;
    (d.file == base).then_some(d)
}

/// The row for a transferring model: a bullet (or pause bar) and yellow text.
fn downloading_row<'a>(name: &str, width: usize, size: &str, d: &Download) -> ListItem<'a> {
    let yellow = Style::default().fg(Color::Yellow);
    let (mark, tag) = if d.paused {
        ("‖ ", "paused")
    } else {
        ("• ", "downloading…")
    };
    ListItem::new(Line::from(vec![
        Span::styled(mark, yellow),
        Span::styled(format!("{name:<width$}"), yellow),
        Span::styled(format!("{size:>9}  {tag}"), yellow),
    ]))
}

/// Header text + rows for whichever list is showing.
fn list_content<'a>(app: &'a App, visible: &HubList<'a>) -> (String, Vec<ListItem<'a>>) {
    let metal_tag = if hub::metal_available() {
        "Metal ✓"
    } else {
        "CPU"
    };
    match visible {
        HubList::Files(view) => (
            " models — ✓ already downloaded · Enter downloads (folders download whole)".into(),
            view.variants
                .iter()
                .map(|v| {
                    // A directory-model variant downloads as its own folder
                    let name = hub::variant_dir_name(&view.repo, &v.subdir);
                    let size = if v.size_bytes > 0 {
                        human_size(v.size_bytes)
                    } else {
                        "?".into()
                    };
                    if let Some(d) = app.hub.download.as_ref().filter(|d| d.file == name) {
                        return downloading_row(&v.label(&view.repo), 44, &size, d);
                    }
                    let installed = app.library.models.iter().any(|m| m.name == name);
                    let marker = if installed { "✓ " } else { "  " };
                    ListItem::new(Line::from(vec![
                        Span::styled(marker, Style::default().fg(Color::Green)),
                        Span::raw(format!("{:<44}", v.label(&view.repo))),
                        Span::styled(
                            format!("{size:>9}  {} files  ", v.file_count),
                            Style::default().fg(DIM),
                        ),
                        Span::styled("MLX", Style::default().fg(Color::Green)),
                    ]))
                })
                .chain(view.files.iter().map(|f| {
                    let size = if f.size_bytes > 0 {
                        human_size(f.size_bytes)
                    } else {
                        "?".into()
                    };
                    if let Some(d) = active_download(app, &f.name) {
                        return downloading_row(&f.name, 44, &size, d);
                    }
                    let installed = app.library.models.iter().any(|m| m.name == f.name);
                    let marker = if installed { "✓ " } else { "  " };
                    ListItem::new(Line::from(vec![
                        Span::styled(marker, Style::default().fg(Color::Green)),
                        Span::raw(format!("{:<44}", f.name)),
                        Span::styled(format!("{size:>9}  "), Style::default().fg(DIM)),
                        Span::styled(metal_tag, Style::default().fg(Color::Green)),
                    ]))
                }))
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
        HubList::Default(entries) => (
            " ✓ selected default · downloaded in white · Enter selects/downloads · Del removes"
                .into(),
            entries
                .iter()
                .map(|e| {
                    // Directory models run on the MLX engine, not Metal/whisper
                    let engine_tag = if e.is_dir { "MLX" } else { metal_tag };
                    if let Some(d) = active_download(app, &e.file) {
                        return downloading_row(e.name, 26, &e.size, d);
                    }
                    if e.installed {
                        // On disk: white text. The checkmark marks only the
                        // model chosen as the current default.
                        let selected = app
                            .library
                            .selected
                            .as_ref()
                            .is_some_and(|m| m.name == e.file);
                        let white = Style::default().fg(Color::White);
                        let marker = if selected {
                            Span::styled("✓ ", Style::default().fg(Color::Green))
                        } else {
                            Span::styled("  ", white)
                        };
                        let mut spans = vec![
                            marker,
                            Span::styled(format!("{:<26}", e.name), white),
                            Span::styled(format!("{:>9}  ", e.size), white),
                            Span::styled(engine_tag, Style::default().fg(Color::Green)),
                        ];
                        if !e.note.is_empty() {
                            spans.push(Span::styled(
                                format!("  {}", e.note),
                                Style::default().fg(DIM),
                            ));
                        }
                        ListItem::new(Line::from(spans))
                    } else {
                        // Not downloaded yet: greyed out. Runnable suggestions
                        // keep their size; MLX-only ones spell out the format.
                        let tail = if e.supported {
                            format!("{:>9}  {}", e.size, e.note)
                        } else {
                            format!("{:>9}  {}  {}", e.size, e.format.to_uppercase(), e.note)
                        };
                        ListItem::new(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(format!("{:<26}", e.name), Style::default().fg(DIM)),
                            Span::styled(tail, Style::default().fg(DIM)),
                        ]))
                    }
                })
                .collect(),
        ),
    }
}

/// Bottom line: download progress (or a paused bar) wins over the info message.
fn draw_bottom_line(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(d) = &app.hub.download {
        let ratio = if d.total > 0 {
            (d.got as f64 / d.total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let sizes = if d.total > 0 {
            format!("{} / {}", human_size(d.got), human_size(d.total))
        } else {
            format!("{}…", human_size(d.got))
        };
        let (color, label) = if d.paused {
            (
                Color::Yellow,
                format!("{}  paused at {sizes}  ·  p resumes · Esc cancels", d.file),
            )
        } else {
            (
                ACCENT,
                format!("{}  {sizes}  ·  p pauses · Esc cancels", d.file),
            )
        };
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::default().fg(color).bg(Color::Black))
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

/// The delete-model confirmation, drawn on top of the hub modal.
pub(super) fn draw_hub_delete_prompt(frame: &mut Frame, app: &App) {
    let Some(prompt) = &app.hub.delete_prompt else {
        return;
    };
    let area = centered_rect(64, 36, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .title(" delete model? ");

    let option = |label: &str, selected: bool| {
        if selected {
            Span::styled(format!("[ {label} ]"), highlight_style())
        } else {
            Span::styled(format!("[ {label} ]"), Style::default().fg(DIM))
        }
    };

    let what = if prompt.path.is_dir() {
        "  Permanently delete this model folder (all its files) from disk?"
    } else {
        "  Permanently delete this model file from disk?"
    };
    let lines = vec![
        Line::raw(""),
        Line::raw(what),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  model: ", Style::default().fg(DIM)),
            Span::styled(
                prompt.name.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  size:  ", Style::default().fg(DIM)),
            Span::raw(human_size(prompt.size)),
        ]),
        Line::from(vec![
            Span::styled("  path:  ", Style::default().fg(DIM)),
            Span::raw(prompt.path.display().to_string()),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            option("Yes, delete", prompt.yes_selected),
            Span::raw("   "),
            option("No, keep it", !prompt.yes_selected),
        ]),
        Line::raw(""),
        Line::from(Span::styled(
            "  ←→ choose · Enter confirm · y / n shortcuts · Esc cancels",
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}
