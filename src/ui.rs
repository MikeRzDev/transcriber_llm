use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, DirRow, FileEntry, Focus, SettingsRow, TranscriptState, WorkState};
use crate::format::{clock_time, fmt_count, human_size};
use crate::hub;

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;

/// The five fixed regions of the base screen.
struct Areas {
    header: Rect,
    files: Rect,
    transcript: Rect,
    status: Rect,
    keys: Rect,
}

fn areas(area: Rect) -> Areas {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(3),    // body
            Constraint::Length(1), // status
            Constraint::Length(1), // keys
        ])
        .split(area);
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(20)])
        .split(outer[1]);
    Areas {
        header: outer[0],
        files: body[0],
        transcript: body[1],
        status: outer[2],
        keys: outer[3],
    }
}

/// Pin the transcript scroll to its content for this frame. Runs in the
/// update phase (before `draw`) so rendering itself never mutates state.
pub fn clamp_transcript(app: &mut App, area: Rect) {
    if app.transcript.segments.is_empty() {
        return;
    }
    let transcript_area = areas(area).transcript;
    // Block borders take one cell on each side
    let inner_width = transcript_area.width.saturating_sub(2) as usize;
    let viewport = transcript_area.height.saturating_sub(2) as usize;
    let text_width = inner_width.saturating_sub(2).max(20);
    let total = transcript_lines(&app.transcript, text_width).len();
    app.transcript.clamp(total, viewport);
}

pub fn draw(frame: &mut Frame, app: &App) {
    let areas = areas(frame.area());

    draw_header(frame, areas.header, app);
    draw_files(frame, areas.files, app);
    draw_transcript(frame, areas.transcript, app);
    draw_status(frame, areas.status, app);
    draw_keys(frame, areas.keys, app);

    if app.settings.open {
        draw_settings(frame, app);
    }
    if app.picker.open {
        draw_model_picker(frame, app);
    }
    if app.hub.open {
        draw_hub(frame, app);
    }
    if app.settings.dir_picker.is_some() {
        draw_dir_picker(frame, app);
    }
    if app.settings.move_prompt.is_some() {
        draw_move_prompt(frame, app);
    }
}

fn draw_move_prompt(frame: &mut Frame, app: &App) {
    let Some(prompt) = &app.settings.move_prompt else {
        return;
    };
    let area = centered_rect(64, 30, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" move models? ");

    let option = |label: &str, selected: bool| {
        if selected {
            Span::styled(
                format!("[ {label} ]"),
                Style::default()
                    .bg(ACCENT)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(format!("[ {label} ]"), Style::default().fg(DIM))
        }
    };

    let lines = vec![
        Line::raw(""),
        Line::raw(format!(
            "  The previous folder still holds {} model(s).",
            prompt.count
        )),
        Line::raw("  Move them to the new models folder?"),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  from: ", Style::default().fg(DIM)),
            Span::raw(prompt.from.display().to_string()),
        ]),
        Line::from(vec![
            Span::styled("  to:   ", Style::default().fg(DIM)),
            Span::raw(prompt.to.display().to_string()),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::raw("  "),
            option("Yes, move them", prompt.yes_selected),
            Span::raw("   "),
            option("No, leave them", !prompt.yes_selected),
        ]),
        Line::raw(""),
        Line::from(Span::styled(
            "  ←→ choose · Enter confirm · y / n shortcuts · Esc leaves them",
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_dir_picker(frame: &mut Frame, app: &App) {
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
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
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
    let list = List::new(items).highlight_style(
        Style::default()
            .bg(ACCENT)
            .fg(Color::Black)
            .add_modifier(Modifier::BOLD),
    );
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

fn draw_hub(frame: &mut Frame, app: &App) {
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

    // Search bar (or breadcrumb when inside a repo's file list)
    if let Some((repo, _)) = &app.hub.files {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" repo: ", Style::default().fg(DIM)),
                Span::styled(repo.clone(), Style::default().add_modifier(Modifier::BOLD)),
            ])),
            rows[0],
        );
    } else {
        const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
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
                format!("  {}", SPINNER[app.tick % SPINNER.len()]),
                Style::default().fg(Color::Yellow),
            ));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), rows[0]);
    }

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(
                " engine: whisper.cpp · {}  ·  downloads to {}",
                hub::backend_label(),
                app.library.dir.display()
            ),
            Style::default().fg(DIM),
        ))),
        rows[1],
    );

    let metal_tag = if hub::metal_available() {
        "Metal ✓"
    } else {
        "CPU"
    };

    // Header + items for whichever list is showing
    let (header, items): (String, Vec<ListItem>) = if let Some((_, files)) = &app.hub.files {
        (
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
        )
    } else if let Some(results) = &app.hub.results {
        (
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
        )
    } else {
        (
            " suggested models — Enter downloads · type to search speech-to-text on Hugging Face"
                .into(),
            app.hub
                .suggested
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
                            Span::styled(
                                format!("{:<26}", s.name),
                                Style::default().fg(DIM),
                            ),
                            Span::styled(
                                format!("{:>7}  {}  {}", s.size, s.format.to_uppercase(), s.note),
                                Style::default().fg(DIM),
                            ),
                        ]))
                    }
                })
                .collect(),
        )
    };

    frame.render_widget(
        Paragraph::new(Span::styled(header, Style::default().fg(DIM))),
        rows[2],
    );

    let list = List::new(items).highlight_style(
        Style::default()
            .bg(ACCENT)
            .fg(Color::Black)
            .add_modifier(Modifier::BOLD),
    );
    let mut state = ListState::default();
    let len = if let Some((_, files)) = &app.hub.files {
        files.len()
    } else if let Some(r) = &app.hub.results {
        r.len()
    } else {
        app.hub.suggested.len()
    };
    if len > 0 {
        state.select(Some(app.hub.selected.min(len - 1)));
    }
    frame.render_stateful_widget(list, rows[3], &mut state);

    // Bottom line: download progress wins over the info message
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
            rows[4],
        );
    } else if !app.hub.info.is_empty() {
        let style = if app.hub.info.contains("failed") || app.hub.info.contains("can only run") {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Green)
        };
        frame.render_widget(
            Paragraph::new(format!(" {}", app.hub.info)).style(style),
            rows[4],
        );
    }
}

fn draw_settings(frame: &mut Frame, app: &App) {
    let area = centered_rect(70, 55, frame.area());
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(" settings ");

    let model = app.library
        .selected
        .as_ref()
        .map(|m| format!("{} ({})", m.name, m.size_human()))
        .unwrap_or_else(|| "none".into());

    let row_style = |row: SettingsRow| {
        if app.settings.selected == row && app.settings.language_input.is_none() {
            Style::default()
                .bg(ACCENT)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        }
    };

    let diarize_state = if app.config.diarize {
        "ON  (uses the tdrz model, English only)"
    } else {
        "OFF"
    };

    let language_line: Line = if let Some(input) = &app.settings.language_input {
        Line::from(vec![
            Span::raw("  Language:       "),
            Span::styled(
                format!("{input}▏"),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ])
    } else {
        Line::from(Span::styled(
            format!(
                "  Language:       {}",
                app.config.language.as_deref().unwrap_or("auto-detect")
            ),
            row_style(SettingsRow::Language),
        ))
    };

    let lines = vec![
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Default model:  {model}"),
            row_style(SettingsRow::DefaultModel),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Models folder:  {}", app.library.dir.display()),
            row_style(SettingsRow::ModelsFolder),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Output folder:  {}", app.output_dir.display()),
            row_style(SettingsRow::OutputFolder),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "  Model management  →  search & download from Hugging Face",
            row_style(SettingsRow::ModelManagement),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Diarization:    {diarize_state}"),
            row_style(SettingsRow::Diarize),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            format!("  Split mode:     {}", app.config.split_mode.label()),
            row_style(SettingsRow::SplitMode),
        )),
        Line::raw(""),
        language_line,
        Line::raw(""),
        Line::from(Span::styled(
            if app.settings.language_input.is_some() {
                "  Type an ISO 639-1 code (en, es, de, fr…) or auto, Enter to save"
            } else {
                "  ↑↓ select · Enter change · Esc close  (saved to ~/.config/transcribe-stt)"
            },
            Style::default().fg(DIM),
        )),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let model = app.library
        .selected
        .as_ref()
        .map(|m| format!("{} ({})", m.name, m.size_human()))
        .unwrap_or_else(|| "no model — press m".into());
    let line = Line::from(vec![
        Span::styled(
            " transcribe-stt ",
            Style::default().fg(Color::Black).bg(ACCENT),
        ),
        Span::raw("  "),
        Span::styled(model, Style::default().fg(ACCENT)),
        Span::styled("  •  Metal GPU", Style::default().fg(DIM)),
        if app.config.diarize {
            Span::styled("  •  diarize", Style::default().fg(Color::Magenta))
        } else {
            Span::raw("")
        },
    ]);
    frame.render_widget(Paragraph::new(line), area);

    // Live process resource usage, right-aligned over the same line
    let stats_style = if app.busy() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(DIM)
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("{} ", app.stats.label()),
            stats_style,
        )))
        .alignment(Alignment::Right),
        area,
    );
}

fn draw_files(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Files;
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };

    let items: Vec<ListItem> = app.browser
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

/// Wrapped display lines: "[mm:ss → mm:ss] text", speaker-tagged when
/// diarized. Shared by rendering and the pre-draw scroll clamp.
fn transcript_lines(transcript: &TranscriptState, text_width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();
    for seg in &transcript.segments {
        let stamp = format!(
            "[{} → {}] ",
            clock_time(seg.start_ms),
            clock_time(seg.end_ms)
        );
        let speaker = seg.speaker.map(|s| {
            (
                format!("{}: ", if s == 0 { "A" } else { "B" }),
                if s == 0 { Color::Cyan } else { Color::Magenta },
            )
        });
        let prefix_len = stamp.len() + speaker.as_ref().map(|(t, _)| t.len()).unwrap_or(0);
        let indent = " ".repeat(prefix_len);
        let wrapped = textwrap::wrap(
            seg.text.trim(),
            text_width.saturating_sub(prefix_len).max(10),
        );
        for (i, piece) in wrapped.iter().enumerate() {
            if i == 0 {
                let mut spans = vec![Span::styled(stamp.clone(), Style::default().fg(DIM))];
                if let Some((tag, color)) = &speaker {
                    spans.push(Span::styled(
                        tag.clone(),
                        Style::default().fg(*color).add_modifier(Modifier::BOLD),
                    ));
                }
                spans.push(Span::raw(piece.to_string()));
                lines.push(Line::from(spans));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::raw(piece.to_string()),
                ]));
            }
        }
    }
    lines
}

fn draw_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Transcript;
    let border_style = if focused {
        Style::default().fg(ACCENT)
    } else {
        Style::default().fg(DIM)
    };

    let title = app
        .transcript
        .source
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| format!(" {} ", n.to_string_lossy()))
        .unwrap_or_else(|| " transcript ".into());

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.transcript.segments.is_empty() {
        let placeholder = match app.work {
            WorkState::Idle => {
                "No transcript yet.\n\nPick an audio file on the left and press Enter."
            }
            _ => "Waiting for first segment…",
        };
        frame.render_widget(
            Paragraph::new(placeholder)
                .style(Style::default().fg(DIM))
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    }

    let text_width = (inner.width as usize).saturating_sub(2).max(20);
    let lines = transcript_lines(&app.transcript, text_width);
    frame.render_widget(
        Paragraph::new(lines).scroll((app.transcript.scroll as u16, 0)),
        inner,
    );
}

fn draw_status(frame: &mut Frame, area: Rect, app: &App) {
    match &app.work {
        WorkState::Transcribing { progress } | WorkState::LoadingModel { progress, .. } => {
            let loading = matches!(app.work, WorkState::LoadingModel { .. });
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(30), Constraint::Min(10)])
                .split(area);
            let color = if loading { Color::Yellow } else { ACCENT };
            let gauge = Gauge::default()
                .gauge_style(Style::default().fg(color).bg(Color::Black))
                .ratio((*progress as f64 / 100.0).clamp(0.0, 1.0))
                .label(format!("{progress}%"));
            frame.render_widget(gauge, chunks[0]);
            frame.render_widget(Paragraph::new(format!(" {}", app.status)), chunks[1]);
        }
        WorkState::UnloadingModel | WorkState::Decoding => {
            const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let frame_char = SPINNER[app.tick % SPINNER.len()];
            frame.render_widget(
                Paragraph::new(format!("{frame_char} {}", app.status))
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

fn draw_keys(frame: &mut Frame, area: Rect, app: &App) {
    let keys: &[(&str, &str)] = if app.hub.open {
        &[
            ("type", "search"),
            ("↑↓", "select"),
            ("Enter", "open / download"),
            ("Esc", "back / cancel / close"),
        ]
    } else if app.settings.move_prompt.is_some() {
        &[
            ("←→", "choose"),
            ("Enter", "confirm"),
            ("y/n", "shortcuts"),
            ("Esc", "leave them"),
        ]
    } else if app.settings.dir_picker.is_some() {
        &[
            ("↑↓", "select"),
            ("Enter", "open / choose folder"),
            ("Esc", "cancel"),
        ]
    } else if app.picker.open {
        &[
            ("↑↓", "select"),
            ("Enter", "set default model"),
            ("Esc", "close"),
        ]
    } else if app.settings.open {
        &[("↑↓", "select"), ("Enter", "change"), ("Esc", "close")]
    } else {
        &[
            ("Tab", "pane"),
            ("Enter", "transcribe"),
            ("drop", "file → transcribe"),
            ("m", "models"),
            ("s", "settings"),
            ("d", "diarize"),
            ("e", "export"),
            ("c", "cancel"),
            ("q", "quit"),
        ]
    };
    let mut spans: Vec<Span> = Vec::new();
    for (key, desc) in keys {
        spans.push(Span::styled(
            format!(" {key} "),
            Style::default().fg(Color::Black).bg(DIM),
        ));
        spans.push(Span::styled(format!(" {desc}  "), Style::default().fg(DIM)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_model_picker(frame: &mut Frame, app: &App) {
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

    let items: Vec<ListItem> = app.library
        .models
        .iter()
        .map(|m| {
            let selected_marker = app.library
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

    let list = List::new(items).block(block).highlight_style(
        Style::default()
            .bg(ACCENT)
            .fg(Color::Black)
            .add_modifier(Modifier::BOLD),
    );

    let mut state = ListState::default();
    state.select(Some(app.picker.selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1]);
    horizontal[1]
}

fn truncate_left(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        let tail: String = chars[chars.len() - max.saturating_sub(1).max(1)..]
            .iter()
            .collect();
        format!("…{tail}")
    }
}
