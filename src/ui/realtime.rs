use super::theme::{ACCENT, DIM};
use crate::app::App;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::widgets::canvas::{Canvas, Line};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

pub(super) fn draw_recording(frame: &mut Frame, area: Rect, app: &App) {
    let live = &app.live;
    let (label, color) = if live.stopping {
        ("FINISHING", Color::Yellow)
    } else if live.recording {
        ("● RECORDING", Color::Red)
    } else if live.active {
        ("PREPARING", Color::Yellow)
    } else {
        ("STOPPED", DIM)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {label} "))
        .border_style(Style::default().fg(color));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(6),
    ])
    .split(inner);
    let time = crate::format::clock_time((live.seconds * 1000.0) as i64);
    frame.render_widget(
        Paragraph::new(format!(
            " {time} elapsed\n {}",
            if live.device.is_empty() {
                "Opening default microphone…"
            } else {
                &live.device
            }
        ))
        .wrap(Wrap { trim: true }),
        rows[0],
    );
    let waveform = Canvas::default()
        .x_bounds([0.0, 120.0])
        .y_bounds([-1.0, 1.0])
        .paint(|ctx| {
            ctx.draw(&Line {
                x1: 0.0,
                x2: 120.0,
                y1: 0.0,
                y2: 0.0,
                color: DIM,
            });
            let offset = 120usize.saturating_sub(live.levels.len());
            for (i, value) in live.levels.iter().enumerate() {
                let amplitude = (*value as f64).sqrt().clamp(0.0, 1.0);
                ctx.draw(&Line {
                    x1: (offset + i) as f64,
                    x2: (offset + i) as f64,
                    y1: -amplitude,
                    y2: amplitude,
                    color: if *value > 0.98 { Color::Red } else { ACCENT },
                });
            }
        });
    frame.render_widget(waveform, rows[1]);
    let db = 20.0 * live.rms.max(0.00001).log10();
    let signal = if !live.recording {
        "Microphone inactive"
    } else if live.peak > 0.98 {
        "Clipping — lower input volume"
    } else if live.rms < 0.003 {
        "Quiet — waiting for speech"
    } else {
        "Receiving audio"
    };
    frame.render_widget(
        Paragraph::new(format!(
            " {db:.0} dBFS  ·  {signal}\n Waveform · last 12 seconds"
        ))
        .style(Style::default().fg(DIM))
        .wrap(Wrap { trim: true }),
        rows[2],
    );
    let speed = match live.inference_rtf {
        Some(rtf) => format!(
            "{rtf:.2}s/audio s · {}",
            if rtf > 1.0 {
                "falling behind"
            } else {
                "keeping up"
            }
        ),
        None => "Measuring inference speed…".into(),
    };
    frame.render_widget(
        Paragraph::new(format!(
            " {}\n {:.1}s pending audio\n {}\n R  stop & export\n G  follow latest text\n a  choose microphone",
            live.mode,
            (live.seconds - live.decoded_seconds).max(0.0),
            speed,
        ))
        .style(Style::default().fg(ACCENT))
        .wrap(Wrap { trim: true }),
        rows[3],
    );
}
