//! Screen geometry: the fixed regions of the base screen and small
//! rect/text helpers shared by the widgets.

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// The five fixed regions of the base screen.
pub struct Areas {
    pub header: Rect,
    pub files: Rect,
    pub transcript: Rect,
    pub status: Rect,
    pub keys: Rect,
}

/// `key_rows`: height of the key-hint bar — it wraps onto multiple rows
/// when the window is too narrow to fit every hint on one line.
pub fn areas(area: Rect, key_rows: u16) -> Areas {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),        // header
            Constraint::Min(3),           // body
            Constraint::Length(1),        // status
            Constraint::Length(key_rows), // keys
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

pub fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
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

/// Truncate keeping the tail — paths are more recognizable by their end.
pub fn truncate_left(s: &str, max: usize) -> String {
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
