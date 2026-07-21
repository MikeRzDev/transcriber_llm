//! Shared display formatting: timestamps, byte sizes, and compact counts.
//! One implementation each, used by exports, the TUI, and headless output.

/// Decompose milliseconds into hours, minutes, seconds, and leftover millis.
fn hms(ms: i64) -> (i64, i64, i64, i64) {
    (
        ms / 3_600_000,
        (ms % 3_600_000) / 60_000,
        (ms % 60_000) / 1000,
        ms % 1000,
    )
}

/// SubRip timestamp: `HH:MM:SS,mmm`.
pub fn srt_time(ms: i64) -> String {
    let (h, m, s, millis) = hms(ms);
    format!("{h:02}:{m:02}:{s:02},{millis:03}")
}

/// Always `HH:MM:SS` so anchors are uniform for machine parsing.
pub fn llm_time(ms: i64) -> String {
    let (h, m, s, _) = hms(ms);
    format!("{h:02}:{m:02}:{s:02}")
}

/// Compact form for the transcript pane: `MM:SS`, hours only when needed.
pub fn clock_time(ms: i64) -> String {
    let (h, m, s, _) = hms(ms);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Human-readable byte size: GB/MB/KB, decimal units.
pub fn human_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{:.0} KB", b / 1e3)
    }
}

/// Compact count for download/like badges: `1.2M`, `3.4k`, `999`.
pub fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_formats() {
        assert_eq!(srt_time(0), "00:00:00,000");
        assert_eq!(srt_time(3_723_456), "01:02:03,456");
        assert_eq!(llm_time(3_723_456), "01:02:03");
        assert_eq!(clock_time(59_000), "00:59");
        assert_eq!(clock_time(3_600_000), "1:00:00");
    }

    #[test]
    fn count_units() {
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(3_400), "3.4k");
        assert_eq!(fmt_count(1_250_000), "1.2M");
    }
}
