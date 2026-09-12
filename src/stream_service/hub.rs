//! Transport-independent transcript state. A subscription atomically captures
//! a snapshot and the cursor for subsequent updates, so connecting never loses
//! text between those two steps. Slow consumers resynchronize via a snapshot.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;

use crate::transcribe::{Event, Segment};

const QUEUE_CAPACITY: usize = 256;
const MAX_SEGMENTS: usize = 2048;
const MAX_TEXT_BYTES: usize = 1024 * 1024;
const MAX_PARTIAL_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub session_id: Option<String>,
    pub source: Option<String>,
    pub model: Option<String>,
    pub live: bool,
    pub status: &'static str,
    pub device: Option<String>,
    pub mode: Option<String>,
    pub language: Option<String>,
    pub duration_secs: Option<f32>,
    pub segments: VecDeque<Segment>,
    pub partial: Option<Segment>,
    /// Older segments were removed or unusually large text was clipped.
    pub truncated: bool,
    pub error: Option<String>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            session_id: None,
            source: None,
            model: None,
            live: false,
            status: "idle",
            device: None,
            mode: None,
            language: None,
            duration_secs: None,
            segments: VecDeque::new(),
            partial: None,
            truncated: false,
            error: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Message {
    pub version: u8,
    /// Process-unique prefix plus a sequence number; shared by both transports.
    pub id: String,
    pub session_id: Option<String>,
    pub timestamp_ms: u64,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub data: Value,
}

struct State {
    snapshot: Snapshot,
    sequence: u64,
    session_count: u64,
    text_bytes: usize,
}

struct Inner {
    epoch: String,
    state: Mutex<State>,
    events: broadcast::Sender<Arc<Message>>,
}

#[derive(Clone)]
pub struct TextStream(Arc<Inner>);

pub struct Subscription {
    pub initial: Arc<Message>,
    pub receiver: broadcast::Receiver<Arc<Message>>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl Default for TextStream {
    fn default() -> Self {
        Self::new()
    }
}

impl TextStream {
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(QUEUE_CAPACITY);
        static INSTANCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = INSTANCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(Arc::new(Inner {
            epoch: format!("{}-{}-{unique}", now_ms(), std::process::id()),
            state: Mutex::new(State {
                snapshot: Snapshot::default(),
                sequence: 0,
                session_count: 0,
                text_bytes: 0,
            }),
            events,
        }))
    }

    fn message(&self, state: &State, kind: &'static str, data: Value) -> Arc<Message> {
        Arc::new(Message {
            version: 1,
            id: format!("{}:{}", self.0.epoch, state.sequence),
            session_id: state.snapshot.session_id.clone(),
            timestamp_ms: now_ms(),
            kind,
            data,
        })
    }

    pub fn snapshot(&self) -> Arc<Message> {
        let state = self.0.state.lock().unwrap();
        self.message(&state, "snapshot", json!(state.snapshot))
    }

    pub fn subscribe(&self) -> Subscription {
        let state = self.0.state.lock().unwrap();
        // Publishing holds this same lock through send().
        let receiver = self.0.events.subscribe();
        let initial = self.message(&state, "snapshot", json!(state.snapshot));
        Subscription { initial, receiver }
    }

    pub fn publish(&self, event: &Event) {
        let mut state = self.0.state.lock().unwrap();
        let (kind, data) = match event {
            Event::SessionStarted {
                source,
                model,
                live,
            } => {
                state.session_count += 1;
                state.snapshot = Snapshot {
                    session_id: Some(format!("{}-{}", self.0.epoch, state.session_count)),
                    source: Some(source.clone()),
                    model: Some(model.clone()),
                    live: *live,
                    status: "preparing",
                    ..Default::default()
                };
                state.text_bytes = 0;
                ("session_started", json!(state.snapshot))
            }
            Event::RecordingStarted { device, mode } => {
                state.snapshot.status = "recording";
                state.snapshot.device = Some(device.clone());
                state.snapshot.mode = Some(mode.clone());
                (
                    "recording_started",
                    json!({ "device": device, "mode": mode }),
                )
            }
            Event::RecordingStopped => {
                state.snapshot.status = "finishing";
                ("recording_stopped", json!({}))
            }
            Event::LivePartial(segment) => {
                let clipped = segment.text.len() > MAX_PARTIAL_BYTES;
                state.snapshot.truncated |= clipped;
                let segment = bounded_segment(segment);
                state.snapshot.partial = (!segment.text.is_empty()).then_some(segment.clone());
                let mut data = json!(segment);
                data["text_truncated"] = json!(clipped);
                ("partial", data)
            }
            Event::Segment(segment) => {
                let clipped = segment.text.len() > MAX_PARTIAL_BYTES;
                state.snapshot.truncated |= clipped;
                let segment = bounded_segment(segment);
                state.snapshot.partial = None;
                state.push_segment(segment.clone());
                let mut data = json!(segment);
                data["text_truncated"] = json!(clipped);
                ("segment", data)
            }
            Event::SegmentsFinal(segments) => {
                state.snapshot.segments.clear();
                state.snapshot.truncated = false;
                state.text_bytes = 0;
                for segment in segments {
                    state.snapshot.truncated |= segment.text.len() > MAX_PARTIAL_BYTES;
                    state.push_segment(bounded_segment(segment));
                }
                state.snapshot.partial = None;
                (
                    "segments_replaced",
                    json!({ "segments": state.snapshot.segments, "truncated": state.snapshot.truncated }),
                )
            }
            Event::Done {
                elapsed_secs,
                audio_secs,
                language,
            } => {
                state.snapshot.status = "completed";
                state.snapshot.language = language.clone();
                state.snapshot.duration_secs = Some(*audio_secs);
                state.snapshot.partial = None;
                (
                    "completed",
                    json!({ "elapsed_secs": elapsed_secs, "audio_secs": audio_secs, "language": language }),
                )
            }
            Event::Cancelled => {
                state.snapshot.status = "cancelled";
                ("cancelled", json!({}))
            }
            Event::Error(error) => {
                state.snapshot.status = "error";
                state.snapshot.error = Some(error.clone());
                ("error", json!({ "message": error }))
            }
            Event::LoadingModel(_) => {
                state.snapshot.status = "loading_model";
                ("status", json!({ "status": "loading_model" }))
            }
            Event::Decoding => {
                state.snapshot.status = "decoding";
                ("status", json!({ "status": "decoding" }))
            }
            Event::AudioInfo { duration_secs } => {
                state.snapshot.duration_secs = Some(*duration_secs);
                state.snapshot.status = "transcribing";
                (
                    "status",
                    json!({ "status": "transcribing", "duration_secs": duration_secs }),
                )
            }
            // Meter, engine logs and percent/heartbeat chatter stay in the TUI.
            _ => return,
        };
        state.sequence += 1;
        let message = self.message(&state, kind, data);
        let _ = self.0.events.send(message);
    }
}

fn bounded_segment(segment: &Segment) -> Segment {
    let mut segment = segment.clone();
    if segment.text.len() > MAX_PARTIAL_BYTES {
        let mut end = MAX_PARTIAL_BYTES;
        while !segment.text.is_char_boundary(end) {
            end -= 1;
        }
        segment.text.truncate(end);
    }
    segment
}

impl State {
    fn push_segment(&mut self, segment: Segment) {
        self.text_bytes += segment.text.len();
        self.snapshot.segments.push_back(segment);
        while self.snapshot.segments.len() > MAX_SEGMENTS || self.text_bytes > MAX_TEXT_BYTES {
            if let Some(old) = self.snapshot.segments.pop_front() {
                self.text_bytes -= old.text.len();
            }
            self.snapshot.truncated = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn segment(text: &str) -> Segment {
        Segment {
            start_ms: 0,
            end_ms: 1000,
            text: text.into(),
            speaker: None,
        }
    }
    #[test]
    fn subscribers_share_ids_and_partials_are_replaceable() {
        let hub = TextStream::new();
        hub.publish(&Event::SessionStarted {
            source: "microphone".into(),
            model: "Qwen".into(),
            live: true,
        });
        let mut a = hub.subscribe();
        let mut b = hub.subscribe();
        assert_eq!(a.initial.id, b.initial.id);
        hub.publish(&Event::LivePartial(segment("hola")));
        assert_eq!(
            a.receiver.try_recv().unwrap().id,
            b.receiver.try_recv().unwrap().id
        );
        hub.publish(&Event::LivePartial(segment("hola mundo")));
        assert_eq!(hub.snapshot().data["partial"]["text"], "hola mundo");
        hub.publish(&Event::Segment(segment("hola mundo")));
        let snapshot = hub.snapshot();
        assert!(snapshot.data["partial"].is_null());
        assert_eq!(snapshot.data["segments"].as_array().unwrap().len(), 1);
        hub.publish(&Event::SessionStarted {
            source: "next".into(),
            model: "Voxtral".into(),
            live: true,
        });
        assert_ne!(hub.snapshot().session_id, snapshot.session_id);
        assert!(hub.snapshot().data["segments"]
            .as_array()
            .unwrap()
            .is_empty());
    }
    #[test]
    fn late_clients_and_slow_clients_can_resynchronize() {
        let hub = TextStream::new();
        let mut old = hub.subscribe();
        for i in 0..300 {
            hub.publish(&Event::Segment(segment(&format!("line {i}"))));
        }
        assert!(matches!(
            old.receiver.try_recv(),
            Err(broadcast::error::TryRecvError::Lagged(_))
        ));
        let current = hub.subscribe();
        assert_eq!(
            current.initial.data["segments"].as_array().unwrap().len(),
            300
        );
        assert_eq!(current.initial.data["segments"][299]["text"], "line 299");
    }
    #[test]
    fn unusually_large_unicode_text_is_bounded_and_marked() {
        let hub = TextStream::new();
        let mut subscriber = hub.subscribe();
        hub.publish(&Event::Segment(segment(&"中文".repeat(20_000))));
        let message = subscriber.receiver.try_recv().unwrap();
        assert_eq!(message.data["text_truncated"], true);
        let text = message.data["text"].as_str().unwrap();
        assert!(text.len() <= MAX_PARTIAL_BYTES);
        assert!(text.ends_with('中') || text.ends_with('文'));
        assert_eq!(hub.snapshot().data["truncated"], true);
    }

    #[test]
    fn relabeling_replaces_segments_and_retention_is_explicit() {
        let hub = TextStream::new();
        hub.publish(&Event::Segment(segment("old")));
        let mut labeled = segment("new");
        labeled.speaker = Some(1);
        hub.publish(&Event::SegmentsFinal(vec![labeled]));
        assert_eq!(hub.snapshot().data["segments"][0]["speaker"], 1);
        for _ in 0..MAX_SEGMENTS {
            hub.publish(&Event::Segment(segment("other")));
        }
        let snapshot = hub.snapshot();
        assert_eq!(
            snapshot.data["segments"].as_array().unwrap().len(),
            MAX_SEGMENTS
        );
        assert_eq!(snapshot.data["truncated"], true);
    }
}
