//! Actual loopback HTTP/SSE/WebSocket integration tests. No models, microphone,
//! external network or Python runtime are needed.
use std::io::{BufRead, BufReader, Read};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::Value;
use transcribe_stt::stream_service::{StreamServer, TextStream};
use transcribe_stt::transcribe::{Event, Segment};

fn segment(text: &str) -> Segment {
    Segment {
        start_ms: 120,
        end_ms: 1600,
        text: text.into(),
        speaker: None,
    }
}
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(5))
        .build()
}
fn connect_sse(server: &StreamServer) -> BufReader<Box<dyn Read + Send + Sync>> {
    let response = agent()
        .get(&format!("http://{}/events", server.address()))
        .call()
        .unwrap();
    assert!(response
        .header("content-type")
        .unwrap()
        .starts_with("text/event-stream"));
    assert_eq!(response.header("x-accel-buffering"), Some("no"));
    BufReader::new(response.into_reader())
}
fn sse_message(reader: &mut impl BufRead) -> Value {
    let mut data = None;
    let mut event = String::new();
    let mut id = String::new();
    loop {
        let mut line = String::new();
        assert!(
            reader.read_line(&mut line).unwrap() > 0,
            "SSE ended before an event arrived"
        );
        if let Some(value) = line
            .strip_prefix("data: ")
            .or_else(|| line.strip_prefix("data:"))
        {
            data = Some(serde_json::from_str::<Value>(value.trim()).unwrap());
        }
        if let Some(value) = line.strip_prefix("event:") {
            event = value.trim().into();
        }
        if let Some(value) = line.strip_prefix("id:") {
            id = value.trim().into();
        }
        if line.trim().is_empty() {
            if let Some(message) = data {
                assert_eq!(message["type"], event);
                assert_eq!(message["id"], id);
                return message;
            }
        }
    }
}

#[test]
fn sse_streams_ordered_partials_finals_and_reconnect_snapshot() {
    let hub = TextStream::new();
    let server = StreamServer::start(0, hub.clone()).unwrap();
    let mut client = connect_sse(&server);
    assert_eq!(sse_message(&mut client)["type"], "snapshot");
    hub.publish(&Event::SessionStarted {
        source: "microphone".into(),
        model: "Qwen".into(),
        live: true,
    });
    hub.publish(&Event::LivePartial(segment("Hola, ¿qué")));
    hub.publish(&Event::LivePartial(segment("Hola, ¿qué tal?\n中文")));
    hub.publish(&Event::Segment(segment("Hola, ¿qué tal?\n中文")));
    hub.publish(&Event::Done {
        elapsed_secs: 2.0,
        audio_secs: 1.6,
        language: Some("es".into()),
    });
    let mut ids = Vec::new();
    for kind in [
        "session_started",
        "partial",
        "partial",
        "segment",
        "completed",
    ] {
        let message = sse_message(&mut client);
        assert_eq!(message["type"], kind);
        ids.push(message["id"].as_str().unwrap().to_owned());
    }
    assert_eq!(
        ids.iter().collect::<std::collections::HashSet<_>>().len(),
        ids.len()
    );
    let response = agent()
        .get(&format!("http://{}/events", server.address()))
        .set("Last-Event-ID", &ids[1])
        .call()
        .unwrap();
    let mut reconnect = BufReader::new(response.into_reader());
    let snapshot = sse_message(&mut reconnect);
    assert_eq!(snapshot["type"], "snapshot");
    assert_eq!(
        snapshot["data"]["segments"][0]["text"],
        "Hola, ¿qué tal?\n中文"
    );
    assert!(snapshot["data"]["partial"].is_null());
    assert_eq!(snapshot["data"]["status"], "completed");
    assert_eq!(snapshot["id"], *ids.last().unwrap());
}

#[test]
fn websocket_and_sse_deliver_identical_json_and_session_boundaries() {
    let hub = TextStream::new();
    let server = StreamServer::start(0, hub.clone()).unwrap();
    let mut sse = connect_sse(&server);
    let initial = sse_message(&mut sse);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{}/ws", server.address()))
            .await
            .unwrap();
        let snapshot = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let snapshot: Value = serde_json::from_str(snapshot.to_text().unwrap()).unwrap();
        assert_eq!(snapshot["id"], initial["id"]);
        for event in [
            Event::SessionStarted {
                source: "microphone".into(),
                model: "Voxtral".into(),
                live: true,
            },
            Event::LivePartial(segment("Hello")),
            Event::Segment(segment("Hello there")),
            Event::RecordingStopped,
            Event::Done {
                elapsed_secs: 1.0,
                audio_secs: 1.0,
                language: None,
            },
            Event::SessionStarted {
                source: "microphone".into(),
                model: "Qwen".into(),
                live: true,
            },
        ] {
            hub.publish(&event);
            let ws_message = tokio::time::timeout(Duration::from_secs(5), ws.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            let ws_message: Value = serde_json::from_str(ws_message.to_text().unwrap()).unwrap();
            assert_eq!(ws_message, sse_message(&mut sse));
        }
        ws.close(None).await.unwrap();
    });
}

#[test]
fn local_web_apps_get_cors_but_external_origins_and_hosts_are_rejected() {
    let server = StreamServer::start(0, TextStream::new()).unwrap();
    let url = format!("http://{}/transcript", server.address());
    let local = agent()
        .get(&url)
        .set("Origin", "http://localhost:3000")
        .call()
        .unwrap();
    assert_eq!(
        local.header("access-control-allow-origin"),
        Some("http://localhost:3000")
    );
    for (header, value) in [
        ("Origin", "https://example.com"),
        ("Host", "unrelated.example:8765"),
    ] {
        assert!(matches!(
            agent().get(&url).set(header, value).call(),
            Err(ureq::Error::Status(403, _))
        ));
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let mut request = format!("ws://{}/ws", server.address()).into_client_request().unwrap();
        request.headers_mut().insert("Origin", "https://example.com".parse().unwrap());
        assert!(matches!(tokio_tungstenite::connect_async(request).await, Err(tokio_tungstenite::tungstenite::Error::Http(response)) if response.status().as_u16() == 403));
    });
}

#[test]
fn shutdown_closes_streams_and_releases_port_without_stopping_publisher() {
    let hub = TextStream::new();
    let server = StreamServer::start(0, hub.clone()).unwrap();
    let address = server.address();
    let mut client = connect_sse(&server);
    sse_message(&mut client);
    hub.publish(&Event::Done {
        elapsed_secs: 1.0,
        audio_secs: 1.0,
        language: None,
    });
    let started = Instant::now();
    drop(server);
    assert!(started.elapsed() < Duration::from_secs(3));
    assert_eq!(sse_message(&mut client)["type"], "completed");
    let listener = std::net::TcpListener::bind(address).unwrap();
    drop(listener);
    hub.publish(&Event::Segment(segment("still available")));
    assert_eq!(
        hub.snapshot().data["segments"][0]["text"],
        "still available"
    );
}

#[test]
fn connection_limit_and_port_conflicts_are_reported() {
    let hub = TextStream::new();
    let server = StreamServer::start(0, hub.clone()).unwrap();
    assert!(StreamServer::start(server.address().port(), hub).is_err());
    let clients = (0..32).map(|_| connect_sse(&server)).collect::<Vec<_>>();
    assert_eq!(server.clients(), 32);
    assert!(matches!(
        agent()
            .get(&format!("http://{}/events", server.address()))
            .call(),
        Err(ureq::Error::Status(503, _))
    ));
    drop(clients);
}

#[test]
fn worker_publishes_without_waiting_for_tui_event_consumption() {
    let (tx, _ui_rx) = channel();
    let mut worker = transcribe_stt::transcribe::spawn(tx);
    let hub = worker.text_stream();
    worker.submit_live(transcribe_stt::transcribe::realtime::LiveJob {
        model: "/nonexistent/test.bin".into(),
        language: None,
        input_device: None,
        noise_suppression: Default::default(),
    });
    worker.shutdown();
    let snapshot = hub.snapshot();
    assert_eq!(snapshot.data["source"], "microphone");
    assert_eq!(snapshot.data["model"], "test.bin");
    assert_eq!(snapshot.data["status"], "error");
    assert!(snapshot.session_id.is_some());
}

#[test]
fn tui_toggle_preserves_recording_and_exposes_endpoint_banner() {
    use crossterm::event::{KeyCode, KeyModifiers};
    use ratatui::{backend::TestBackend, Terminal};
    use transcribe_stt::app::{App, WorkState};
    let (tx, _rx) = channel();
    let mut app = App::new(
        std::env::temp_dir(),
        None,
        transcribe_stt::transcribe::spawn(tx),
    );
    app.start_stream_service(0).unwrap();
    app.live.active = true;
    app.work = WorkState::Recording;
    app.transcriber
        .text_stream()
        .publish(&Event::Segment(segment("Retained text")));
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal
        .draw(|frame| transcribe_stt::ui::draw(frame, &app))
        .unwrap();
    let screen = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(screen.contains("Text service 127.0.0.1:"));
    app.on_key(KeyCode::Char('v'), KeyModifiers::NONE);
    assert!(app.stream_server.is_none());
    assert_eq!(app.work, WorkState::Recording);
    assert!(!app
        .transcriber
        .cancel
        .load(std::sync::atomic::Ordering::Relaxed));
    app.start_stream_service(0).unwrap();
    let mut client = connect_sse(app.stream_server.as_ref().unwrap());
    assert_eq!(
        sse_message(&mut client)["data"]["segments"][0]["text"],
        "Retained text"
    );
    app.stream_server.take();
    app.transcriber.shutdown();
}
