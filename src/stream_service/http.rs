use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, Method, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures_util::Stream;
use serde_json::json;
use tokio::sync::{broadcast, watch, OwnedSemaphorePermit, Semaphore};

use super::hub::Subscription;
use super::{wait_shutdown, Message, TextStream};

#[derive(Clone)]
pub(super) struct HttpState {
    pub stream: TextStream,
    pub shutdown: watch::Receiver<bool>,
    pub slots: Arc<Semaphore>,
    pub port: u16,
}

pub(super) fn router(state: HttpState) -> Router {
    Router::new()
        .route("/", get(info))
        .route("/health", get(health))
        .route("/transcript", get(snapshot))
        .route("/events", get(sse))
        .route("/sse", get(sse))
        .route("/ws", get(ws))
        .layer(middleware::from_fn_with_state(state.clone(), local_clients))
        .with_state(state)
}

async fn info() -> Json<serde_json::Value> {
    Json(
        json!({"name": "transcribe-stt", "version": 1, "sse": "/events", "websocket": "/ws", "snapshot": "/transcript", "read_only": true}),
    )
}

async fn health(State(state): State<HttpState>) -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "clients": super::MAX_CLIENTS-state.slots.available_permits()}))
}

async fn snapshot(State(state): State<HttpState>) -> Json<serde_json::Value> {
    Json(json!(&*state.stream.snapshot()))
}

fn permit(state: &HttpState) -> Result<OwnedSemaphorePermit, StatusCode> {
    state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
}

async fn sse(State(state): State<HttpState>) -> Result<Response, StatusCode> {
    let permit = permit(&state)?;
    // On every connect/reconnect the consumer replaces its local state with
    // this snapshot. Last-Event-ID is deliberately not a replay promise.
    let Subscription {
        initial,
        mut receiver,
    } = state.stream.subscribe();
    let output = async_stream::stream! {
        let _permit = permit;
        yield Ok::<_, Infallible>(sse_event(&initial));
        loop {
            tokio::select! {
                // Drain queued completion events before shutting connections.
                biased;
                received = receiver.recv() => match received {
                    Ok(message) => yield Ok(sse_event(&message)),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        let replacement = state.stream.subscribe();
                        receiver = replacement.receiver;
                        yield Ok(sse_event(&replacement.initial));
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                _ = wait_shutdown(state.shutdown.clone()) => break,
            }
        }
    };
    Ok(sse_response(output))
}

fn sse_event(message: &Message) -> SseEvent {
    SseEvent::default()
        .id(&message.id)
        .event(message.kind)
        .json_data(message)
        .expect("stream messages are JSON values")
}

fn sse_response(
    output: impl Stream<Item = Result<SseEvent, Infallible>> + Send + 'static,
) -> Response {
    let mut response = Sse::new(output)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(10))
                .text("keep-alive"),
        )
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

async fn ws(
    State(state): State<HttpState>,
    upgrade: WebSocketUpgrade,
) -> Result<Response, StatusCode> {
    let permit = permit(&state)?;
    Ok(upgrade
        .max_message_size(1024)
        .max_frame_size(1024)
        .on_upgrade(move |socket| websocket(socket, state, permit)))
}

async fn send(socket: &mut WebSocket, message: WsMessage) -> bool {
    matches!(
        tokio::time::timeout(Duration::from_secs(2), socket.send(message)).await,
        Ok(Ok(()))
    )
}

async fn send_json(socket: &mut WebSocket, message: &Message) -> bool {
    let Ok(json) = serde_json::to_string(message) else {
        return false;
    };
    send(socket, WsMessage::Text(json.into())).await
}

async fn websocket(mut socket: WebSocket, state: HttpState, _permit: OwnedSemaphorePermit) {
    let Subscription {
        initial,
        mut receiver,
    } = state.stream.subscribe();
    if !send_json(&mut socket, &initial).await {
        return;
    }
    let mut heartbeat = tokio::time::interval(Duration::from_secs(15));
    heartbeat.tick().await;
    loop {
        tokio::select! {
            biased;
            received = receiver.recv() => match received {
                Ok(message) => { if !send_json(&mut socket, &message).await { break; } }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let replacement = state.stream.subscribe();
                    receiver = replacement.receiver;
                    if !send_json(&mut socket, &replacement.initial).await { break; }
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
            _ = wait_shutdown(state.shutdown.clone()) => break,
            incoming = socket.recv() => match incoming {
                Some(Ok(WsMessage::Close(_))) | Some(Err(_)) | None => break,
                Some(Ok(WsMessage::Ping(data))) => {
                    let sent = send(&mut socket, WsMessage::Pong(data)).await;
                    if !sent { break; }
                }
                _ => {} // Receive-only service: clients cannot trigger recording or jobs.
            },
            _ = heartbeat.tick() => {
                if !send(&mut socket, WsMessage::Ping(Vec::new().into())).await { break; }
            }
        }
    }
    let _ = send(&mut socket, WsMessage::Close(None)).await;
}

fn local_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

fn allowed_origin(origin: &str) -> bool {
    let Ok(uri) = origin.parse::<Uri>() else {
        return false;
    };
    matches!(uri.scheme_str(), Some("http" | "https"))
        && uri.host().is_some_and(local_host)
        && uri.authority().is_some_and(|a| !a.as_str().contains('@'))
        && uri.path_and_query().is_none_or(|path| path.as_str() == "/")
}

/// Native clients use loopback directly. Local web apps may read the service
/// using CORS; unrelated website origins and rebinding hostnames are rejected.
async fn local_clients(State(state): State<HttpState>, request: Request, next: Next) -> Response {
    let host_ok = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .and_then(|host| host.parse::<axum::http::uri::Authority>().ok())
        .is_some_and(|host| local_host(host.host()) && host.port_u16().unwrap_or(80) == state.port);
    if !host_ok {
        return StatusCode::FORBIDDEN.into_response();
    }
    let origin = request.headers().get(header::ORIGIN).cloned();
    if origin
        .as_ref()
        .is_some_and(|value| !value.to_str().is_ok_and(allowed_origin))
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let mut response = if request.method() == Method::OPTIONS {
        StatusCode::NO_CONTENT.into_response()
    } else {
        next.run(request).await
    };
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    if let Some(origin) = origin {
        let headers = response.headers_mut();
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        headers.insert(header::VARY, HeaderValue::from_static("Origin"));
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, OPTIONS"),
        );
        headers.insert(
            header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("Accept, Last-Event-ID"),
        );
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn origins_allow_local_apps_but_not_unrelated_websites() {
        for origin in [
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://[::1]:8000",
        ] {
            assert!(allowed_origin(origin), "{origin}");
        }
        for origin in [
            "https://example.com",
            "http://localhost.evil:8765",
            "null",
            "file:///tmp/client.html",
            "http://localhost:3000/path",
            "http://evil@localhost:3000",
        ] {
            assert!(!allowed_origin(origin), "{origin}");
        }
    }
}
