# Local text stream service

`transcribe-stt` can broadcast the same transcript to several local applications
while its TUI records or transcribes a file. No extra model instance is loaded.

## Start it

```sh
# Microphone or Bluetooth input, using your existing models:
./scripts/run-realtime.sh /Volumes/MikeExternal/ai_models/speech-to-text-realtime --serve

# The assembled executable supports the same option:
./binary/transcribe-stt --serve --realtime --models-dir /path/to/models

# Another port; the equals sign keeps a file argument unambiguous:
./binary/transcribe-stt --serve=9000

# Also works during a headless file job:
./binary/transcribe-stt --headless --serve -m /path/to/model recording.wav
```

In the TUI, **v** starts or stops the text service on port **8765**. Turning it
off leaves transcription running. The footer shows the address and connected
client count. Enabling it partway through a recording includes the current
transcript in the first snapshot. **R** controls recording as before, and **a**
selects the microphone. The service exits when the app exits; a headless file
job ends its service after completing that file. This option does not install
an OS background daemon or start recording on behalf of a subscriber.

## Endpoints

All endpoints bind to **127.0.0.1**, so they are for applications on this Mac.

| Transport | URL | Purpose |
| --- | --- | --- |
| SSE | `http://127.0.0.1:8765/events` | Continuous transcript events; `/sse` is an alias |
| WebSocket | `ws://127.0.0.1:8765/ws` | The same events as individual JSON messages |
| HTTP GET | `http://127.0.0.1:8765/transcript` | Snapshot of the current/most recent session |
| HTTP GET | `http://127.0.0.1:8765/health` | Service status and connected client count |
| HTTP GET | `http://127.0.0.1:8765/` | Endpoint discovery |

Quick check from another terminal:

```sh
curl -N http://127.0.0.1:8765/events
curl http://127.0.0.1:8765/transcript
```

Native applications can use any HTTP or WebSocket client. Browser apps served
from a loopback origin, such as `http://localhost:3000`, receive CORS headers.
Unrelated website origins, `file://`/`null` origins, and non-loopback Host
headers are rejected. Serve browser examples from a local HTTP server instead
of opening an HTML file directly. There is no LAN binding or remote control
API; subscribers cannot start recording, choose models, or submit audio.

## Message contract (version 1)

SSE frames carry an `id:`, a named `event:`, and `data:` containing the same JSON
envelope sent by WebSocket:

```json
{
  "version": 1,
  "id": "server-instance:42",
  "session_id": "recording-session-id",
  "timestamp_ms": 1789212345678,
  "type": "partial",
  "data": {
    "start_ms": 120,
    "end_ms": 1600,
    "text": "Hola, ¿qué tal?",
    "speaker": null,
    "text_truncated": false
  }
}
```

`id` is unique within a publisher instance and increases across sessions. A
snapshot uses the current event ID rather than consuming a new one. The
`session_id` changes when a new microphone or file job starts and is `null`
before the first job. Audio timestamps are relative to that session;
`timestamp_ms` is the server's publication time. Reconnect snapshots have a
new publication timestamp but retain the latest event ID.

| Event | Client action |
| --- | --- |
| `snapshot` | Replace your entire local state with `data` |
| `session_started` | Replace state; a new session has started, with empty text |
| `partial` | **Replace** the current draft with `data.text`; empty text clears it |
| `segment` | Append this committed segment and clear the draft |
| `segments_replaced` | Replace all committed segments; for example, diarization has labeled speakers |
| `recording_started` | Show `data.device` and `data.mode`; state is recording |
| `recording_stopped` | Microphone is stopped; queued speech is still being finalized |
| `status` | Update status (`loading_model`, `decoding`, or `transcribing`) |
| `completed` | Clear the draft; read `audio_secs`, `elapsed_secs`, and `language` |
| `cancelled` | Session ended early; keep the already received text |
| `error` | Show `data.message`; keep received text |

Partials are cumulative replacements, **not text deltas**. Do not append every
partial: doing so duplicates words. Commit a phrase only on `segment`.
Speaker values are numeric indices or `null`. Native live timestamps are
approximate, as they are in the TUI. Ignore unknown fields for forward
compatibility. Waveform samples, percentage progress and engine logs are not
broadcast.

## SSE browser client

```js
let segments = [];
let partial = null;

function receive(message) {
  switch (message.type) {
    case "snapshot":
    case "session_started":
      segments = message.data.segments;
      partial = message.data.partial;
      break;
    case "partial":
      partial = message.data.text ? message.data : null;
      break;
    case "segment":
      segments.push(message.data);
      partial = null;
      break;
    case "segments_replaced":
      segments = message.data.segments;
      partial = null;
      break;
    case "completed":
      partial = null;
      break;
  }
  const committedText = segments.map(s => s.text).join(" ");
  console.log({committedText, draft: partial?.text ?? ""});
}

const events = new EventSource("http://127.0.0.1:8765/events");
for (const type of ["snapshot", "session_started", "partial", "segment",
                    "segments_replaced", "completed"]) {
  events.addEventListener(type, event => receive(JSON.parse(event.data)));
}
// Later: events.close();
```

For WebSocket, reuse the same `receive` function:

```js
const socket = new WebSocket("ws://127.0.0.1:8765/ws");
socket.onmessage = event => receive(JSON.parse(event.data));
// Later: socket.close();
```

WebSocket clients should reconnect after an unexpected close; each new
connection starts with a fresh snapshot. WebSocket messages sent by a client
are ignored except for connection control frames.

## Delivery and limits

- Updates are published directly from the worker's event forwarder, before the
  TUI receives them, avoiding the terminal loop's roughly 100 ms polling delay.
- Every new or reconnected subscriber gets an atomic snapshot plus subsequent
  events. A new recording resets the state, preventing sessions from mixing.
- SSE's `Last-Event-ID` is accepted but **does not replay a historical event
  log**. Reconnection always sends a replacement snapshot. This restores the
  retained current session without duplicating committed text. Sessions that
  finished while a client was disconnected are not archived by the service.
- Each subscriber has a 256-event broadcast buffer. A client that falls behind
  gets a new replacement snapshot instead of silently continuing with gaps.
  Slow WebSocket writes time out after two seconds. Inference never waits for
  a subscriber's network writes.
- Snapshots keep the newest 2,048 segments, limited to 1 MiB of committed text.
  Individual text fields are capped at 64 KiB without splitting UTF-8 characters;
  clipped updates carry `text_truncated: true`. The snapshot's `truncated` flag
  tells consumers when text has been removed or clipped. Full TUI/export text is
  unaffected by these service limits.
- There are at most 32 simultaneous SSE/WebSocket clients. Further stream
  connections return HTTP 503; snapshot and health requests still work.
- SSE sends keep-alive comments every ten seconds; WebSocket sends ping frames
  every fifteen seconds. Quitting closes subscribers and releases the port.

## Which transport to choose

SSE is the simplest choice for receiving live text: it uses HTTP and browsers
reconnect automatically. WebSocket is appropriate when an application already
has a WebSocket client, or a future protocol needs two-way messages. This
service exposes the same receive-only contract on both.

For a tightly coupled native producer and consumer on one machine, a Unix
domain socket or a child-process stdout pipe can avoid HTTP/WebSocket framing.
Those are possible future transports, not implemented here. Benchmark the full
ASR-to-consumer path before adding another transport; the current server sends
updates as they arrive and does not wait for the TUI.

References: [Axum SSE](https://docs.rs/axum/0.8.9/axum/response/sse/index.html),
[Axum WebSocket](https://docs.rs/axum/0.8.9/axum/extract/ws/index.html),
[Tokio broadcast behavior](https://docs.rs/tokio/1.53.1/tokio/sync/broadcast/index.html),
[Unix domain sockets](https://man7.org/linux/man-pages/man7/unix.7.html).

## Verification

```sh
cargo test --lib
cargo test --test stream_service  # needs permission to bind loopback sockets
cargo clippy --all-targets -- -D warnings
```

Integration tests exercise actual SSE and WebSocket connections, matching event
IDs and JSON, cumulative drafts, completed segments, session boundaries,
reconnect snapshots, CORS/Host checks, connection limits, port conflicts, clean
shutdown, worker publication independent of the UI, and the TUI service toggle.
