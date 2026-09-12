//! Optional loopback SSE/WebSocket service. Its Tokio runtime lives on a
//! dedicated thread; capture, inference, and the terminal remain synchronous.
mod http;
mod hub;

pub use hub::{Message, Snapshot, TextStream};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::{watch, Semaphore};

pub const DEFAULT_PORT: u16 = 8765;
pub(crate) const MAX_CLIENTS: usize = 32;

pub struct StreamServer {
    address: SocketAddr,
    stop: watch::Sender<bool>,
    slots: Arc<Semaphore>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StreamServer {
    pub fn start(port: u16, stream: TextStream) -> Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).with_context(|| {
            format!("Cannot start text service on 127.0.0.1:{port}; the port may already be in use")
        })?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let listener = {
            let _entered = runtime.enter();
            tokio::net::TcpListener::from_std(listener)?
        };
        let (stop, shutdown) = watch::channel(false);
        let slots = Arc::new(Semaphore::new(MAX_CLIENTS));
        let state = http::HttpState {
            stream,
            shutdown: shutdown.clone(),
            slots: slots.clone(),
            port: address.port(),
        };
        let thread = std::thread::Builder::new().name("text-stream-server".into()).spawn(move || {
            runtime.block_on(async move {
                let graceful = shutdown.clone();
                let serving = axum::serve(listener, http::router(state))
                    .with_graceful_shutdown(wait_shutdown(graceful));
                let serving = std::future::IntoFuture::into_future(serving);
                tokio::pin!(serving);
                tokio::select! {
                    result = &mut serving => { if let Err(error) = result { log::error!("text stream server: {error}"); } }
                    _ = wait_shutdown(shutdown) => {
                        // Slow sockets must not prevent quitting the TUI.
                        let _ = tokio::time::timeout(Duration::from_secs(2), &mut serving).await;
                    }
                }
            });
        })?;
        Ok(Self {
            address,
            stop,
            slots,
            thread: Some(thread),
        })
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }
    pub fn clients(&self) -> usize {
        MAX_CLIENTS - self.slots.available_permits()
    }
}

impl Drop for StreamServer {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(crate) async fn wait_shutdown(mut shutdown: watch::Receiver<bool>) {
    if !*shutdown.borrow() {
        let _ = shutdown.changed().await;
    }
}
