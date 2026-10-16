//! The router as logical process `router` of a simulated run.
//!
//! With `ATOM_COMPASS_CLOCK_ENDPOINT` set, the router binds LP `router` at
//! startup and handles the messages the clock releases to it one at a time, in
//! (arrival, channel, seq) order: each request from traffic, each prefill
//! response and each decode stream event. A message waits at its hold point
//! until it is released; its handler then runs with the policies' clock at its
//! arrival, and the next message is released only once it is handled. Each
//! message sends exactly one on, on the outbound channel of its kind (`http`,
//! `relay` or `stream`), stamped at its arrival plus that channel's lookahead,
//! from `ATOM_COMPASS_ROUTER_LOOKAHEADS` (`{"<channel>": seconds}`). The
//! policies draw from the run's seed, `ATOM_COMPASS_SEED`. A periodic task
//! fires on the LP clock instead of a thread ([`periodic`]): its next deadline
//! goes out as the daemon deadline of each request, and it runs before any
//! message arriving at or after that deadline. Any clock error ends the
//! process. Without the endpoint nothing here runs.

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use axum::{extract::Request, http::HeaderValue, middleware::Next, response::Response};
use bytes::Bytes;
use futures_util::{stream::BoxStream, Stream, StreamExt};
use memchr::memmem;
use serde_json::Value;
use tokio::sync::{oneshot, Notify};
use tracing::{error, warn};

use crate::{
    compass_clock::{ClockClient, ClockError, ENDPOINT_ENV},
    core::WorkerLoadGuard,
    determinism,
};

pub const SEED_ENV: &str = "ATOM_COMPASS_SEED";
pub const LOOKAHEADS_ENV: &str = "ATOM_COMPASS_ROUTER_LOOKAHEADS";

/// A released message being handled; dropping it marks it handled.
pub struct Turn {
    /// The stamp of the message it sends on.
    arrival: f64,
    seq: u64,
    _handled: oneshot::Sender<()>,
}

impl Turn {
    fn entry(&self) -> String {
        format!("a:{:?};s:{}", self.arrival, self.seq)
    }
}

struct Lp {
    /// Outbound channel -> lookahead.
    lookaheads: HashMap<String, f64>,
    /// (kind, seq) of each message read and not yet released.
    waiting: parking_lot::Mutex<HashMap<(String, u64), oneshot::Sender<Turn>>>,
    arrived: Notify,
    /// Periodic tasks, in registration order.
    timers: parking_lot::Mutex<Vec<Timer>>,
}

struct Timer {
    due: f64,
    every: f64,
    stopped: Arc<AtomicBool>,
    task: Box<dyn Fn() + Send>,
}

static LP: OnceLock<Lp> = OnceLock::new();

tokio::task_local! {
    static REQUEST: RefCell<Option<Turn>>;
}

fn fail(why: impl std::fmt::Display) -> ! {
    error!("router LP: {why}; ending the run");
    eprintln!("router LP: {why}; ending the run");
    std::process::exit(1)
}

/// Bind LP `router` when the clock endpoint is set; call it before the
/// policies are built.
pub fn start() {
    let Ok(endpoint) = std::env::var(ENDPOINT_ENV) else {
        return;
    };
    let seed = std::env::var(SEED_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| fail(format!("{SEED_ENV} is not an unsigned integer")));
    let lookaheads = std::env::var(LOOKAHEADS_ENV)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| fail(format!("{LOOKAHEADS_ENV} is not a channel map")));
    determinism::enable(seed);
    let lp = LP.get_or_init(|| Lp::new(lookaheads));
    tokio::spawn(async move {
        let result = async {
            let client = ClockClient::bind(&endpoint, "router").await?;
            lp.serve(client).await
        };
        if let Err(error) = result.await {
            fail(error);
        }
    });
}

fn kind(channel: &str) -> &str {
    channel.rsplit(':').next().unwrap_or_default()
}

impl Lp {
    fn new(lookaheads: HashMap<String, f64>) -> Self {
        Self {
            lookaheads,
            waiting: Default::default(),
            arrived: Notify::new(),
            timers: Default::default(),
        }
    }

    /// The earliest deadline of a task still running.
    fn next_due(&self) -> f64 {
        let mut timers = self.timers.lock();
        timers.retain(|t| !t.stopped.load(Ordering::Relaxed));
        timers.iter().map(|t| t.due).fold(f64::INFINITY, f64::min)
    }

    /// Run each task due by `time`, in deadline then registration order, with
    /// the policies' clock at its deadline.
    fn fire(&self, time: f64) {
        while self.next_due() <= time {
            let mut timers = self.timers.lock();
            let first = timers.iter_mut().min_by(|a, b| a.due.total_cmp(&b.due));
            let timer = first.expect("a due task");
            determinism::set_now(timer.due);
            (timer.task)();
            timer.due += timer.every;
        }
    }

    /// Ask for the next event until the `+inf` grant, handling each release.
    async fn serve(&self, mut client: ClockClient) -> Result<(), ClockError> {
        loop {
            let grant = client.next_event(f64::INFINITY, self.next_due()).await?;
            for message in grant.released {
                self.fire(message.arrival);
                let kind = kind(&message.channel);
                let Some((out, lookahead)) = self
                    .lookaheads
                    .iter()
                    .find(|(ch, _)| self::kind(ch) == kind)
                else {
                    fail(format!("no outbound channel for {}", message.channel));
                };
                determinism::set_now(message.arrival);
                let arrival = message.arrival + lookahead;
                let seq = client.send(out, arrival);
                let take = self.taken(kind, message.seq, &message.channel).await;
                let (handled, done) = oneshot::channel();
                if take
                    .send(Turn {
                        arrival,
                        seq,
                        _handled: handled,
                    })
                    .is_ok()
                {
                    let _ = done.await;
                }
            }
            if grant.time == f64::INFINITY {
                return Ok(());
            }
            self.fire(grant.time);
        }
    }

    /// The hold point that read (`kind`, `seq`), once it has.
    async fn taken(&self, kind: &str, seq: u64, channel: &str) -> oneshot::Sender<Turn> {
        let key = (kind.to_owned(), seq);
        loop {
            if let Some(take) = self.waiting.lock().remove(&key) {
                return take;
            }
            let wait = tokio::time::timeout(Duration::from_secs(30), self.arrived.notified());
            if wait.await.is_err() {
                warn!("router LP: released {channel} seq {seq} has not reached the router");
            }
        }
    }

    /// Hold a message read on the `kind` channel until it is released.
    async fn hold(&self, kind: &str, seq: u64) -> Turn {
        let (take, turn) = oneshot::channel();
        let key = (kind.to_owned(), seq);
        if self.waiting.lock().insert(key, take).is_some() {
            fail(format!("{kind} seq {seq} arrived twice"));
        }
        self.arrived.notify_one();
        turn.await
            .unwrap_or_else(|_| fail("the clock loop ended with a message held"))
    }
}

/// The (arrival, seq) of a stamp written `a:<arrival>;s:<seq>` or
/// `a=<arrival> s=<seq>`.
fn stamp(text: &str) -> (f64, u64) {
    let parsed = text
        .strip_prefix("a")
        .and_then(|t| t.get(1..))
        .and_then(|t| t.split_once([';', ' ']))
        .and_then(|(a, s)| Some((a.parse().ok()?, s.get(2..)?.parse().ok()?)));
    parsed.unwrap_or_else(|| fail(format!("malformed compass stamp {text:?}")))
}

/// Ingress layer: a stamped request waits until it is released, then goes on
/// with its `tracestate` entry restamped for prefill. It is handled once the
/// router has placed it (`placed`).
pub async fn ingress(mut request: Request, next: Next) -> Response {
    let lp = LP.get();
    let state = request.headers().get("tracestate");
    let Some((lp, header)) = lp.zip(state.and_then(|h| h.to_str().ok()).map(str::to_owned)) else {
        return next.run(request).await;
    };
    let mut entries: Vec<String> = header.split(',').map(str::to_owned).collect();
    let Some(entry) = entries
        .iter_mut()
        .find(|e| e.trim().starts_with("compass="))
    else {
        return next.run(request).await;
    };
    let turn = lp
        .hold("http", stamp(&entry.trim()["compass=".len()..]).1)
        .await;
    *entry = format!("compass={}", turn.entry());
    let restamped = HeaderValue::from_str(&entries.join(",")).expect("an ASCII tracestate");
    request.headers_mut().insert("tracestate", restamped);
    REQUEST
        .scope(RefCell::new(Some(turn)), next.run(request))
        .await
}

/// The request being handled has its workers and their load: it is handled.
pub fn placed() {
    let _ = REQUEST.try_with(|turn| turn.borrow_mut().take());
}

/// Run `task` every `interval_secs` of the LP clock, from now, until `stopped`
/// is set. Without the router LP, hands `task` back.
pub fn periodic<F: Fn() + Send + 'static>(
    interval_secs: u64,
    stopped: &Arc<AtomicBool>,
    task: F,
) -> Result<(), F> {
    let Some(lp) = LP.get() else {
        return Err(task);
    };
    let every = interval_secs as f64;
    lp.timers.lock().push(Timer {
        due: determinism::seconds() + every,
        every,
        stopped: stopped.clone(),
        task: Box::new(task),
    });
    Ok(())
}

/// Hold a prefill response until it is released and restamp its relay entry
/// for decode; it is handled when the returned turn drops.
pub async fn relay(body: &mut Value) -> Option<Turn> {
    let lp = LP.get()?;
    let entry = body.get_mut("kv_transfer_params")?.get_mut("compass")?;
    let seq = stamp(entry.as_str().unwrap_or_default()).1;
    let turn = lp.hold("relay", seq).await;
    *entry = Value::String(turn.entry());
    Some(turn)
}

/// The decode stream, each event held until it is released and restamped for
/// traffic. The terminal event drops `guard`, the decode load, before it
/// counts as handled. Without the clock: the stream as is, and the guard.
pub fn stream<S>(
    upstream: S,
    guard: WorkerLoadGuard,
) -> (
    BoxStream<'static, Result<Bytes, reqwest::Error>>,
    Option<WorkerLoadGuard>,
)
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let Some(lp) = LP.get() else {
        return (upstream.boxed(), Some(guard));
    };
    let events = futures_util::stream::unfold(
        (upstream.boxed(), Vec::new(), Some(guard)),
        move |(mut upstream, mut buf, mut guard)| async move {
            loop {
                if let Some(end) = memmem::find(&buf, b"\n\n") {
                    let event: Vec<u8> = buf.drain(..end + 2).collect();
                    let event = lp.restamp(event, &mut guard).await;
                    return Some((Ok(Bytes::from(event)), (upstream, buf, guard)));
                }
                match upstream.next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(error)) => return Some((Err(error), (upstream, buf, guard))),
                    None if buf.is_empty() => return None,
                    None => {
                        let rest = Bytes::from(std::mem::take(&mut buf));
                        return Some((Ok(rest), (upstream, buf, guard)));
                    }
                }
            }
        },
    );
    (events.boxed(), None)
}

impl Lp {
    async fn restamp(&self, event: Vec<u8>, guard: &mut Option<WorkerLoadGuard>) -> Vec<u8> {
        let line = event.split(|b| *b == b'\n').next().unwrap_or_default();
        let Some(text) = line.strip_prefix(b": compass ") else {
            return event;
        };
        let seq = stamp(&String::from_utf8_lossy(text)).1;
        let turn = self.hold("stream", seq).await;
        let mut out = format!(": compass a={:?} s={}", turn.arrival, turn.seq).into_bytes();
        out.extend_from_slice(&event[line.len()..]);
        if memmem::find(&out, b"data: [DONE]").is_some() {
            guard.take();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::stamp;

    #[test]
    fn reads_both_stamp_spellings() {
        assert_eq!(stamp("a:0.000244140625;s:17"), (0.000244140625, 17));
        assert_eq!(stamp("a=1e-07 s=3"), (1e-7, 3));
        assert_eq!(stamp("a=inf s=0"), (f64::INFINITY, 0));
    }
}
