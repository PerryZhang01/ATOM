//! The router as logical process `router` of a simulated run.
//!
//! With `ATOM_COMPASS_CLOCK_ENDPOINT` set, the router binds LP `router` at
//! startup and handles the messages the clock releases to it one at a time, in
//! (arrival, channel, seq) order: each request from traffic, each prefill
//! response and each decode stream event. A message waits at its hold point
//! until it is released; its handler then runs with the policies' clock at its
//! arrival, and the next message is released only once it is handled. Each
//! message sends exactly one on, on the outbound channel of its kind (`http`,
//! `relay` or `stream`) to the frontend it goes to, stamped at its arrival plus
//! that channel's lookahead, from `ATOM_COMPASS_ROUTER_LOOKAHEADS`
//! (`{"<channel>": seconds}`); a request is sent on once it is placed. Where
//! several frontends take one kind, `ATOM_COMPASS_ROUTER_WORKERS`
//! (`{"<worker base url>": "<frontend LP>"}`) names each worker's. The
//! policies draw from the run's seed, `ATOM_COMPASS_SEED`. Any clock error
//! ends the process. Without the endpoint nothing here runs.

use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    sync::OnceLock,
    time::Duration,
};

use axum::{extract::Request, middleware::Next, response::Response};
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
pub const WORKERS_ENV: &str = "ATOM_COMPASS_ROUTER_WORKERS";

/// A released message being handled. It holds the clock client, for the one
/// message it sends on; dropping it hands the client back and marks it handled.
pub struct Turn {
    arrival: f64,
    client: Option<ClockClient>,
    back: Option<oneshot::Sender<ClockClient>>,
}

impl Turn {
    /// Sends this message on, on `channel`, at its arrival plus that channel's
    /// lookahead; the stamp it goes with.
    fn send(&mut self, channel: &str) -> (f64, u64) {
        let lookahead = LP.get().and_then(|lp| lp.lookaheads.get(channel));
        let Some(lookahead) = lookahead else {
            fail(format!("no outbound channel {channel}"));
        };
        let arrival = self.arrival + lookahead;
        let client = self.client.as_mut().expect("a turn holds the clock");
        (arrival, client.send(channel, arrival))
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        if let (Some(client), Some(back)) = (self.client.take(), self.back.take()) {
            let _ = back.send(client);
        }
    }
}

struct Lp {
    /// Outbound channel -> lookahead.
    lookaheads: HashMap<String, f64>,
    /// Worker base URL -> its frontend LP.
    workers: BTreeMap<String, String>,
    /// (channel, seq) of each message read and not yet released.
    waiting: parking_lot::Mutex<HashMap<(String, u64), oneshot::Sender<Turn>>>,
    arrived: Notify,
}

static LP: OnceLock<Lp> = OnceLock::new();

/// The request being handled: its turn until it is placed, then its prefill
/// stamp and its (prefill, decode) frontends.
#[derive(Default)]
struct Handling {
    turn: Option<Turn>,
    stamp: Option<String>,
    pair: Option<(String, String)>,
}

tokio::task_local! {
    static REQUEST: RefCell<Handling>;
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
    let workers = std::env::var(WORKERS_ENV)
        .map_or_else(|_| Some(BTreeMap::new()), |s| serde_json::from_str(&s).ok());
    let workers = workers.unwrap_or_else(|| fail(format!("{WORKERS_ENV} is not a worker map")));
    determinism::enable(seed);
    let lp = LP.get_or_init(|| Lp::new(lookaheads, workers));
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

fn entry((arrival, seq): (f64, u64)) -> String {
    format!("a:{arrival:?};s:{seq}")
}

impl Lp {
    fn new(lookaheads: HashMap<String, f64>, workers: BTreeMap<String, String>) -> Self {
        Self {
            lookaheads,
            workers,
            waiting: Default::default(),
            arrived: Notify::new(),
        }
    }

    /// Ask for the next event until the `+inf` grant, handing each release
    /// the client until it is handled.
    async fn serve(&self, mut client: ClockClient) -> Result<(), ClockError> {
        loop {
            let grant = client.next_event(f64::INFINITY, f64::INFINITY).await?;
            for message in grant.released {
                determinism::set_now(message.arrival);
                let take = self.taken(&message.channel, message.seq).await;
                let (back, done) = oneshot::channel();
                let _ = take.send(Turn {
                    arrival: message.arrival,
                    client: Some(client),
                    back: Some(back),
                });
                client = done
                    .await
                    .unwrap_or_else(|_| fail("a turn ended holding the clock"));
            }
            if grant.time == f64::INFINITY {
                return Ok(());
            }
        }
    }

    /// The frontend LP of the worker at base URL `url`, among those the
    /// outbound `kind` channels reach: the one `WORKERS_ENV` names, or the only one.
    fn frontend(&self, url: Option<&str>, kind: &str) -> String {
        if let Some(lp) = url.and_then(|u| self.workers.get(u)) {
            return lp.clone();
        }
        let reached: Vec<&str> = (self.lookaheads.keys())
            .filter(|ch| self::kind(ch) == kind)
            .filter_map(|ch| ch.strip_prefix("router->")?.split(':').next())
            .collect();
        match reached[..] {
            [only] => only.to_owned(),
            _ => fail(format!(
                "{WORKERS_ENV} names no frontend LP for worker {url:?}"
            )),
        }
    }

    /// The hold point that read (`channel`, `seq`), once it has.
    async fn taken(&self, channel: &str, seq: u64) -> oneshot::Sender<Turn> {
        let key = (channel.to_owned(), seq);
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

    /// Hold a message read on `channel` until it is released.
    async fn hold(&self, channel: &str, seq: u64) -> Turn {
        let (take, turn) = oneshot::channel();
        let key = (channel.to_owned(), seq);
        if self.waiting.lock().insert(key, take).is_some() {
            fail(format!("{channel} seq {seq} arrived twice"));
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

/// Ingress layer: a stamped request waits until it is released, and is
/// handled once the router has placed it (`placed`).
pub async fn ingress(request: Request, next: Next) -> Response {
    let header = request.headers().get("tracestate");
    let entry = header
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.split(',').find_map(|e| e.trim().strip_prefix("compass=")));
    let Some((lp, seq)) = LP.get().zip(entry.map(|e| stamp(e).1)) else {
        return next.run(request).await;
    };
    let turn = lp.hold("traffic->router:http", seq).await;
    let handling = Handling {
        turn: Some(turn),
        ..Default::default()
    };
    REQUEST
        .scope(RefCell::new(handling), next.run(request))
        .await
}

/// The request being handled holds the load of the workers at base URLs
/// `prefill` and `decode`: it goes on to that prefill, and is handled.
pub fn placed(prefill: &str, decode: &str) {
    let Some(lp) = LP.get() else {
        return;
    };
    let _ = REQUEST.try_with(|handling| {
        let mut handling = handling.borrow_mut();
        let Some(mut turn) = handling.turn.take() else {
            return;
        };
        let p = lp.frontend(Some(prefill), "http");
        handling.stamp = Some(entry(turn.send(&format!("router->{p}:http"))));
        handling.pair = Some((p, lp.frontend(Some(decode), "relay")));
    });
}

/// Request header `name`'s `value` as it goes to a worker: a `tracestate`
/// with its `compass` entry restamped for the prefill the request was placed on.
pub fn header(name: &str, value: &str) -> String {
    let stamp = (name == "tracestate")
        .then(|| {
            REQUEST
                .try_with(|h| h.borrow().stamp.clone())
                .ok()
                .flatten()
        })
        .flatten();
    let Some(stamp) = stamp else {
        return value.to_owned();
    };
    let entries = value
        .split(',')
        .map(|e| match e.trim().starts_with("compass=") {
            true => format!("compass={stamp}"),
            false => e.to_owned(),
        });
    entries.collect::<Vec<_>>().join(",")
}

/// Hold a prefill response until it is released and restamp its relay entry
/// for the decode the request was placed on; it is handled when the returned
/// turn drops.
pub async fn relay(body: &mut Value) -> Option<Turn> {
    let lp = LP.get()?;
    let entry = body.get_mut("kv_transfer_params")?.get_mut("compass")?;
    let seq = stamp(entry.as_str().unwrap_or_default()).1;
    let pair = REQUEST.try_with(|h| h.borrow().pair.clone()).ok().flatten();
    let (p, d) = pair.unwrap_or_else(|| (lp.frontend(None, "http"), lp.frontend(None, "relay")));
    let mut turn = lp.hold(&format!("{p}->router:relay"), seq).await;
    *entry = Value::String(self::entry(turn.send(&format!("router->{d}:relay"))));
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
    let decode = lp.frontend(Some(guard.worker().base_url()), "relay");
    let from = format!("{decode}->router:stream");
    let events = futures_util::stream::unfold(
        (upstream.boxed(), Vec::new(), Some(guard)),
        move |(mut upstream, mut buf, mut guard)| {
            let from = from.clone();
            async move {
                loop {
                    if let Some(end) = memmem::find(&buf, b"\n\n") {
                        let event: Vec<u8> = buf.drain(..end + 2).collect();
                        let event = lp.restamp(event, &mut guard, &from).await;
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
            }
        },
    );
    (events.boxed(), None)
}

impl Lp {
    async fn restamp(
        &self,
        event: Vec<u8>,
        guard: &mut Option<WorkerLoadGuard>,
        from: &str,
    ) -> Vec<u8> {
        let line = event.split(|b| *b == b'\n').next().unwrap_or_default();
        let Some(text) = line.strip_prefix(b": compass ") else {
            return event;
        };
        let seq = stamp(&String::from_utf8_lossy(text)).1;
        let mut turn = self.hold(from, seq).await;
        let (arrival, seq) = turn.send("router->traffic:stream");
        let mut out = format!(": compass a={arrival:?} s={seq}").into_bytes();
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
