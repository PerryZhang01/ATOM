//! A client of the Compass clock authority, for an LP in this process.
//!
//! The wire is the authority's TCP carrier: each frame is its length as four
//! big-endian bytes, then a JSON object with sorted keys. The first frame binds
//! the connection to an LP and is answered `JOINED`; each later request, an
//! `NER`, carries the sends logged since the previous one and is answered
//! `GRANT`, with the messages it releases, or `REFUSED`. Times are seconds, and
//! the two infinities travel as the strings `+inf` and `-inf`.

use std::collections::HashMap;

use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

/// The variable naming the authority's endpoint, `tcp://<host>:<port>`.
pub const ENDPOINT_ENV: &str = "ATOM_COMPASS_CLOCK_ENDPOINT";
/// A frame longer than this is a peer not speaking the protocol.
const LARGEST_FRAME: u32 = 1 << 22;

#[derive(Debug, thiserror::Error)]
pub enum ClockError {
    #[error("clock connection: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a clock message: {0}")]
    Malformed(String),
    #[error("the clock refused the request with {error}: {reason}")]
    Refused { error: String, reason: String },
}

/// A message the authority released to this LP.
#[derive(Debug, Clone, PartialEq)]
pub struct Released {
    pub arrival: f64,
    pub channel: String,
    pub seq: u64,
}

/// The authority's answer to a request.
#[derive(Debug, Clone, PartialEq)]
pub struct Grant {
    /// The time the LP may advance to; `+inf` ends the run.
    pub time: f64,
    /// In (arrival, channel, seq) order, the order the LP handles them in.
    pub released: Vec<Released>,
}

/// One LP's bound connection to the authority.
pub struct ClockClient {
    stream: TcpStream,
    next_seq: HashMap<String, u64>,
    log: Vec<(String, u64, f64)>,
}

impl ClockClient {
    /// Connects to the authority at `tcp://<host>:<port>` and binds `lp`.
    pub async fn bind(endpoint: &str, lp: &str) -> Result<Self, ClockError> {
        let address = endpoint.strip_prefix("tcp://").ok_or_else(|| {
            ClockError::Malformed(format!("{endpoint:?} is not tcp://<host>:<port>"))
        })?;
        let mut client = Self {
            stream: TcpStream::connect(address).await?,
            next_seq: HashMap::new(),
            log: Vec::new(),
        };
        client.write(&json!({"kind": "BIND", "lp": lp})).await?;
        let joined = client.reply("JOINED").await?;
        let seqs = joined["seqs"].as_object().into_iter().flatten();
        client.next_seq = seqs
            .map(|(ch, seq)| Ok((ch.clone(), seq_in(seq)?)))
            .collect::<Result<_, ClockError>>()?;
        Ok(client)
    }

    /// Logs one message sent on `channel`, arriving at `arrival`; it goes to
    /// the authority with the next request. Returns its sequence number.
    pub fn send(&mut self, channel: &str, arrival: f64) -> u64 {
        let next = self.next_seq.entry(channel.to_owned()).or_insert(0);
        let seq = *next;
        *next += 1;
        self.log.push((channel.to_owned(), seq, arrival));
        seq
    }

    /// Idles until `t`, the daemon deadline `t_daemon` or the earliest arrival.
    pub async fn next_event(&mut self, t: f64, t_daemon: f64) -> Result<Grant, ClockError> {
        let (t, t_daemon) = (time_out(t)?, time_out(t_daemon)?);
        let log = std::mem::take(&mut self.log)
            .into_iter()
            .map(|(ch, seq, arrival)| Ok(json!([ch, seq, time_out(arrival)?])))
            .collect::<Result<Vec<_>, ClockError>>()?;
        let frame = json!({"kind": "NER", "log": log, "t": t, "t_daemon": t_daemon});
        self.write(&frame).await?;
        let grant = self.reply("GRANT").await?;
        let mut released = Vec::new();
        for (channel, pairs) in grant["released"].as_object().into_iter().flatten() {
            for pair in pairs.as_array().into_iter().flatten() {
                released.push(Released {
                    arrival: time_in(&pair[1])?,
                    channel: channel.clone(),
                    seq: seq_in(&pair[0])?,
                });
            }
        }
        released.sort_by(|a, b| {
            (a.arrival.total_cmp(&b.arrival))
                .then_with(|| a.channel.cmp(&b.channel))
                .then(a.seq.cmp(&b.seq))
        });
        Ok(Grant {
            time: time_in(&grant["G"])?,
            released,
        })
    }

    async fn write(&mut self, message: &Value) -> Result<(), ClockError> {
        let body = message.to_string().into_bytes();
        let mut frame = (body.len() as u32).to_be_bytes().to_vec();
        frame.extend(body);
        Ok(self.stream.write_all(&frame).await?)
    }

    /// The next frame, which must be `kind`; a refusal is an error.
    async fn reply(&mut self, kind: &str) -> Result<Value, ClockError> {
        let size = self.stream.read_u32().await?;
        if size > LARGEST_FRAME {
            return Err(ClockError::Malformed(format!("a {size}-byte frame")));
        }
        let mut body = vec![0; size as usize];
        self.stream.read_exact(&mut body).await?;
        // serde_json may read a 17-digit number as a neighbouring f64, and a time
        // must come back exactly, so numbers are read as text.
        let message: Value = serde_json::from_slice(&quote_numbers(&body)).map_err(|e| {
            ClockError::Malformed(format!("{e}: {}", String::from_utf8_lossy(&body)))
        })?;
        match message["kind"].as_str() {
            Some(k) if k == kind => Ok(message),
            Some("REFUSED") => Err(ClockError::Refused {
                error: message["error"].as_str().unwrap_or_default().to_owned(),
                reason: message["reason"].as_str().unwrap_or_default().to_owned(),
            }),
            _ => Err(ClockError::Malformed(format!(
                "expected {kind}, got {message}"
            ))),
        }
    }
}

fn time_out(t: f64) -> Result<Value, ClockError> {
    match t {
        f64::INFINITY => Ok(json!("+inf")),
        f64::NEG_INFINITY => Ok(json!("-inf")),
        t if t.is_nan() => Err(ClockError::Malformed("a NaN time has no spelling".into())),
        t => Ok(json!(t)),
    }
}

/// `body` with each number outside a string quoted, so it is read as text.
fn quote_numbers(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 64);
    let (mut string, mut escaped, mut number) = (false, false, false);
    for &b in body {
        let digit = !string
            && if number {
                b"0123456789.eE+-".contains(&b)
            } else {
                b == b'-' || b.is_ascii_digit()
            };
        if digit != number {
            out.push(b'"');
            number = digit;
        }
        if string {
            (escaped, string) = (!escaped && b == b'\\', escaped || b != b'"');
        } else {
            string = b == b'"';
        }
        out.push(b);
    }
    if number {
        out.push(b'"');
    }
    out
}

fn time_in(v: &Value) -> Result<f64, ClockError> {
    match v.as_str() {
        Some("+inf") => Some(f64::INFINITY),
        Some("-inf") => Some(f64::NEG_INFINITY),
        Some(t) => t.parse().ok(),
        None => None,
    }
    .ok_or_else(|| ClockError::Malformed(format!("{v} is not a time")))
}

fn seq_in(v: &Value) -> Result<u64, ClockError> {
    v.as_str()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| ClockError::Malformed(format!("{v} is not a sequence number")))
}
