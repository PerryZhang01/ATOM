//! The clock authority client against a scripted authority, and, driven by
//! `tests/compass/test_mesh_clock_client.py`, against the Python one.

use mesh::compass_clock::{ClockClient, ClockError, Grant, Released, ENDPOINT_ENV};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

const INF: f64 = f64::INFINITY;

async fn read_frame(sock: &mut TcpStream) -> String {
    let mut body = vec![0; sock.read_u32().await.unwrap() as usize];
    sock.read_exact(&mut body).await.unwrap();
    String::from_utf8(body).unwrap()
}

async fn write_frame(sock: &mut TcpStream, message: Value) {
    let body = message.to_string();
    sock.write_u32(body.len() as u32).await.unwrap();
    sock.write_all(body.as_bytes()).await.unwrap();
}

/// An authority that answers each frame it reads with the next of `replies`,
/// and returns the frames it read.
async fn scripted(replies: Vec<Value>) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
    let served = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut read = Vec::new();
        for reply in replies {
            read.push(read_frame(&mut sock).await);
            write_frame(&mut sock, reply).await;
        }
        read
    });
    (endpoint, served)
}

#[tokio::test]
async fn sends_its_log_and_reads_grants_and_refusals() {
    let (endpoint, served) = scripted(vec![
        json!({"kind": "JOINED", "now": 1.5, "seqs": {"router->traffic:stream": 3}}),
        json!({"kind": "GRANT", "G": 2.5, "released": {
            "traffic->router:http": [[1, 2.0], [0, 1.75]],
            "frontend-P->router:relay": [[2, 2.0]],
        }}),
        json!({"kind": "GRANT", "G": "+inf", "released": {}, "ended": 0}),
        json!({"kind": "REFUSED", "error": "BackdatedEvent", "reason": "late", "table": null}),
    ])
    .await;
    let mut clock = ClockClient::bind(&endpoint, "router").await.unwrap();
    assert_eq!(clock.send("router->traffic:stream", 2.0), 3);
    assert_eq!(clock.send("router->frontend-D:relay", 2.25), 0);
    let grant = clock.next_event(INF, 10.0).await.unwrap();
    let released = |arrival, channel: &str, seq| Released {
        arrival,
        channel: channel.into(),
        seq,
    };
    assert_eq!(
        grant,
        Grant {
            time: 2.5,
            released: vec![
                released(1.75, "traffic->router:http", 0),
                released(2.0, "frontend-P->router:relay", 2),
                released(2.0, "traffic->router:http", 1),
            ],
        }
    );
    let last = clock.next_event(3.0, INF).await.unwrap();
    assert_eq!(last.time, INF);
    assert!(matches!(
        clock.next_event(f64::NAN, INF).await,
        Err(ClockError::Malformed(_))
    ));
    let refused = clock.next_event(4.0, INF).await;
    assert!(
        matches!(&refused, Err(ClockError::Refused { error, reason }) if error == "BackdatedEvent" && reason == "late"),
        "{refused:?}"
    );
    assert_eq!(
        served.await.unwrap(),
        [
            r#"{"kind":"BIND","lp":"router"}"#,
            r#"{"kind":"NER","log":[["router->traffic:stream",3,2.0],["router->frontend-D:relay",0,2.25]],"t":"+inf","t_daemon":10.0}"#,
            r#"{"kind":"NER","log":[],"t":3.0,"t_daemon":"+inf"}"#,
            r#"{"kind":"NER","log":[],"t":4.0,"t_daemon":"+inf"}"#,
        ]
    );
}

/// A 17-digit time, as the authority writes it, reads back as the same f64.
#[tokio::test]
async fn reads_each_time_exactly() {
    let t = 10.759668308661709;
    let (endpoint, _served) = scripted(vec![
        json!({"kind": "JOINED", "now": t, "seqs": {"router->traffic:stream": 2}}),
        json!({"kind": "GRANT", "G": t, "released": {"traffic->router:http": [[1, t]]}}),
    ])
    .await;
    let mut clock = ClockClient::bind(&endpoint, "router").await.unwrap();
    assert_eq!(clock.send("router->traffic:stream", t), 2);
    let grant = clock.next_event(INF, INF).await.unwrap();
    assert_eq!(grant.time.to_bits(), f64::to_bits(t));
    assert_eq!(grant.released[0].arrival.to_bits(), f64::to_bits(t));
    assert_eq!(grant.released[0].seq, 1);
}

/// Run by `tests/compass/test_mesh_clock_client.py`, which serves the Python
/// authority with LPs `traffic` and `router` and sends one request from
/// `traffic` at 0.5; the router answers on its stream channel 0.25 later.
#[tokio::test]
#[ignore = "needs the Python authority; run by tests/compass/test_mesh_clock_client.py"]
async fn python_authority_releases_a_message_to_the_router() {
    let endpoint = std::env::var(ENDPOINT_ENV).expect(ENDPOINT_ENV);
    let mut clock = ClockClient::bind(&endpoint, "router").await.unwrap();
    let grant = clock.next_event(INF, INF).await.unwrap();
    let request = Released {
        arrival: 0.5,
        channel: "traffic->router:http".into(),
        seq: 0,
    };
    assert_eq!(grant.released, [request]);
    clock.send("router->traffic:stream", grant.released[0].arrival + 0.25);
    assert_eq!(clock.next_event(INF, INF).await.unwrap().time, INF);
}
