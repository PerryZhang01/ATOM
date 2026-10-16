//! The router LP against a scripted clock authority: it hands a request, a
//! prefill response and stream events on one at a time in release order,
//! whatever order they reached the router in, restamps each at its arrival
//! plus its outbound lookahead, and logs those sends with its next request.

use std::{sync::Arc, time::Duration};

use axum::{
    body::Body,
    http::{HeaderMap, Request},
    routing::post,
    Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::BodyExt;
use mesh::{
    compass_clock::ENDPOINT_ENV,
    compass_router::{self, LOOKAHEADS_ENV, SEED_ENV},
    core::{BasicWorkerBuilder, Worker, WorkerLoadGuard},
    determinism,
};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{oneshot, Mutex},
};
use tower::ServiceExt;

async fn read_frame(sock: &mut TcpStream) -> Value {
    let mut body = vec![0; sock.read_u32().await.unwrap() as usize];
    sock.read_exact(&mut body).await.unwrap();
    serde_json::from_slice(&body).unwrap()
}

async fn write_frame(sock: &mut TcpStream, message: Value) {
    let body = message.to_string();
    sock.write_u32(body.len() as u32).await.unwrap();
    sock.write_all(body.as_bytes()).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn releases_are_handled_one_at_a_time_in_order_and_restamped() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
    let replies = [
        json!({"kind": "JOINED", "now": 0.0, "seqs": {}}),
        json!({"kind": "GRANT", "G": 2.0, "released": {
            "traffic->router:http": [[0, 0.5]],
            "frontend-D->router:stream": [[1, 2.0], [0, 1.0]],
            "frontend-P->router:relay": [[0, 1.5]],
        }}),
        json!({"kind": "GRANT", "G": "+inf", "released": {}}),
    ];
    let served = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut read = Vec::new();
        for reply in replies {
            read.push(read_frame(&mut sock).await);
            write_frame(&mut sock, reply).await;
        }
        read
    });
    std::env::set_var(ENDPOINT_ENV, &endpoint);
    std::env::set_var(SEED_ENV, "7");
    std::env::set_var(
        LOOKAHEADS_ENV,
        r#"{"router->traffic:stream": 0.25, "router->frontend-D:relay": 0.5,
            "router->frontend-P:http": 0.125}"#,
    );
    // The policies' clock origin; `start` enables it again at zero.
    determinism::enable(0);
    let origin = determinism::now();
    compass_router::start();

    // The prefill response reaches the router first, but is released second.
    let relay = tokio::spawn(async {
        let mut body = json!({"kv_transfer_params": {"compass": "a:1.5;s:0", "k": 1}});
        let turn = compass_router::relay(&mut body).await.unwrap();
        (body, turn)
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!relay.is_finished());

    // The request is released first, and handled with the policies' clock at
    // its arrival; until it is placed nothing else is released.
    let (go, placed) = oneshot::channel::<()>();
    let placed = Arc::new(Mutex::new(Some(placed)));
    let app = Router::new()
        .route(
            "/",
            post(move |headers: HeaderMap| async move {
                let clock = (determinism::now() - origin).as_secs_f64();
                compass_router::placed("http://p", "http://d");
                let wait = placed.lock().await.take().unwrap();
                wait.await.unwrap();
                let state = headers["tracestate"].to_str().unwrap();
                format!("{} at {clock}", compass_router::header("tracestate", state))
            }),
        )
        .layer(axum::middleware::from_fn(compass_router::ingress));
    let request = tokio::spawn(async move {
        let request = Request::post("/")
            .header("tracestate", "rojo=00f067aa0ba902b7,compass=a:0.5;s:0")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        String::from_utf8(body.to_vec()).unwrap()
    });

    let decode: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://d").build());
    let guard = WorkerLoadGuard::new(decode.clone(), None);
    let chunk = ": compass a=1.0 s=0\ndata: x\n\n: compass a=2.0 s=1\ndata: [DONE]\n\n";
    let upstream = futures_util::stream::iter([Ok(Bytes::from(chunk))]);
    let (mut events, kept) = compass_router::stream(upstream, guard);
    assert!(kept.is_none());
    let first = tokio::time::timeout(Duration::from_secs(5), events.next());
    let first = first.await.expect("released once the request is placed");
    let first = first.unwrap().unwrap();
    assert_eq!(first, ": compass a=1.25 s=0\ndata: x\n\n");
    go.send(()).unwrap();
    assert_eq!(
        request.await.unwrap(),
        "rojo=00f067aa0ba902b7,compass=a:0.625;s:0 at 0.5"
    );

    let (body, turn) = relay.await.unwrap();
    assert_eq!(
        body,
        json!({"kv_transfer_params": {"compass": "a:2.0;s:0", "k": 1}})
    );
    // The last event waits until the relay is handled.
    let next = tokio::time::timeout(Duration::from_millis(200), events.next()).await;
    assert!(next.is_err());
    assert_eq!(decode.load(), 1);
    drop(turn);
    let last = events.next().await.unwrap().unwrap();
    assert_eq!(last, ": compass a=2.25 s=1\ndata: [DONE]\n\n");
    assert_eq!(decode.load(), 0);
    assert!(events.next().await.is_none());

    let read = served.await.unwrap();
    assert_eq!(read[0], json!({"kind": "BIND", "lp": "router"}));
    assert_eq!(read[1]["log"], json!([]));
    assert_eq!(
        read[2]["log"],
        json!([
            ["router->frontend-P:http", 0, 0.625],
            ["router->traffic:stream", 0, 1.25],
            ["router->frontend-D:relay", 0, 2.0],
            ["router->traffic:stream", 1, 2.25],
        ])
    );
}
