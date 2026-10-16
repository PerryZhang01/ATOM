//! The router LP on the ATOM PD router's own request path, against a scripted
//! clock authority and two scripted workers: the request is held at ingress,
//! the prefill response at the relay and each decode event in the stream, and
//! each goes on restamped at its arrival plus its outbound lookahead.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Request},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use http_body_util::BodyExt;
use mesh::{
    app_context::AppContext,
    compass_clock::ENDPOINT_ENV,
    compass_router::{self, LOOKAHEADS_ENV, SEED_ENV},
    config::{types::BackendType, RouterConfig, RoutingMode},
    core::{BasicWorkerBuilder, Worker, WorkerType},
    routers::RouterFactory,
    server::{build_app, AppState},
};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tower::ServiceExt;

mod compass_frames;
use compass_frames::{read_frame, write_frame};

/// What each worker was sent: (prefill, tracestate, body).
type Seen = Arc<Mutex<Vec<(bool, Option<String>, Value)>>>;

async fn worker(
    State((seen, prefill)): State<(Seen, bool)>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let state = headers
        .get("tracestate")
        .map(|h| h.to_str().unwrap().to_owned());
    seen.lock().unwrap().push((prefill, state, body));
    if prefill {
        Json(json!({"kv_transfer_params": {"compass": "a:1.5;s:0", "dp_rank": 0}})).into_response()
    } else {
        let events = ": compass a=2.0 s=0\ndata: x\n\n: compass a=2.5 s=1\ndata: [DONE]\n\n";
        ([("content-type", "text/event-stream")], events).into_response()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pd_request_is_held_and_restamped_at_ingress_relay_and_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("tcp://{}", listener.local_addr().unwrap());
    let replies = [
        json!({"kind": "JOINED", "now": 0.0, "seqs": {}}),
        json!({"kind": "GRANT", "G": 3.0, "released": {
            "traffic->router:http": [[0, 0.5]],
            "frontend-P->router:relay": [[0, 1.5]],
            "frontend-D->router:stream": [[0, 2.0], [1, 2.5]],
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
    compass_router::start();

    let config = RouterConfig {
        backend: BackendType::Atom,
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        ..Default::default()
    };
    let context = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let seen = Seen::default();
    let mut workers = Vec::new();
    for prefill in [true, false] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/v1/completions", post(worker))
            .route(
                "/kv_transfer_info",
                get(|| async { Json(json!({"tp_size": 1, "kv_role": "kv_producer"})) }),
            )
            .with_state((seen.clone(), prefill));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let kind = if prefill {
            WorkerType::Prefill {
                bootstrap_port: None,
            }
        } else {
            WorkerType::Decode
        };
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{address}"))
                .model_id("m")
                .worker_type(kind)
                .build(),
        );
        context.worker_registry.register(worker.clone());
        workers.push(worker);
    }
    let router = RouterFactory::create_router(&context).await.unwrap();
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: context.clone(),
        router_manager: None,
    });
    let app = build_app(state, 1 << 20, vec![]);

    let request = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .header("tracestate", "compass=a:0.5;s:0")
        .body(Body::from(
            json!({"model": "m", "prompt": "hi", "stream": true}).to_string(),
        ))
        .unwrap();
    let answered = async {
        let response = app.oneshot(request).await.unwrap();
        response.into_body().collect().await.unwrap().to_bytes()
    };
    let body = tokio::time::timeout(Duration::from_secs(10), answered)
        .await
        .expect("each hold point is released in turn");
    assert_eq!(
        String::from_utf8(body.to_vec()).unwrap(),
        ": compass a=2.25 s=0\ndata: x\n\n: compass a=2.75 s=1\ndata: [DONE]\n\n"
    );

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        (seen[0].0, seen[0].1.as_deref()),
        (true, Some("compass=a:0.625;s:0"))
    );
    assert!(!seen[1].0);
    assert_eq!(seen[1].2["kv_transfer_params"]["compass"], "a:2.0;s:0");
    assert_eq!(workers[0].load() + workers[1].load(), 0);
    assert_eq!(
        served.await.unwrap()[2]["log"],
        json!([
            ["router->frontend-P:http", 0, 0.625],
            ["router->frontend-D:relay", 0, 2.0],
            ["router->traffic:stream", 0, 2.25],
            ["router->traffic:stream", 1, 2.75],
        ])
    );
}
