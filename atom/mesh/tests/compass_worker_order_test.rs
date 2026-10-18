//! Under `determinism::enable`, the ATOM PD router running dp_sticky over
//! DP-aware workers picks the same ranks from one seed whatever order the
//! workers were registered or hashed in: a request naming its model reads the
//! model's list, one naming none reads every worker. A set worker order
//! (`determinism::set_worker_order`) lists them in it instead.

use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    extract::State,
    http::Request,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use http_body_util::BodyExt;
use mesh::{
    app_context::AppContext,
    config::{types::BackendType, PolicyConfig, RouterConfig, RoutingMode},
    core::{DPAwareWorkerBuilder, Worker, WorkerRegistry, WorkerType},
    determinism,
    routers::RouterFactory,
    server::{build_app, AppState},
};
use serde_json::{json, Value};
use serial_test::serial;
use tokio::net::TcpListener;
use tower::ServiceExt;

/// (server, data_parallel_rank) of every request a worker was sent.
type Seen = Arc<Mutex<Vec<(usize, Value)>>>;

async fn worker(
    State((seen, server)): State<(Seen, usize)>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let rank = body["data_parallel_rank"].clone();
    seen.lock().unwrap().push((server, rank.clone()));
    Json(json!({"kv_transfer_params": {"dp_rank": rank}, "text": "x"})).into_response()
}

/// One run: a prefill server and two decode servers of four DP ranks each,
/// registered in the order given, then twelve requests over six sessions.
async fn picks(seed: u64, order: [usize; 3], path: &str, body: Value) -> Vec<(usize, Value)> {
    determinism::enable(seed);
    let config = RouterConfig {
        backend: BackendType::Atom,
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        policy: PolicyConfig::DpSticky,
        ..Default::default()
    };
    let context = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let seen = Seen::default();
    // Server numbers follow URL order, so they name the same ranks in every run.
    let mut listeners = Vec::new();
    for _ in 0..3 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        listeners.push((
            format!("http://{}", listener.local_addr().unwrap()),
            listener,
        ));
    }
    listeners.sort_by(|a, b| a.0.cmp(&b.0));
    let mut servers = Vec::new();
    for (server, (url, listener)) in listeners.into_iter().enumerate() {
        let app = Router::new()
            .route("/v1/chat/completions", post(worker))
            .route("/generate", post(worker))
            .route(
                "/kv_transfer_info",
                get(|| async { Json(json!({"tp_size": 1, "kv_role": "kv_producer"})) }),
            )
            .with_state((seen.clone(), server));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        servers.push(url);
    }
    for server in order {
        let kind = if server == 0 {
            WorkerType::Prefill {
                bootstrap_port: None,
            }
        } else {
            WorkerType::Decode
        };
        for rank in 0..4 {
            let worker: Arc<dyn Worker> = Arc::new(
                DPAwareWorkerBuilder::new(servers[server].clone(), rank, 4)
                    .model_id("m")
                    .worker_type(kind.clone())
                    .build(),
            );
            context.worker_registry.register(worker);
        }
    }
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = build_app(
        Arc::new(AppState {
            router: Arc::from(router),
            context,
            router_manager: None,
        }),
        1 << 20,
        vec![],
    );
    for i in 0..12 {
        let request = Request::post(path)
            .header("content-type", "application/json")
            .header("x-session-id", format!("s{}", i % 6))
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert!(response.status().is_success(), "{}", response.status());
        response.into_body().collect().await.unwrap();
    }
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 24);
    seen
}

#[tokio::test(flavor = "multi_thread")]
#[serial]
async fn a_request_naming_its_model_picks_the_same_ranks_in_any_registration_order() {
    let chat = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let path = "/v1/chat/completions";
    assert_eq!(
        picks(20260707, [0, 1, 2], path, chat.clone()).await,
        picks(20260707, [2, 1, 0], path, chat).await
    );
}

#[tokio::test(flavor = "multi_thread")]
#[serial]
async fn a_request_naming_no_model_picks_the_same_ranks_from_one_seed() {
    let generate = json!({"text": "hi"});
    assert_eq!(
        picks(20260707, [0, 1, 2], "/generate", generate.clone()).await,
        picks(20260707, [0, 1, 2], "/generate", generate).await
    );
}

/// Each worker's (base URL, DP rank), in the order `listed` gives them.
fn named(listed: &[Arc<dyn Worker>]) -> Vec<(String, usize)> {
    listed
        .iter()
        .map(|w| (w.base_url().to_string(), w.dp_rank().unwrap()))
        .collect()
}

#[test]
#[serial]
fn a_set_worker_order_lists_the_workers_in_it_and_the_rest_after_them_sorted() {
    determinism::enable(20260707);
    let at = |url: &str, rank| (url.to_string(), rank);
    let order = vec![at("http://b", 1), at("http://a", 0), at("http://b", 0)];
    determinism::set_worker_order(order.clone());
    let registry = WorkerRegistry::new();
    for (url, rank) in [
        ("http://a", 1),
        ("http://b", 0),
        ("http://a", 0),
        ("http://b", 1),
    ] {
        registry.register(Arc::new(
            DPAwareWorkerBuilder::new(url, rank, 2)
                .model_id("m")
                .worker_type(WorkerType::Decode)
                .build(),
        ));
    }
    let mut expected = order;
    expected.push(at("http://a", 1));
    assert_eq!(named(&registry.get_by_model("m")), expected);
    assert_eq!(named(&registry.get_all()), expected);
    determinism::enable(20260707);
    assert_eq!(
        named(&registry.get_all()),
        [
            at("http://a", 0),
            at("http://a", 1),
            at("http://b", 0),
            at("http://b", 1)
        ]
    );
}
