//! cache_aware's tree eviction under the router LP, against a scripted clock
//! authority: the router reports the eviction deadline as its daemon deadline
//! and evicts when granted it, or before a request arriving at or after it, so
//! one seed picks the same ranks whatever the wall timing and whichever grants
//! it gets, and the eviction moves picks.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{body::Body, http::Request, routing::post, Router};
use mesh::{
    compass_clock::ENDPOINT_ENV,
    compass_router::{self, LOOKAHEADS_ENV, SEED_ENV},
    core::{DPAwareWorkerBuilder, Worker},
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
};
use rand::{seq::SliceRandom, Rng};
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::oneshot};
use tower::ServiceExt;

mod compass_frames;
use compass_frames::{read_frame, write_frame};

const HTTP: &str = "traffic->router:http";

/// Eight distinct 16-character prompts, each sent twice: seqs 0-7 before the
/// first deadline at 1 s, seqs 8-15 at or after it.
fn prompt(seq: u64) -> String {
    let c = char::from_u32(0x4E00 + (seq % 8) as u32).unwrap();
    std::iter::repeat_n(c, 16).collect()
}

/// One run: the ranks picked in handling order, and the daemon deadline of
/// each request the router made. With `deadline_granted`, the authority grants
/// 1 s on its own and releases seqs 8-15 from 1.0625 s; without it, seqs 8-15
/// come from exactly 1 s in the next grant, as when the router's first request
/// went out before its policies registered the deadline.
async fn run(
    seed: u64,
    eviction_interval_secs: u64,
    deadline_granted: bool,
) -> (Vec<usize>, Vec<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    std::env::set_var(
        ENDPOINT_ENV,
        format!("tcp://{}", listener.local_addr().unwrap()),
    );
    std::env::set_var(SEED_ENV, seed.to_string());
    let lookaheads = r#"{"router->frontend-P:http": 0.125, "router->frontend-D:relay": 0.125}"#;
    std::env::set_var(LOOKAHEADS_ENV, lookaheads);
    let released = |seqs: std::ops::Range<u64>, from: f64| {
        let pairs: Vec<Value> = seqs
            .map(|s| json!([s, from + (s % 8 + 1) as f64 / 16.0]))
            .collect();
        json!({ HTTP: pairs })
    };
    let mut grants = vec![
        json!({"kind": "GRANT", "G": 0.5, "released": released(0..8, 0.0)}),
        json!({"kind": "GRANT", "G": 1.5, "released": released(8..16, 0.9375)}),
        json!({"kind": "GRANT", "G": "+inf", "released": {}}),
    ];
    if deadline_granted {
        grants[1] = json!({"kind": "GRANT", "G": 1.5, "released": released(8..16, 1.0)});
        grants.insert(1, json!({"kind": "GRANT", "G": 1.0, "released": {}}));
    }
    let (ready, built) = oneshot::channel::<()>();
    let served = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        read_frame(&mut sock).await;
        built.await.unwrap();
        write_frame(&mut sock, json!({"kind": "JOINED", "now": 0.0, "seqs": {}})).await;
        let mut daemons = Vec::new();
        for grant in grants {
            daemons.push(read_frame(&mut sock).await["t_daemon"].clone());
            write_frame(&mut sock, grant).await;
        }
        daemons
    });

    compass_router::start();
    let policy = Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs,
        max_tree_size: 24,
        ..Default::default()
    }));
    let workers: Arc<Vec<Arc<dyn Worker>>> = Arc::new(
        (0..4)
            .map(|rank| {
                Arc::new(DPAwareWorkerBuilder::new("http://w:8000", rank, 4).build())
                    as Arc<dyn Worker>
            })
            .collect(),
    );
    policy.init_workers(&workers);
    ready.send(()).unwrap();

    let picks = Arc::new(Mutex::new(Vec::new()));
    let handler = {
        let picks = picks.clone();
        move |text: String| {
            let (policy, workers, picks) = (policy.clone(), workers.clone(), picks.clone());
            async move {
                let info = SelectWorkerInfo {
                    request_text: Some(&text),
                    ..Default::default()
                };
                let rank = policy.select_worker(&workers, &info).await.unwrap();
                picks.lock().unwrap().push(rank);
                compass_router::placed("http://p", "http://d");
            }
        }
    };
    let app = Router::new()
        .route("/", post(handler))
        .layer(axum::middleware::from_fn(compass_router::ingress));

    let mut order: Vec<u64> = (0..16).collect();
    order.shuffle(&mut rand::rng());
    let requests: Vec<_> = order
        .into_iter()
        .map(|seq| {
            let pause = Duration::from_micros(rand::rng().random_range(0..3000));
            let request = Request::post("/")
                .header("tracestate", format!("compass=a:0.0;s:{seq}"))
                .body(Body::from(prompt(seq)))
                .unwrap();
            let app = app.clone();
            tokio::spawn(async move {
                tokio::time::sleep(pause).await;
                app.oneshot(request).await.unwrap();
            })
        })
        .collect();
    for request in requests {
        request.await.unwrap();
    }
    // The policy, and so its task, lives until the last request is read.
    let daemons = served.await.unwrap();
    drop(app);
    let picks = picks.lock().unwrap().clone();
    (picks, daemons)
}

#[tokio::test(flavor = "multi_thread")]
async fn eviction_fires_on_the_lp_clock_and_one_seed_picks_the_same_ranks() {
    let (picks, daemons) = run(20261016, 1, true).await;
    assert_eq!(daemons, [1.0, 1.0, 2.0, 2.0].map(|t| json!(t)));
    assert_eq!(run(20261016, 1, true).await, (picks.clone(), daemons));

    // A deadline past the run: no eviction, and every resent prompt keeps its
    // rank. The first run's task, dropped with its policy, reports nothing.
    let (kept, daemons) = run(20261016, 100, true).await;
    assert_eq!(daemons, [100.0; 4].map(|t| json!(t)));
    assert_eq!(kept[..8], kept[8..]);
    assert_eq!(picks[..8], kept[..8]);
    assert_ne!(picks[8..], kept[8..], "the eviction must move a pick");

    // No grant at the deadline, and seq 8 arrives exactly on it: the eviction
    // runs before seq 8, so the picks are the first run's.
    let (late, daemons) = run(20261016, 1, false).await;
    assert_eq!(late, picks, "the eviction must run before seq 8");
    assert_eq!(daemons, [1.0, 1.0, 2.0].map(|t| json!(t)));
}
