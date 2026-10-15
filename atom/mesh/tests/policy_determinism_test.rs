//! Under `determinism::enable`, the routing policies pick ranks from the
//! seed and the simulated clock alone, whatever the wall time between
//! requests. The tests share the process-wide clock and RNG, hence serial.

use std::{sync::Arc, time::Duration};

use mesh::{
    core::{DPAwareWorkerBuilder, Worker},
    determinism,
    policies::{
        tree::Tree, CacheAwareConfig, CacheAwarePolicy, DpStickyPolicy, LoadBalancingPolicy,
        SelectWorkerInfo,
    },
};
use rand::Rng;
use serial_test::serial;

fn ranks() -> Vec<Arc<dyn Worker>> {
    (0..4)
        .map(|rank| {
            Arc::new(DPAwareWorkerBuilder::new("http://w:8000", rank, 4).build()) as Arc<dyn Worker>
        })
        .collect()
}

fn session(id: &str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-session-id", id.parse().unwrap());
    headers
}

/// One run: cache_aware breaking four-way load ties, then two-way ties when
/// imbalanced, and placing at random with no tree; dp_sticky sessions
/// idling past expiry on the simulated clock, and radix-tree owner picks
/// after each node's cached owner leaves. A random wall-clock pause precedes
/// every request.
async fn run(seed: u64) -> Vec<String> {
    determinism::enable(seed);
    let mut picks = Vec::new();

    let workers = ranks();
    let cache_aware = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    cache_aware.init_workers(&workers);
    let sticky_workers = ranks();
    let sticky = DpStickyPolicy::new();
    for i in 0..48u32 {
        std::thread::sleep(Duration::from_micros(rand::rng().random_range(0..300)));
        determinism::set_now(f64::from(i) * 3000.0);

        let text: String = std::iter::repeat_n(char::from_u32(0x4E00 + i).unwrap(), 8).collect();
        let info = SelectWorkerInfo {
            request_text: Some(&text),
            ..Default::default()
        };
        picks.push(format!(
            "cache_aware {:?}",
            cache_aware.select_worker(&workers, &info).await
        ));

        let headers = session(&format!("s{}", i % 3));
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let rank = sticky.select_worker(&sticky_workers, &info).await.unwrap();
        sticky_workers[rank].increment_load();
        picks.push(format!("dp_sticky {rank}"));
    }

    // Ranks 0 and 1 past balance_abs_threshold: the imbalanced branch breaks
    // a tie between ranks 2 and 3. A policy that never ran init_workers has
    // no tree and places idle ranks at random.
    for worker in &workers[..2] {
        for _ in 0..33 {
            worker.increment_load();
        }
    }
    let idle = ranks();
    let unseeded = CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    });
    let info = SelectWorkerInfo {
        request_text: Some("imbalanced"),
        ..Default::default()
    };
    for _ in 0..16 {
        picks.push(format!(
            "imbalanced {:?}",
            cache_aware.select_worker(&workers, &info).await
        ));
        picks.push(format!(
            "no tree {:?}",
            unseeded.select_worker(&idle, &info).await
        ));
    }

    let tree = Tree::new();
    for i in 0..16 {
        for rank in 0..4 {
            tree.insert(&format!("{i:02} shared prefix"), &format!("rank{rank}"));
        }
    }
    tree.remove_tenant("rank0");
    for i in 0..16 {
        picks.push(format!(
            "tree {}",
            tree.prefix_match(&format!("{i:02} shared prefix")).1
        ));
    }
    picks
}

#[tokio::test]
#[serial]
async fn one_seed_picks_the_same_ranks_under_shuffled_timing() {
    let first = run(20261015).await;
    assert_eq!(first, run(20261015).await);
    assert_ne!(first, run(7).await, "the seed must reach the tie-breaks");
}

#[tokio::test]
#[serial]
async fn a_session_idle_past_expiry_on_the_simulated_clock_moves() {
    determinism::enable(0);
    let workers = ranks();
    let sticky = DpStickyPolicy::new();
    let headers = session("s");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };

    let first = sticky.select_worker(&workers, &info).await.unwrap();
    workers[first].increment_load();
    determinism::set_now(119.0 * 60.0);
    assert_eq!(sticky.select_worker(&workers, &info).await, Some(first));
    determinism::set_now(240.0 * 60.0);
    assert_ne!(sticky.select_worker(&workers, &info).await, Some(first));
}
