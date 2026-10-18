//! The clock, random numbers and hasher the routing policies read.
//!
//! Normally these are the host's: `Instant::now()`, the thread RNG and a
//! randomly keyed hasher. A simulated run calls [`enable`] with its seed
//! before the router builds its policies. From then on [`now`] reads the
//! logical-process clock that [`set_now`] advances, [`rng`] draws from one
//! generator seeded by the run, and [`hasher`] uses fixed keys, so one seed
//! and one simulated history pick the same workers whatever the host timing.

use std::{
    collections::hash_map::{DefaultHasher, RandomState},
    hash::BuildHasher,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, MutexGuard, OnceLock,
    },
    time::{Duration, Instant},
};

use rand::{
    rngs::{StdRng, ThreadRng},
    RngCore, SeedableRng,
};

/// Set once [`enable`] runs; the instant that clock time zero maps to.
static ORIGIN: OnceLock<Instant> = OnceLock::new();
/// Logical-process clock in seconds, stored as `f64` bits.
static CLOCK: AtomicU64 = AtomicU64::new(0);
static RUN_RNG: Mutex<Option<StdRng>> = Mutex::new(None);
static WORKER_ORDER: Mutex<Vec<(String, usize)>> = Mutex::new(Vec::new());

/// Whether [`enable`] has run.
pub fn simulated() -> bool {
    ORIGIN.get().is_some()
}

/// Switch the policies to the simulated clock, an RNG seeded with `seed`
/// and fixed hasher keys, with the clock at zero. Call it before the
/// policies are built: a map built earlier keeps its random keys. Calling it
/// again restarts the RNG and the clock and clears the worker order.
pub fn enable(seed: u64) {
    *run_rng() = Some(StdRng::seed_from_u64(seed));
    set_worker_order(Vec::new());
    set_now(0.0);
    ORIGIN.get_or_init(Instant::now);
}

/// List workers in this order of (base URL, DP rank) under [`enable`], a
/// worker with no DP rank as rank 0; the ones it leaves out follow, sorted.
pub fn set_worker_order(order: Vec<(String, usize)>) {
    *worker_order() = order;
}

/// The order [`set_worker_order`] set, empty when none was.
pub fn worker_order() -> MutexGuard<'static, Vec<(String, usize)>> {
    WORKER_ORDER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Advance the simulated clock to `seconds` of simulated time.
pub fn set_now(seconds: f64) {
    CLOCK.store(seconds.to_bits(), Ordering::Release);
}

/// The simulated clock in seconds.
pub fn seconds() -> f64 {
    f64::from_bits(CLOCK.load(Ordering::Acquire))
}

/// `Instant::now()`, or the simulated clock once [`enable`] has run.
pub fn now() -> Instant {
    match ORIGIN.get() {
        None => Instant::now(),
        Some(origin) => *origin + Duration::from_secs_f64(seconds()),
    }
}

fn run_rng() -> MutexGuard<'static, Option<StdRng>> {
    RUN_RNG.lock().unwrap_or_else(|error| error.into_inner())
}

/// The thread RNG, or the run's seeded RNG once [`enable`] has run. The
/// seeded one is locked while the returned value lives.
pub fn rng() -> PolicyRng {
    if simulated() {
        PolicyRng::Run(run_rng())
    } else {
        PolicyRng::Host(rand::rng())
    }
}

pub enum PolicyRng {
    Host(ThreadRng),
    Run(MutexGuard<'static, Option<StdRng>>),
}

impl PolicyRng {
    fn inner(&mut self) -> &mut dyn RngCore {
        match self {
            Self::Host(rng) => rng,
            Self::Run(guard) => guard.as_mut().expect("enable() seeds the RNG"),
        }
    }
}

impl RngCore for PolicyRng {
    fn next_u32(&mut self) -> u32 {
        self.inner().next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner().next_u64()
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.inner().fill_bytes(dst)
    }
}

/// Random keys, or fixed ones once [`enable`] has run.
pub fn hasher() -> PolicyHasher {
    if simulated() {
        PolicyHasher::Fixed
    } else {
        PolicyHasher::Random(RandomState::new())
    }
}

#[derive(Clone, Debug)]
pub enum PolicyHasher {
    Random(RandomState),
    Fixed,
}

impl BuildHasher for PolicyHasher {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        match self {
            Self::Random(state) => state.build_hasher(),
            Self::Fixed => DefaultHasher::new(),
        }
    }
}
