<!-- SPDX-License-Identifier: MIT -->
# Blocking calls on ATOM's serving path

A simulated run substitutes a predicted duration for the work a batch does. Any
call that parks a thread on the **real** clock therefore has to be accounted
for: some of them are the durations being predicted, some are where a message
crosses from one logical process to another, some are bounds, and most are
invisible to the result and are left exactly as they are.

Two files:

| File | What |
|---|---|
| `sync_scan.py` | finds the call sites. It reports candidates and judges none of them. |
| `sync_sites.json` | says what each one is: the mechanism a simulated run applies there, who it is waiting for, and one line of why. |

`tests/compass/test_sync_inventory.py` asserts the two agree. **A blocking call
added to ATOM later fails that test** rather than being missed, and the failure
message says which of the two things went wrong: an unclassified site, or a
pinned piece of text that is no longer exactly once in its symbol. The same file
checks each row's mechanism against what the row says about its peer and its
bound.

## The mechanisms

A simulated run groups ATOM's processes into logical processes (LPs), each a
group that shares one clock. Each LP has one clock owner — its step loop or its
asyncio event loop — and only the clock owner moves the LP's clock; every other
thread in the LP is transport. A clock authority grants each LP the time it may
advance to.

| | Mechanism | What a simulated run does there |
|---|---|---|
| **K1** | event cost | the clock owner advances by a duration that the cost model or a resource station prices |
| **K2** | clock read | the read returns the LP's logical time |
| **K3** | idle point | the clock owner waits for the LP's next event instead of spinning, or parking on the real clock |
| **K4** | channel send | the message is stamped with the LP's time, and the send is logged on the clock owner |
| **K5** | channel receive | the receive is counted where the thread waits, and the message is delivered in timestamp order once the LP's clock passes its arrival |
| **K6** | wait inside one LP | nothing: it is never reported to the clock authority, and its duration belongs to the event the clock owner is pricing |
| **K7** | virtual timer | the bound runs on the LP clock |
| **K8** | real bound | the clock authority cannot reach it, or it stays real on purpose; it is configured, not simulated |
| **K9** | outside the model | nothing: it is outside the simulated window, inside code the simulator replaces, or cannot park |

**The boundary that matters is the LP, not the process.** A tensor-parallel
group is one LP, so the engine's RPC to its own workers is K6 although its peer
is another process, and so is a frontend coroutine waiting for data its own
output thread delivers. Only a wait whose sender is in another LP is K5. **A
bound in Python is a virtual timer**; only what the clock authority cannot reach
— the router's bounds and switches, and OS-level process-death
detection — is a real bound. **Sends are counted**, because a send is one end
of the count of messages in transit, even on a socket that cannot park.

Every row also keeps the `category` and `why` of the first classification,
which sorted waits into A (simulated time decides it), B (an unbounded wait for
another process), C1 (a bound that declares something broken), C2 (a bound that
sets a cadence), `ignore` and `undecided`. They are history, and rows added
since carry no category. Each category named one mechanism — A K1, B K5, C1
K8, C2 K7, `ignore` K9 — and wherever a row's mechanism is a different one,
`mechanism_why` says why; the test requires it there.

## Rows added since the first classification

The `send_multipart` calls the scanner found once it knew that shape carry no
category: the front end's request and control sends (K4), and the sends in the
KV event publisher and the transfer backends (K9). Compass's own runtime is
scanned too, for the waits it adds to a simulated run: the data-parallel
group's per-step all_gather in the simulated runner (K6), the two ends of the
KV write request between a decode and a prefill deployment (K4, K5), the
engine's RPCs to its own workers for its hand-ins (K6), and the joins and
waits that close a run (K9). The rows per mechanism,
crossed with the first classification's categories, are printed by
`pytest tests/compass/test_sync_inventory.py -s -k crosstab`.

## What the scanner does, and where it can be wrong

It parses every file under `SCANNED_ROOTS` and reports calls whose **shape** can
park a thread: sleeps, socket receives, pollers, queue operations, joins, lock
and event waits, collectives, device synchronization, sends, thread-pool
hand-offs, and the worker RPC. It also reports `while` loops that go around
without calling anything that parks. Shapes match on the call's own text and
argument form, never on what a name is bound to at runtime.

Where a name alone is ambiguous, the argument form settles it, because only the
blocking reading can take that form: `d.get(key)` always passes the key
positionally and `q.get()` never does; `",".join(parts)` always passes one
iterable and `t.join(5)` a lone number. Those rules have their own tests.

**Roots are directories, not a list of files.** A file allowlist declares its
blind spot at a finer grain than it excludes: a module added beside a scanned
one is invisible while the exclusion list still reads complete. Scanning the
directory makes a new file a test failure on the day it lands, and a test
asserts no scanned file is also declared unscanned.

It over-reports on purpose. A `.send()` that never blocks in practice is still
listed, because that is where a message leaves one process for another. The
classification is what removes it from consideration, and a K9 row with
its reason is a better record than a silent omission.

It can be wrong in two directions, and both are visible rather than hidden:

- **Toward noise.** Some listed calls cannot park — an `enqueue` that appends to
  a thread-local buffer, a `result()` on a task already finished, a `while` loop
  walking a string. They carry a K9 row saying so.
- **Toward silence.** A blocking call reached through a name the shapes do not
  know, or in a directory outside `SCANNED_ROOTS`, is invisible.
  `UNSCANNED_ROOTS` names each excluded directory and why, so the boundary is
  arguable rather than implicit, and a test asserts every named root exists.

## How a row keeps naming the same call

A row's identity is `file::symbol::expression::shape#ordinal` — **not** the
line, so an edit elsewhere in a file changes no row, and **with** the
arguments, so two calls to one method in one function are told apart. An
earlier version keyed on the callee alone: inserting one
`call_func("flush_pp_send", ...)` above a `call_func("forward", ...)` then
silently moved every later row onto the wrong site. Two sites that still share
an ordinal are the same call written the same way in one function, and a test
asserts the inventory answers both the same way, so the ordinal never carries
meaning of its own.

## What the rows do not cover

Every reading of the wall clock, as opposed to every wait. The rows under
`anchors` carry the readings that gate scheduling or that the result reports —
the delay gate, the queue-age guard, and the arrival, first-token and finish
stamps — because a mechanism already applies to them. An anchor names its
enclosing symbol the way a site id does (`EngineCore.busy_loop`, or `<module>`),
and its text must occur exactly once there; the Rust anchors name no symbol and
must occur once in their file. Every clock read on this path is classified
by the clock-source lint instead (`atom/compass/detect/clock_source.py`): a
read either goes through `atom.utils.clock.now` and returns the LP's time, or
stays real under a `CORE_ALLOW_LIST` entry naming its site, class and reason.
