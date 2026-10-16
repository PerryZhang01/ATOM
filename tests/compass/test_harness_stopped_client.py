# SPDX-License-Identifier: MIT
"""A client bound when a simulated server is stopped is granted ``+inf`` at the
stop, and compass_harness's traffic LP raises on it naming its open request.

A frontend, an engine and the traffic LP each hold their own TCP connection to
the run's authority, as a deployment's processes do. The traffic LP sends one
request and the frontend takes it; the frontend's runtime then stops the run
as a SIGTERM does, and the engine takes its ``+inf`` grant and leaves, which
ends its connection. Skips by name as ``test_harness_pacing.py`` does.
"""

import asyncio
import math
import threading
from importlib import metadata

import pytest


def _missing() -> str | None:
    try:
        metadata.version("compass-harness")
        import compass_harness  # noqa: F401  checks aiperf against its pinned source
    except metadata.PackageNotFoundError as e:
        return f"{e.name} is not installed"
    except RuntimeError as e:
        return str(e)
    return None


if _why := _missing():
    pytest.skip(
        f"needs agentx-harness 56a0cf70 and compass-harness: {_why}",
        allow_module_level=True,
    )

from compass_harness.traffic_lp import TrafficLP, UnansweredRequests
from compass_harness.transport import credit_key

from atom.compass import clock_transport
from atom.compass import run as compass_run
from atom.utils.clock import LPRuntime

INF = math.inf
KEY = credit_key("profiling", 0, 0)


def test_a_client_bound_at_a_stop_names_its_open_request(tmp_path):
    run = {"admission_path": "serving", "ipc_s": 2.0**-10, "stream_s": 2.0**-9}
    run.update(bound_s=100.0, client_runs=None, out_dir=str(tmp_path))
    authority = compass_run._RecordingAuthority(run)
    server = clock_transport.serve(authority, "tcp://127.0.0.1:0")
    table = compass_run.channel_table(run)

    def runtime(lp, cls=LPRuntime):
        return cls(lp, table, clock_transport.connect(lp, server.endpoint))

    frontend = runtime(compass_run.FRONTEND, compass_run._StoppingClock)
    engine = runtime(compass_run.ENGINE)
    engine.ask_next_event(INF, 50.0)
    frontend.ask_next_event(INF)

    def stop():
        frontend.take_grant()  # the request's admission
        frontend.end_run()  # the stop, as the frontend's event loop makes it
        assert engine.take_grant() == INF
        engine.conn.close()  # the engine's process leaves

    async def client():
        traffic = TrafficLP(runtime(compass_run.TRAFFIC), str(tmp_path / "lp"))
        traffic.send(KEY)
        await asyncio.wait_for(traffic.done, 10)

    stopper = threading.Thread(target=stop, daemon=True)
    stopper.start()
    try:
        with pytest.raises(UnansweredRequests) as raised:
            asyncio.run(client())
    finally:
        stopper.join(10)
        server.close()
    print(f"\n{raised.value}")
    assert str(raised.value) == (
        f"the run finished with requests [{KEY}] sent but never given their final "
        "response"
    )
    assert not stopper.is_alive() and engine.now == INF
    assert (authority.stopped, authority.stopped_whole) == (True, False)
