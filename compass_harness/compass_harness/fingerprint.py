# SPDX-License-Identifier: MIT
"""The sha256 of each aiperf function compass_harness overrides, calls or relies on.

Read from the installed aiperf's source files with ``ast``, never by importing
them: the package bootstrap runs this while aiperf's plugin registry is still
loading.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path

PINNED = {
    # ClockPacedLoopScheduler overrides or calls these.
    "aiperf.common.loop_scheduler:LoopScheduler.__init__": "b14a1de8978777687ae825113e71cc1d29216c695ac793c8923f59635ae993a6",
    "aiperf.common.loop_scheduler:LoopScheduler._safe_callback": "e9020b50d48e297a6204765930424fe7be8b3e9d386e28c44d0f6ab201022684",
    "aiperf.common.loop_scheduler:LoopScheduler._track_handle_and_return_id": "87701ddc1c6f57bed48b3c1892dbfc3916763ac00ad6f4bdd9f88432b770f10d",
    "aiperf.common.loop_scheduler:LoopScheduler.execute_async": "b8c686189b3ef7773ec9f04f8c368265f34ed2396f6f5b571071057a3b8626da",
    "aiperf.common.loop_scheduler:LoopScheduler.schedule_later": "be79c87f7893ebd4d89e2c80975992c1ad773492885b1ef3683a1cc616e08afa",
    "aiperf.common.loop_scheduler:LoopScheduler.schedule_at": "0f1b5a3dc48dd4b390fe0962b96ebef3c75f615f04a1dd10184c6eba7fbb09b5",
    "aiperf.common.loop_scheduler:LoopScheduler.schedule_at_perf_sec": "a5c2654fb4d79eaaa9440a2687cd048834fd8eb3c5a039966d052c2a8bfc656a",
    "aiperf.common.loop_scheduler:LoopScheduler.schedule_at_perf_ns": "c09aa30daf6fcbca9b1349f133beedeedc4f331d4496e873481bf96a6bfcc260",
    "aiperf.common.loop_scheduler:LoopScheduler.cancel_handle_id": "922247e9c9b64881c96ccd65ef6d7f564be7771c117ab3eb13c4ab7903f57702",
    "aiperf.common.loop_scheduler:LoopScheduler.cancel_all_pending": "9a3c64a1548d0c8ffc0a518d4cd84672065ce1ac12be447d1a6776197b4e314d",
    "aiperf.common.loop_scheduler:LoopScheduler.cap_pending_delay": "aa479467f2be8ac91fd1e8915f4b3d85b6ca72f96a50fff9276dfd22a633adf5",
    "aiperf.common.loop_scheduler:LoopScheduler.cap_pending_delay_for_group": "05f62588664e2aa36ab99fbc4986c64421e5d4933042bfe7a692152f0cba26a7",
    # CompassTransport: headers, the send, the per-event callback on the SSE
    # read path, the comment packet, and the record fields read after the send.
    "aiperf.transports.base_transports:BaseTransport.build_headers": "1125033199929680f3cc4d9520cdbb571e808aa90476559b93f7c00b6c4dc5f4",
    "aiperf.transports.aiohttp_transport:AioHttpTransport.send_request": "87def9e8157b8e1f11c5461320153c30d4f5afc9e807559cd6f0322f56ddded7",
    "aiperf.transports.aiohttp_client:AioHttpClient.post_request": "fd2102cb373cd08df7712ab366ba2ef6d5e2902241edcda26f69d500d416f435",
    "aiperf.transports.aiohttp_client:AioHttpClient._request": "0435fa598b8182cb0c52d2e9fd35ede3d45af6e469361572d91bc06212abe9a5",
    "aiperf.common.models.record_models:SSEMessage.parse": "eb6bb061a28675669699fa2f34c7a2e5a0b1b968c27d9ef0df109ada64ab3f9b",
    "aiperf.workers.inference_client:InferenceClient._finalize_request_record": "fbe74fd8a5d4754cbe295bb17f187e961da1c958dc746c2e75f8c383823c324b",
    # CompassCreditRouter: the stamp on the one send, the hold on the one return
    # path, the worker wait, the finish, and the manager that builds the router.
    "aiperf.credit.sticky_router:StickyCreditRouter.__init__": "fba91c833a57f0eaabb3b8ba458bb1de7dda142db147c121cd33660b983cbdd8",
    "aiperf.credit.sticky_router:StickyCreditRouter.send_credit": "458eeb8e5eea035abc9fc15f4af3ace27a792e76bec6a93a697c0be589ef60f6",
    "aiperf.credit.sticky_router:StickyCreditRouter.set_return_callback": "4debf458e16ad7f804e0700572ace0c210c2bb3a586906545a72643445c401de",
    "aiperf.credit.sticky_router:StickyCreditRouter._handle_router_message": "0fbe77fb29177ae21af763546c60f08383d02adc2a4c95056f7d4780866497f3",
    "aiperf.credit.sticky_router:StickyCreditRouter.wait_for_workers": "c4caa24397e021c49299899cef718e6038f2dc838c1217d68198a8cb2d0665ed",
    "aiperf.credit.sticky_router:StickyCreditRouter.mark_credits_complete": "65c1844f45499a4d94207e446f5fc58919c9ff8341295d9f0fe46134f913b119",
    "aiperf.credit.issuer:CreditIssuer._issue_credit_internal": "11e0184be4fc3af1ef5b4e5d71035f97623c05f0c8d1e9047e67967efdd13964",
    "aiperf.timing.phase_orchestrator:PhaseOrchestrator._start_orchestrator": "2f84c93d885c6e68b77d0de540043dc84e62ff996e5b587f39c8722b350b8b30",
    "aiperf.timing.manager:TimingManager.__init__": "fb1667460f69c2f48a8c04a0610cd64c59cf3ff48aba734f183f8c0d5885548a",
    # The time source: what reads the rebound ``time`` and ``uuid4``, the idle
    # watchdog the strategy moves onto the clock, the deadline wait
    # ClockPhaseRunner overrides, and the orchestrator that builds the runner.
    "aiperf.timing.phase.lifecycle:PhaseLifecycle.start": "019b5bfaa18a7e0cb59bd53867dbc6169c2601d0e052f4d59b56d819d7d2b254",
    "aiperf.timing.phase.lifecycle:PhaseLifecycle.mark_sending_complete": "b231e566784164c635580f796f650f800c0307430247a29e2a2640f60ae383a3",
    "aiperf.timing.phase.lifecycle:PhaseLifecycle.mark_complete": "4a3f74cbfddc4290b67c2bc05f3fd8844cb0655bee9dd228c5d02dc89f0e2b31",
    "aiperf.timing.phase.lifecycle:PhaseLifecycle.time_left_in_seconds": "7dfd8c609c21c196a12bbe322e86e0bf495a4f1f409262fd764c363e2fa54337",
    "aiperf.timing.strategies.agentic_replay:AgenticReplayStrategy.enforce_system_idle_cap": "410f76de8a195bb73cc6b1f920c27a48f8fb42f9c38e6ad7f6c45e20772ff22b",
    "aiperf.timing.strategies.agentic_replay:AgenticReplayStrategy._arm_system_idle_watchdog": "a63e64ba62d956ae784fa2aefb480f25dffec6d6cae10599b1157399d8e3f785",
    "aiperf.timing.strategies.agentic_replay:AgenticReplayStrategy._run_system_idle_watchdog": "be75be66b6069177145e672f4ffa7aa26252b95895bfd2c99faf8e49a11b7939",
    "aiperf.timing.strategies.agentic_replay:AgenticReplayStrategy._cancel_system_idle_watchdog": "272b852cce3db18c82e8a85d01588bb280ad9633fa0e88b68f180fc9fef53160",
    "aiperf.timing.phase.runner:PhaseRunner._wait_for_event_with_timeout": "02074d907a91cc440a678e5e8648f17d9a89c8cdad83f80e683204de8867c55b",
    "aiperf.timing.phase_orchestrator:PhaseOrchestrator._execute_phases": "7181bf33a2d4e6c440c2291390feb96a8c31f76bab258e9a3df633006f125abd",
    "aiperf.cli_runner:_make_benchmark_run": "f804f11878e985466855f13604390f7b68143da9f8fccb9f22255d67b884f1cc",
}

#: Call sites the router relies on being the only ones: every credit leaves
#: through one ``send_credit`` call, every return reaches aiperf through one
#: callback call.
SCANS = {".send_credit(": 1, "._on_return_callback(": 1}


def _digest(root: Path, name: str) -> str | None:
    module, _, qualname = name.partition(":")
    path = root.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():  # a package
        path = path.with_suffix("") / "__init__.py"
    if not path.is_file():
        return None
    text = path.read_text()
    node = ast.parse(text)
    for part in qualname.split("."):
        node = next(
            (n for n in getattr(node, "body", ()) if getattr(n, "name", None) == part),
            None,
        )
        if node is None:
            return None
    return hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()


def _calls(root: Path, pattern: str) -> int:
    return sum(
        line.count(pattern)
        for path in (root / "aiperf").rglob("*.py")
        for line in path.read_text().splitlines()
        if "def " not in line
    )


def changed(root: Path | None = None) -> list[str]:
    """The pinned functions whose source differs in the aiperf under `root`, and
    the scanned call sites whose count does."""
    if root is None:
        spec = importlib.util.find_spec("aiperf")
        if spec is None:
            raise RuntimeError("compass_harness needs aiperf, and it is not installed")
        root = Path(spec.origin).parent.parent
    return [name for name, sha in PINNED.items() if _digest(root, name) != sha] + [
        f"{n} {pattern} call sites, not {want}"
        for pattern, want in SCANS.items()
        if (n := _calls(root, pattern)) != want
    ]


def check(root: Path | None = None) -> None:
    """Raise naming every pinned aiperf function or call-site count that changed."""
    if names := changed(root):
        raise RuntimeError(
            "compass_harness was built against other aiperf source; these "
            "functions changed: " + ", ".join(names)
        )
